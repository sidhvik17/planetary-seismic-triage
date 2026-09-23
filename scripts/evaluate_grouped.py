"""Evaluate freshly trained models on the acquisition-grouped lunar benchmark.

Example:
  python scripts/evaluate_grouped.py --data-dir data/cache/lunar_grouped_v1 \
      --cnn runs/lunar_grouped_v1/best.pt --unet runs/unet_lunar_grouped_v1/best.pt \
      --output results/lunar_grouped_v1_seed42.json

The original detectors, scoring tolerance, validation grids and tie rules are
unchanged. Both models' validation operating points are written to an exclusive
``*.selection.json`` file BEFORE any test waveform is opened. Original results
are never overwritten. Each invocation is one preregistered held-out evaluation;
use a new output name only for a genuinely separate seed/experiment.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.baseline import sta_lta_detect
from planetseis.config import CODA_SEC, DEFAULT as CFG, PROJECT_ROOT
from planetseis.detect import _forward_windows, _window_starts, cluster_detections
from planetseis.detect_spec import SEC_PER_BIN, compute_curve, curve_to_detections
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN
from planetseis.unet import UNET_ARCHS, SpecUNet
from scripts.eval_unet import DUR_GRID, THR_GRID


BENCHMARK_ID = "lunar_grouped_v1"
CNN_GRID = [float(x) for x in np.arange(0.2, 0.95, 0.05)] + [
    0.95, 0.97, 0.98, 0.99, 0.995
]
STALTA_GRID = [2.0, 2.5, 3.0, 4.0, 5.0, 7.0]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_manifest(data_dir):
    """Check metadata and the file inventory without opening held-out arrays."""
    data_dir = Path(data_dir).resolve()
    path = data_dir / "manifest.json"
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError(f"expected benchmark_id={BENCHMARK_ID}")
    traces = manifest.get("traces", [])
    groups = manifest.get("groups", [])
    if not traces or not groups:
        raise ValueError("manifest needs nonempty traces and groups")
    by_id = {r["id"]: r for r in traces}
    by_group = {g["id"]: g for g in groups}
    if len(by_id) != len(traces) or len(by_group) != len(groups):
        raise ValueError("duplicate trace/group ID in manifest")

    source_owners, hash_splits, expected = {}, {}, {}
    for split in ("train", "val", "test"):
        ids = manifest.get("splits", {}).get(split, [])
        actual_ids = [r["id"] for r in traces if r["split"] == split]
        if not ids or len(ids) != len(set(ids)) or set(ids) != set(actual_ids):
            raise ValueError(f"{split}: empty or inconsistent split membership")
        expected[split] = set()
    for r in traces:
        split = r["split"]
        if split not in expected:
            raise ValueError(f"{r['id']}: unknown split")
        group = by_group.get(r["group_id"])
        if group is None or group["split"] != split:
            raise ValueError(f"{r['id']}: group/split mismatch")
        path = (data_dir / r["cache_file"]).resolve()
        if (path.parent != data_dir / "continuous" / split
                or path.suffix != ".npz" or path in expected[split]):
            raise ValueError(f"{r['id']}: invalid or reused cache_file")
        expected[split].add(path)
        start, end = _timestamp(r["start_time"]), _timestamp(r["end_time"])
        if start.tzinfo is None or end.tzinfo is None or end <= start:
            raise ValueError(f"{r['id']}: expected positive UTC acquisition interval")
        if r["n_samples"] < CFG.window.n_samples:
            raise ValueError(f"{r['id']}: trace too short for benchmark detector")
        if not np.isclose(r["rate_hz"], CFG.preproc.target_rate_hz):
            raise ValueError(f"{r['id']}: unexpected processed sample rate")
        picks = np.asarray(r["picks_rel_sec"], dtype=float)
        if (picks.ndim != 1 or not np.isfinite(picks).all()
                or np.any(np.diff(picks) <= 0) or np.any(picks < 0)
                or np.any(picks >= r["n_samples"] / r["rate_hz"])):
            raise ValueError(f"{r['id']}: invalid unioned picks")
        for source in r["source_files"]:
            if source in source_owners:
                raise ValueError(f"source file assigned to multiple traces: {source}")
            source_owners[source] = r["id"]
        digest = r["trace_sha256"]
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"{r['id']}: invalid trace SHA256")
        if digest in hash_splits and hash_splits[digest] != split:
            raise ValueError("identical waveform hash crosses train/val/test")
        hash_splits[digest] = split
    for group in groups:
        ids = group["trace_ids"]
        actual = {r["id"] for r in traces if r["group_id"] == group["id"]}
        if not ids or len(ids) != len(set(ids)) or set(ids) != actual:
            raise ValueError(f"{group['id']}: inconsistent group membership")
    for i, left in enumerate(traces):
        for right in traces[i + 1:]:
            if left["channel"] != right["channel"]:
                continue
            overlaps = (max(_timestamp(left["start_time"]), _timestamp(right["start_time"]))
                        < min(_timestamp(left["end_time"]), _timestamp(right["end_time"])))
            if overlaps and left["group_id"] != right["group_id"]:
                raise ValueError("overlapping acquisition intervals span different groups")
    for split, paths in expected.items():
        actual = {p.resolve() for p in (data_dir / "continuous" / split).glob("*.npz")}
        if actual != paths:
            raise ValueError(f"{split}: cache file inventory differs from manifest")
    return manifest, hashlib.sha256(raw).hexdigest()


def load_checkpoint(path, family, manifest_sha256):
    """Historical/warm-started checkpoints cannot qualify for this protocol."""
    path = Path(path)
    raw = path.read_bytes()
    # best.pt holds tensors and plain metadata only; never unpickle code.
    checkpoint = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if checkpoint.get("data_manifest_sha256") != manifest_sha256:
        raise ValueError(f"{path}: checkpoint/data manifest SHA256 mismatch")
    if checkpoint.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError(f"{path}: checkpoint benchmark_id mismatch")
    if checkpoint.get("initialized_from_scratch") is not True:
        raise ValueError(f"{path}: checkpoint must record initialization from scratch")
    config = checkpoint.get("config", {})
    if config.get("finetune_from") or config.get("hardneg"):
        raise ValueError(f"{path}: external warm-start/hard-negative provenance rejected")
    architecture = checkpoint.get("arch", "base")
    if family == "cnn":
        model = SeisCNN(channels=ARCHS[architecture])
    elif family == "unet":
        model = SpecUNet(base=UNET_ARCHS[architecture])
    else:
        raise ValueError(f"unknown model family {family}")
    model.load_state_dict(checkpoint["model"])
    model.eval()
    try:
        display_path = path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = str(path.resolve())
    provenance = {
        "path": display_path,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "data_manifest_sha256": manifest_sha256,
        "benchmark_id": BENCHMARK_ID,
        "initialized_from_scratch": True,
        "arch": architecture,
        "params": sum(p.numel() for p in model.parameters()),
        "epoch": checkpoint.get("epoch"),
        "seed": config.get("seed"),
        "training_config": config,
    }
    return model, provenance


def load_trace(data_dir, record):
    """Validate the actual held-out payload immediately before inference."""
    with np.load(Path(data_dir) / record["cache_file"], allow_pickle=False) as z:
        trace = np.asarray(z["trace"])
        rate = float(z["rate"])
        picks = np.asarray(z["picks"], dtype=float)
    if (trace.dtype != np.float32 or trace.ndim != 1
            or len(trace) != record["n_samples"] or not np.isfinite(trace).all()):
        raise ValueError(f"{record['id']}: cached trace shape/dtype/values mismatch")
    if hashlib.sha256(np.ascontiguousarray(trace).tobytes()).hexdigest() != record["trace_sha256"]:
        raise ValueError(f"{record['id']}: cached waveform SHA256 mismatch")
    if not np.isfinite(rate) or rate != record["rate_hz"]:
        raise ValueError(f"{record['id']}: cached rate mismatch")
    if not np.array_equal(picks, np.asarray(record["picks_rel_sec"], dtype=float)):
        raise ValueError(f"{record['id']}: cached picks differ from manifest union")
    return trace, rate, list(picks)


@torch.no_grad()
def cnn_outputs(model, trace, rate, device):
    """Same forward/normalization as detect_events; cache once for the sweep."""
    model.eval().to(device)
    trace, starts = _window_starts(trace, CFG.window.n_samples, CFG.window.hop)
    probs, offsets = _forward_windows(model, trace, starts, CFG.window.n_samples,
                                     device, batch_size=64)
    return {"probs": probs, "offsets": offsets,
            "start_secs": np.asarray(starts) / rate,
            "win_sec": CFG.window.n_samples / rate}


def infer_split(data_dir, manifest, split, models, device, baseline_thresholds):
    cached = []
    records = sorted((r for r in manifest["traces"] if r["split"] == split),
                     key=lambda r: r["id"])
    for index, record in enumerate(records, 1):
        print(f"{split}: {index}/{len(records)} {record['id']}", flush=True)
        trace, rate, picks = load_trace(data_dir, record)
        row = {"record": record, "picks": picks}
        if "cnn" in models:
            row["cnn"] = cnn_outputs(models["cnn"], trace, rate, device)
        if "unet" in models:
            row["unet"] = compute_curve(models["unet"], trace, device, batch_size=8)
        if baseline_thresholds:
            row["sta_lta"] = {}
            for threshold in baseline_thresholds:
                times = sta_lta_detect(trace, rate, thr_on=threshold,
                                       thr_off=max(1.2, threshold / 2))
                kept = []
                for t in sorted(times):
                    if not kept or t - kept[-1] >= CODA_SEC["lunar"]:
                        kept.append(t)
                row["sta_lta"][threshold] = kept
        cached.append(row)
    return cached


def score_outputs(outputs, family, operating_point):
    """Use the frozen scorer independently within each unique acquisition span."""
    total, per_trace = Scores(), []
    for row in outputs:
        if family == "cnn":
            c = row[family]
            detections = cluster_detections(
                c["probs"], c["offsets"], c["start_secs"], c["win_sec"],
                operating_point["threshold"], CODA_SEC["lunar"])
        elif family == "unet":
            min_bins = max(1, int(round(operating_point["min_dur_sec"] / SEC_PER_BIN)))
            detections = curve_to_detections(
                row[family], operating_point["threshold"],
                suppress_sec=CODA_SEC["lunar"], min_bins=min_bins)
        else:
            detections = None
        if detections is None:
            predictions = [{"time_sec": t} for t in row["sta_lta"][operating_point["thr_on"]]]
        else:
            predictions = [{"time_sec": float(d.time_sec), "confidence": float(d.confidence)}
                           for d in detections]
        scores = score_trace([p["time_sec"] for p in predictions], row["picks"],
                             CFG.window.match_tolerance_sec)
        total.merge(scores)
        record = row["record"]
        per_trace.append({"id": record["id"], "group_id": record["group_id"],
                          "source_files": record["source_files"],
                          "trace_sha256": record["trace_sha256"],
                          "picks_rel_sec": row["picks"],
                          "predictions": predictions, "scores": scores.as_dict()})
    return total, per_trace


def select_operating_point(outputs, family):
    """Exact historical validation grids/ties; no test argument is accepted."""
    if not outputs or any(row["record"]["split"] != "val" for row in outputs):
        raise ValueError("operating points may be selected only on validation traces")
    if family == "cnn":
        grid = [{"threshold": t} for t in CNN_GRID]
    elif family == "unet":
        grid = [{"threshold": t, "min_dur_sec": d} for d in DUR_GRID for t in THR_GRID]
    elif family == "sta_lta":
        grid = [{"thr_on": t} for t in STALTA_GRID]
    else:
        raise ValueError(f"unknown model family {family}")
    sweep = []
    for point in grid:
        scores, _ = score_outputs(outputs, family, point)
        sweep.append({**point, "f1_unrounded": scores.f1, "scores": scores.as_dict()})
    best_f1 = max(row["f1_unrounded"] for row in sweep)
    if family == "cnn":
        selected = min((r for r in sweep if r["f1_unrounded"] >= best_f1 - 0.02),
                       key=lambda r: r["threshold"])
    elif family == "unet":
        tied = sorted((r for r in sweep if r["f1_unrounded"] >= best_f1 - 0.02),
                      key=lambda r: (r["threshold"], r["min_dur_sec"]))
        selected = tied[len(tied) // 2]
    else:
        selected = next(r for r in sweep if r["f1_unrounded"] == best_f1)
    point = {key: selected[key] for key in grid[0]}
    scores, per_trace = score_outputs(outputs, family, point)
    return {"operating_point": point, "best_val_f1": best_f1,
            "selected_val_f1": scores.f1, "validation_scores": scores.as_dict(),
            "validation_per_trace": per_trace, "validation_sweep": sweep}


def write_new_json(path, value):
    """Exclusive creation prevents overwriting either historical or new results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def resolve_device(requested):
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if requested not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or cuda")
    return requested


def evaluate(data_dir, output, cnn=None, unet=None, device="auto", baseline=True):
    started = time.perf_counter()
    if not cnn and not unet:
        raise ValueError("provide at least one freshly trained checkpoint: --cnn or --unet")
    output = Path(output)
    selection_path = output.with_name(output.stem + ".selection.json")
    if output.exists() or selection_path.exists():
        raise FileExistsError("evaluation/selection already exists; refusing to retune or overwrite")
    manifest, manifest_hash = load_manifest(data_dir)
    models, provenance = {}, {}
    for family, path in (("cnn", cnn), ("unet", unet)):
        if path:
            models[family], provenance[family] = load_checkpoint(path, family, manifest_hash)
    device = resolve_device(device)
    code_files = ["scripts/evaluate_grouped.py", "scripts/eval_unet.py",
                  "planetseis/detect.py", "planetseis/detect_spec.py",
                  "planetseis/evaluate.py", "planetseis/baseline.py",
                  "planetseis/config.py", "planetseis/model.py", "planetseis/unet.py"]
    metadata = {
        "schema_version": 1, "benchmark_id": manifest["benchmark_id"],
        "data_manifest_sha256": manifest_hash,
        "checkpoints": provenance,
        "code_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in code_files},
        "protocol": {
            "tuned_on": "val", "reported_on": "test", "body": "lunar",
            "tolerance_sec": CFG.window.match_tolerance_sec,
            "suppress_sec": CODA_SEC["lunar"],
            "cnn_threshold_grid": CNN_GRID, "unet_threshold_grid": THR_GRID,
            "unet_min_dur_grid": DUR_GRID, "sta_lta_threshold_grid": STALTA_GRID,
            "near_tie_f1": 0.02,
            "cnn_tie_rule": "lowest threshold within 0.02 of maximum validation F1",
            "unet_tie_rule": "sorted(threshold,duration) upper median within 0.02 of maximum validation F1",
            "sta_lta_tie_rule": "first threshold at maximum validation F1",
            "cnn_batch_size": 64, "unet_batch_size": 8,
            "scoring_unit": "distinct acquisition span; grouped before splitting",
            "optional_arrival_refinement": False,
        },
        "hardware": {"device": device, "device_name": torch.cuda.get_device_name(0)
                     if device == "cuda" else platform.processor(),
                     "platform": platform.platform(), "logical_cpu_count": os.cpu_count(),
                     "python": platform.python_version(), "torch": torch.__version__,
                     "cuda_runtime": torch.version.cuda},
    }
    validation = infer_split(data_dir, manifest, "val", models, device,
                             STALTA_GRID if baseline else [])
    families = list(models) + (["sta_lta"] if baseline else [])
    selected = {family: select_operating_point(validation, family) for family in families}
    selection = {
        **metadata, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_trace_ids": [row["record"]["id"] for row in validation],
        "selected": selected, "validation_elapsed_sec": time.perf_counter() - started,
        "test_waveforms_opened": False,
    }
    write_new_json(selection_path, selection)
    selection_hash = sha256_file(selection_path)
    print(f"Locked validation selection -> {selection_path}", flush=True)
    del validation
    test_started = time.perf_counter()
    test = infer_split(data_dir, manifest, "test", models, device,
                       [selected["sta_lta"]["operating_point"]["thr_on"]] if baseline else [])
    results = {}
    for family in families:
        operating_point = selected[family]["operating_point"]
        scores, per_trace = score_outputs(test, family, operating_point)
        results[family] = {"operating_point": operating_point,
                           "selected_val_f1": selected[family]["selected_val_f1"],
                           "test_scores": scores.as_dict(), "test_per_trace": per_trace}
    result = {
        **metadata, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_file": selection_path.name, "selection_sha256": selection_hash,
        "test_trace_count": len(test),
        "test_group_count": len({row["record"]["group_id"] for row in test}),
        "results": results, "test_elapsed_sec": time.perf_counter() - test_started,
        "total_elapsed_sec": time.perf_counter() - started,
        "interpretation": "New acquisition-grouped split, not directly comparable to historical filename-split scores. Single-seed results do not establish seed stability or broader archive generalization.",
    }
    write_new_json(output, result)
    for family, result_row in results.items():
        print(f"{family}: {json.dumps(result_row['test_scores'])}", flush=True)
    print(f"Saved held-out results -> {output}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cnn", type=Path)
    parser.add_argument("--unet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args(argv)
    if not args.cnn and not args.unet:
        parser.error("at least one of --cnn or --unet is required")
    try:
        evaluate(args.data_dir, args.output, args.cnn, args.unet, args.device,
                 baseline=not args.skip_baseline)
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(2, f"evaluation stopped: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
