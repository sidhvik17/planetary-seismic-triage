"""Denoise-then-detect chain on lunar_grouped_v1 (corrected rerun of eval_chain.py).

    python scripts/chain_grouped.py --data-dir data/cache/lunar_grouped_v1 \
        --cnn runs/lunar_grouped_v1/best.pt --unet runs/unet_lunar_grouped_v1/best.pt

Three SeisCNN variants on every span:
  raw     SeisCNN on the preprocessed trace (must reproduce the locked evaluation)
  chain   SeisCNN on the SpecUNet-denoised trace
  fusion  per-window max of the raw and denoised probabilities; the arrival
          offset comes from the stream with the higher probability

Each variant uses the grouped evaluation's own validation grid, tie rule and
scorer. All three operating points are written to an exclusive
*.selection.json before any test waveform is opened. No model is trained and
no locked result is touched; outputs are new files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT
from planetseis.detect_spec import denoise_trace
from scripts.evaluate_grouped import (cnn_outputs, load_checkpoint, load_manifest, load_trace,
                                      resolve_device, score_outputs, select_operating_point,
                                      sha256_file, write_new_json)

VARIANTS = ("raw", "chain", "fusion")


def fuse(raw: dict, den: dict) -> dict:
    if len(raw["probs"]) != len(den["probs"]) or not np.array_equal(raw["start_secs"], den["start_secs"]):
        raise ValueError("raw and denoised window grids differ")
    stronger = raw["probs"] >= den["probs"]
    return {"probs": np.maximum(raw["probs"], den["probs"]),
            "offsets": np.where(stronger, raw["offsets"], den["offsets"]),
            "start_secs": raw["start_secs"], "win_sec": raw["win_sec"]}


def infer(data_dir, manifest, split, cnn, unet, device):
    """Per span: record, picks and the three variants' window outputs."""
    rows = {v: [] for v in VARIANTS}
    records = sorted((r for r in manifest["traces"] if r["split"] == split), key=lambda r: r["id"])
    for index, record in enumerate(records, 1):
        print(f"{split}: {index}/{len(records)} {record['id']}", flush=True)
        trace, rate, picks = load_trace(data_dir, record)
        with torch.no_grad():
            den = denoise_trace(unet, trace, rate, device)
        if len(den) != len(trace) or not np.isfinite(den).all():
            raise ValueError(f"{record['id']}: invalid denoised trace")
        raw = cnn_outputs(cnn, trace, rate, device)
        chain = cnn_outputs(cnn, den, rate, device)
        for variant, out in zip(VARIANTS, (raw, chain, fuse(raw, chain))):
            rows[variant].append({"record": record, "picks": picks, "cnn": out})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--cnn", type=Path, required=True)
    ap.add_argument("--unet", type=Path, required=True)
    ap.add_argument("--locked", type=Path,
                    default=PROJECT_ROOT / "results" / "lunar_grouped_v1_seed42_cnn.json",
                    help="locked SeisCNN evaluation the raw variant must reproduce")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / "chain_lunar_grouped_v1_seed42.json")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args(argv)
    selection_path = args.output.with_name(args.output.stem + ".selection.json")
    if args.output.exists() or selection_path.exists():
        sys.exit("result or selection already exists; refusing to retune or overwrite")

    manifest, manifest_hash = load_manifest(args.data_dir)
    cnn, cnn_prov = load_checkpoint(args.cnn, "cnn", manifest_hash)
    unet, unet_prov = load_checkpoint(args.unet, "unet", manifest_hash)
    locked = json.loads(args.locked.read_text(encoding="utf-8"))
    if locked["checkpoints"]["cnn"]["sha256"] != cnn_prov["sha256"]:
        raise ValueError("--locked evaluation scored a different SeisCNN checkpoint")
    device = resolve_device(args.device)
    unet.to(device)

    common = {
        "benchmark_id": manifest["benchmark_id"], "data_manifest_sha256": manifest_hash,
        "checkpoints": {"cnn": cnn_prov, "unet": unet_prov},
        "variants": {"raw": "SeisCNN on the preprocessed trace",
                     "chain": "SeisCNN on the SpecUNet-denoised trace",
                     "fusion": "per-window max of raw and denoised probabilities"},
        "selection_rule": "grouped SeisCNN grid and tie rule (evaluate_grouped.select_operating_point)",
        "code_sha256": {name: sha256_file(PROJECT_ROOT / name) for name in
                        ("scripts/chain_grouped.py", "scripts/evaluate_grouped.py",
                         "planetseis/detect_spec.py", "planetseis/evaluate.py")},
    }

    validation = infer(args.data_dir, manifest, "val", cnn, unet, device)
    selections = {v: select_operating_point(validation[v], "cnn") for v in VARIANTS}
    if selections["raw"]["operating_point"] != locked["results"]["cnn"]["operating_point"]:
        raise ValueError("raw variant does not reproduce the locked validation selection")
    write_new_json(selection_path, {
        **common, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selections": {v: {k: s[k] for k in ("operating_point", "best_val_f1", "selected_val_f1",
                                              "validation_scores")}
                       for v, s in selections.items()}})
    print(f"Locked validation selection -> {selection_path}", flush=True)

    test = infer(args.data_dir, manifest, "test", cnn, unet, device)
    results = {}
    for v in VARIANTS:
        point = selections[v]["operating_point"]
        scores, per_trace = score_outputs(test[v], "cnn", point)
        results[v] = {"operating_point": point, "test_scores": scores.as_dict(),
                      "selected_val_f1": selections[v]["selected_val_f1"],
                      "per_trace": per_trace}
        print(f"  {v:7s} thr={point['threshold']} {scores.as_dict()}", flush=True)
    if results["raw"]["test_scores"] != locked["results"]["cnn"]["test_scores"]:
        raise ValueError("raw variant does not reproduce the locked test scores")
    write_new_json(args.output, {
        **common, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_file": selection_path.name, "selection_sha256": sha256_file(selection_path),
        "results": results,
        "interpretation": "Single designated seed (42) per model. Raw reproduces the locked "
                          "evaluation; chain and fusion are new validation-selected variants."})
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
