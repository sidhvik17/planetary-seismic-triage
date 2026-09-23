"""Audit whether stored Nakamura matches add labels or only widen timing tolerance.

    python scripts/analyze_catalog_tolerance.py

Uses stored detection CSVs, corrected cache UTC starts/picks and the local
catalog. No inference, training, threshold selection or raw-result mutation.
Detection CSV times are rounded to 0.1 s, so this is an attribution audit,
not a replacement for full-precision benchmark scoring.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT
from scripts.crosscheck_nakamura import load_nakamura

BENCHMARK_ID = "lunar_grouped_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def utc_text(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat().replace("+00:00", "Z")


def classify_catalog_match(detection_utc: float, catalog_utc, grade_a_utc,
                           catalog_tolerance_sec=300.0, label_identity_tolerance_sec=60.0):
    """Separate catalog membership from whether its onset is already labeled.

    Identity uses the catalog onset versus Grade-A onset, never detection
    versus Grade-A. A late detector arrival does not make a label new.
    """
    catalog = np.asarray(catalog_utc, dtype=float)
    labels = np.asarray(grade_a_utc, dtype=float)
    if catalog.size == 0:
        raise ValueError("catalog has no station-detected events")
    nearest = float(catalog[np.argmin(np.abs(catalog - detection_utc))])
    catalog_error = abs(nearest - detection_utc)
    label_error = float(np.min(np.abs(labels - nearest))) if labels.size else None
    detection_label_error = float(np.min(np.abs(labels - detection_utc))) if labels.size else None
    return {
        "detection_utc": utc_text(detection_utc),
        "nearest_catalog_utc": utc_text(nearest),
        "detection_to_catalog_sec": round(catalog_error, 6),
        "nearest_catalog_to_grade_a_sec": round(label_error, 6) if label_error is not None else None,
        "detection_to_grade_a_sec": round(detection_label_error, 6) if detection_label_error is not None else None,
        "catalog_match": catalog_error <= catalog_tolerance_sec,
        "catalog_entry_already_grade_a": label_error is not None and label_error <= label_identity_tolerance_sec,
        "catalog_entry_exact_grade_a_time": label_error is not None and label_error <= 1e-5,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/cache/lunar_grouped_v1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results/lunar_grouped_v1_catalog_tolerance_audit.json")
    args = ap.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        ap.error("seeds must be unique")
    manifest_path = args.data_dir / "manifest.json"
    manifest_sha = sha256(manifest_path)
    catalog_path = PROJECT_ROOT / "data/raw/levent.1008.dat"
    catalog_times = np.asarray([float(t) for t in load_nakamura()["utc"]])
    source_hashes = {"data_manifest": manifest_sha, "catalog": sha256(catalog_path)}
    spans = {}
    per_seed = []
    additional = {}

    for seed in args.seeds:
        tag = f"{BENCHMARK_ID}_seed{seed}"
        result_path = args.results_dir / f"nakamura_crosscheck_{tag}.json"
        csv_path = args.results_dir / f"nakamura_crosscheck_detections_{tag}.csv"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result["data_manifest_sha256"] != manifest_sha or result["benchmark_id"] != BENCHMARK_ID:
            raise ValueError(f"seed {seed}: catalog result uses a different dataset")
        source_hashes[result_path.name] = sha256(result_path)
        source_hashes[csv_path.name] = sha256(csv_path)
        rows = list(csv.DictReader(csv_path.open(encoding="utf-8", newline="")))
        false_rows = [r for r in rows if r["status"] != "benchmark_tp"]
        if len(false_rows) != result["benchmark"]["fp"]:
            raise ValueError(f"seed {seed}: CSV FP count differs from raw JSON")
        associations = []
        for row in false_rows:
            stem = row["file"]
            if stem not in spans:
                path = args.data_dir / "continuous/test" / f"{stem}.npz"
                with np.load(path, allow_pickle=False) as data:
                    start = datetime.fromisoformat(str(data["start_time"]).replace("Z", "+00:00")).timestamp()
                    spans[stem] = (start, start + np.asarray(data["picks"], dtype=float))
                source_hashes[f"continuous/test/{path.name}"] = sha256(path)
            start, picks = spans[stem]
            association = classify_catalog_match(start + float(row["time_sec"]), catalog_times, picks)
            association.update({"span": stem, "stored_time_sec": float(row["time_sec"]),
                                "stored_status": row["status"]})
            if association["catalog_match"] != (row["status"] == "fp_matches_nakamura"):
                raise ValueError(f"seed {seed} {stem}: rounded CSV cannot reproduce match classification")
            associations.append(association)
            if association["catalog_match"] and not association["catalog_entry_already_grade_a"]:
                event = additional.setdefault(association["nearest_catalog_utc"], {"seeds": [], "spans": []})
                event["seeds"].append(seed)
                event["spans"].append(stem)
        matched = [r for r in associations if r["catalog_match"]]
        labeled = sum(r["catalog_entry_already_grade_a"] for r in matched)
        if len(matched) != result["fp_matching_nakamura"]:
            raise ValueError(f"seed {seed}: match count differs from raw JSON")
        per_seed.append({
            "seed": seed, "benchmark_fp": len(false_rows), "catalog_matches": len(matched),
            "matches_to_existing_grade_a": labeled,
            "matches_to_exact_grade_a_timestamp": sum(r["catalog_entry_exact_grade_a_time"] for r in matched),
            "matches_to_catalog_entries_absent_from_grade_a": len(matched) - labeled,
            "match_count_by_tolerance": result["match_count_by_tolerance"],
            "associations": associations,
        })

    output = {
        "benchmark_id": BENCHMARK_ID, "data_manifest_sha256": manifest_sha,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "platform": platform.platform(), "command": " ".join(sys.argv)},
        "analysis_script_sha256": sha256(Path(__file__)), "source_sha256": source_hashes,
        "seeds": args.seeds, "benchmark_tolerance_sec": 120,
        "catalog_tolerance_sec": 300, "catalog_label_identity_tolerance_sec": 60,
        "csv_detection_time_resolution_sec": 0.1,
        "pooled_benchmark_fp": sum(r["benchmark_fp"] for r in per_seed),
        "pooled_catalog_matches": sum(r["catalog_matches"] for r in per_seed),
        "pooled_matches_to_existing_grade_a": sum(r["matches_to_existing_grade_a"] for r in per_seed),
        "pooled_matches_to_exact_grade_a_timestamp": sum(r["matches_to_exact_grade_a_timestamp"] for r in per_seed),
        "pooled_matches_to_catalog_entries_absent_from_grade_a": sum(r["matches_to_catalog_entries_absent_from_grade_a"] for r in per_seed),
        "n_unique_additional_catalog_timestamps": len(additional),
        "additional_catalog_candidates": [{"catalog_utc": k, "seeds": sorted(set(v["seeds"])),
                                            "spans": sorted(set(v["spans"]))}
                                           for k, v in sorted(additional.items())],
        "interpretation": "Pooled counts are model-seed detection associations, not independent events. "
                          "Existing Grade-A associations chiefly measure wider arrival tolerance; "
                          "remaining catalog associations are unverified additional-event candidates. "
                          "The uniform random-placement null does not test discovery of new events.",
        "per_seed": per_seed,
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in output.items() if k not in ("source_sha256", "per_seed", "runtime")}, indent=2))


if __name__ == "__main__":
    main()
