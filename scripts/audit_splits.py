"""Audit frozen splits for repeated acquisition spans without changing data.

Examples (run from the repository root):
    python scripts/audit_splits.py
    python scripts/audit_splits.py --cache data/cache/lunar/continuous \
        --output results/split_integrity_audit.json

Exit 1 means acquisition names overlap across splits, even if local waveforms
are unavailable. Exact cache-array comparisons can confirm those candidates.
This targeted check does not prove the absence of other forms of leakage.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
ACQUISITION = re.compile(
    r"^(?P<stem>.+\.(?P<time>\d{4}-\d{2}-\d{2}HR\d{2}))_evid[^.]+$"
)


def trace_stem(filename: str) -> str:
    """Accept packet filenames and extensionless names in extended manifests."""
    path = Path(filename)
    return path.stem if path.suffix.lower() in {".mseed", ".sac", ".csv", ".npz"} else filename


def acquisition_stem(filename: str) -> str | None:
    """Recognize packet names with an explicit acquisition date and hour."""
    match = ACQUISITION.fullmatch(trace_stem(filename))
    if match is None:
        # Extended Mars names such as XB.ELYSE.02.BHV.S0133a do not encode
        # acquisition time; guessing a grouping for them would be unsafe.
        return None
    try:
        datetime.strptime(match["time"], "%Y-%m-%dHR%H")
    except ValueError:
        return None
    return match["stem"]


def _waveform(cache: Path, entry: dict):
    import numpy as np

    relative = Path(entry["split"]) / (trace_stem(entry["filename"]) + ".npz")
    with np.load(cache / relative, allow_pickle=False) as data:
        trace = data["trace"]
        rate = float(data["rate"])
        if trace.ndim != 1 or not np.isfinite(trace).all() or rate <= 0 or not np.isfinite(rate):
            raise ValueError(f"invalid waveform in {relative.as_posix()}")
        evidence = {
            "cache_file": relative.as_posix(),
            "n_samples": len(trace),
            "rate_hz": rate,
            "dtype": str(trace.dtype),
            "trace_sha256": hashlib.sha256(trace.tobytes()).hexdigest(),
        }
    return trace, evidence


def compare_candidate(cache: Path | None, left: dict, right: dict) -> dict:
    comparison = {"left": left, "right": right, "status": "unverified"}
    if cache is None:
        comparison["reason"] = "cache comparison not requested"
        return comparison
    import numpy as np

    try:
        left_trace, left_info = _waveform(cache, left)
        right_trace, right_info = _waveform(cache, right)
    except FileNotFoundError:
        comparison["reason"] = "one or both cached waveforms are unavailable"
        return comparison
    arrays_equal = bool(np.array_equal(left_trace, right_trace))
    rates_equal = left_info["rate_hz"] == right_info["rate_hz"]
    comparison.update(
        status="confirmed_identical" if arrays_equal and rates_equal else "not_identical",
        arrays_equal=arrays_equal,
        rates_equal=rates_equal,
        waveforms=[left_info, right_info],
    )
    return comparison


def audit_manifest(manifest: dict, cache: Path | None = None) -> dict:
    """Return name-level candidates and optional exact waveform evidence."""
    grouped: dict[str, list[dict]] = {}
    unchecked = []
    counts = {}
    for split in SPLITS:
        filenames = manifest.get(split, [])
        if not isinstance(filenames, list):
            raise ValueError(f"{split} must be a list of filenames")
        counts[split] = len(filenames)
        for filename in filenames:
            if not isinstance(filename, str) or "/" in filename or "\\" in filename:
                raise ValueError(f"{split} must contain plain filenames")
            entry = {"split": split, "filename": filename}
            stem = acquisition_stem(filename)
            if stem is None:
                unchecked.append(entry)
            else:
                grouped.setdefault(stem, []).append(entry)

    duplicates = []
    for stem, entries in sorted(grouped.items()):
        if len(entries) < 2:
            continue
        comparisons = [
            compare_candidate(cache, left, right)
            for left, right in itertools.combinations(entries, 2)
            if left["split"] != right["split"]
        ]
        duplicates.append({
            "acquisition_stem": stem,
            "entries": entries,
            "cross_split": bool(comparisons),
            "comparisons": comparisons,
        })

    overlaps = sum(group["cross_split"] for group in duplicates)
    confirmed = sum(
        comparison["status"] == "confirmed_identical"
        for group in duplicates for comparison in group["comparisons"]
    )
    return {
        "schema_version": 1,
        "method": "group explicit acquisition timestamps before _evid; optionally compare cached trace arrays and sample rates",
        "limitations": [
            "Filename overlap is a candidate, not waveform proof; confirmed_identical requires exact array and sample-rate equality.",
            "This does not detect differently named, partially overlapping, or shifted acquisition spans.",
            "Unchecked filenames require a separate metadata-based audit; no candidates is not proof of independence.",
        ],
        "summary": {
            "files_by_split": counts,
            "timestamped_acquisitions": len(grouped),
            "unchecked_filenames": len(unchecked),
            "duplicate_acquisitions": len(duplicates),
            "cross_split_acquisitions": overlaps,
            "confirmed_identical_cross_split_pairs": confirmed,
            "status": "overlap_detected" if overlaps else "no_candidates_detected",
        },
        "duplicate_acquisitions": duplicates,
        "unchecked_filenames": unchecked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "benchmark/lunar_splits.json")
    parser.add_argument("--cache", type=Path, help="optional continuous cache containing train/, val/, test/")
    parser.add_argument("--output", type=Path, help="also write the JSON report to this file")
    args = parser.parse_args(argv)
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or not any(split in manifest for split in SPLITS):
            raise ValueError("manifest must contain train, val, or test lists")
        report = audit_manifest(manifest, args.cache)
        # Keep checked-in evidence portable and independent of user directories.
        report["manifest"] = args.manifest.name
        encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.write_text(encoded, encoding="utf-8")
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f"split audit failed: {error}\n")
    print(encoded, end="")
    return int(report["summary"]["cross_split_acquisitions"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
