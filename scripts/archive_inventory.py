"""Inventory of the continuous-archive cache: day files, valid station-days, gaps.

    python scripts/archive_inventory.py                 # reads PLANETSEIS_ARCHIVE
    python scripts/archive_inventory.py --stations S12 S15

Writes results/archive_scan/archive_inventory.json, the record behind the
archive size the paper quotes. Valid time is counted inside each file's own
day, the denominator the archive scan uses; the padded total is reported next
to it because the ingest manifests written before 2026-10-15 summed the
context pads and so counted every boundary hour twice. Only the small
validity members of each day file are read, never the trace.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.archive import DAY_SEC, day_valid_samples
from planetseis.config import ARCHIVE_CACHE, PROJECT_ROOT

RATE = 6.625
OUT = PROJECT_ROOT / "results" / "archive_scan" / "archive_inventory.json"


def _epoch(text: str) -> float:
    stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp()


def station_inventory(directory: Path) -> dict:
    files = sorted(directory.glob("*.npz"))
    day_samples = padded_samples = n_bytes = 0
    days = []
    for path in files:
        with np.load(path, allow_pickle=False) as z:
            valid = np.unpackbits(z["valid"])[: int(z["n_valid_bits"])].astype(bool)
            seg_start = _epoch(str(z["starttime"]))
            day = str(z["day"])
        padded_samples += int(valid.sum())
        day_samples += day_valid_samples(valid, seg_start, _epoch(day + "T00:00:00+00:00"), RATE)
        n_bytes += path.stat().st_size
        days.append(date.fromisoformat(day))
    gap = {"days": 0}
    for before, after in zip(days, days[1:]):
        missing = (after - before).days - 1
        if missing > gap["days"]:
            gap = {"days": missing, "years": round(missing / 365.25, 2),
                   "after": before.isoformat(), "before": after.isoformat()}
    return {"day_files": len(files),
            "first_day": days[0].isoformat() if days else None,
            "last_day": days[-1].isoformat() if days else None,
            "valid_station_days": round(day_samples / RATE / DAY_SEC, 2),
            "padded_valid_station_days": round(padded_samples / RATE / DAY_SEC, 2),
            "npz_bytes": n_bytes,
            "longest_gap": gap}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_CACHE)
    parser.add_argument("--stations", nargs="+", default=["S12", "S14", "S15", "S16"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    stations = {}
    for station in args.stations:
        stations[station] = station_inventory(args.archive / station)
        print(station, json.dumps(stations[station]), flush=True)
    totals = {key: round(sum(s[key] for s in stations.values()), 2)
              for key in ("day_files", "valid_station_days", "padded_valid_station_days", "npz_bytes")}
    record = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "archive": str(args.archive), "rate_hz": RATE,
              "definition": "valid_station_days counts validity inside each file's own UTC day; "
                            "padded_valid_station_days also counts the context pads and double-counts "
                            "every day boundary",
              "stations": stations, "totals": totals}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("totals", json.dumps(totals))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
