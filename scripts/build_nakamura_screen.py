"""Precompute Nakamura exclusion times for the injection noise pool (A1).

    python scripts/build_nakamura_screen.py --body lunar

`injection.NoisePool` harvests "event-free" noise by excluding a guard zone
around every catalog pick — but the benchmark's picks are the packet's 76
Grade-A labels, and those are a strict subset of the Nakamura catalog (audit
2026-07-25: 75/75 checked picks match a Nakamura S12 event within +/-60 s).
The remaining ~5,300 S12-detected Nakamura events are invisible to that
guard, so their energy is harvested as noise and trained against a zero mask.

Measured contamination on the lunar train split: **2.0 M of 24.0 M eligible
window starts (8.4 %) fall inside the guard radius of a real catalogued
event.** That is textbook positive-unlabeled contamination — unlabeled data
treated as negative, a known fraction of which are hidden positives — and
teaching a detector to suppress real moonquakes is the one failure mode this
project has already documented (the first hard-negative attempt, where 23 of
64 mined "negatives" turned out to be genuine events).

This runs as a SEPARATE step, writing a plain-JSON cache, for a blunt
practical reason recorded in the project notes: mass obspy UTCDateTime
construction with a live CUDA context, and pandas.read_fwf's pyarrow string
backend, have both segfaulted this machine mid-session. Training therefore
never parses the catalog — it loads this cache.

Output: data/cache/{body}/nakamura_screen.json
  {stem: [event_time_sec, ...]}  event times relative to each trace's start.

Versioned datasets (--data-dir, e.g. data/cache/lunar_grouped_v1) use the
start_time stored in each cached span and write a NEW file beside the cache
(default <data-dir>_nakamura_screen.json), so the frozen cache is untouched:

    python scripts/build_nakamura_screen.py --data-dir data/cache/lunar_grouped_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, PROJECT_ROOT
from planetseis.spectral import RATE_HZ

CATALOG = PROJECT_ROOT / "data" / "raw" / "levent.1008.dat"


def load_nakamura_times() -> np.ndarray:
    """S12-detected event epochs, via a plain fixed-width parse.

    Column 19:23 is the station-11/12 amplitude; a parseable number there
    means the event was detected at S12 (the column covers S11 in 1969 only,
    and every benchmark file is 1970+ S12).
    """
    times: list[float] = []
    with open(CATALOG) as f:
        for line in f:
            try:
                float(line[19:23].strip())
            except ValueError:
                continue
            try:
                y, doy, hhmm = 1900 + int(line[2:4]), int(line[5:8]), int(line[9:13])
            except ValueError:
                continue
            t = (datetime(y, 1, 1, tzinfo=timezone.utc)
                 + timedelta(days=doy - 1, hours=hhmm // 100, minutes=hhmm % 100))
            times.append(t.timestamp())
    return np.array(sorted(times))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="lunar")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--data-dir", type=Path,
                    help="versioned dataset root; span start times come from its npz files")
    ap.add_argument("--output", type=Path,
                    help="new screen file (default <data-dir>_nakamura_screen.json)")
    args = ap.parse_args()
    if "test" in args.splits:
        ap.error("the screen shapes training noise; test spans must not be read")

    # imported here so obspy's UTCDateTime work stays in this process only
    from scripts.crosscheck_nakamura import trace_start_utc

    nak = load_nakamura_times()
    print(f"Nakamura S12-detected events: {len(nak)}")

    root = args.data_dir or DATA_CACHE / args.body
    screen: dict[str, list[float]] = {}
    n_hit = n_files = 0
    for split in args.splits:
        d = root / "continuous" / split
        for p in sorted(d.glob("*.npz")):
            z = np.load(p, allow_pickle=False)
            if args.data_dir is not None:
                t0 = datetime.fromisoformat(
                    str(z["start_time"]).replace("Z", "+00:00")).timestamp()
            else:
                t0 = trace_start_utc(p.stem)
            if t0 is None:
                print(f"  {p.stem}: no mseed header, skipped")
                continue
            n_files += 1
            span = len(z["trace"]) / RATE_HZ
            start = float(t0)
            inside = nak[(nak >= start - 3600) & (nak <= start + span + 3600)]
            rel = sorted(float(t - start) for t in inside)
            screen[p.stem] = rel
            n_hit += len(rel)
            print(f"  {split}/{p.stem}: {len(rel)} Nakamura events in span "
                  f"({span / 3600:.1f} h)")

    if args.data_dir is None:
        out = DATA_CACHE / args.body / "nakamura_screen.json"
        out.write_text(json.dumps(screen, indent=1))
    else:
        out = args.output or args.data_dir.parent / f"{args.data_dir.name}_nakamura_screen.json"
        with out.open("x", encoding="utf-8", newline="\n") as handle:  # never overwrite
            handle.write(json.dumps(screen, indent=1))
    print(f"\n{n_hit} event times across {n_files} files -> {out}")
    print("train with:  python scripts/train_unet.py --body lunar --screen-nakamura")


if __name__ == "__main__":
    main()
