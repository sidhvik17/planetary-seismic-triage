"""Operating-point sweep over the cached archive curves (B3).

    python scripts/sweep_archive.py --tag screened --stations S12 S15 S16

The benchmark-tuned operating point is tuned for the Grade-A selection
function — the largest, clearest moonquakes — so on the full archive it is
far too conservative to be the whole story. Prior work reports detector
performance as a curve, and a single point invites the obvious objection
that it was chosen after seeing the answer. This sweeps (threshold,
min-duration) over the cached stitched curves, so no GPU pass is needed per
point, and reports recall against the full Nakamura catalog versus unmatched
detections per station-day.

Requires `scan_archive.py --save-curves` to have run for the same --tag.
Writes results/archive_scan/{tag}/sweep.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.detect_spec import curve_to_detections
from planetseis.config import PROJECT_ROOT
from scripts.score_archive_vs_nakamura import load_catalog

SCAN_ROOT = PROJECT_ROOT / "results" / "archive_scan"
DAY_SEC = 86400.0

THR_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
DUR_GRID = [120.0, 240.0, 600.0]


def load_curves(path: Path):
    z = np.load(path, allow_pickle=False)
    return (z["curve"].astype(np.float32), z["valid_pct"], z["offsets"],
            z["seg_start"], z["day_start"], z["loc"], float(z["sec_per_bin"]))


def detections_at(curve, valid_pct, seg_start, day_start, spb,
                  thr, min_dur, coda, min_valid):
    """Detections for one day at one operating point, gated exactly as the
    scan gates them: attributed to a single day, and rejected where the data
    around them was not genuinely observed."""
    out = []
    min_bins = max(1, int(round(min_dur / spb)))
    for d in curve_to_detections(curve, thr, suppress_sec=coda,
                                 min_bins=min_bins):
        t_abs = seg_start + d.time_sec
        if not (day_start <= t_abs < day_start + DAY_SEC):
            continue
        b = int(round(d.time_sec / spb))
        lo, hi = max(b - 6, 0), min(b + min_bins, len(valid_pct))
        if hi <= lo or valid_pct[lo:hi].mean() / 100.0 < min_valid:
            continue
        out.append(t_abs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="screened")
    ap.add_argument("--stations", nargs="+", default=["S12", "S15", "S16"])
    ap.add_argument("--tol", type=float, default=300.0)
    ap.add_argument("--min-valid", type=float, default=0.9)
    ap.add_argument("--coda-sec", type=float, default=1800.0)
    ap.add_argument("--thr-grid", nargs="+", type=float, default=None)
    ap.add_argument("--dur-grid", nargs="+", type=float, default=None)
    args = ap.parse_args()

    global THR_GRID, DUR_GRID
    if args.thr_grid:
        THR_GRID = args.thr_grid
    if args.dur_grid:
        DUR_GRID = args.dur_grid

    scan_dir = SCAN_ROOT / args.tag
    cat = load_catalog()
    print(f"Nakamura catalog: {len(cat)} events")

    report = {"tag": args.tag, "tolerance_sec": args.tol,
              "min_valid": args.min_valid, "coda_sec": args.coda_sec,
              "stations": {}}

    for station in args.stations:
        p = scan_dir / f"curves_{station}.npz"
        if not p.exists():
            print(f"{station}: no curve cache — rerun scan_archive.py "
                  f"--save-curves --tag {args.tag}")
            continue
        curve, valid_pct, off, seg_start, day_start, loc, spb = load_curves(p)
        n_days = len(off) - 1
        station_days = float(valid_pct.mean() / 100.0 * n_days)
        t_lo, t_hi = day_start.min(), day_start.max() + DAY_SEC

        # observability is threshold-independent, so resolve it once
        ev_t, ev_type, ev_gain = [], [], []
        day_idx = {float(d): i for i, d in enumerate(day_start)}
        day_sorted = np.sort(day_start)
        for ev in cat:
            if station not in ev["stations"] or not (t_lo <= ev["epoch"] < t_hi):
                continue
            j = int(np.searchsorted(day_sorted, ev["epoch"], side="right") - 1)
            if j < 0:
                continue
            i = day_idx.get(float(day_sorted[j]))
            if i is None:
                continue
            b = int(round((ev["epoch"] - seg_start[i]) / spb))
            a, c = off[i] + max(b - 6, 0), off[i] + min(b + 60, off[i + 1] - off[i])
            if c <= a or valid_pct[a:c].mean() / 100.0 < args.min_valid:
                continue
            ev_t.append(ev["epoch"])
            ev_type.append(ev["etype"])
            ev_gain.append("peaked" if str(loc[i]) == "00" else "flat")
        ev_t = np.array(ev_t)
        print(f"\n=== {station}: {n_days} days, {station_days:.0f} valid "
              f"station-days, {len(ev_t)} observable catalog events ===")

        rows = []
        for min_dur in DUR_GRID:
            for thr in THR_GRID:
                dets = []
                for i in range(n_days):
                    dets += detections_at(
                        curve[off[i]:off[i + 1]], valid_pct[off[i]:off[i + 1]],
                        float(seg_start[i]), float(day_start[i]), spb,
                        thr, min_dur, args.coda_sec, args.min_valid)
                dets = np.sort(np.array(dets))
                if len(ev_t) and len(dets):
                    k = np.searchsorted(dets, ev_t)
                    best = np.full(len(ev_t), np.inf)
                    for shift in (-1, 0):
                        idx = np.clip(k + shift, 0, len(dets) - 1)
                        best = np.minimum(best, np.abs(dets[idx] - ev_t))
                    hit = best <= args.tol
                else:
                    hit = np.zeros(len(ev_t), dtype=bool)
                by_type = defaultdict(lambda: [0, 0])
                for h, t in zip(hit, ev_type):
                    by_type[t][0] += int(h)
                    by_type[t][1] += 1
                # a detection is "explained" if some catalog event matches it
                if len(dets) and len(ev_t):
                    k2 = np.searchsorted(ev_t, dets)
                    best2 = np.full(len(dets), np.inf)
                    for shift in (-1, 0):
                        idx = np.clip(k2 + shift, 0, len(ev_t) - 1)
                        best2 = np.minimum(best2, np.abs(ev_t[idx] - dets))
                    matched = int((best2 <= args.tol).sum())
                else:
                    matched = 0
                unmatched = len(dets) - matched
                rows.append({
                    "threshold": thr, "min_dur_sec": min_dur,
                    "detections": int(len(dets)),
                    "recall": round(float(hit.mean()), 4) if len(ev_t) else None,
                    "matched": matched,
                    "survey_precision": round(matched / len(dets), 4) if len(dets) else None,
                    "unmatched_per_station_day": round(unmatched / station_days, 4)
                    if station_days else None,
                    "recall_by_type": {k: {"found": v[0], "n": v[1],
                                           "recall": round(v[0] / v[1], 4)}
                                       for k, v in sorted(by_type.items())},
                })
                r = rows[-1]
                print(f"  thr {thr:.2f} dur {min_dur:4.0f}s -> "
                      f"{r['detections']:6d} dets  recall {r['recall']}  "
                      f"surv.P {r['survey_precision']}  "
                      f"{r['unmatched_per_station_day']}/day")
        report["stations"][station] = {
            "valid_station_days": round(station_days, 2),
            "observable_catalog_events": int(len(ev_t)),
            "events_by_type": {k: int(sum(1 for t in ev_type if t == k))
                               for k in sorted(set(ev_type))},
            "events_by_gain": {k: int(sum(1 for g in ev_gain if g == k))
                               for k in sorted(set(ev_gain))},
            "sweep": rows,
        }

    report["note"] = ("unmatched detections include real uncatalogued events "
                      "- 45% of the benchmark's false positives were genuine "
                      "Nakamura events - so unmatched_per_station_day is an "
                      "UPPER BOUND on the false-alarm rate, not the rate.")
    out = scan_dir / "sweep.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
