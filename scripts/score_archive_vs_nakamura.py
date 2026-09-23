"""Score an archive scan against the full Nakamura catalog (B3).

    python scripts/score_archive_vs_nakamura.py --stations S12 S15 S16

This is the measurement the whole archive pull exists for. The frozen
benchmark scores 19 files against 76 Grade-A labels; here the denominator is
every Nakamura event the station actually detected, over thousands of valid
station-days, broken out by event type.

Three things make the numbers defensible rather than flattering:

* **Per-station denominators.** Nakamura records a per-station amplitude, so
  an event only enters a station's recall denominator if that station
  detected it (column 19:23 covers S11/S12, then S14, S15, S16).
* **Observability gating.** An event inside a dropout cannot be found, and
  counting it as a miss understates recall exactly as counting it as
  observable overstates it. Every catalog event is checked against the
  ingest validity mask at one-minute resolution and excluded from the
  denominator if the data around it was not genuinely observed.
* **Type and gain-state stratification.** Recall is reported per event class
  (deep moonquake, meteoroid impact, artificial impact, and the rest) and
  per gain state, because flat mode sits 5.6x closer to the digitiser LSB
  and low-SNR recall genuinely differs there.

Unmatched detections are written out as the B4 candidate list — they are NOT
called false alarms, because the benchmark already showed 45 % of the
SpecUNet's "false positives" were real catalogued events. The raw unmatched
rate is an upper bound on the false-alarm rate, stated as such.

Writes results/archive_scan/nakamura_score.json and candidates_{station}.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import ARCHIVE_CACHE, PROJECT_ROOT

CATALOG = PROJECT_ROOT / "data" / "raw" / "levent.1008.dat"
SCAN_DIR = PROJECT_ROOT / "results" / "archive_scan"
RATE, DAY_SEC = 6.625, 86400.0

# Nakamura per-station amplitude columns: a parseable number means that
# station detected the event. The first also covers S11 (1969 only).
AMP_COL = {"S12": (19, 23), "S14": (23, 27), "S15": (27, 31), "S16": (31, 35)}

# Event classification, established by inspection of the file itself
# (2026-07-25) rather than assumed from column names:
#
#   cols 81-84  deep-moonquake NEST id, 'A###' or 'T###'. 7,318 events carry
#               one across 320 distinct nests (top: A1 441, A8 327, A10 230),
#               i.e. 56% of the catalog — matching the published ">55% deep
#               moonquakes" and Nakamura (2003)'s ~7,245.
#   col 76      NOT an event class. Events sharing a single nest carry
#               different col-76 codes (M-coded and blank-coded events share
#               247 of 250 nests), so it encodes something else — quality or
#               identification route. Its 'A' code was previously read as
#               "artificial impact", which is provably wrong: only 9
#               artificial impacts exist in the literature, while col-76 'A'
#               appears 1,359 times and co-occurs with a deep-moonquake nest
#               1,327 of those times.
#
# So: nest presence decides deep moonquakes; 'H' (28 events, matching the
# published 28 shallow moonquakes exactly) decides shallow. The remaining
# codes are reported under their raw letter, NOT given invented names, until
# the UTIG TR-18 documentation settles them.
CODE_NAME = {"H": "shallow_moonquake", "C": "code_C", "Z": "code_Z",
             "X": "code_X", "S": "code_S", "L": "code_L", "M": "code_M",
             "A": "code_A"}


def load_catalog() -> list[dict]:
    """Plain fixed-width parse — pandas.read_fwf's pyarrow string backend has
    segfaulted this machine on this file."""
    out = []
    with open(CATALOG) as f:
        for line in f:
            try:
                y, doy, hhmm = 1900 + int(line[2:4]), int(line[5:8]), int(line[9:13])
            except ValueError:
                continue
            t = (datetime(y, 1, 1, tzinfo=timezone.utc)
                 + timedelta(days=doy - 1, hours=hhmm // 100, minutes=hhmm % 100))
            stations = set()
            for st, (a, b) in AMP_COL.items():
                try:
                    float(line[a:b].strip())
                    stations.add(st)
                except ValueError:
                    pass
            code = line[76:77].strip().upper() if len(line) > 76 else ""
            nest = line[81:85].strip() if len(line) > 84 else ""
            if nest:
                etype, nest_id = "deep_moonquake", nest.replace(" ", "")
            elif code == "H":
                etype, nest_id = "shallow_moonquake", ""
            elif code:
                etype, nest_id = CODE_NAME.get(code, f"code_{code}"), ""
            else:
                etype, nest_id = "unclassified", ""
            out.append(dict(epoch=t.timestamp(), stations=stations,
                            etype=etype, deep_nest=nest_id))
    return out


def validity_index(station: str, cache: Path):
    """Per-minute observed fraction for every scanned day, built once.

    Full masks are ~600 k bits per day; one-minute resolution is ample for
    asking 'was this catalogue event observable' and keeps the whole station
    in memory.
    """
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["day_epoch"], z["minutes"], (z["loc"] if "loc" in z else None)
    days, mins, locs = [], [], []
    files = sorted((ARCHIVE_CACHE / station).glob("*.npz"))
    for p in files:
        z = np.load(p)
        valid = np.unpackbits(z["valid"])[: int(z["n_valid_bits"])].astype(bool)
        seg_start = datetime.strptime(
            str(z["starttime"]).replace("Z", "").split(".")[0],
            "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp()
        day_start = datetime.strptime(str(z["day"]), "%Y-%m-%d").replace(
            tzinfo=timezone.utc).timestamp()
        lo = int(round((day_start - seg_start) * RATE))
        per_min = np.zeros(1440, dtype=np.float32)
        for m in range(1440):
            a = lo + int(round(m * 60 * RATE))
            b = a + int(round(60 * RATE))
            a2, b2 = max(a, 0), min(b, len(valid))
            per_min[m] = valid[a2:b2].mean() if b2 > a2 else 0.0
        days.append(day_start)
        mins.append(per_min)
        locs.append(str(z["loc"]))
    day_epoch = np.array(days)
    minutes = np.stack(mins) if mins else np.zeros((0, 1440), np.float32)
    loc_arr = np.array(locs)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, day_epoch=day_epoch, minutes=minutes, loc=loc_arr)
    return day_epoch, minutes, loc_arr


def observable(epoch, day_epoch, minutes, min_valid, window_min=10):
    """Was the data around `epoch` genuinely observed?"""
    if not len(day_epoch):
        return False, None
    i = int(np.searchsorted(day_epoch, epoch, side="right") - 1)
    if i < 0 or epoch >= day_epoch[i] + DAY_SEC:
        return False, None
    m = int((epoch - day_epoch[i]) // 60)
    lo, hi = max(m - 1, 0), min(m + window_min, 1440)
    return bool(minutes[i, lo:hi].mean() >= min_valid), i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", nargs="+", default=["S12", "S15", "S16"])
    ap.add_argument("--tol", type=float, default=300.0,
                    help="match tolerance (s); detector arrival error, not "
                         "catalogue coarseness, sets this")
    ap.add_argument("--min-valid", type=float, default=0.9)
    ap.add_argument("--tag", default="",
                    help="scan subdirectory written by scan_archive.py --tag")
    args = ap.parse_args()

    global SCAN_DIR
    if args.tag:
        SCAN_DIR = SCAN_DIR / args.tag

    cat = load_catalog()
    print(f"Nakamura catalog: {len(cat)} events")
    report = {"tolerance_sec": args.tol, "min_valid": args.min_valid,
              "stations": {}}

    for station in args.stations:
        scan_csv = SCAN_DIR / f"{station}.csv"
        if not scan_csv.exists():
            print(f"{station}: no scan csv — run scan_archive.py first")
            continue
        with open(scan_csv) as f:
            dets = [r for r in csv.DictReader(f)]
        det_t = np.array([float(r["epoch"]) for r in dets])
        order = np.argsort(det_t)
        det_t = det_t[order]
        dets = [dets[i] for i in order]
        det_loc = np.array([r["loc"] for r in dets])

        day_epoch, minutes, _ = validity_index(
            station, SCAN_DIR / f"validity_{station}.npz")
        if not len(day_epoch):
            print(f"{station}: no day files")
            continue
        t_lo, t_hi = day_epoch.min(), day_epoch.max() + DAY_SEC

        # --- recall, per type and per gain state -------------------------
        by_type = defaultdict(lambda: {"observable": 0, "found": 0})
        by_gain = defaultdict(lambda: {"observable": 0, "found": 0})
        n_excluded = 0
        matched_det = np.zeros(len(det_t), dtype=bool)
        for ev in cat:
            if station not in ev["stations"]:
                continue
            if not (t_lo <= ev["epoch"] < t_hi):
                continue
            ok, day_i = observable(ev["epoch"], day_epoch, minutes,
                                   args.min_valid)
            if not ok:
                n_excluded += 1
                continue
            gain = "peaked" if _loc_of(station, day_i) == "00" else "flat"
            hit = False
            if len(det_t):
                j = int(np.argmin(np.abs(det_t - ev["epoch"])))
                if abs(det_t[j] - ev["epoch"]) <= args.tol:
                    hit = True
                    matched_det[j] = True
            for d in (by_type[ev["etype"]], by_gain[gain]):
                d["observable"] += 1
                d["found"] += int(hit)

        valid_days = float(minutes.mean(axis=1).sum())
        n_unmatched = int((~matched_det).sum())
        tot_obs = sum(v["observable"] for v in by_type.values())
        tot_found = sum(v["found"] for v in by_type.values())

        rows = [dict(dets[i], nakamura_match=int(matched_det[i]))
                for i in range(len(dets))]
        cand = [r for r in rows if not r["nakamura_match"]]
        with open(SCAN_DIR / f"candidates_{station}.csv", "w", newline="") as f:
            if cand:
                wr = csv.DictWriter(f, fieldnames=list(cand[0]))
                wr.writeheader()
                wr.writerows(cand)

        report["stations"][station] = {
            "valid_station_days": round(valid_days, 2),
            "detections": len(dets),
            "catalog_events_in_span": tot_obs,
            "catalog_events_excluded_unobservable": n_excluded,
            "recall_overall": round(tot_found / tot_obs, 4) if tot_obs else None,
            "matched_detections": int(matched_det.sum()),
            "survey_precision": round(float(matched_det.mean()), 4) if len(dets) else None,
            "unmatched_detections": n_unmatched,
            "unmatched_per_station_day": round(n_unmatched / valid_days, 4)
            if valid_days else None,
            "recall_by_type": {k: {**v, "recall": round(v["found"] / v["observable"], 4)
                                   if v["observable"] else None}
                               for k, v in sorted(by_type.items())},
            "recall_by_gain_state": {k: {**v, "recall": round(v["found"] / v["observable"], 4)
                                         if v["observable"] else None}
                                     for k, v in sorted(by_gain.items())},
        }
        s = report["stations"][station]
        print(f"\n=== {station} ===")
        print(f"  {valid_days:.0f} valid station-days, {len(dets)} detections")
        print(f"  recall {s['recall_overall']} over {tot_obs} observable "
              f"catalog events ({n_excluded} excluded as unobservable)")
        print(f"  survey precision {s['survey_precision']}, "
              f"{s['unmatched_per_station_day']} unmatched/station-day")
        for k, v in s["recall_by_type"].items():
            print(f"    {k:20s} {v['found']:5d}/{v['observable']:5d} = {v['recall']}")

    report["note"] = ("unmatched detections are candidates, not confirmed "
                      "false alarms — 45% of the benchmark's 'false positives' "
                      "were real catalogued events. unmatched_per_station_day "
                      "is an UPPER BOUND on the false-alarm rate.")
    out = SCAN_DIR / "nakamura_score.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nsaved -> {out}")


_LOC_CACHE: dict[str, np.ndarray] = {}


def _loc_of(station: str, day_i):
    if day_i is None:
        return "00"
    if station not in _LOC_CACHE:
        z = np.load(SCAN_DIR / f"validity_{station}.npz", allow_pickle=False)
        _LOC_CACHE[station] = z["loc"] if "loc" in z else np.array([])
    arr = _LOC_CACHE[station]
    return str(arr[day_i]) if day_i < len(arr) else "00"


if __name__ == "__main__":
    main()
