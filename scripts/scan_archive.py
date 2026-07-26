"""Scan the continuous Apollo archive with SpecUNet (B2/B3).

    python scripts/scan_archive.py --model runs/unet_lunar_screened/best.pt \
        --stations S12 S15 S16 --threshold 0.3 --min-dur 600

Turns the frozen benchmark's 19 test files into ~7,850 valid station-days.
That is the whole point of the archive pull: recall measured against
thousands of Nakamura events instead of 19, and a false-alarm rate expressed
per station-day, which curated single-event snippets cannot express at all.

Two rules keep the output honest:

* **Pad de-duplication.** Day files carry PAD_SEC of context each side so the
  detector's edge guard never blanks a real event near midnight. A detection
  is attributed to exactly one day — the file whose [day, day+24 h) span
  contains it — so the overlap cannot double-count.
* **Validity gating.** Long dropouts were interpolated for filter continuity
  but marked invalid at ingest. A detection is discarded unless the data
  around it is genuinely observed (`--min-valid`, default 0.9), and the same
  mask supplies the station-day denominator. Without this, invented signal
  would both manufacture detections and inflate the observed-time divisor.

Writes results/archive_scan/{station}.csv (one row per detection: UTC time,
confidence, gain state, local validity) plus a per-station summary json with
the valid-station-day totals the false-alarm rate is computed from. Re-running
skips stations already scanned unless --force.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import ARCHIVE_CACHE, PROJECT_ROOT
from planetseis.detect_spec import SEC_PER_BIN, compute_curve, curve_to_detections
from planetseis.unet import UNET_ARCHS, SpecUNet

OUT_DIR = PROJECT_ROOT / "results" / "archive_scan"
RATE = 6.625
DAY_SEC = 86400.0


def _iso(epoch: float) -> str:
    """UTC ISO string without dragging obspy into a CUDA-live process —
    mass UTCDateTime construction alongside a live context has segfaulted
    this machine before.

    Built by arithmetic from the epoch rather than datetime.fromtimestamp:
    Apollo runs 1969-1977, so roughly the first two months of S12 carry
    NEGATIVE Unix timestamps, which fromtimestamp rejects on Windows with
    OSError 22.
    """
    from datetime import datetime, timedelta, timezone
    return (datetime(1970, 1, 1, tzinfo=timezone.utc)
            + timedelta(seconds=float(epoch))).strftime("%Y-%m-%dT%H:%M:%S")


def _epoch_of(iso: str) -> float:
    from datetime import datetime, timezone
    s = str(iso).replace("Z", "").split(".")[0]
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp()


def scan_station(model, station, device, thr, min_dur, min_valid,
                 coda_sec, limit=None, save_curves=False):
    files = sorted((ARCHIVE_CACHE / station).glob("*.npz"))
    if limit:
        files = files[:limit]
    min_bins = max(1, int(round(min_dur / SEC_PER_BIN)))
    rows = []
    valid_samples = 0
    t_start = time.time()
    # Optional curve cache: the model forward is the entire cost of this scan,
    # so storing the stitched curves once turns every later (threshold,
    # duration) sweep into pure numpy. ~130 MB per station at float16, versus
    # ~15 minutes of GPU per re-scan — and a FAR-vs-recall curve needs many
    # operating points, not one.
    cur_chunks, cur_meta = ([], []) if save_curves else (None, None)

    for k, p in enumerate(files, 1):
        z = np.load(p)
        trace = z["trace"]
        valid = np.unpackbits(z["valid"])[: int(z["n_valid_bits"])].astype(bool)
        seg_start = _epoch_of(str(z["starttime"]))
        day_start = _epoch_of(str(z["day"]) + "T00:00:00")
        loc = str(z["loc"])

        # denominator: valid samples inside the day proper only, so the pad
        # overlap between neighbouring files is never counted twice
        d_lo = int(round((day_start - seg_start) * RATE))
        d_hi = d_lo + int(round(DAY_SEC * RATE))
        d_lo, d_hi = max(d_lo, 0), min(d_hi, len(valid))
        if d_hi > d_lo:
            valid_samples += int(valid[d_lo:d_hi].sum())

        curve = compute_curve(model, trace, device)
        if save_curves:
            # validity averaged onto the curve's bin grid, so a sweep can
            # apply the same observability gate without reloading day files
            nb = len(curve)
            spb = int(round(SEC_PER_BIN * RATE))
            pad = nb * spb - len(valid)
            v = np.pad(valid, (0, max(pad, 0)))[: nb * spb]
            cur_chunks.append(curve.astype(np.float16))
            cur_meta.append((seg_start, day_start, loc,
                             (v.reshape(nb, spb).mean(axis=1) * 100)
                             .astype(np.uint8)))
        for d in curve_to_detections(curve, thr, suppress_sec=coda_sec,
                                     min_bins=min_bins):
            t_abs = seg_start + d.time_sec
            if not (day_start <= t_abs < day_start + DAY_SEC):
                continue                      # belongs to a neighbouring file
            lo = int(round((d.time_sec - 60.0) * RATE))
            hi = int(round((d.time_sec + min_dur) * RATE))
            lo, hi = max(lo, 0), min(hi, len(valid))
            frac = float(valid[lo:hi].mean()) if hi > lo else 0.0
            if frac < min_valid:
                continue                      # sits in invented signal
            rows.append(dict(station=station, time_utc=_iso(t_abs),
                             epoch=round(t_abs, 2),
                             confidence=round(d.confidence, 4),
                             loc=loc, valid_frac=round(frac, 4),
                             day=str(z["day"])))
        if k % 250 == 0 or k == len(files):
            el = time.time() - t_start
            print(f"  {station} {k}/{len(files)} files, {len(rows)} dets, "
                  f"{el / k:.2f} s/file, eta {(len(files) - k) * el / k / 60:.0f} min")

    station_days = valid_samples / RATE / DAY_SEC
    if save_curves and cur_chunks:
        lens = np.array([len(c) for c in cur_chunks], dtype=np.int64)
        np.savez_compressed(
            OUT_DIR / f"curves_{station}.npz",
            curve=np.concatenate(cur_chunks),
            valid_pct=np.concatenate([m[3] for m in cur_meta]),
            offsets=np.concatenate(([0], np.cumsum(lens))),
            seg_start=np.array([m[0] for m in cur_meta]),
            day_start=np.array([m[1] for m in cur_meta]),
            loc=np.array([m[2] for m in cur_meta]),
            sec_per_bin=SEC_PER_BIN)
        print(f"  curve cache -> {OUT_DIR / f'curves_{station}.npz'}")
    return rows, station_days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--stations", nargs="+", default=["S12", "S15", "S16"])
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--min-dur", type=float, default=600.0)
    ap.add_argument("--min-valid", type=float, default=0.9,
                    help="min observed-data fraction around a detection")
    ap.add_argument("--coda-sec", type=float, default=1800.0,
                    help="dead time after a detection (lunar ring-down)")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N day files per station (smoke test)")
    ap.add_argument("--tag", default="",
                    help="output subdirectory, so two models can be scanned "
                         "and compared at archive scale")
    ap.add_argument("--save-curves", action="store_true",
                    help="cache stitched curves so operating-point sweeps "
                         "need no further GPU passes (~130 MB/station)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global OUT_DIR
    if args.tag:
        OUT_DIR = OUT_DIR / args.tag

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"model {args.model} on {device}; thr={args.threshold} "
          f"min_dur={args.min_dur}s min_valid={args.min_valid}")

    summary = {}
    for station in args.stations:
        out_csv = OUT_DIR / f"{station}.csv"
        if out_csv.exists() and not args.force:
            print(f"{station}: already scanned, skipping (--force to redo)")
            continue
        n_files = len(list((ARCHIVE_CACHE / station).glob("*.npz")))
        if not n_files:
            print(f"{station}: no day files under {ARCHIVE_CACHE} — skipped")
            continue
        print(f"\n=== {station}: {n_files} day files ===")
        rows, station_days = scan_station(
            model, station, device, args.threshold, args.min_dur,
            args.min_valid, args.coda_sec, args.limit, args.save_curves)
        with open(out_csv, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["station", "time_utc", "epoch",
                                               "confidence", "loc",
                                               "valid_frac", "day"])
            wr.writeheader()
            wr.writerows(rows)
        far = len(rows) / station_days if station_days else float("nan")
        summary[station] = {"detections": len(rows),
                            "valid_station_days": round(station_days, 2),
                            "detections_per_station_day": round(far, 4)}
        print(f"  -> {len(rows)} detections over {station_days:.1f} valid "
              f"station-days ({far:.3f}/day) -> {out_csv}")

    if summary:
        meta = {"model": str(args.model), "threshold": args.threshold,
                "min_dur_sec": args.min_dur, "min_valid": args.min_valid,
                "coda_sec": args.coda_sec, "stations": summary,
                "note": "detections_per_station_day is the RAW rate before "
                        "any Nakamura matching; it is an upper bound on the "
                        "false-alarm rate, since real uncatalogued events are "
                        "counted here too"}
        (OUT_DIR / "scan_summary.json").write_text(json.dumps(meta, indent=2))
        print(f"\n{json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
