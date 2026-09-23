"""MC-Dropout uncertainty + calibration for the injection-trained SpecUNet.

The repo's deployment story (uncertainty-aware triage) was built on the
supervised SeisCNN: MC-Dropout sigma separated false alarms from true
events 5.9x. This ports the same analysis to the injection-trained model —
the question being whether a network that never saw a real labeled positive
is still usable for triage:

  1. sigma separation: MC-Dropout std of the detection-curve peak for
     benchmark TPs vs FPs on the lunar test split (10 passes).
  2. Nakamura-aware variant: FPs that match the full catalog counted with
     the TPs ('real events'), so uncatalogued-event contamination cannot
     mask a real separation.
  3. Confidence calibration: detection confidence binned vs empirical
     precision (against Grade-A and against Grade-A + Nakamura).

Usage: python scripts/uncertainty_unet.py --model runs/unet_lunar/best.pt
           --threshold 0.3 --min-dur 600
Writes results/uncertainty_unet.json. Corrected split (exclusive output):
       --data-dir data/cache/lunar_grouped_v1 --output results/<new>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect_spec import detect_events_spec_mc
from planetseis.unet import UNET_ARCHS, SpecUNet
from scripts.crosscheck_nakamura import trace_start_utc

NAK_TOL_SEC = 300.0


def load_nakamura_times() -> np.ndarray:
    times = []
    with open(PROJECT_ROOT / "data" / "raw" / "levent.1008.dat") as f:
        for line in f:
            try:
                float(line[19:23].strip())
                y = 1900 + int(line[2:4])
                doy = int(line[5:8])
                hhmm = int(line[9:13])
            except ValueError:
                continue
            t = (datetime(y, 1, 1, tzinfo=timezone.utc)
                 + timedelta(days=doy - 1, hours=hhmm // 100,
                             minutes=hhmm % 100))
            times.append(t.timestamp())
    return np.array(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--min-dur", type=float, required=True)
    ap.add_argument("--passes", type=int, default=10)
    ap.add_argument("--data-dir", type=Path,
                    help="versioned dataset root; requires --output")
    ap.add_argument("--output", type=Path,
                    help="result JSON (created exclusively with --data-dir)")
    args = ap.parse_args()
    root = args.data_dir or DATA_CACHE / "lunar"
    out = args.output or PROJECT_ROOT / "results" / "uncertainty_unet.json"
    manifest_sha, benchmark_id = None, "historical filename split"
    if args.data_dir is not None:
        if args.output is None:
            ap.error("--data-dir requires --output")
        if out.exists():
            sys.exit(f"refusing to overwrite {out}")
        raw_manifest = (root / "manifest.json").read_bytes()
        manifest_sha = hashlib.sha256(raw_manifest).hexdigest()
        benchmark_id = json.loads(raw_manifest)["benchmark_id"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])

    nak_times = load_nakamura_times()
    tol = CFG.window.match_tolerance_sec
    rows = []
    for p in sorted((root / "continuous" / "test").glob("*.npz")):
        z = np.load(p)
        trace, picks = z["trace"], list(z["picks"])
        t0 = (datetime.fromisoformat(str(z["start_time"]).replace("Z", "+00:00")).timestamp()
              if "start_time" in z.files else trace_start_utc(p.stem))
        dets, _, _ = detect_events_spec_mc(
            model, trace, 6.625, CFG, args.threshold, device,
            suppress_sec=CODA_SEC["lunar"], n_passes=args.passes,
            review_low_bar=1.0, min_dur_sec=args.min_dur)
        unmatched = list(picks)
        for d in dets:
            errs = [abs(d.time_sec - pk) for pk in unmatched]
            if errs and min(errs) <= tol:
                unmatched.pop(int(np.argmin(errs)))
                cls = "tp"
            else:
                cls = "fp"
                if t0 is not None and len(nak_times):
                    dt = np.abs(nak_times - (float(t0) + d.time_sec)).min()
                    if dt <= NAK_TOL_SEC:
                        cls = "fp_nakamura"
            rows.append((cls, d.confidence, d.uncertainty))
        print(f"{p.stem}: {len(dets)} dets")

    arr_cls = np.array([r[0] for r in rows])
    conf = np.array([r[1] for r in rows])
    sig = np.array([r[2] for r in rows])

    def med_sig(mask):
        return float(np.median(sig[mask])) if mask.any() else None

    s_tp = med_sig(arr_cls == "tp")
    s_fp = med_sig(np.isin(arr_cls, ["fp", "fp_nakamura"]))
    s_real = med_sig(np.isin(arr_cls, ["tp", "fp_nakamura"]))
    s_fp_clean = med_sig(arr_cls == "fp")

    # calibration: confidence bins vs empirical precision
    bins = [(0.3, 0.4), (0.4, 0.55), (0.55, 0.7), (0.7, 1.01)]
    calib = []
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        if not m.any():
            continue
        real = np.isin(arr_cls[m], ["tp", "fp_nakamura"])
        calib.append({
            "conf_bin": f"{lo}-{hi}",
            "n": int(m.sum()),
            "precision_grade_a": round(float((arr_cls[m] == "tp").mean()), 3),
            "precision_incl_nakamura": round(float(real.mean()), 3),
            "mean_conf": round(float(conf[m].mean()), 3),
        })

    result = {
        "model": Path(args.model).parent.name if args.data_dir is None else Path(args.model).name,
        "benchmark_id": benchmark_id,
        "data_manifest_sha256": manifest_sha,
        "model_sha256": hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        "threshold": args.threshold,
        "min_dur_sec": args.min_dur,
        "mc_passes": args.passes,
        "n_detections": len(rows),
        "counts": {c: int((arr_cls == c).sum())
                   for c in ("tp", "fp_nakamura", "fp")},
        "median_sigma": {
            "tp": s_tp,
            "fp_all_unmatched": med_sig(np.isin(arr_cls, ["fp", "fp_nakamura"])),
            "fp_excluding_nakamura": s_fp_clean,
            "real_events_tp_plus_nakamura": s_real,
        },
        "sigma_separation_fp_over_tp":
            round(s_fp / s_tp, 2) if s_tp and s_fp else None,
        "sigma_separation_catalog_unmatched_fp_over_benchmark_tp":
            round(s_fp_clean / s_tp, 2) if s_tp and s_fp_clean else None,
        "sigma_separation_cleanfp_over_real":
            round(s_fp_clean / s_real, 2) if s_real and s_fp_clean else None,
        "calibration": calib,
    }
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
