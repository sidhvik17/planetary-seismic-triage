"""Validate benchmark 'false positives' against the full Nakamura catalog.

The frozen lunar benchmark scores against the Space Apps packet's Grade-A
labels — 76 events. The complete Apollo PSE long-period catalog (Nakamura et
al., 2008 revision; UTIG Technical Report 18) holds 13,058, most of which
the packet simply does not label. MarsQuakeNet's headline was exactly such
detections beyond the working catalog, verified manually; here the
verification is automatic: every U-Net detection on the lunar test split
that the benchmark counts as a false positive is cross-referenced against
Nakamura events flagged as detected at station 12. A match means the model
found a real, catalogued moonquake the benchmark cannot credit.

Reports benchmark precision alongside 'survey precision' (Nakamura-matched
detections counted as true). Match tolerance is generous-but-bounded: the
catalog's signal-start times are minute-quantized, so ±5 min.

Usage: python scripts/crosscheck_nakamura.py --model runs/unet_lunar/best.pt
           --threshold 0.3 --min-dur 600
Requires data/raw/levent.1008.dat. Writes results/nakamura_crosscheck.json
plus a per-detection CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import obspy
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PACKET_ROOT, PROJECT_ROOT
from planetseis.detect_spec import SEC_PER_BIN, compute_curve, curve_to_detections
from planetseis.evaluate import score_trace
from planetseis.unet import UNET_ARCHS, SpecUNet

MATCH_TOL_SEC = 300.0
COLSPECS = [(2, 4), (5, 8), (9, 13), (14, 18), (19, 23), (23, 27), (27, 31),
            (31, 35), (36, 40), (41, 45), (46, 76), (76, 77), (77, 80)]
NAMES = ["Year", "DOY", "StartTime", "StopTime", "A1112Amp", "A14Amp",
         "A15Amp", "A16Amp", "Availability", "Quality", "Comments",
         "EventType", "DeepClass"]


def load_nakamura() -> pd.DataFrame:
    df = pd.read_fwf(PROJECT_ROOT / "data" / "raw" / "levent.1008.dat",
                     colspecs=COLSPECS, header=None, names=NAMES)
    # station-12 column also covers station 11 (1969 only); benchmark files
    # are all 1970+ S12, so a nonzero amplitude there = detected at S12
    df = df[pd.to_numeric(df["A1112Amp"], errors="coerce").notna()]
    hhmm = df["StartTime"].astype(int)
    df = df.assign(
        utc=[obspy.UTCDateTime(year=1900 + int(y), julday=int(d))
             + (t // 100) * 3600 + (t % 100) * 60
             for y, d, t in zip(df["Year"], df["DOY"], hhmm)])
    return df


def trace_start_utc(stem: str) -> obspy.UTCDateTime | None:
    data_dir = PACKET_ROOT / "data" / "lunar" / "training" / "data"
    hits = list(data_dir.rglob(stem + ".mseed"))
    if not hits:
        return None
    return obspy.read(str(hits[0]), headonly=True)[0].stats.starttime


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--min-dur", type=float, required=True)
    ap.add_argument("--split", default="test")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    nak = load_nakamura()
    nak_times = np.array([float(t) for t in nak["utc"]])
    print(f"Nakamura S12-detected events: {len(nak)}")
    rng = np.random.default_rng(42)
    null_hits, null_draws = 0, 0

    tol = CFG.window.match_tolerance_sec
    min_bins = max(1, int(round(args.min_dur / SEC_PER_BIN)))
    rows = []
    n_tp = n_fp = n_fn = 0
    n_fp_nak = 0
    for p in sorted((DATA_CACHE / "lunar" / "continuous" / args.split).glob("*.npz")):
        z = np.load(p)
        trace, picks = z["trace"], list(z["picks"])
        t0 = trace_start_utc(p.stem)
        curve = compute_curve(model, trace, device)
        dets = curve_to_detections(curve, args.threshold,
                                   suppress_sec=CODA_SEC["lunar"],
                                   min_bins=min_bins)
        s = score_trace([d.time_sec for d in dets], picks, tol)
        n_tp += s.tp
        n_fp += s.fp
        n_fn += s.fn
        unmatched = list(picks)
        for d in dets:
            errs = [abs(d.time_sec - pk) for pk in unmatched]
            if errs and min(errs) <= tol:
                unmatched.pop(int(np.argmin(errs)))
                status = "benchmark_tp"
                nak_dt = None
            else:
                status = "benchmark_fp"
                nak_dt = None
                if t0 is not None:
                    abs_t = float(t0) + d.time_sec
                    k = int(np.argmin(np.abs(nak_times - abs_t)))
                    dt = abs(nak_times[k] - abs_t)
                    if dt <= MATCH_TOL_SEC:
                        status = "fp_matches_nakamura"
                        nak_dt = round(float(dt), 0)
                        n_fp_nak += 1
            rows.append(dict(file=p.stem, time_sec=round(d.time_sec, 1),
                             confidence=round(d.confidence, 3),
                             status=status, nakamura_dt_sec=nak_dt))
        print(f"{p.stem}: {len(dets)} dets, {s.tp} tp, {s.fp} fp "
              f"({sum(1 for r in rows if r['file'] == p.stem and r['status'] == 'fp_matches_nakamura')} nak-matched)")
        # empirical null: how often do random times in this file match?
        if t0 is not None and dets:
            span = len(trace) / 6.625
            rand_abs = float(t0) + rng.uniform(0, span, size=1000 * len(dets))
            d_near = np.abs(nak_times[None, :] - rand_abs[:, None]).min(axis=1)
            null_hits += int((d_near <= MATCH_TOL_SEC).sum())
            null_draws += len(rand_abs)

    survey_tp = n_tp + n_fp_nak
    result = {
        "model": Path(args.model).parent.name,
        "split": args.split,
        "threshold": args.threshold,
        "min_dur_sec": args.min_dur,
        "nakamura_match_tol_sec": MATCH_TOL_SEC,
        "benchmark": {"tp": n_tp, "fp": n_fp, "fn": n_fn,
                      "precision": round(n_tp / max(n_tp + n_fp, 1), 4)},
        "fp_matching_nakamura": n_fp_nak,
        "fp_nakamura_match_rate": round(n_fp_nak / max(n_fp, 1), 4),
        "chance_match_rate": round(null_hits / max(null_draws, 1), 4),
        "survey_precision": round(survey_tp / max(n_tp + n_fp, 1), 4),
        "note": "survey_precision counts detections matching ANY Nakamura "
                "S12 event as true; recall is not restated because the "
                "benchmark's pick list stays the recall denominator",
    }
    out_dir = PROJECT_ROOT / "results"
    pd.DataFrame(rows).to_csv(out_dir / "nakamura_crosscheck_detections.csv",
                              index=False)
    (out_dir / "nakamura_crosscheck.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
