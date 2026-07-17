"""Cross-validate Mars detections against MarsQuakeNet's published catalog.

The mars_ext benchmark scores against MQS v14 picks only. MQNet's released
detection list (Zenodo 10.5281/zenodo.7157843) additionally contains the
D-named events MQNet found beyond MQS plus every S-named MQS match, each as
a [utc_start, utc_end] interval. Scanning our frozen test spans in survey
mode (lower threshold than the benchmark operating point) and matching
every detection against both references answers two questions:

  1. does the injection-trained SpecUNet's survey mode stay clean on Mars
     (precision vs the union of MQS + MQNet), and
  2. does it recover detections that only MQNet — a model trained on 3
     Mars-years of data — reports?

Match rule: a detection's absolute UTC falls inside a MQNet interval padded
by ±120 s. Chance rate from uniform random times in the same spans.

Usage: python scripts/crosscheck_mqnet.py --model runs/unet_mars_ext/best.pt
           [--threshold 0.15] [--min-dur 120]
Writes results/mqnet_crosscheck.json + per-detection CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from obspy import UTCDateTime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect_spec import SEC_PER_BIN, compute_curve, curve_to_detections
from planetseis.evaluate import score_trace
from planetseis.unet import UNET_ARCHS, SpecUNet

PAD_SEC = 120.0


def load_mqnet_intervals() -> np.ndarray:
    df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "MQNet_alldetections.csv",
                     header=None,
                     names=["event_name", "utc_start", "utc_end", "family",
                            "sol", "value", "above", "lf_amp", "hf_amp",
                            "model", "comment"])
    lo = np.array([float(UTCDateTime(str(t))) for t in df["utc_start"]])
    hi = np.array([float(UTCDateTime(str(t))) for t in df["utc_end"]])
    names = df["event_name"].astype(str).to_numpy()
    return lo, hi, names


def file_abs_start(stem: str, picks: list[float]) -> float | None:
    """Absolute UTC of a mars_ext trace's first sample, reconstructed from
    the MQS arrival table (arrival_utc of the file's first pick minus its
    relative offset in the refetched trace)."""
    arr = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "mqs_arrivals.csv")
    rows = arr[arr.filename == stem]
    if not len(rows) or not picks:
        return None
    return float(UTCDateTime(str(rows.iloc[0]["arrival_utc"]))) - picks[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--min-dur", type=float, default=120.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    mq_lo, mq_hi, mq_names = load_mqnet_intervals()
    tol = CFG.window.match_tolerance_sec
    min_bins = max(1, int(round(args.min_dur / SEC_PER_BIN)))
    rng = np.random.default_rng(42)

    rows = []
    n_tp = n_fp = n_fn = n_fp_mqnet = 0
    null_hits = null_draws = 0
    for p in sorted((DATA_CACHE / "mars_ext" / "continuous" / "test").glob("*.npz")):
        z = np.load(p)
        trace, picks = z["trace"], list(z["picks"])
        t0 = file_abs_start(p.stem, picks)
        curve = compute_curve(model, trace, device)
        dets = curve_to_detections(curve, args.threshold,
                                   suppress_sec=CODA_SEC["mars_ext"],
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
                status, mq_name = "matches_mqs_pick", None
            else:
                status, mq_name = "unmatched", None
                if t0 is not None:
                    abs_t = t0 + d.time_sec
                    inside = (abs_t >= mq_lo - PAD_SEC) & (abs_t <= mq_hi + PAD_SEC)
                    if inside.any():
                        status = "matches_mqnet_catalog"
                        mq_name = str(mq_names[np.argmax(inside)])
                        n_fp_mqnet += 1
            rows.append(dict(file=p.stem, time_sec=round(d.time_sec, 1),
                             confidence=round(d.confidence, 3),
                             status=status, mqnet_event=mq_name))
        # chance null for this span
        if t0 is not None and dets:
            span = len(trace) / 6.625
            rand_abs = t0 + rng.uniform(0, span, size=1000 * len(dets))
            hit = ((rand_abs[:, None] >= mq_lo[None, :] - PAD_SEC)
                   & (rand_abs[:, None] <= mq_hi[None, :] + PAD_SEC)).any(axis=1)
            null_hits += int(hit.sum())
            null_draws += len(rand_abs)
        print(f"{p.stem}: {len(dets)} dets, {s.tp} mqs-tp, {s.fp} unmatched")

    survey_tp = n_tp + n_fp_mqnet
    result = {
        "model": Path(args.model).parent.name,
        "threshold": args.threshold,
        "min_dur_sec": args.min_dur,
        "mode": "survey (below benchmark operating point)",
        "vs_mqs_only": {"tp": n_tp, "fp": n_fp, "fn": n_fn,
                        "precision": round(n_tp / max(n_tp + n_fp, 1), 4),
                        "recall": round(n_tp / max(n_tp + n_fn, 1), 4)},
        "unmatched_matching_mqnet": n_fp_mqnet,
        "chance_match_rate": round(null_hits / max(null_draws, 1), 4),
        "survey_precision_vs_mqs_plus_mqnet":
            round(survey_tp / max(n_tp + n_fp, 1), 4),
        "reference": "MQNet detections: Zenodo 10.5281/zenodo.7157843",
    }
    out = PROJECT_ROOT / "results"
    pd.DataFrame(rows).to_csv(out / "mqnet_crosscheck_detections.csv",
                              index=False)
    (out / "mqnet_crosscheck.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
