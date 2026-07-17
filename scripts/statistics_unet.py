"""Statistical rigor for the U-Net result: CIs + paired delta-F1 vs SeisCNN.

Same protocol as statistics_rigor.py / the existing paired bootstrap:
resample lunar test FILES with replacement (10k draws), compute each
detector's F1 on the resampled set, report each CI and the CI of the
PAIRED difference (U-Net minus SeisCNN on identical draws — the difference
distribution absorbs file-level difficulty correlation).

Usage: python scripts/statistics_unet.py --unet-thr <thr> [--cnn-thr 0.99]
Writes results/statistics_unet.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect import detect_events
from planetseis.detect_spec import detect_events_spec
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN
from planetseis.unet import UNET_ARCHS, SpecUNet

N_BOOT = 10_000
TOL = CFG.window.match_tolerance_sec


def per_file(device, unet_thr, cnn_thr, min_dur):
    ck = torch.load(PROJECT_ROOT / "runs" / "unet_lunar" / "best.pt",
                    map_location=device, weights_only=False)
    unet = SpecUNet(base=UNET_ARCHS[ck.get("arch", "base")]).to(device)
    unet.load_state_dict(ck["model"])
    ck2 = torch.load(PROJECT_ROOT / "runs" / "lunar" / "best.pt",
                     map_location=device, weights_only=False)
    cnn = SeisCNN(channels=ARCHS[ck2.get("arch", "base")]).to(device)
    cnn.load_state_dict(ck2["model"])

    files = []
    for p in sorted((DATA_CACHE / "lunar" / "continuous" / "test").glob("*.npz")):
        z = np.load(p)
        trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
        du, _ = detect_events_spec(unet, trace, rate, CFG, unet_thr, device,
                                   suppress_sec=CODA_SEC["lunar"],
                                   min_dur_sec=min_dur)
        dc, _, _ = detect_events(cnn, trace, rate, CFG, cnn_thr, device,
                                 suppress_sec=CODA_SEC["lunar"])
        files.append({
            "unet": score_trace([d.time_sec for d in du], picks, TOL),
            "cnn": score_trace([d.time_sec for d in dc], picks, TOL),
        })
    return files


def f1_of(files, key):
    s = Scores()
    for f in files:
        s.merge(Scores(tp=f[key].tp, fp=f[key].fp, fn=f[key].fn))
    return s.f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet-thr", type=float, required=True)
    ap.add_argument("--cnn-thr", type=float, default=0.99)
    ap.add_argument("--min-dur", type=float, default=600.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = per_file(device, args.unet_thr, args.cnn_thr, args.min_dur)
    rng = np.random.default_rng(42)
    n = len(files)
    f1_u, f1_c, delta = [], [], []
    for _ in range(N_BOOT):
        draw = [files[i] for i in rng.integers(0, n, n)]
        u, c = f1_of(draw, "unet"), f1_of(draw, "cnn")
        f1_u.append(u)
        f1_c.append(c)
        delta.append(u - c)

    def ci(a):
        return [round(float(np.percentile(a, q)), 3) for q in (2.5, 97.5)]

    result = {
        "n_test_files": n,
        "unet_threshold": args.unet_thr,
        "unet_min_dur_sec": args.min_dur,
        "cnn_threshold": args.cnn_thr,
        "f1_unet": round(f1_of(files, "unet"), 4),
        "f1_cnn": round(f1_of(files, "cnn"), 4),
        "ci_unet": ci(f1_u),
        "ci_cnn": ci(f1_c),
        "delta_f1_unet_minus_cnn": round(float(np.mean(delta)), 4),
        "ci_delta": ci(delta),
        "p_delta_le_0": round(float(np.mean(np.array(delta) <= 0)), 4),
        "n_bootstrap": N_BOOT,
    }
    out = PROJECT_ROOT / "results" / "statistics_unet.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
