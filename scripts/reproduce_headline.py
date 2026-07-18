"""One-command reproduction of the headline table from shipped checkpoints.

    python scripts/reproduce_headline.py

Runs both lunar detectors at their published operating points on the frozen
test split using the checkpoints committed under models/, prints the
headline rows, and exits nonzero if any number drifts from the published
values (tag v1.0-results-freeze). Requires the data cache
(data/cache/lunar) built per the README; no training, no tuning, no val
access. ~2 minutes on GPU, ~15 on CPU.
"""
from __future__ import annotations

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

EXPECTED = {  # frozen at v1.0-results-freeze
    "SeisCNN": dict(precision=0.5556, recall=0.5263, f1=0.5405),
    "SpecUNet": dict(precision=0.3548, recall=0.5789, f1=0.4400),
}


def load_test():
    d = DATA_CACHE / "lunar" / "continuous" / "test"
    files = sorted(d.glob("*.npz"))
    if len(files) != 19:
        sys.exit(f"expected 19 lunar test files in {d}, found {len(files)} — "
                 "build the cache first (README step 3)")
    return [(np.load(p)["trace"], float(np.load(p)["rate"]),
             list(np.load(p)["picks"])) for p in files]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = load_test()
    tol = CFG.window.match_tolerance_sec
    rows = {}

    ck = torch.load(PROJECT_ROOT / "models" / "lunar_best.pt",
                    map_location=device, weights_only=False)
    cnn = SeisCNN(channels=ARCHS[ck.get("arch", "base")])
    cnn.load_state_dict(ck["model"])
    s = Scores()
    for trace, rate, picks in test:
        dets, _, _ = detect_events(cnn, trace, rate, CFG, 0.99, device,
                                   suppress_sec=CODA_SEC["lunar"])
        s.merge(score_trace([d.time_sec for d in dets], picks, tol))
    rows["SeisCNN"] = s

    ck = torch.load(PROJECT_ROOT / "models" / "unet_lunar_best.pt",
                    map_location=device, weights_only=False)
    unet = SpecUNet(base=UNET_ARCHS[ck.get("arch", "base")])
    unet.load_state_dict(ck["model"])
    s = Scores()
    for trace, rate, picks in test:
        dets, _ = detect_events_spec(unet, trace, rate, CFG, 0.30, device,
                                     suppress_sec=CODA_SEC["lunar"],
                                     min_dur_sec=600.0)
        s.merge(score_trace([d.time_sec for d in dets], picks, tol))
    rows["SpecUNet"] = s

    print(f"\n{'model':10s} {'P':>7s} {'R':>7s} {'F1':>7s}   expected F1")
    ok = True
    for name, sc in rows.items():
        exp = EXPECTED[name]
        match = (round(sc.precision, 4) == exp["precision"]
                 and round(sc.recall, 4) == exp["recall"]
                 and round(sc.f1, 4) == exp["f1"])
        ok &= match
        print(f"{name:10s} {sc.precision:7.4f} {sc.recall:7.4f} "
              f"{sc.f1:7.4f}   {exp['f1']:.4f}  "
              f"{'OK' if match else 'MISMATCH'}")
    if not ok:
        sys.exit("headline table did NOT reproduce")
    print("\nheadline table reproduces (tag v1.0-results-freeze)")


if __name__ == "__main__":
    main()
