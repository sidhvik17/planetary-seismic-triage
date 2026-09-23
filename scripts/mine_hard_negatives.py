"""Bootstrap hard-negative mining (PRD F-2).

Run the current model over its own TRAIN continuous traces; windows that fire
confidently but overlap no catalogued event (nor its coda) are exactly the
noise transients the model confuses for events. Append them as negatives to
train_windows.npz (original backed up once as train_windows_orig.npz), then
retrain.

Usage: python scripts/mine_hard_negatives.py --body lunar
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE
from planetseis.model import SeisCNN
from planetseis.preprocessing import normalize_window


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True, choices=["lunar", "mars"])
    ap.add_argument("--prob-min", type=float, default=0.5)
    ap.add_argument("--cap", type=int, default=1500)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(f"runs/{args.body}/best.pt", map_location=device, weights_only=True)
    model = SeisCNN().to(device).eval()
    model.load_state_dict(ckpt["model"])

    n_win, hop = CFG.window.n_samples, CFG.window.hop
    coda = CODA_SEC[args.body]
    tol = CFG.window.match_tolerance_sec
    mined, mined_probs = [], []

    cont = DATA_CACHE / args.body / "continuous" / "train"
    for p in sorted(cont.glob("*.npz")):
        z = np.load(p)
        trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
        win_sec = n_win / rate
        starts = list(range(0, max(len(trace) - n_win + 1, 1), hop))
        with torch.no_grad():
            for b in range(0, len(starts), 256):
                chunk = starts[b : b + 256]
                x = torch.from_numpy(
                    np.stack([normalize_window(trace[s : s + n_win]) for s in chunk])
                ).unsqueeze(1).to(device)
                logit, _, _ = model(x)
                probs = torch.sigmoid(logit).cpu().numpy()
                for s, pr in zip(chunk, probs):
                    if pr < args.prob_min:
                        continue
                    s_sec = s / rate
                    # keep only windows clear of every event and its coda
                    if any(s_sec < pk + coda + tol and s_sec + win_sec > pk - tol
                           for pk in picks):
                        continue
                    mined.append(trace[s : s + n_win].copy())
                    mined_probs.append(pr)

    if not mined:
        print("no hard negatives found")
        return
    order = np.argsort(mined_probs)[::-1][: args.cap]
    hard = np.stack([mined[i] for i in order])
    print(f"mined {len(mined)} candidates, keeping top {len(hard)}")

    tw = DATA_CACHE / args.body / "train_windows.npz"
    orig = DATA_CACHE / args.body / "train_windows_orig.npz"
    if not orig.exists():
        shutil.copy(tw, orig)
    z = np.load(orig)  # always mine on top of the pristine set, never stack repeatedly
    X = np.concatenate([z["X"], hard])
    y = np.concatenate([z["y"], np.zeros(len(hard), dtype=np.int64)])
    off = np.concatenate([z["offset"], np.full(len(hard), -1.0, dtype=np.float32)])
    np.savez_compressed(tw, X=X, y=y, offset=off)
    print(f"train_windows.npz: {len(z['y'])} -> {len(y)} windows "
          f"({int(y.sum())} pos / {len(y) - int(y.sum())} neg)")


if __name__ == "__main__":
    main()
