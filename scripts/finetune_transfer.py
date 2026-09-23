"""Few-shot transfer: fine-tune the lunar model on the tiny Mars training set.

MANet (2025) and Civilini et al. (2021) both show cross-domain seismic
transfer succeeds through fine-tuning, not zero-shot application. This is the
matching experiment for our study: zero-shot lunar->Mars collapsed (F1 0);
does adapting on ONE labeled Martian file (16 windows) recover detection?

Protocol: init from runs/lunar/best.pt, low LR, few epochs, train on mars
train windows only; evaluate on the held-out mars test file (never seen).
Saves runs/lunar_ft_mars/best.pt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DEFAULT as CFG, RUNS_DIR, SEED
from planetseis.dataset import WindowDataset
from planetseis.model import ARCHS, SeisCNN, count_params
from planetseis.train import load_split, run_epoch, set_seed

EPOCHS, LR = 30, 1e-4


def main():
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(RUNS_DIR / "lunar" / "best.pt", map_location=device, weights_only=True)
    arch = ck.get("arch", "base")
    model = SeisCNN(channels=ARCHS[arch]).to(device)
    model.load_state_dict(ck["model"])

    X, y, off = load_split("mars", "train")
    print(f"fine-tuning lunar model ({count_params(model)} params) on "
          f"{len(y)} mars windows ({int(y.sum())} pos), lr={LR}")
    dl = DataLoader(WindowDataset(X, y, off, augment=True, seed=SEED),
                    batch_size=16, shuffle=True)
    n_pos = int(y.sum())
    pw = torch.tensor((len(y) - n_pos) / max(n_pos, 1), dtype=torch.float32, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    out = RUNS_DIR / "lunar_ft_mars"
    out.mkdir(parents=True, exist_ok=True)
    for ep in range(1, EPOCHS + 1):
        tl, ta = run_epoch(model, dl, device, opt, CFG.train.lambda_reg, pw)
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:2d} loss {tl:.4f} acc {ta:.3f}")
    torch.save({"model": model.state_dict(), "arch": arch,
                "config": {"body": "mars", "init": "lunar", "lr": LR, "epochs": EPOCHS},
                "params": count_params(model)}, out / "best.pt")
    print(f"saved -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
