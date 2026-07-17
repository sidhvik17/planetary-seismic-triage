"""Train the dual-head CNN on cached windows.

Usage: python -m planetseis.train --body lunar [--no-augment] [--tag ablation]
Outputs: runs/{body}[_{tag}]/best.pt, log.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DEFAULT as CFG, DATA_CACHE, RUNS_DIR, SEED
from .dataset import WindowDataset
from .model import ARCHS, SeisCNN, count_params


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(body: str, split: str):
    path = DATA_CACHE / body / f"{split}_windows.npz"
    if split == "val" and not path.exists():
        # Mars has too few labeled files for a real val split; fall back to
        # train windows for early stopping and say so loudly.
        print("WARNING: no val split — using train windows for early stopping")
        path = DATA_CACHE / body / "train_windows.npz"
    z = np.load(path)
    return z["X"], z["y"], z["offset"]


def run_epoch(model, loader, device, opt=None, lambda_reg=2.0, pos_weight=None):
    train = opt is not None
    model.train(train)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    reg = nn.SmoothL1Loss()
    tot, n = 0.0, 0
    correct = 0
    with torch.set_grad_enabled(train):
        for x, y, off in loader:
            x, y, off = x.to(device), y.to(device), off.to(device)
            logit, offset, _ = model(x)
            loss = bce(logit, y)
            mask = y > 0.5
            if mask.any():
                loss = loss + lambda_reg * reg(offset[mask], off[mask])
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(y)
            n += len(y)
            correct += ((torch.sigmoid(logit) > 0.5) == (y > 0.5)).sum().item()
    return tot / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True,
                    choices=["lunar", "mars", "lunar_dn"])
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--arch", default="base", choices=list(ARCHS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=CFG.train.epochs)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    augment = CFG.train.augment and not args.no_augment

    Xtr, ytr, otr = load_split(args.body, "train")
    Xva, yva, ova = load_split(args.body, "val")
    dl_tr = DataLoader(
        WindowDataset(Xtr, ytr, otr, augment=augment, seed=SEED),
        batch_size=CFG.train.batch_size, shuffle=True, num_workers=0, drop_last=False,
    )
    dl_va = DataLoader(
        WindowDataset(Xva, yva, ova, augment=False),
        batch_size=CFG.train.batch_size, shuffle=False,
    )

    n_pos = int(ytr.sum())
    pos_weight = torch.tensor((len(ytr) - n_pos) / max(n_pos, 1), device=device)

    model = SeisCNN(channels=ARCHS[args.arch]).to(device)
    print(f"device={device} params={count_params(model)} "
          f"train={len(ytr)} ({n_pos} pos) val={len(yva)} augment={augment}")
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.train.lr,
                            weight_decay=CFG.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    run_name = args.body + (f"_{args.tag}" if args.tag else "")
    out = RUNS_DIR / run_name
    out.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"])
        for ep in range(1, args.epochs + 1):
            tl, ta = run_epoch(model, dl_tr, device, opt,
                               CFG.train.lambda_reg, pos_weight)
            vl, va = run_epoch(model, dl_va, device, None,
                               CFG.train.lambda_reg, pos_weight)
            sched.step()
            wr.writerow([ep, f"{tl:.4f}", f"{ta:.4f}", f"{vl:.4f}", f"{va:.4f}",
                         f"{opt.param_groups[0]['lr']:.2e}"])
            f.flush()
            marker = ""
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "config": vars(args),
                            "arch": args.arch,
                            "params": count_params(model)}, out / "best.pt")
                marker = " *"
            print(f"ep {ep:3d} train {tl:.4f}/{ta:.3f} val {vl:.4f}/{va:.3f}{marker}")
    print(f"done. best val loss {best_val:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
