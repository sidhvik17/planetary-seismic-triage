"""Train the dual-head CNN on cached windows.

Usage: python -m planetseis.train --body lunar [--no-augment] [--tag ablation]
Outputs: runs/{body}[_{tag}]/best.pt, log.csv
"""
from __future__ import annotations

import argparse
import csv
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import DEFAULT as CFG, DATA_CACHE, RUNS_DIR, SEED
from .dataset import WindowDataset
from .model import ARCHS, SeisCNN, count_params
from .training_state import (capture_rng, checkpoint_metadata, dataset_provenance,
                             prepare_log, prepare_run, restore_rng,
                             save_checkpoint, training_config)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(body: str, split: str, data_dir: str | Path | None = None):
    root = Path(data_dir) if data_dir is not None else DATA_CACHE / body
    path = root / f"{split}_windows.npz"
    if split == "val" and not path.exists() and data_dir is None:
        # Mars has too few labeled files for a real val split; fall back to
        # train windows for early stopping and say so loudly.
        print("WARNING: no val split — using train windows for early stopping")
        path = root / "train_windows.npz"
    if not path.is_file():
        raise ValueError(f"Missing {split} windows: {path}")
    with np.load(path) as z:
        X, y, offset = z["X"], z["y"], z["offset"]
    if not len(X) or not (len(X) == len(y) == len(offset)):
        raise ValueError(f"{split} windows must contain nonempty, matching X/y/offset arrays.")
    return X, y, offset


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
    ap.add_argument("--data-dir", type=Path, help="Versioned dataset root, with manifest.json and separate val data")
    ap.add_argument("--out-dir", type=Path, help="Fresh run directory; existing runs require --resume")
    ap.add_argument("--resume", action="store_true", help="Continue this run from last.pt")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    if args.epochs < 1:
        ap.error("--epochs must be positive")

    provenance = dataset_provenance(args.body, args.data_dir)
    config = training_config(args, asdict(CFG))
    run_name = args.body + (f"_{args.tag}" if args.tag else "")
    out = args.out_dir or RUNS_DIR / run_name
    saved = prepare_run(out, args.resume, config, provenance)

    set_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    augment = CFG.train.augment and not args.no_augment

    Xtr, ytr, otr = load_split(args.body, "train", args.data_dir)
    Xva, yva, ova = load_split(args.body, "val", args.data_dir)
    ds_tr = WindowDataset(Xtr, ytr, otr, augment=augment,
                          seed=args.seed if args.data_dir is not None else SEED)
    dl_tr = DataLoader(
        ds_tr,
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

    best_val = float("inf")
    completed_epoch = 0
    if saved is not None:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        best_val, completed_epoch = saved["best_val"], saved["epoch"]
        restore_rng(saved["rng"], ds_tr, device)
        print(f"resumed after epoch {completed_epoch} (best val {best_val:.4f})")
    metadata = checkpoint_metadata(args, config, provenance)
    log_path = out / "log.csv"
    fields = ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"]
    prepare_log(log_path, fields, completed_epoch)
    with open(log_path, "a", newline="") as f:
        wr = csv.writer(f)
        for ep in range(completed_epoch + 1, args.epochs + 1):
            tl, ta = run_epoch(model, dl_tr, device, opt,
                               CFG.train.lambda_reg, pos_weight)
            vl, va = run_epoch(model, dl_va, device, None,
                               CFG.train.lambda_reg, pos_weight)
            sched.step()
            wr.writerow([ep, f"{tl:.4f}", f"{ta:.4f}", f"{vl:.4f}", f"{va:.4f}",
                         f"{opt.param_groups[0]['lr']:.2e}"])
            f.flush()
            marker = ""
            improved = vl < best_val
            if improved:
                best_val = vl
                marker = " *"
            weights = {**metadata, "model": model.state_dict(), "arch": args.arch,
                       "params": count_params(model), "epoch": ep, "seed": args.seed,
                       "best_val": best_val}
            if improved:
                save_checkpoint(out / "best.pt", weights)
            save_checkpoint(out / "last.pt", {
                **weights, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": capture_rng(ds_tr),
            })
            print(f"ep {ep:3d} train {tl:.4f}/{ta:.3f} val {vl:.4f}/{va:.3f}{marker}")
    print(f"done. best val loss {best_val:.4f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
