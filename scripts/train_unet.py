"""Train the spectrogram U-Net on synthetic-injection data (MQNet protocol).

Usage:
  python scripts/train_unet.py --body lunar [--arch base] [--tag x] [--epochs 40]

Training samples are generated on the fly by planetseis.injection from the
TRAIN split only; validation loss uses a fixed, seeded injection set built
from the VAL split's files (never seen in training). Outputs
runs/unet_{body}[_{tag}]/best.pt + log.csv, same layout as planetseis.train.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, RUNS_DIR, SEED
from planetseis.injection import InjectionDataset
from planetseis.model import count_params
from planetseis.train import set_seed
from planetseis.training_state import (capture_rng, checkpoint_metadata, dataset_provenance,
                                      prepare_log, prepare_run, restore_rng,
                                      save_checkpoint, training_config)
from planetseis.unet import UNET_ARCHS, SpecUNet


def weighted_bce(logits, target, event_weight: float = 3.0):
    """BCE with per-pixel weights 1 + w*mask: event pixels are rare (masks are
    mostly zero), so they get emphasis — MQNet's optional weight map."""
    loss = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none")
    w = 1.0 + event_weight * target
    return (loss * w).sum() / w.sum()


def run_epoch(model, loader, device, opt=None, event_weight=3.0):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, m in loader:
            x, m = x.to(device), m.to(device)
            loss = weighted_bce(model(x), m, event_weight)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += loss.item() * len(x)
            n += len(x)
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", required=True,
                    choices=["lunar", "mars", "mars_ext"])
    ap.add_argument("--arch", default="base", choices=list(UNET_ARCHS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--epoch-len", type=int, default=6400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--event-weight", type=float, default=3.0)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--val-split", default="val",
                    help="split for the fixed validation injection set "
                         "(mars has no val files: falls back to train)")
    ap.add_argument("--finetune-from", default=None,
                    help="checkpoint to continue from (hard-negative loop)")
    ap.add_argument("--hardneg", default=None,
                    help="npz of mined hard-negative windows "
                         "(scripts/mine_unet_hardneg.py)")
    ap.add_argument("--p-hardneg", type=float, default=0.35)
    ap.add_argument("--screen-nakamura", action="store_true",
                    help="exclude noise windows near ANY catalogued Nakamura "
                         "event, not just the Grade-A picks "
                         "(scripts/build_nakamura_screen.py)")
    ap.add_argument("--data-dir", type=Path, help="Versioned dataset root with manifest.json and a held-out val split")
    ap.add_argument("--out-dir", type=Path, help="Fresh run directory; existing runs require --resume")
    ap.add_argument("--resume", action="store_true", help="Continue this run from last.pt")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    if min(args.epochs, args.epoch_len, args.batch_size, args.patience) < 1:
        ap.error("--epochs, --epoch-len, --batch-size and --patience must be positive")
    if args.data_dir is not None and args.val_split != "val":
        ap.error("An explicit dataset requires the separate val split (--val-split val)")
    provenance = dataset_provenance(args.body, args.data_dir)
    if provenance["benchmark_id"] == "lunar_grouped_v1" and (args.finetune_from or args.hardneg):
        ap.error("Corrected grouped training must start from scratch without legacy checkpoints or hard negatives")
    config = training_config(args, asdict(CFG))
    run_name = f"unet_{args.body}" + (f"_{args.tag}" if args.tag else "")
    out = args.out_dir or RUNS_DIR / run_name
    saved = prepare_run(out, args.resume, config, provenance)

    set_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    # loaded before any CUDA context exists: plain JSON, no obspy, no pandas
    screen = None
    if args.screen_nakamura:
        sp = (args.data_dir or DATA_CACHE / args.body) / "nakamura_screen.json"
        if not sp.exists():
            sys.exit(f"missing {sp} — run scripts/build_nakamura_screen.py "
                     f"--body {args.body} first")
        screen = json.loads(sp.read_text())
        print(f"Nakamura screen: {sum(len(v) for v in screen.values())} event "
              f"times across {len(screen)} files")

    hardneg = None
    if args.hardneg:
        hardneg = np.load(args.hardneg)["X"]
        print(f"hard negatives: {len(hardneg)} windows "
              f"(p_hardneg={args.p_hardneg})")
    ds_tr = InjectionDataset(args.body, "train", epoch_len=args.epoch_len,
                             p_hardneg=args.p_hardneg if hardneg is not None else 0.0,
                             hardneg_windows=hardneg,
                             seed=args.seed if args.data_dir is not None else None,
                             screen=screen, data_dir=args.data_dir)
    if screen is not None:
        print(f"screened out {ds_tr.pool.n_screened:,} noise window starts "
              f"beyond the Grade-A guard")
    val_split = args.val_split
    try:
        ds_va = InjectionDataset(args.body, val_split, epoch_len=1280,
                                 seed=args.seed, screen=screen, data_dir=args.data_dir)
    except RuntimeError:
        if args.data_dir is not None:
            raise
        print(f"WARNING: no {val_split} files — validating on train-split "
              "injections (mars fallback)")
        ds_va = InjectionDataset(args.body, "train", epoch_len=1280,
                                 seed=args.seed + 1, screen=screen, data_dir=args.data_dir)
    # Windows + CUDA parent: spawned workers have repeatedly triggered CUDA
    # 'unknown error' crashes here; generation is 7 ms/sample so the main
    # process keeps the GPU >90% busy anyway.
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, num_workers=0)

    model = SpecUNet(base=UNET_ARCHS[args.arch]).to(device)
    if args.finetune_from and saved is None:
        ck = torch.load(args.finetune_from, map_location=device,
                        weights_only=True)
        model.load_state_dict(ck["model"])
        print(f"fine-tuning from {args.finetune_from}")
    print(f"device={device} arch={args.arch} params={count_params(model):,} "
          f"templates={len(ds_tr.bank)} epoch_len={args.epoch_len}")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val, since_best = float("inf"), 0
    completed_epoch = 0
    if saved is not None:
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        sched.load_state_dict(saved["scheduler"])
        best_val = saved["best_val"]
        since_best, completed_epoch = saved["since_best"], saved["epoch"]
        restore_rng(saved["rng"], device=device)
        print(f"resumed after epoch {completed_epoch} (best val {best_val:.5f})")
    if since_best >= args.patience:
        print(f"run already early-stopped after epoch {completed_epoch}")
        return
    metadata = checkpoint_metadata(args, config, provenance,
                                   initialized_from_scratch=not bool(args.finetune_from))
    log_path = out / "log.csv"
    prepare_log(log_path, ["epoch", "train_loss", "val_loss", "lr", "sec"], completed_epoch)
    with open(log_path, "a", newline="") as f:
        wr = csv.writer(f)
        for ep in range(completed_epoch + 1, args.epochs + 1):
            t0 = time.time()
            ds_tr.set_epoch(ep)
            tl = run_epoch(model, dl_tr, device, opt, args.event_weight)
            vl = run_epoch(model, dl_va, device, None, args.event_weight)
            sched.step()
            dt = time.time() - t0
            wr.writerow([ep, f"{tl:.5f}", f"{vl:.5f}",
                         f"{opt.param_groups[0]['lr']:.2e}", f"{dt:.0f}"])
            f.flush()
            marker = ""
            improved = vl < best_val
            if improved:
                best_val, since_best = vl, 0
                marker = " *"
            else:
                since_best += 1
            weights = {**metadata, "model": model.state_dict(), "arch": args.arch,
                       "params": count_params(model), "epoch": ep, "seed": args.seed,
                       "best_val": best_val}
            if improved:
                save_checkpoint(out / "best.pt", weights)
            save_checkpoint(out / "last.pt", {
                **weights, "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "rng": capture_rng(), "since_best": since_best,
            })
            print(f"ep {ep:3d} train {tl:.5f} val {vl:.5f} [{dt:.0f}s]{marker}")
            if since_best >= args.patience:
                print(f"early stop at epoch {ep} (no val gain for "
                      f"{args.patience} epochs)")
                break
    print(f"done. best val {best_val:.5f} -> {out / 'best.pt'}")


if __name__ == "__main__":
    main()
