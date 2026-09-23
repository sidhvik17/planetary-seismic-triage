"""Pretrain the masked autoencoder on the full unlabeled corpus (Goal 1).

Usage: python scripts/pretrain_ssl.py [--epochs 60] [--mask 0.5]
Saves runs/ssl/encoder.pt (backbone weights + config) and loss log.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, RUNS_DIR, SEED
from planetseis.ssl import MaskedAutoencoder, UnlabeledWindows
from planetseis.train import set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--mask", type=float, default=0.5)
    ap.add_argument("--windows-per-epoch", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted((DATA_CACHE / "unlabeled").glob("*.npz"))
    ds = UnlabeledWindows(paths, CFG.window.n_samples, args.windows_per_epoch, seed=SEED)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"corpus: {len(ds.traces)} traces, {args.windows_per_epoch} windows/epoch, "
          f"device={device}")

    mae = MaskedAutoencoder(mask_ratio=args.mask).to(device)
    opt = torch.optim.AdamW(mae.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out = RUNS_DIR / "ssl"
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    start_ep = 1
    full = out / "mae_full.pt"
    if args.resume and full.exists():
        st = torch.load(full, map_location=device, weights_only=True)
        mae.load_state_dict(st["mae"])
        opt.load_state_dict(st["opt"])
        best = st["best"]
        start_ep = st["epoch"] + 1
        for _ in range(st["epoch"]):
            sched.step()
        print(f"resumed at epoch {start_ep} (best {best:.5f})")
    with open(out / "log.csv", "a" if args.resume else "w", newline="") as f:
        wr = csv.writer(f)
        if not args.resume:
            wr.writerow(["epoch", "recon_loss", "lr"])
        for ep in range(start_ep, args.epochs + 1):
            mae.train()
            tot, nb = 0.0, 0
            for x in dl:
                x = x.to(device)
                loss, _, _ = mae(x)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item()
                nb += 1
            sched.step()
            avg = tot / nb
            wr.writerow([ep, f"{avg:.5f}", f"{opt.param_groups[0]['lr']:.2e}"])
            f.flush()
            mark = ""
            if avg < best:
                best = avg
                torch.save({"backbone": mae.backbone.state_dict(),
                            "arch": "base", "mask_ratio": args.mask,
                            "epoch": ep, "recon_loss": avg}, out / "encoder.pt")
                mark = " *"
            torch.save({"mae": mae.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep, "best": best}, full)
            print(f"ep {ep:3d} recon {avg:.5f}{mark}")
    print(f"best recon {best:.5f} -> {out / 'encoder.pt'}")


if __name__ == "__main__":
    main()
