"""Build a denoised twin of the lunar window/continuous caches.

Completes the denoise-then-detect chain experiment: eval_chain.py showed the
raw-trained SeisCNN collapses on denoised input (P 0.556 -> 0.067) while
probability fusion recovers 3 of its 8 false negatives — the denoiser
surfaces the missing events but the detector has never seen denoised
statistics. Fix: denoise the SeisCNN's own training windows with the frozen
SpecUNet and retrain on them ('lunar_dn' body).

Windows are denoised individually (mask x STFT, one window = one grid cell);
continuous val/test traces reuse the cache eval_chain.py already writes.

Usage: python scripts/build_denoised_windows.py --unet runs/unet_lunar/best.pt
Writes data/cache/lunar_dn/{train,val}_windows.npz and links continuous/.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE
from planetseis.spectral import istft_window, normalize_stft, stft_window
from planetseis.unet import UNET_ARCHS, SpecUNet


def denoise_windows(model, X, device, batch=64):
    out = np.empty_like(X)
    for b in range(0, len(X), batch):
        chunk = X[b : b + batch]
        Zs = [stft_window(w) for w in chunk]
        x = torch.from_numpy(np.stack([normalize_stft(Z) for Z in Zs])).to(device)
        with torch.no_grad():
            masks = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
        for i, (Z, mk) in enumerate(zip(Zs, masks)):
            out[b + i] = istft_window(mk * Z)
        if b % 640 == 0:
            print(f"  {b}/{len(X)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet", required=True)
    ap.add_argument("--body", default="lunar")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.unet, map_location=device, weights_only=False)
    model = SpecUNet(base=UNET_ARCHS[ck.get("arch", "base")])
    model.load_state_dict(ck["model"])
    model.eval().to(device)

    src = DATA_CACHE / args.body
    dst = DATA_CACHE / f"{args.body}_dn"
    dst.mkdir(exist_ok=True)
    for split in ("train", "val"):
        z = np.load(src / f"{split}_windows.npz")
        print(f"{split}: denoising {len(z['X'])} windows")
        Xd = denoise_windows(model, z["X"], device)
        np.savez_compressed(dst / f"{split}_windows.npz", X=Xd,
                            y=z["y"], offset=z["offset"])

    # continuous denoised traces: reuse eval_chain's cache
    for split in ("val", "test"):
        cont_src = src / "denoised" / split
        cont_dst = dst / "continuous" / split
        cont_dst.mkdir(parents=True, exist_ok=True)
        for p in sorted(cont_src.glob("*.npz")):
            raw = np.load(src / "continuous" / split / p.name)
            den = np.load(p)["trace"]
            np.savez_compressed(cont_dst / p.name, trace=den,
                                rate=raw["rate"], picks=raw["picks"])
            print(f"continuous/{split}/{p.name} linked")
    print(f"done -> {dst}")


if __name__ == "__main__":
    main()
