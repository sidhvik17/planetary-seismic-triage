"""Mine hard-negative windows for U-Net fine-tuning.

The injection-trained U-Net learns 'any event-like energy = event'. The
Grade-A catalog is a SELECTION: long-ringing energy the analysts did not
catalog counts as false positive on the benchmark. SeisCNN closed this gap
with hard-negative mining (scripts/mine_hard_negatives.py); this is the same
loop for the mask model: run detection over the TRAIN split, keep confident
regions that are far from every catalog pick, save their windows as a
negative pool with all-zero mask targets.

Usage: python scripts/mine_unet_hardneg.py --model runs/unet_lunar/best.pt
           --body lunar [--threshold 0.25] [--min-dur 300]
Writes data/cache/{body}/unet_hardneg.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DATA_CACHE
from planetseis.detect_spec import SEC_PER_BIN, compute_curve, curve_to_detections
from planetseis.spectral import N_SAMPLES, RATE_HZ
from planetseis.unet import UNET_ARCHS, SpecUNet

# a mined region must be at least this far from every catalog pick — beyond
# match tolerance plus the full lunar ring-down, so no true event's coda is
# ever taught as negative
GUARD_SEC = 2400.0

# The first fine-tune attempt taught the model to suppress real events:
# ~45% of confident 'false positives' turned out to be genuine moonquakes
# absent from the Grade-A labels (see crosscheck_nakamura.py). Mined
# negatives are therefore ALSO screened against the full Nakamura catalog —
# anything within NAK_GUARD_SEC of any S12-detected Nakamura event is
# assumed to be a real event and excluded from the negative pool.
NAK_GUARD_SEC = 600.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--body", default="lunar",
                    choices=["lunar", "mars", "mars_ext"])
    ap.add_argument("--threshold", type=float, default=0.25)
    ap.add_argument("--min-dur", type=float, default=300.0)
    ap.add_argument("--max-per-file", type=int, default=6)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location=device, weights_only=True)
    model = SpecUNet(base=UNET_ARCHS[ckpt.get("arch", "base")])
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)

    nak_times = None
    if args.body == "lunar":
        # plain-Python fixed-width parse: both pandas.read_fwf (pyarrow
        # string backend access violation) and mass obspy UTCDateTime
        # construction have segfaulted on this machine mid-session
        from datetime import datetime, timedelta, timezone

        from scripts.crosscheck_nakamura import trace_start_utc
        from planetseis.config import PROJECT_ROOT

        times = []
        with open(PROJECT_ROOT / "data" / "raw" / "levent.1008.dat") as f:
            for line in f:
                try:
                    float(line[19:23].strip())   # detected at S12 (or S11)?
                except ValueError:
                    continue
                try:
                    y = 1900 + int(line[2:4])
                    doy = int(line[5:8])
                    hhmm = int(line[9:13])
                except ValueError:
                    continue
                t = (datetime(y, 1, 1, tzinfo=timezone.utc)
                     + timedelta(days=doy - 1, hours=hhmm // 100,
                                 minutes=hhmm % 100))
                times.append(t.timestamp())
        nak_times = np.array(times)
        print(f"screening negatives against {len(nak_times)} Nakamura events")

    min_bins = max(1, int(round(args.min_dur / SEC_PER_BIN)))
    windows = []
    src = []
    n_nak_dropped = 0
    for p in sorted((DATA_CACHE / args.body / "continuous" / "train").glob("*.npz")):
        z = np.load(p)
        trace, picks = z["trace"], list(z["picks"])
        t0 = trace_start_utc(p.stem) if nak_times is not None else None
        curve = compute_curve(model, trace, device)
        dets = curve_to_detections(curve, args.threshold,
                                   suppress_sec=CODA_SEC[args.body],
                                   min_bins=min_bins)
        kept = 0
        for d in sorted(dets, key=lambda d: -d.confidence):
            if kept >= args.max_per_file:
                break
            if any(abs(d.time_sec - pk) < GUARD_SEC for pk in picks):
                continue
            if t0 is not None:
                dt = np.abs(nak_times - (float(t0) + d.time_sec)).min()
                if dt <= NAK_GUARD_SEC:
                    n_nak_dropped += 1
                    continue
            s = int(d.time_sec * RATE_HZ)
            if s + N_SAMPLES > len(trace):
                s = len(trace) - N_SAMPLES
            windows.append(trace[s : s + N_SAMPLES].astype(np.float32))
            src.append(f"{p.stem}@{d.time_sec:.0f}s conf={d.confidence:.2f}")
            kept += 1
        print(f"{p.stem}: {len(dets)} dets -> {kept} hard negatives")

    print(f"dropped {n_nak_dropped} mined regions matching Nakamura events")
    out = DATA_CACHE / args.body / "unet_hardneg.npz"
    np.savez_compressed(out, X=np.stack(windows) if windows else
                        np.zeros((0, N_SAMPLES), np.float32),
                        src=np.array(src))
    print(f"{len(windows)} hard-negative windows -> {out}")


if __name__ == "__main__":
    main()
