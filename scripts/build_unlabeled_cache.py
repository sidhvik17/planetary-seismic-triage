"""Preprocess every trace we have (labeled or not) into the unlabeled SSL cache.

Sources: lunar training days (75), lunar test-station files S12B/S15/S16 (96),
Mars training + test files. No labels stored — this corpus only teaches the
noise structure. Writes data/cache/unlabeled/*.npz.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import DATA_CACHE, DEFAULT as CFG, PACKET_ROOT
from planetseis.preprocessing import load_trace, preprocess

OUT = DATA_CACHE / "unlabeled"


def add(path: Path, done: set):
    if path.stem in done:
        return
    try:
        raw, rate, _ = load_trace(path)
        proc, prate = preprocess(raw, rate, CFG.preproc)
        np.savez_compressed(OUT / (path.stem + ".npz"), trace=proc, rate=prate)
        done.add(path.stem)
    except Exception as e:
        print(f"skip {path.name}: {e}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in OUT.glob("*.npz")}
    # already-preprocessed labeled traces: just link content across
    for body in ("lunar", "mars"):
        for split in ("train", "val", "test"):
            for p in (DATA_CACHE / body / "continuous" / split).glob("*.npz"):
                if p.stem not in done:
                    z = np.load(p)
                    np.savez_compressed(OUT / p.name, trace=z["trace"], rate=z["rate"])
                    done.add(p.stem)
    # unlabeled station files + mars test
    for d in [PACKET_ROOT / "data" / "lunar" / "test" / "data",
              PACKET_ROOT / "data" / "mars" / "test" / "data"]:
        for f in sorted(d.rglob("*.mseed")):
            add(f, done)
    print(f"unlabeled cache: {len(list(OUT.glob('*.npz')))} traces")


if __name__ == "__main__":
    main()
