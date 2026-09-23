"""Run the corrected-split secondary analyses for every trained seed.

    python scripts/run_grouped_secondary.py            # seeds 42 1 2 3 4
    python scripts/run_grouped_secondary.py --seeds 1 2

For each seed, the operating points are read from that seed's locked
evaluation files (results/lunar_grouped_v1_seed<N>_{cnn,unet}.json), never
typed by hand and never chosen on test data. Each analysis writes new
*_lunar_grouped_v1_seed<N>.* files and is skipped when its output already
exists, so the runner is safe to repeat. One model job runs at a time.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA = "data/cache/lunar_grouped_v1"


def run_dir(family: str, seed: int) -> Path:
    base = "lunar_grouped_v1" if family == "cnn" else "unet_lunar_grouped_v1"
    return ROOT / "runs" / (base if seed == 42 else f"{base}_s{seed}")


def locked_point(family: str, seed: int) -> dict:
    result = json.loads((ROOT / "results" / f"lunar_grouped_v1_seed{seed}_{family}.json")
                        .read_text(encoding="utf-8"))
    checkpoint = result["checkpoints"][family]
    if checkpoint["seed"] != seed:
        raise ValueError(f"seed {seed}: evaluation file records seed {checkpoint['seed']}")
    return result["results"][family]["operating_point"]


def commands(seed: int):
    tag = f"lunar_grouped_v1_seed{seed}"
    cnn, unet = locked_point("cnn", seed), locked_point("unet", seed)
    cnn_model = str(run_dir("cnn", seed) / "best.pt")
    unet_model = str(run_dir("unet", seed) / "best.pt")
    yield (ROOT / "results" / f"nakamura_crosscheck_{tag}.json",
           ["scripts/crosscheck_nakamura.py", "--data-dir", DATA, "--model", unet_model,
            "--threshold", str(unet["threshold"]), "--min-dur", str(unet["min_dur_sec"]),
            "--tag", tag])
    yield (ROOT / "results" / f"uncertainty_{tag}.json",
           ["scripts/uncertainty_eval.py", "--data-dir", DATA, "--model", cnn_model,
            "--threshold", str(cnn["threshold"]),
            "--output", f"results/uncertainty_{tag}.json"])
    yield (ROOT / "results" / f"uncertainty_unet_{tag}.json",
           ["scripts/uncertainty_unet.py", "--data-dir", DATA, "--model", unet_model,
            "--threshold", str(unet["threshold"]), "--min-dur", str(unet["min_dur_sec"]),
            "--output", f"results/uncertainty_unet_{tag}.json"])
    yield (ROOT / "results" / f"snr_recall_{tag}.json",
           ["scripts/extra_analysis.py", "--data-dir", DATA, "--model", cnn_model,
            "--threshold", str(cnn["threshold"]), "--tag", tag])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    args = ap.parse_args()
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "PYTHONUNBUFFERED": "1"}
    for seed in args.seeds:
        for output, cmd in commands(seed):
            if output.exists():
                print(f"seed {seed}: {output.name} exists, skipped", flush=True)
                continue
            print(f"seed {seed}: {' '.join(cmd)}", flush=True)
            subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
