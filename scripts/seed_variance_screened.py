"""Aggregate 5-seed variance for the screened SpecUNet (A3).

    python scripts/seed_variance_screened.py

Three seeds were never enough to support the paired-bootstrap reading: the
original addendum already showed SpecUNet's spread (+/-0.086) was comparable
to the SeisCNN-vs-SpecUNet gap itself, which is why the headline claim was
softened from "statistically indistinguishable" to "same band, not resolved".
Five seeds is the standard bar and makes that statement defensible rather
than merely cautious.

Each seed is evaluated with its OWN val-tuned operating point (eval_unet.py
does the tuning), so no seed borrows another's threshold and test is never
touched during selection. Expects runs/unet_lunar_screened{,_s1..s4} to exist
and to have been evaluated; pass --eval to run the evaluations first.

Writes results/seed_variance_unet_screened.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT

RUNS = {42: "unet_lunar_screened", 1: "unet_lunar_screened_s1",
        2: "unet_lunar_screened_s2", 3: "unet_lunar_screened_s3",
        4: "unet_lunar_screened_s4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true",
                    help="run eval_unet.py for any seed missing its result")
    args = ap.parse_args()

    per_seed, per_op, missing = {}, {}, []
    for seed, run in RUNS.items():
        res = PROJECT_ROOT / "results" / f"{run}_to_lunar.json"
        ckpt = PROJECT_ROOT / "runs" / run / "best.pt"
        if not res.exists():
            if not ckpt.exists():
                missing.append(run)
                continue
            if args.eval:
                print(f"evaluating {run} ...")
                subprocess.run(
                    [sys.executable, "scripts/eval_unet.py", "--model",
                     str(ckpt), "--eval-body", "lunar"],
                    cwd=PROJECT_ROOT, check=True)
            else:
                missing.append(run)
                continue
        d = json.loads(res.read_text())
        per_seed[f"seed{seed}"] = d["unet"]["f1"]
        per_op[f"seed{seed}"] = [d["threshold"], d["min_dur_sec"]]

    if missing:
        print(f"missing (train or --eval first): {missing}")
    if len(per_seed) < 2:
        sys.exit("need at least two seeds")

    f1 = np.array(list(per_seed.values()), dtype=float)
    out = {
        "addendum": "5-seed variance for the Nakamura-screened SpecUNet "
                    "(v1.1). Each seed evaluated at its OWN val-tuned "
                    "operating point; test untouched during selection.",
        "n_seeds": len(f1),
        "specunet_screened": {
            "per_seed_f1": per_seed,
            "per_seed_operating_point": per_op,
            "mean_f1": round(float(f1.mean()), 4),
            "std_f1": round(float(f1.std(ddof=1)), 4),
            "min_f1": round(float(f1.min()), 4),
            "max_f1": round(float(f1.max()), 4),
        },
        "reference_unscreened_3seed": {
            "per_seed_f1": {"seed42": 0.44, "seed1": 0.2807, "seed2": 0.4151},
            "mean_f1": 0.3786, "std_f1": 0.0857,
        },
        "reference_seiscnn_3seed": {
            "per_seed_f1": {"seed42": 0.5405, "seed1": 0.5, "seed2": 0.4545},
            "mean_f1": 0.4983, "std_f1": 0.043,
        },
        "note": "Read the SpecUNet-vs-SeisCNN comparison as overlap, not a "
                "tie: the seed spread is comparable to the point estimate of "
                "the gap. With 19 test files the two are not resolved, and "
                "no amount of reseeding fixes that - only a larger evaluation "
                "set would.",
    }
    p = PROJECT_ROOT / "results" / "seed_variance_unet_screened.json"
    p.write_text(json.dumps(out, indent=2))
    print(json.dumps(out["specunet_screened"], indent=2))
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
