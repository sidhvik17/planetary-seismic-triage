"""Screened vs unscreened SpecUNet on lunar_grouped_v1 (positive-unlabeled ablation).

    python scripts/compare_screened_grouped.py

Analysis fixed before any screened test result existed (tasks/todo.md, E1):
seed-mean test F1 of five screened seeds against the five existing unscreened
seeds, Welch t-test, each seed at its own validation-locked operating point.
p >= 0.05 is reported as "no difference detected", never as "no effect".
Checks that both arms use the same manifest, that only the screen differs in
the training arguments, and that each evaluation scored the checkpoint on disk.
Writes results/lunar_grouped_v1_screening_ablation.json (exclusive).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy import stats
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import PROJECT_ROOT

SEEDS = [42, 1, 2, 3, 4]
ARMS = {"unscreened": "lunar_grouped_v1_seed{seed}_unet.json",
        "screened": "lunar_grouped_v1_screened_seed{seed}_unet.json"}
# Arguments that must agree between arms apart from the screen itself.
SHARED = ("body", "arch", "epochs", "epoch_len", "batch_size", "lr", "event_weight",
          "patience", "val_split", "finetune_from", "hardneg")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_arm(results: Path, arm: str) -> list[dict]:
    rows = []
    for seed in SEEDS:
        path = results / ARMS[arm].format(seed=seed)
        data = json.loads(path.read_text(encoding="utf-8"))
        prov = data["checkpoints"]["unet"]
        config = prov["training_config"]
        if prov["seed"] != seed or bool(config.get("screen_nakamura")) != (arm == "screened"):
            raise ValueError(f"{path.name}: seed or screen flag does not match its arm")
        checkpoint = PROJECT_ROOT / prov["path"]
        if sha256(checkpoint) != prov["sha256"]:
            raise ValueError(f"{path.name}: checkpoint on disk differs from the one evaluated")
        ck = torch.load(checkpoint, map_location="cpu", weights_only=True)
        rows.append({"seed": seed, "file": path.name, "sha256_result": sha256(path),
                     "manifest": data["data_manifest_sha256"],
                     "shared_args": {k: config.get(k) for k in SHARED},
                     "screen_sha256": ck["training_config"].get("nakamura_screen_sha256"),
                     "operating_point": data["results"]["unet"]["operating_point"],
                     "val_f1": data["results"]["unet"]["selected_val_f1"],
                     "test": data["results"]["unet"]["test_scores"]})
    return rows


def summary(values):
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(a.mean()), 4), "sd": round(float(a.std(ddof=1)), 4),
            "min": round(float(a.min()), 4), "max": round(float(a.max()), 4), "n": len(a)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results")
    ap.add_argument("--output", type=Path,
                    default=PROJECT_ROOT / "results" / "lunar_grouped_v1_screening_ablation.json")
    args = ap.parse_args(argv)
    arms = {arm: load_arm(args.results_dir, arm) for arm in ARMS}
    rows = arms["unscreened"] + arms["screened"]
    if len({r["manifest"] for r in rows}) != 1:
        raise ValueError("arms use different data manifests")
    if any(r["shared_args"] != rows[0]["shared_args"] for r in rows):
        raise ValueError("training recipe differs between runs beyond the screen")
    screens = {r["screen_sha256"] for r in arms["screened"]}
    if len(screens) != 1 or None in screens or any(r["screen_sha256"] for r in arms["unscreened"]):
        raise ValueError("screened runs must share one recorded screen; unscreened runs none")

    f1 = {arm: [r["test"]["f1"] for r in arms[arm]] for arm in ARMS}
    t, p = stats.ttest_ind(f1["screened"], f1["unscreened"], equal_var=False)
    diff = float(np.mean(f1["screened"]) - np.mean(f1["unscreened"]))
    result = {
        "benchmark_id": "lunar_grouped_v1", "data_manifest_sha256": rows[0]["manifest"],
        "nakamura_screen_sha256": screens.pop(), "seeds": SEEDS,
        "protocol": "Five seeds per arm from scratch, identical recipe except --screen-nakamura; "
                    "each at its own validation-locked operating point; Welch t-test on "
                    "seed-level test F1, fixed before the screened results existed.",
        "test_f1": {arm: summary(v) for arm, v in f1.items()},
        "val_f1": {arm: summary([r["val_f1"] for r in arms[arm]]) for arm in ARMS},
        "precision": {arm: summary([r["test"]["precision"] for r in arms[arm]]) for arm in ARMS},
        "recall": {arm: summary([r["test"]["recall"] for r in arms[arm]]) for arm in ARMS},
        "difference_screened_minus_unscreened": round(diff, 4),
        "welch": {"t": round(float(t), 3), "p_value": round(float(p), 4)},
        "reading": ("difference detected at p < 0.05" if p < 0.05 else
                    "no difference detected at p < 0.05; not an equivalence test"),
        "per_seed": arms,
        "analysis_script_sha256": sha256(Path(__file__)),
    }
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("test_f1", "difference_screened_minus_unscreened",
                                               "welch", "reading")}, indent=2))


if __name__ == "__main__":
    main()
