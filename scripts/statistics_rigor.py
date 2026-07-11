"""Goal 4 — statistical rigor for every headline number.

* Bootstrap 95% CIs (resampling test FILES with replacement, 10k draws) for
  CNN and STA/LTA precision/recall/F1 on the lunar test split.
* Permutation test vs chance: null = same number of detections per file
  placed uniformly at random; p-value = P(null F1 >= observed F1).
* Marks Mars results as anecdotal (n=1) — no CI pretense.

Writes results/statistics.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.baseline import sta_lta_detect
from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect import detect_events
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN

THR = 0.99
STA_ON = 7.0
N_BOOT = 10_000
N_PERM = 2_000
rng = np.random.default_rng(42)


def per_file_results(device):
    """(detections, picks, trace_dur_sec) per lunar test file, CNN + STA/LTA."""
    ck = torch.load(PROJECT_ROOT / "runs" / "lunar" / "best.pt",
                    map_location=device, weights_only=False)
    model = SeisCNN(channels=ARCHS[ck.get("arch", "base")]).to(device)
    model.load_state_dict(ck["model"])
    files = []
    for p in sorted((DATA_CACHE / "lunar" / "continuous" / "test").glob("*.npz")):
        z = np.load(p)
        trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
        dets, _, _ = detect_events(model, trace, rate, CFG, THR, device,
                                   suppress_sec=CODA_SEC["lunar"])
        st = sta_lta_detect(trace, rate, thr_on=STA_ON, thr_off=STA_ON / 2)
        kept = []
        for t in sorted(st):
            if not kept or t - kept[-1] >= CODA_SEC["lunar"]:
                kept.append(t)
        files.append({
            "cnn": [d.time_sec for d in dets],
            "sta": kept,
            "picks": picks,
            "dur": len(trace) / rate,
        })
    return files


def scores_for(files, key) -> Scores:
    s = Scores()
    for f in files:
        s.merge(score_trace(f[key], f["picks"], CFG.window.match_tolerance_sec))
    return s


def bootstrap_ci(files, key):
    stats = {"precision": [], "recall": [], "f1": []}
    n = len(files)
    for _ in range(N_BOOT):
        sample = [files[i] for i in rng.integers(0, n, size=n)]
        s = scores_for(sample, key)
        stats["precision"].append(s.precision)
        stats["recall"].append(s.recall)
        stats["f1"].append(s.f1)
    return {m: [round(float(np.percentile(v, 2.5)), 3),
                round(float(np.percentile(v, 97.5)), 3)]
            for m, v in stats.items()}


def permutation_p(files, key):
    obs = scores_for(files, key).f1
    null = []
    for _ in range(N_PERM):
        s = Scores()
        for f in files:
            fake = list(rng.uniform(0, f["dur"], size=len(f[key])))
            s.merge(score_trace(sorted(fake), f["picks"],
                                CFG.window.match_tolerance_sec))
        null.append(s.f1)
    null = np.array(null)
    p = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    return {"observed_f1": round(obs, 4), "null_f1_mean": round(float(null.mean()), 4),
            "null_f1_p97.5": round(float(np.percentile(null, 97.5)), 4),
            "p_value": round(p, 5)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    files = per_file_results(device)
    out = {"n_test_files": len(files),
           "n_test_events": int(sum(len(f["picks"]) for f in files)),
           "protocol": {"threshold": THR, "tolerance_sec": CFG.window.match_tolerance_sec,
                        "bootstrap_draws": N_BOOT, "permutation_draws": N_PERM,
                        "resampling_unit": "file"}}
    for key, label in [("cnn", "cnn"), ("sta", "sta_lta")]:
        s = scores_for(files, key)
        out[label] = {
            "point": {"precision": round(s.precision, 4), "recall": round(s.recall, 4),
                      "f1": round(s.f1, 4)},
            "ci95": bootstrap_ci(files, key),
            "permutation_vs_chance": permutation_p(files, key),
        }
        print(label, json.dumps(out[label], indent=2))
    out["mars_note"] = ("Mars same-body results are single-event (n=1) and are "
                        "reported as anecdotal; no CI is meaningful.")
    (PROJECT_ROOT / "results" / "statistics.json").write_text(json.dumps(out, indent=2))
    print("saved -> results/statistics.json")


if __name__ == "__main__":
    main()
