"""Zero-shot pretrained terrestrial baselines: PhaseNet + EQTransformer (Goal 2).

Both models ship pretrained (STEAD) in SeisBench. They expect 3-component
100 Hz terrestrial data; we feed the same preprocessed single-channel planetary
stream the CNN sees, channel-replicated to 3 components, and let SeisBench
resample. That is the honest 'apply the terrestrial giant as-is' protocol —
degradation is expected and is part of the finding.

Peaks are extracted once per trace at a low bar, cached, then the decision
threshold is swept on val (test never touched), with the same dead-time rule
and scorer as every other detector in this study.

Usage: python scripts/seisbench_baseline.py --eval-body lunar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import obspy
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import seisbench.models as sbm

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.evaluate import Scores, score_trace

RESULTS = PROJECT_ROOT / "results"
CACHE = RESULTS / "seisbench_peaks"


def to_stream(trace: np.ndarray, rate: float) -> obspy.Stream:
    trs = []
    for ch in ("HHZ", "HHN", "HHE"):
        t = obspy.Trace(trace.astype(np.float64))
        t.stats.sampling_rate = rate
        t.stats.channel = ch
        t.stats.network, t.stats.station = "XX", "DEMO"
        trs.append(t)
    return obspy.Stream(trs)


def detection_prob(model_name: str, ann: obspy.Stream) -> obspy.Trace | None:
    """The channel that means 'an event is here' for each model."""
    if model_name == "eqt":
        sel = ann.select(channel="*Detection*")
        return sel[0] if sel else None
    # PhaseNet: max of P and S phase probabilities
    p = ann.select(channel="*_P")
    s = ann.select(channel="*_S")
    if not p:
        return None
    tr = p[0].copy()
    if s:
        n = min(len(p[0].data), len(s[0].data))
        tr.data = np.maximum(p[0].data[:n], s[0].data[:n])
    return tr


def peaks_for_trace(model, model_name, trace, rate) -> list[tuple[float, float]]:
    ann = model.annotate(to_stream(trace, rate))
    tr = detection_prob(model_name, ann)
    if tr is None or len(tr.data) == 0:
        return []
    prob = np.nan_to_num(np.asarray(tr.data, dtype=np.float64))
    dist = max(1, int(10 * tr.stats.sampling_rate))  # >=10 s apart
    idx, props = find_peaks(prob, height=0.05, distance=dist)
    return [(float(i / tr.stats.sampling_rate), float(h))
            for i, h in zip(idx, props["peak_heights"])]


def load_split(body, split):
    d = DATA_CACHE / body / "continuous" / split
    for p in sorted(d.glob("*.npz")):
        z = np.load(p)
        yield p.stem, z["trace"], float(z["rate"]), list(z["picks"])


def get_peaks(model, model_name, body, split) -> dict:
    """Cached peak extraction (annotate is the expensive part)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{model_name}_{body}_{split}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())
    out = {}
    for stem, trace, rate, picks in load_split(body, split):
        pk = peaks_for_trace(model, model_name, trace, rate)
        out[stem] = {"peaks": pk, "picks": picks}
        print(f"  {stem}: {len(pk)} raw peaks")
    cache_file.write_text(json.dumps(out))
    return out


def score_at(peaks_by_trace: dict, thr: float, coda: float, tol: float) -> Scores:
    total = Scores()
    for rec in peaks_by_trace.values():
        cand = sorted([t for t, h in rec["peaks"] if h >= thr])
        kept = []
        for t in cand:
            if not kept or t - kept[-1] >= coda:
                kept.append(t)
        total.merge(score_trace(kept, list(rec["picks"]), tol))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-body", required=True, choices=["lunar", "mars"])
    args = ap.parse_args()
    body = args.eval_body
    coda, tol = CODA_SEC[body], CFG.window.match_tolerance_sec
    device_ok = torch.cuda.is_available()

    results = {}
    for model_name, loader in [("phasenet", sbm.PhaseNet), ("eqt", sbm.EQTransformer)]:
        print(f"=== {model_name} (pretrained: stead) ===")
        model = loader.from_pretrained("stead")
        if device_ok:
            model.cuda()
        val = get_peaks(model, model_name, body, "val")
        test = get_peaks(model, model_name, body, "test")
        tune_on = val if val else get_peaks(model, model_name, body, "train")

        best_thr, best_f1 = 0.1, -1.0
        for thr in [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            s = score_at(tune_on, thr, coda, tol)
            print(f"  thr={thr:.2f} P={s.precision:.3f} R={s.recall:.3f} F1={s.f1:.3f}")
            if s.f1 > best_f1:
                best_thr, best_f1 = thr, s.f1
        s = score_at(test, best_thr, coda, tol)
        results[model_name] = {"threshold": best_thr, **s.as_dict()}
        print(f"  TEST {model_name}: {s.as_dict()}")

    out = RESULTS / f"seisbench_zeroshot_{body}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
