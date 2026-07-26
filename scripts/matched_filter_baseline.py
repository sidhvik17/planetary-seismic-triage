"""Template-matching (matched-filter) baseline on the frozen benchmark (A2).

    python scripts/matched_filter_baseline.py --body lunar

Template matching is THE established method for extending lunar catalogs —
Nakamura (2003) and Bulow et al. (2005, 2007) used waveform cross-correlation
and stacking to add hundreds of deep moonquakes precisely because those
events repeat. PhaseNet and EQTransformer scoring ~0 is a domain mismatch,
not a baseline: they are 100 Hz terrestrial P/S pickers applied to 6.625 Hz
emergent, scattered signals. A reviewer in this field will ask for this
comparison first, so it is the one that matters.

Protocol, identical to every other detector here so the numbers are directly
comparable:
  * templates cut from the TRAIN split only — the same 45 Grade-A events the
    injection engine gets, so the information budget matches
  * detection statistic = max normalised cross-correlation across templates,
    thresholded at k x MAD of that trace (the standard convention; k ~ 8-12
    in the terrestrial literature)
  * k tuned on VAL, never on test
  * same dead-time suppression (lunar ring-down), same scorer, same +/-120 s
  * CPU only (scipy FFT), so it can run beside GPU training

Writes results/matched_filter_{body}.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DATA_CACHE, DEFAULT as CFG, PROJECT_ROOT
from planetseis.evaluate import Scores, score_trace

PRE_SEC = 120.0      # template starts before the pick: lunar onsets are
                     # emergent and picks are minute-quantised
K_GRID = [4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0]


def build_templates(body: str, split: str, rate: float, tpl_sec: float):
    """Raw (bandpassed) waveform cutouts around each train-split pick.

    Deliberately NOT the spectrally gated templates the injection engine
    builds: classical template matching correlates the recorded waveform, and
    gating would make this a different, non-standard method.
    """
    n = int(tpl_sec * rate)
    pre = int(PRE_SEC * rate)
    out = []
    for p in sorted((DATA_CACHE / body / "continuous" / split).glob("*.npz")):
        z = np.load(p)
        trace = np.asarray(z["trace"], dtype=np.float64)
        for pick in z["picks"]:
            s = int(float(pick) * rate) - pre
            if s < 0 or s + n > len(trace):
                continue
            w = trace[s : s + n].copy()
            w -= w.mean()
            nrm = np.linalg.norm(w)
            if nrm < 1e-12:
                continue
            out.append((w / nrm, p.stem))
    return out


def max_cc(trace: np.ndarray, templates) -> np.ndarray:
    """Max normalised cross-correlation over all templates, per sample.

    Normalisation uses the running mean/energy of the data window, so the
    statistic is a true correlation coefficient in [-1, 1] and a threshold in
    MAD units means the same thing everywhere on the trace.
    """
    x = np.asarray(trace, dtype=np.float64)
    n = len(x)
    L = len(templates[0][0])
    if n < L:
        return np.zeros(0)
    # running sum and sum-of-squares over each length-L window
    c1 = np.concatenate(([0.0], np.cumsum(x)))
    c2 = np.concatenate(([0.0], np.cumsum(x * x)))
    s1 = c1[L:] - c1[:-L]
    s2 = c2[L:] - c2[:-L]
    var = np.maximum(s2 - s1 * s1 / L, 0.0)
    den = np.sqrt(var)
    den[den < 1e-12] = np.inf                 # flat/dead windows -> CC 0

    best = np.full(n - L + 1, -np.inf)
    for w, _ in templates:
        num = fftconvolve(x, w[::-1], mode="valid")
        np.maximum(best, num / den, out=best)
    best[~np.isfinite(best)] = 0.0
    return best


def detections_from_cc(cc, rate, thr, suppress_sec, offset_sec):
    """Peaks above `thr`, with the same dead-time rule as the other detectors:
    the strongest arrival wins each ring-down slot."""
    hot = cc >= thr
    if not hot.any():
        return []
    idx = np.flatnonzero(hot)
    brk = np.flatnonzero(np.diff(idx) != 1)
    starts = np.concatenate(([idx[0]], idx[brk + 1]))
    stops = np.concatenate((idx[brk], [idx[-1]])) + 1
    dets = []
    for a, b in zip(starts, stops):
        pk = a + int(np.argmax(cc[a:b]))
        dets.append((pk / rate + offset_sec, float(cc[pk])))
    kept = []
    for t, c in dets:
        if kept and t - kept[-1][0] < suppress_sec:
            if c > kept[-1][1]:
                kept[-1] = (t, c)
        else:
            kept.append((t, c))
    return kept


def cc_for_split(body, split, templates, rate, tpl_sec):
    out = []
    for p in sorted((DATA_CACHE / body / "continuous" / split).glob("*.npz")):
        z = np.load(p)
        cc = max_cc(z["trace"], templates)
        out.append((p.stem, cc, list(z["picks"])))
        print(f"  {split}/{p.stem}: cc len {len(cc)}")
    return out


def score_at(cc_split, rate, k, coda, tol, offset_sec):
    total = Scores()
    for _, cc, picks in cc_split:
        if not len(cc):
            total.merge(score_trace([], picks, tol))
            continue
        mad = np.median(np.abs(cc - np.median(cc)))
        thr = np.median(cc) + k * 1.4826 * mad
        dets = detections_from_cc(cc, rate, thr, coda, offset_sec)
        total.merge(score_trace([t for t, _ in dets], picks, tol))
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default="lunar")
    ap.add_argument("--tpl-sec", nargs="+", type=float,
                    default=[200.0, 300.0, 600.0, 1200.0],
                    help="template lengths to try, incl. the 120 s pre-onset "
                         "lead. Both length and k are selected on VAL, so the "
                         "baseline gets its best honest shot rather than one "
                         "arbitrary length that could be called a strawman.")
    args = ap.parse_args()
    args.tpl_sec_all = list(args.tpl_sec)

    rate = CFG.preproc.target_rate_hz
    tol = CFG.window.match_tolerance_sec
    coda = CODA_SEC[args.body]
    # a correlation peak marks where the TEMPLATE START aligns, so the arrival
    # sits PRE_SEC later
    offset = PRE_SEC

    best = {"f1": -1.0}
    val_grid = []
    for tpl_sec in args.tpl_sec:
        templates = build_templates(args.body, "train", rate, tpl_sec)
        if not templates:
            print(f"tpl {tpl_sec:.0f}s: no templates fit, skipped")
            continue
        print(f"\ntpl {tpl_sec:.0f}s: {len(templates)} templates; val ...")
        cc_val = cc_for_split(args.body, "val", templates, rate, tpl_sec)
        for k in K_GRID:
            s = score_at(cc_val, rate, k, coda, tol, offset)
            val_grid.append({"tpl_sec": tpl_sec, "k": k,
                             "val_f1": round(s.f1, 4),
                             "val_p": round(s.precision, 4),
                             "val_r": round(s.recall, 4)})
            print(f"  k={k:4.1f} MAD -> val P {s.precision:.3f} "
                  f"R {s.recall:.3f} F1 {s.f1:.3f}")
            if s.f1 > best["f1"]:
                best = {"f1": s.f1, "k": k, "tpl_sec": tpl_sec,
                        "n_templates": len(templates)}
    if best["f1"] < 0:
        sys.exit("no usable templates at any length")
    best_k, best_f1 = best["k"], best["f1"]
    print(f"\nselected tpl={best['tpl_sec']:.0f}s k={best_k} "
          f"(val F1 {best_f1:.3f})")

    templates = build_templates(args.body, "train", rate, best["tpl_sec"])
    print("test correlations ...")
    cc_test = cc_for_split(args.body, "test", templates, rate, best["tpl_sec"])
    s = score_at(cc_test, rate, best_k, coda, tol, offset)
    args.tpl_sec = best["tpl_sec"]

    # ORACLE upper bound: the best this baseline could reach if its operating
    # point were chosen on TEST. Not a reportable score — it is cheating — but
    # it forecloses the objection that the baseline was under-tuned, since
    # even this generous number loses to the learned detectors.
    print("\noracle sweep (test-tuned, NOT the reported number) ...")
    oracle = {"f1": -1.0}
    for tpl_sec in args.tpl_sec_all:
        tps = build_templates(args.body, "train", rate, tpl_sec)
        if not tps:
            continue
        cc_t = cc_for_split(args.body, "test", tps, rate, tpl_sec)
        for k in K_GRID:
            so = score_at(cc_t, rate, k, coda, tol, offset)
            if so.f1 > oracle["f1"]:
                oracle = {"f1": round(so.f1, 4), "k": k, "tpl_sec": tpl_sec,
                          "precision": round(so.precision, 4),
                          "recall": round(so.recall, 4)}
    print(f"oracle best: {oracle}")

    result = {
        "method": "matched filter (max normalised CC over train templates)",
        "body": args.body,
        "n_templates": len(templates),
        "template_sec": args.tpl_sec,
        "pre_sec": PRE_SEC,
        "threshold_k_mad": best_k,
        "threshold_tuned_on": "val (both template length and k)",
        "val_f1": round(best_f1, 4),
        "val_grid": val_grid,
        "oracle_test_tuned_upper_bound": oracle,
        "tolerance_sec": tol,
        "suppress_sec": coda,
        "matched_filter": s.as_dict(),
        "note": "templates come from the TRAIN split only, the same 45 "
                "Grade-A events the injection engine uses, so both methods "
                "see the same labelled information.",
    }
    out = PROJECT_ROOT / "results" / f"matched_filter_{args.body}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
