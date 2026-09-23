"""Goal 4: MC-Dropout uncertainty, calibration quality, human-review queue.

Three questions, answered on lunar continuous held-out data:
 1. Are the model's confidences trustworthy? -> reliability diagram + ECE,
    before/after temperature scaling (T fitted on val, never test).
 2. Does MC-Dropout uncertainty separate correct from wrong detections?
 3. If low-confidence detections go to a human review queue instead of being
    auto-accepted/dropped, how much recall does review recover, and how big
    is the queue per day of data?

Writes results/uncertainty.json. Corrected split (exclusive output):
    python scripts/uncertainty_eval.py --data-dir data/cache/lunar_grouped_v1
        --model models/lunar_grouped_v1_seed42.pt --threshold 0.97
        --output results/uncertainty_lunar_grouped_v1_seed42.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from planetseis.config import CODA_SEC, DEFAULT as CFG, DATA_CACHE, PROJECT_ROOT
from planetseis.detect import detect_events_mc, _forward_windows, _window_starts, _enable_mc_dropout
from planetseis.evaluate import Scores, score_trace
from planetseis.model import ARCHS, SeisCNN

BODY = "lunar"
N_PASSES = 20
ACCEPT_THR = 0.99          # same operating point as the headline result
REVIEW_LOW = 0.5           # detections in [0.5, accept) or high-std go to review
STD_REVIEW = 0.15


def window_labels(trace, rate, picks, n, hop):
    """Window-level ground truth mirroring training labels; coda windows are
    ambiguous (real energy, no onset) and excluded from calibration."""
    trace, starts = _window_starts(trace, n, hop)
    win_sec = n / rate
    coda = CODA_SEC[BODY]
    labels, keep = [], []
    for s in starts:
        s_sec = s / rate
        has_pick = any(0.05 * win_sec <= p - s_sec <= 0.95 * win_sec for p in picks)
        in_coda = any(s_sec < p + coda and s_sec + win_sec > p for p in picks)
        labels.append(1 if has_pick else 0)
        keep.append(has_pick or not in_coda)
    return trace, starts, np.array(labels), np.array(keep)


def collect(model, split, device, root):
    """Deterministic + MC window probabilities with labels, over a split."""
    det_p, mc_p, mc_s, ys = [], [], [], []
    n, hop = CFG.window.n_samples, CFG.window.hop
    for pth in sorted((root / "continuous" / split).glob("*.npz")):
        z = np.load(pth)
        trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
        trace, starts, labels, keep = window_labels(trace, rate, picks, n, hop)
        model.eval()
        with torch.no_grad():
            p_det, _ = _forward_windows(model, trace, starts, n, device)
            _enable_mc_dropout(model)
            stack = np.stack([_forward_windows(model, trace, starts, n, device)[0]
                              for _ in range(N_PASSES)])
            model.eval()
        det_p.append(p_det[keep])
        mc_p.append(stack.mean(0)[keep])
        mc_s.append(stack.std(0)[keep])
        ys.append(labels[keep])
    return (np.concatenate(det_p), np.concatenate(mc_p),
            np.concatenate(mc_s), np.concatenate(ys))


def ece(probs, labels, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    e, n = 0.0, len(probs)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi) if hi < 1 else (probs >= lo) & (probs <= hi)
        if m.sum() == 0:
            bins.append((0.5 * (lo + hi), None, 0))
            continue
        conf, acc = probs[m].mean(), labels[m].mean()
        e += m.sum() / n * abs(conf - acc)
        bins.append((0.5 * (lo + hi), (conf, acc), int(m.sum())))
    return e, bins


def fit_temperature(probs, labels):
    """Single-parameter temperature scaling on logits, NLL objective."""
    logits = torch.tensor(np.log(np.clip(probs, 1e-6, 1 - 1e-6) /
                                 np.clip(1 - probs, 1e-6, 1)), dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    logT = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits / logT.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(logT.exp())


def apply_temperature(probs, T):
    logits = np.log(np.clip(probs, 1e-6, 1 - 1e-6) / np.clip(1 - probs, 1e-6, 1))
    return 1 / (1 + np.exp(-logits / T))


def review_queue_eval(model, device, root, accept_thr):
    """Event-level: auto-accept vs review-queue outcome on the test split."""
    auto, combined = Scores(), Scores()
    n_review, n_days = 0, 0
    review_hits, review_dets = 0, 0
    for pth in sorted((root / "continuous" / "test").glob("*.npz")):
        z = np.load(pth)
        trace, rate, picks = z["trace"], float(z["rate"]), list(z["picks"])
        n_days += len(trace) / rate / 86400
        dets, _, _, _ = detect_events_mc(
            model, trace, rate, CFG, threshold=accept_thr, device=device,
            suppress_sec=CODA_SEC[BODY], n_passes=N_PASSES,
            review_band=(REVIEW_LOW, None), std_review=STD_REVIEW)
        acc = [d.time_sec for d in dets if not d.needs_review]
        rev = [d.time_sec for d in dets if d.needs_review]
        n_review += len(rev)
        auto.merge(score_trace(acc, picks, CFG.window.match_tolerance_sec))
        combined.merge(score_trace(acc + rev, picks, CFG.window.match_tolerance_sec))
        # how many review-queue entries are real events?
        s_rev = score_trace(rev, [p for p in picks], CFG.window.match_tolerance_sec)
        review_hits += s_rev.tp
        review_dets += len(rev)
    return {
        "auto_accept": auto.as_dict(),
        "with_review": combined.as_dict(),
        "review_queue_per_day": round(n_review / max(n_days, 1e-9), 2),
        "review_queue_total": n_review,
        "review_queue_true_events": review_hits,
        "accept_threshold": accept_thr,
        "review_low": REVIEW_LOW,
        "std_review": STD_REVIEW,
        "mc_passes": N_PASSES,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=PROJECT_ROOT / "runs" / BODY / "best.pt")
    ap.add_argument("--threshold", type=float, default=ACCEPT_THR,
                    help="auto-accept threshold: the model's validation-selected point")
    ap.add_argument("--data-dir", type=Path, help="versioned dataset root; requires --output")
    ap.add_argument("--output", type=Path, help="result JSON (exclusive with --data-dir)")
    args = ap.parse_args()
    root = args.data_dir or DATA_CACHE / BODY
    out_path = args.output or PROJECT_ROOT / "results" / "uncertainty.json"
    provenance = {"benchmark_id": "historical filename split", "data_manifest_sha256": None}
    if args.data_dir is not None:
        if args.output is None:
            ap.error("--data-dir requires --output")
        if out_path.exists():
            sys.exit(f"refusing to overwrite {out_path}")
        raw_manifest = (root / "manifest.json").read_bytes()
        provenance = {"benchmark_id": json.loads(raw_manifest)["benchmark_id"],
                      "data_manifest_sha256": hashlib.sha256(raw_manifest).hexdigest()}
    provenance["model"] = args.model.name
    provenance["model_sha256"] = hashlib.sha256(args.model.read_bytes()).hexdigest()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.model, map_location=device, weights_only=True)
    model = SeisCNN(channels=ARCHS[ck.get("arch", "base")]).to(device)
    model.load_state_dict(ck["model"])

    print("collecting val windows (for temperature fit)...")
    vdet, vmc, _, vy = collect(model, "val", device, root)
    print("collecting test windows...")
    tdet, tmc, tstd, ty = collect(model, "test", device, root)

    T = fit_temperature(vdet, vy)
    ece_det, bins_det = ece(tdet, ty)
    ece_temp, bins_temp = ece(apply_temperature(tdet, T), ty)
    ece_mc, bins_mc = ece(tmc, ty)
    print(f"T={T:.3f}  ECE raw={ece_det:.4f}  temp-scaled={ece_temp:.4f}  mc={ece_mc:.4f}")

    # does uncertainty separate hits from false alarms at the window level?
    hot = tmc >= 0.5
    std_tp = float(tstd[hot & (ty == 1)].mean()) if (hot & (ty == 1)).any() else None
    std_fp = float(tstd[hot & (ty == 0)].mean()) if (hot & (ty == 0)).any() else None
    print(f"mean MC std | true-event windows: {std_tp}, false-alarm windows: {std_fp}")

    print("event-level review-queue eval...")
    queue = review_queue_eval(model, device, root, args.threshold)

    out = {
        **provenance,
        "temperature": round(T, 3),
        "ece_raw": round(ece_det, 4),
        "ece_temp_scaled": round(ece_temp, 4),
        "ece_mc": round(ece_mc, 4),
        "mc_std_true_event_windows": round(std_tp, 4) if std_tp is not None else None,
        "mc_std_false_alarm_windows": round(std_fp, 4) if std_fp is not None else None,
        "review_queue": queue,
        "n_test_windows": int(len(ty)),
        "reliability_bins": {
            "raw": [(c, b, n) for c, b, n in bins_det],
            "temp": [(c, b, n) for c, b, n in bins_temp],
            "mc": [(c, b, n) for c, b, n in bins_mc],
        },
    }
    out_path.write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps({k: v for k, v in out.items() if k != "reliability_bins"},
                     indent=2, default=float))


if __name__ == "__main__":
    main()
