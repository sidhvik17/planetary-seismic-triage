"""Detection scoring: precision, recall, arrival-time MAE (FR-9).

A detection matches a catalog pick if within the stated tolerance (F-7).
Greedy one-to-one matching by time distance. Same scorer for CNN and STA/LTA
baseline so numbers are directly comparable (F-4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Scores:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    abs_errors_sec: list = field(default_factory=list)

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mae_sec(self):
        return float(np.mean(self.abs_errors_sec)) if self.abs_errors_sec else float("nan")

    def merge(self, other: "Scores"):
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn
        self.abs_errors_sec += other.abs_errors_sec

    def as_dict(self):
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "mae_sec": round(self.mae_sec, 2) if self.abs_errors_sec else None,
            "median_ae_sec": round(float(np.median(self.abs_errors_sec)), 2)
            if self.abs_errors_sec else None,
        }


def score_trace(
    detected_times: list[float], truth_times: list[float], tolerance_sec: float
) -> Scores:
    s = Scores()
    unmatched_truth = list(truth_times)
    for dt in sorted(detected_times):
        if not unmatched_truth:
            s.fp += 1
            continue
        errs = [abs(dt - t) for t in unmatched_truth]
        k = int(np.argmin(errs))
        if errs[k] <= tolerance_sec:
            s.tp += 1
            s.abs_errors_sec.append(errs[k])
            unmatched_truth.pop(k)
        else:
            s.fp += 1
    s.fn = len(unmatched_truth)
    return s
