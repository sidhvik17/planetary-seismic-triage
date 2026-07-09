"""Segment preprocessed traces into labeled, overlapping windows (FR-4)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WindowConfig


@dataclass
class Window:
    data: np.ndarray        # float32 [n_samples], NOT yet normalized
    start_sec: float        # offset of window start within the trace
    label: int              # 1 if a catalog pick falls in the central region
    offset_frac: float      # pick position within window in [0,1]; -1 if label==0


def make_windows(
    trace: np.ndarray,
    rate_hz: float,
    arrivals_sec: list[float],
    cfg: WindowConfig,
) -> list[Window]:
    n, hop = cfg.n_samples, cfg.hop
    win_sec = n / rate_hz
    margin = cfg.edge_margin_frac
    out: list[Window] = []
    for start in range(0, len(trace) - n + 1, hop):
        start_sec = start / rate_hz
        label, offset = 0, -1.0
        for t in arrivals_sec:
            frac = (t - start_sec) / win_sec
            if margin <= frac <= 1.0 - margin:
                label, offset = 1, float(frac)
                break
        out.append(Window(trace[start : start + n], start_sec, label, offset))
    return out


def positives_around_pick(
    trace: np.ndarray,
    rate_hz: float,
    pick_sec: float,
    cfg: WindowConfig,
    n_shifts: int = 8,
    rng: np.random.Generator | None = None,
) -> list[Window]:
    """Extra positive windows with the pick at random positions.

    This is the time-shift augmentation (FR-5): re-window the same event so the
    model cannot learn 'event is always centered'. Also multiplies the tiny
    positive class (F-5).
    """
    rng = rng or np.random.default_rng()
    n = cfg.n_samples
    win_sec = n / rate_hz
    margin = cfg.edge_margin_frac + 0.05
    out = []
    for _ in range(n_shifts):
        frac = rng.uniform(margin, 1.0 - margin)
        start = int(round((pick_sec - frac * win_sec) * rate_hz))
        if start < 0 or start + n > len(trace):
            continue
        out.append(Window(trace[start : start + n], start / rate_hz, 1, float(frac)))
    return out
