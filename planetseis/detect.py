"""Sliding-window inference over a continuous trace.

Shared by evaluation and the web app so served results always match reported
results (F-16). `detect_events` is the deterministic path; `detect_events_mc`
adds MC-Dropout epistemic uncertainty per detection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .preprocessing import normalize_window


@dataclass
class Detection:
    time_sec: float          # predicted arrival, seconds from trace start
    confidence: float        # max window probability supporting this detection
    uncertainty: float = 0.0  # MC-Dropout std of that window's probability
    needs_review: bool = False


def _window_starts(trace: np.ndarray, n: int, hop: int):
    if len(trace) < n:  # short upload: pad to one window
        trace = np.pad(trace, (0, n - len(trace)))
        return trace, [0]
    return trace, list(range(0, len(trace) - n + 1, hop))


def _forward_windows(model, trace, starts, n, device, batch_size=256):
    probs, offsets = [], []
    for b in range(0, len(starts), batch_size):
        batch = np.stack(
            [normalize_window(trace[s : s + n]) for s in starts[b : b + batch_size]]
        )
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        logit, offset, _ = model(x)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        offsets.append(offset.cpu().numpy())
    return np.concatenate(probs), np.concatenate(offsets)


def cluster_detections(
    probs: np.ndarray,
    offsets: np.ndarray,
    start_secs: np.ndarray,
    win_sec: float,
    threshold: float,
    suppress_sec: float = 0.0,
    stds: np.ndarray | None = None,
) -> list[Detection]:
    """Windows overlap 50%, so one event fires in several windows — but nearby
    events can share one contiguous hot run, so cluster by each hot window's
    PREDICTED ARRIVAL TIME, not by window adjacency. Votes within win_sec/4 of
    the running cluster mean belong to the same physical event. `suppress_sec`
    then applies classical dead time over each event's ring-down."""
    arrival_times = start_secs + offsets * win_sec
    hot_idx = np.where(probs >= threshold)[0]
    order = hot_idx[np.argsort(arrival_times[hot_idx])]
    gap = win_sec / 4
    merged: list[Detection] = []
    cluster: list[int] = []

    def flush(cluster):
        if not cluster:
            return
        w = probs[cluster]
        t = float(np.average(arrival_times[cluster], weights=w))
        k_best = cluster[int(np.argmax(w))]
        unc = float(stds[k_best]) if stds is not None else 0.0
        merged.append(Detection(time_sec=t, confidence=float(w.max()), uncertainty=unc))

    for k in order:
        if cluster and arrival_times[k] - arrival_times[cluster[-1]] > gap:
            flush(cluster)
            cluster = []
        cluster.append(k)
    flush(cluster)
    merged.sort(key=lambda d: d.time_sec)

    if suppress_sec > 0 and merged:
        kept: list[Detection] = []
        for d in merged:
            prev = kept[-1] if kept else None
            if prev and d.time_sec - prev.time_sec < suppress_sec:
                if d.confidence > prev.confidence:
                    kept[-1] = d  # stronger arrival wins the dead-time slot
            else:
                kept.append(d)
        merged = kept
    return merged


@torch.no_grad()
def detect_events(
    model,
    trace: np.ndarray,
    rate_hz: float,
    cfg: Config,
    threshold: float = 0.5,
    device: str = "cpu",
    batch_size: int = 256,
    suppress_sec: float = 0.0,
) -> tuple[list[Detection], np.ndarray, np.ndarray]:
    """Deterministic detection. Returns (detections, window_start_secs, window_probs)."""
    n, hop = cfg.window.n_samples, cfg.window.hop
    model.eval().to(device)
    trace, starts = _window_starts(trace, n, hop)
    probs, offsets = _forward_windows(model, trace, starts, n, device, batch_size)
    start_secs = np.array(starts) / rate_hz
    win_sec = n / rate_hz
    dets = cluster_detections(probs, offsets, start_secs, win_sec, threshold, suppress_sec)
    return dets, start_secs, probs


def _enable_mc_dropout(model):
    """eval() everywhere (BatchNorm must use running stats), dropout stochastic."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def detect_events_mc(
    model,
    trace: np.ndarray,
    rate_hz: float,
    cfg: Config,
    threshold: float = 0.5,
    device: str = "cpu",
    batch_size: int = 256,
    suppress_sec: float = 0.0,
    n_passes: int = 20,
    review_band: tuple[float, float] = (0.5, None),
    std_review: float = 0.15,
) -> tuple[list[Detection], np.ndarray, np.ndarray, np.ndarray]:
    """MC-Dropout detection (Gal & Ghahramani 2016) for planetary label scarcity:
    every detection carries an epistemic uncertainty, and borderline ones are
    flagged for human review instead of silently accepted or dropped.

    Detections are formed at `review_band[0]` (low bar), then split:
      confidence >= threshold and std <= std_review  -> auto-accept
      otherwise                                      -> needs_review
    Returns (detections, window_start_secs, mean_probs, std_probs).
    """
    n, hop = cfg.window.n_samples, cfg.window.hop
    model.to(device)
    _enable_mc_dropout(model)
    trace, starts = _window_starts(trace, n, hop)

    all_probs, all_offsets = [], []
    for _ in range(n_passes):
        p, o = _forward_windows(model, trace, starts, n, device, batch_size)
        all_probs.append(p)
        all_offsets.append(o)
    model.eval()
    probs = np.stack(all_probs)          # [T, W]
    mean_p, std_p = probs.mean(axis=0), probs.std(axis=0)
    mean_off = np.stack(all_offsets).mean(axis=0)

    start_secs = np.array(starts) / rate_hz
    win_sec = n / rate_hz
    low_bar = review_band[0]
    dets = cluster_detections(mean_p, mean_off, start_secs, win_sec,
                              low_bar, suppress_sec, stds=std_p)
    for d in dets:
        d.needs_review = not (d.confidence >= threshold and d.uncertainty <= std_review)
    return dets, start_secs, mean_p, std_p
