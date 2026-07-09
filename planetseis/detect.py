"""Sliding-window inference over a continuous trace.

Shared by evaluation and the web app so served results always match reported
results (F-16).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import Config
from .preprocessing import normalize_window


@dataclass
class Detection:
    time_sec: float        # predicted arrival, seconds from trace start
    confidence: float      # max window probability supporting this detection


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
    """Returns (detections, window_start_secs, window_probs).

    `suppress_sec`: classical dead time — a detection inside this interval
    after a stronger one is treated as its coda, not a new event.
    """
    n, hop = cfg.window.n_samples, cfg.window.hop
    model.eval().to(device)
    starts = list(range(0, max(len(trace) - n + 1, 1), hop))
    if len(trace) < n:  # short upload: pad to one window
        trace = np.pad(trace, (0, n - len(trace)))
        starts = [0]

    probs, offsets = [], []
    for b in range(0, len(starts), batch_size):
        batch = np.stack(
            [normalize_window(trace[s : s + n]) for s in starts[b : b + batch_size]]
        )
        x = torch.from_numpy(batch).unsqueeze(1).to(device)
        logit, offset, _ = model(x)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        offsets.append(offset.cpu().numpy())
    probs = np.concatenate(probs)
    offsets = np.concatenate(offsets)
    start_secs = np.array(starts) / rate_hz
    win_sec = n / rate_hz

    # Windows overlap 50%, so one event fires in several windows — but nearby
    # events can share one contiguous hot run, so we cluster by each hot
    # window's PREDICTED ARRIVAL TIME, not by window adjacency. Votes within
    # win_sec/4 of the running cluster mean belong to the same physical event.
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
        merged.append(Detection(time_sec=t, confidence=float(w.max())))

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
    return merged, start_secs, probs
