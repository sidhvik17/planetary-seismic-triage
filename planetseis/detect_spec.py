"""Continuous-trace detection and denoising with the spectrogram U-Net.

Mirrors detect.py's contract (same Detection dataclass, same dead-time rule,
same scorer downstream) so U-Net results are directly comparable with the
1D CNN on the frozen benchmark. Detection statistic follows MQNet: the
predicted event mask is integrated over the protocol frequency band, stitched
across overlapping windows into one per-trace curve, and thresholded into
event regions; each region's arrival is its threshold-crossing time.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .detect import Detection
from .spectral import (
    BAND_ROWS,
    N_FREQ,
    N_SAMPLES,
    N_TIME,
    NOVERLAP,
    SEC_PER_BIN,
    istft_window,
    normalize_stft,
    stft_window,
)

HOP_BINS = 4096 // NOVERLAP   # window hop expressed in STFT time bins (64)


def _window_starts(n_trace: int, hop: int = 4096) -> list[int]:
    if n_trace <= N_SAMPLES:
        return [0]
    starts = list(range(0, n_trace - N_SAMPLES + 1, hop))
    if starts[-1] != n_trace - N_SAMPLES:   # cover the tail
        starts.append(n_trace - N_SAMPLES)
    return starts


def _stitch_curves(model, trace, starts, device, batch_size, keep_masks=False):
    """Forward all windows, average band-integrated mask into one timeline.

    Returns (curve [n_bins], count [n_bins], masks or None, Zs or None).
    """
    n_bins = int(np.ceil(len(trace) / NOVERLAP))
    acc = np.zeros(n_bins)
    cnt = np.zeros(n_bins)
    masks = [] if keep_masks else None
    Zs = [] if keep_masks else None
    for b in range(0, len(starts), batch_size):
        chunk = starts[b : b + batch_size]
        Zc = [stft_window(trace[s : s + N_SAMPLES]) for s in chunk]
        x = torch.from_numpy(np.stack([normalize_stft(Z) for Z in Zc])).to(device)
        with torch.no_grad():
            m = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()   # [B, F, T]
        for s, Z, mk in zip(chunk, Zc, m):
            b0 = s // NOVERLAP
            score = mk[BAND_ROWS, :].mean(axis=0)                  # [T]
            acc[b0 : b0 + N_TIME] += score[: n_bins - b0]
            cnt[b0 : b0 + N_TIME] += 1
            if keep_masks:
                masks.append(mk)
                Zs.append(Z)
    curve = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
    return curve, cnt, masks, Zs


def _smooth(curve: np.ndarray, k: int = 5) -> np.ndarray:
    if len(curve) < k:
        return curve
    kernel = np.ones(k) / k
    return np.convolve(curve, kernel, mode="same")


def curve_to_detections(
    curve: np.ndarray,
    threshold: float,
    suppress_sec: float = 0.0,
    stds: np.ndarray | None = None,
    min_bins: int = 3,
) -> list[Detection]:
    """Threshold the stitched curve into event regions -> arrivals.

    Arrival = first bin of the supra-threshold region (mask energy starts at
    onset); confidence = region peak. Regions shorter than `min_bins`
    (~29 s — real planetary events ring for minutes, glitch residue does
    not) are discarded as speckle. Dead-time suppression matches
    detect.cluster_detections: strongest arrival wins the ring-down slot.
    """
    hot = curve >= threshold
    dets: list[Detection] = []
    i = 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < len(hot) and hot[j]:
            j += 1
        if j - i >= min_bins:
            peak = int(i + np.argmax(curve[i:j]))
            unc = float(stds[peak]) if stds is not None else 0.0
            dets.append(Detection(time_sec=i * SEC_PER_BIN,
                                  confidence=float(curve[peak]),
                                  uncertainty=unc))
        i = j
    if suppress_sec > 0 and dets:
        kept: list[Detection] = []
        for d in dets:
            if kept and d.time_sec - kept[-1].time_sec < suppress_sec:
                if d.confidence > kept[-1].confidence:
                    kept[-1] = d
            else:
                kept.append(d)
        dets = kept
    return dets


EDGE_GUARD_BINS = 30   # ~290 s: detrend/bandpass edge transients at the very
                       # start/end of a day-file masquerade as onsets


@torch.no_grad()
def compute_curve(model, trace: np.ndarray, device: str = "cpu",
                  batch_size: int = 32) -> np.ndarray:
    """Stitched, smoothed detection curve for a whole trace (model forward
    happens only here — threshold/duration sweeps reuse the cached curve)."""
    model.eval().to(device)
    starts = _window_starts(len(trace))
    curve, _, _, _ = _stitch_curves(model, trace, starts, device, batch_size)
    curve = _smooth(curve)
    if len(curve) > 4 * EDGE_GUARD_BINS:
        curve[:EDGE_GUARD_BINS] = 0.0
        curve[-EDGE_GUARD_BINS:] = 0.0
    return curve


@torch.no_grad()
def detect_events_spec(
    model,
    trace: np.ndarray,
    rate_hz: float,
    cfg: Config,
    threshold: float = 0.3,
    device: str = "cpu",
    batch_size: int = 32,
    suppress_sec: float = 0.0,
    min_dur_sec: float = 30.0,
) -> tuple[list[Detection], np.ndarray]:
    """Deterministic U-Net detection. Returns (detections, stitched curve).

    `min_dur_sec`: minimum supra-threshold region duration. Real planetary
    events ring from minutes (Mars) to the better part of an hour (Moon,
    scattering coda); glitch residue and noise speckle do not — on the lunar
    val split this single gate removes ~94% of false regions at no recall
    cost. Tuned on val alongside the threshold (see eval_unet.py).
    """
    curve = compute_curve(model, trace, device, batch_size)
    min_bins = max(1, int(round(min_dur_sec / SEC_PER_BIN)))
    return curve_to_detections(curve, threshold, suppress_sec,
                               min_bins=min_bins), curve


def _enable_mc_dropout(model):
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout2d):
            m.train()


@torch.no_grad()
def detect_events_spec_mc(
    model,
    trace: np.ndarray,
    rate_hz: float,
    cfg: Config,
    threshold: float = 0.3,
    device: str = "cpu",
    batch_size: int = 32,
    suppress_sec: float = 0.0,
    n_passes: int = 10,
    review_low_bar: float = 0.5,
    std_review: float = 0.05,
    min_dur_sec: float = 30.0,
) -> tuple[list[Detection], np.ndarray, np.ndarray]:
    """MC-Dropout U-Net detection with per-detection epistemic uncertainty.

    Detections form at `review_low_bar * threshold`; those below the full
    threshold or with high curve std are flagged needs_review — the triage
    contract shared with detect.detect_events_mc.
    """
    model.to(device)
    _enable_mc_dropout(model)
    starts = _window_starts(len(trace))
    curves = []
    for _ in range(n_passes):
        c, _, _, _ = _stitch_curves(model, trace, starts, device, batch_size)
        c = _smooth(c)
        if len(c) > 4 * EDGE_GUARD_BINS:
            c[:EDGE_GUARD_BINS] = 0.0
            c[-EDGE_GUARD_BINS:] = 0.0
        curves.append(c)
    model.eval()
    curves = np.stack(curves)
    mean_c, std_c = curves.mean(axis=0), curves.std(axis=0)
    dets = curve_to_detections(mean_c, review_low_bar * threshold,
                               suppress_sec, stds=std_c,
                               min_bins=max(1, int(round(min_dur_sec / SEC_PER_BIN))))
    for d in dets:
        d.needs_review = not (d.confidence >= threshold
                              and d.uncertainty <= std_review)
    return dets, mean_c, std_c


@torch.no_grad()
def denoise_trace(
    model,
    trace: np.ndarray,
    rate_hz: float,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """MQNet denoising: multiply each window's complex STFT by its predicted
    event mask, invert, and overlap-average the reconstructions."""
    model.eval().to(device)
    starts = _window_starts(len(trace))
    out = np.zeros(len(trace))
    weight = np.zeros(len(trace))
    _, _, masks, Zs = _stitch_curves(model, trace, starts, device,
                                     batch_size, keep_masks=True)
    valid = N_SAMPLES - NOVERLAP   # last dropped STFT frame is unreliable
    for s, mk, Z in zip(starts, masks, Zs):
        rec = istft_window(mk * Z)[:valid]
        out[s : s + valid] += rec
        weight[s : s + valid] += 1
    return (out / np.maximum(weight, 1)).astype(np.float32)
