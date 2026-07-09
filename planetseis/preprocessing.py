"""Trace loading and preprocessing shared by training, evaluation and the app.

This module is the ONLY place raw traces are turned into model-ready arrays.
The Streamlit app and the training pipeline both import from here (PRD F-16).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import obspy
import pandas as pd

from .config import PreprocConfig


def load_trace(source, filename: str | None = None) -> tuple[np.ndarray, float, obspy.UTCDateTime | None]:
    """Load a single-channel trace from miniSEED/SAC/CSV.

    `source` may be a path or a file-like object (Streamlit upload).
    Returns (samples, sampling_rate_hz, start_time or None).
    CSV must have a relative-time column and a velocity/amplitude column
    (the Space Apps packet format), or be a single column of samples.
    """
    name = (filename or str(source)).lower()
    if name.endswith(".csv"):
        return _load_csv(source)

    if isinstance(source, (str, Path)):
        st = obspy.read(str(source))
    else:
        st = obspy.read(io.BytesIO(source.read()))
    st.merge(method=1, fill_value=0)  # close gaps; real planetary data has dropouts (F-9)
    tr = st[0]
    data = np.asarray(tr.data, dtype=np.float64)
    data = np.nan_to_num(data, nan=0.0)
    return data, float(tr.stats.sampling_rate), tr.stats.starttime


def _load_csv(source) -> tuple[np.ndarray, float, None]:
    df = pd.read_csv(source)
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 1:
        # bare sample column; assume packet-native rate is unknowable -> caller
        # must resample knowing the true rate, we default to 6.625 Hz
        return num.iloc[:, 0].to_numpy(dtype=np.float64), 6.625, None
    # Space Apps format: time_rel column + velocity column
    rel_col = next((c for c in df.columns if "rel" in c.lower()), num.columns[0])
    val_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ("velocity", "amplitude", "value"))),
        num.columns[-1],
    )
    t = df[rel_col].to_numpy(dtype=np.float64)
    v = np.nan_to_num(df[val_col].to_numpy(dtype=np.float64), nan=0.0)
    dt = np.median(np.diff(t[: min(len(t), 10000)]))
    rate = 1.0 / dt if dt > 0 else 6.625
    return v, float(rate), None


def preprocess(
    data: np.ndarray, rate_hz: float, cfg: PreprocConfig
) -> tuple[np.ndarray, float]:
    """Detrend -> bandpass -> resample to the common rate.

    Returns (processed_samples, target_rate_hz). Normalization is per-window
    (see `normalize_window`), never global, so no train/test statistics can
    leak (PRD F-11).
    """
    tr = obspy.Trace(np.asarray(data, dtype=np.float64))
    tr.stats.sampling_rate = rate_hz
    tr.detrend("demean")
    tr.detrend("linear")
    lo, hi = cfg.band_hz
    hi = min(hi, 0.45 * rate_hz)  # stay under source Nyquist before resampling
    tr.filter("bandpass", freqmin=lo, freqmax=hi, corners=4, zerophase=True)
    if abs(rate_hz - cfg.target_rate_hz) > 1e-6:
        tr.resample(cfg.target_rate_hz, no_filter=True)  # bandpass above is the anti-alias filter
    return np.asarray(tr.data, dtype=np.float32), cfg.target_rate_hz


def normalize_window(w: np.ndarray) -> np.ndarray:
    """Per-window z-score. Statistics come from this window alone."""
    std = w.std()
    if std < 1e-12:
        return np.zeros_like(w, dtype=np.float32)
    return ((w - w.mean()) / std).astype(np.float32)
