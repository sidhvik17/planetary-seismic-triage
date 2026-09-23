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


def _samples(data, min_samples: int = 1) -> np.ndarray:
    """Validate amplitudes before they can contaminate filtering or inference."""
    if np.iscomplexobj(data):
        raise ValueError("Trace amplitudes must be real numbers.")
    try:
        values = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Trace amplitudes must be numeric.") from exc
    if values.ndim != 1:
        raise ValueError("Trace must contain one channel of samples (a 1D array).")
    if values.size < min_samples:
        raise ValueError(f"Trace must contain at least {min_samples} sample(s).")
    if not np.isfinite(values).all():
        raise ValueError("Trace amplitudes contain missing or non-finite values (NaN or infinity).")
    return values


def _positive_rate(rate, name: str = "Sampling rate") -> float:
    try:
        rate = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive, finite number in Hz.") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError(f"{name} must be a positive, finite number in Hz.")
    return rate


def load_trace(source, filename: str | None = None,
               max_samples: int | None = None) -> tuple[np.ndarray, float, obspy.UTCDateTime | None]:
    """Load a single-channel trace from miniSEED/SAC/CSV.

    `source` may be a path or a file-like object (Streamlit upload).
    Returns (samples, sampling_rate_hz, start_time or None).
    CSV must have a relative-time column and a velocity/amplitude column
    (the Space Apps packet format), or be a single column of samples.
    A sample-only CSV, with or without a header, assumes 6.625 Hz because it
    contains no timing information. Supply relative times for other rates.
    `max_samples` bounds the gap-filled span of untrusted input: a small file
    whose records are years apart would otherwise allocate the whole gap.
    """
    name = (filename or str(source)).lower()
    if name.endswith(".csv"):
        return _load_csv(source)

    if isinstance(source, (str, Path)):
        st = obspy.read(str(source))
    else:
        st = obspy.read(io.BytesIO(source.read()))
    if not st:
        raise ValueError("The file contains no traces.")
    if len({tr.id for tr in st}) != 1:
        raise ValueError("The file contains multiple channels; upload one channel at a time.")
    for tr in st:
        _positive_rate(tr.stats.sampling_rate)
        _samples(tr.data)
    if max_samples is not None:
        rate = max(float(tr.stats.sampling_rate) for tr in st)
        span = max(tr.stats.endtime for tr in st) - min(tr.stats.starttime for tr in st)
        if round(span * rate) + 1 > max_samples:
            raise ValueError(f"The recording spans more than {max_samples:,} samples once "
                             "gaps are filled; upload a shorter time range.")
    st.merge(method=1, fill_value=0)  # close gaps; real planetary data has dropouts (F-9)
    tr = st[0]
    data = _samples(tr.data)
    return data, float(tr.stats.sampling_rate), tr.stats.starttime


def _load_csv(source) -> tuple[np.ndarray, float, None]:
    try:
        df = pd.read_csv(source, skip_blank_lines=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ValueError("CSV must contain numeric samples or relative-time and amplitude columns.") from exc
    if len(df.columns) == 1:
        values = df.iloc[:, 0].to_numpy()
        # pandas treats the first sample of a headerless CSV as a header.
        # A numeric column name is therefore restored as the first sample.
        try:
            first_sample = float(str(df.columns[0]).strip())
        except ValueError:
            pass
        else:
            values = np.concatenate(([first_sample], values))
        return _samples(values), 6.625, None
    if df.empty:
        raise ValueError("CSV contains no samples.")
    num = df.select_dtypes(include=[np.number])
    # Space Apps format: time_rel column + velocity column
    rel_col = next((c for c in df.columns if "rel" in c.lower()), None)
    if rel_col is None and num.shape[1] >= 2:
        rel_col = num.columns[0]
    val_col = next(
        (c for c in df.columns if c != rel_col
         and any(k in c.lower() for k in ("velocity", "amplitude", "value"))),
        None,
    )
    if val_col is None:
        val_col = next((c for c in reversed(num.columns) if c != rel_col), None)
    if rel_col is None or val_col is None:
        raise ValueError("CSV needs relative-time and amplitude columns, or one sample column.")
    try:
        t = df[rel_col].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("CSV relative times must be numeric seconds.") from exc
    if len(t) < 2 or not np.isfinite(t).all():
        raise ValueError("CSV needs at least two finite relative times to determine the sampling rate.")
    v = _samples(df[val_col].to_numpy())
    deltas = np.diff(t)
    if not np.isfinite(deltas).all() or np.any(deltas <= 0):
        raise ValueError("CSV relative times must be strictly increasing, with no duplicates.")
    dt = np.median(np.diff(t[: min(len(t), 10000)]))
    if not np.allclose(deltas, dt, rtol=1e-3, atol=1e-6):
        raise ValueError("CSV relative times must be evenly spaced; resample irregular data before uploading.")
    rate = _positive_rate(1.0 / dt)
    return v, rate, None


def preprocess(
    data: np.ndarray, rate_hz: float, cfg: PreprocConfig
) -> tuple[np.ndarray, float]:
    """Detrend -> bandpass -> resample to the common rate.

    Returns (processed_samples, target_rate_hz). Normalization is per-window
    (see `normalize_window`), never global, so no train/test statistics can
    leak (PRD F-11).
    """
    values = _samples(data, min_samples=2)
    rate_hz = _positive_rate(rate_hz)
    target_rate = _positive_rate(cfg.target_rate_hz, "Target sampling rate")
    try:
        lo, hi = (float(value) for value in cfg.band_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError("Bandpass must contain two positive, increasing frequencies in Hz.") from exc
    if not np.isfinite([lo, hi]).all() or not 0 < lo < hi:
        raise ValueError("Bandpass must contain two positive, increasing frequencies in Hz.")
    hi = min(hi, 0.45 * rate_hz)  # stay under source Nyquist before resampling
    if hi <= lo:
        raise ValueError("Sampling rate is too low for the configured bandpass.")
    if hi >= 0.5 * target_rate:
        raise ValueError("Bandpass upper frequency must be below the target sampling rate's Nyquist frequency.")
    tr = obspy.Trace(values)
    tr.stats.sampling_rate = rate_hz
    tr.detrend("demean")
    tr.detrend("linear")
    tr.filter("bandpass", freqmin=lo, freqmax=hi, corners=4, zerophase=True)
    if abs(rate_hz - target_rate) > 1e-6:
        tr.resample(target_rate, no_filter=True)  # bandpass above is the anti-alias filter
    return np.asarray(tr.data, dtype=np.float32), target_rate


def normalize_window(w: np.ndarray) -> np.ndarray:
    """Per-window z-score. Statistics come from this window alone."""
    std = w.std()
    if std < 1e-12:
        return np.zeros_like(w, dtype=np.float32)
    return ((w - w.mean()) / std).astype(np.float32)
