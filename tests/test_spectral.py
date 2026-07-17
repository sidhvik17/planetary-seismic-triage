"""Invariants of the STFT front end — the grid contract everything shares."""
import numpy as np
import pytest

from planetseis.spectral import (
    BAND_ROWS,
    N_FREQ,
    N_SAMPLES,
    N_TIME,
    SEC_PER_BIN,
    band_score,
    istft_window,
    normalize_stft,
    ratio_mask,
    stft_window,
)


def test_grid_shape():
    Z = stft_window(np.random.randn(N_SAMPLES).astype(np.float32))
    assert Z.shape == (N_FREQ, N_TIME) == (128, 128)


def test_wrong_length_rejected():
    with pytest.raises(ValueError):
        stft_window(np.zeros(N_SAMPLES - 1))


def test_istft_reconstructs():
    x = np.random.randn(N_SAMPLES).astype(np.float32)
    xr = istft_window(stft_window(x))
    # last dropped frame corrupts only the final NOVERLAP samples
    assert np.allclose(x[:-64], xr[:-64], atol=1e-5)


def test_time_bin_mapping():
    """An impulse at sample k must light up time bin ~k/64."""
    x = np.zeros(N_SAMPLES, dtype=np.float32)
    x[5000] = 1.0
    Z = np.abs(stft_window(x))
    peak_bin = int(np.argmax(Z.sum(axis=0)))
    assert abs(peak_bin - 5000 // 64) <= 1


def test_normalize_stft_is_local():
    """Scaling the input must not change the normalized output (per-window
    statistics only — no global scale can leak)."""
    x = np.random.randn(N_SAMPLES).astype(np.float32)
    a = normalize_stft(stft_window(x))
    b = normalize_stft(stft_window(x * 1e9))
    assert np.allclose(a, b, atol=1e-3)


def test_ratio_mask_bounds_and_floor():
    S_ev = stft_window(np.random.randn(N_SAMPLES).astype(np.float32))
    S_nz = stft_window(np.random.randn(N_SAMPLES).astype(np.float32))
    m = ratio_mask(S_ev, S_nz)
    assert m.min() >= 0.0 and m.max() <= 1.0
    # zero event -> mask ~ 0 everywhere (the floor kills 0/0 pixels)
    m0 = ratio_mask(np.zeros_like(S_ev), S_nz)
    assert m0.max() == 0.0


def test_band_score_range():
    m = np.random.rand(N_FREQ, N_TIME).astype(np.float32)
    s = band_score(m)
    assert s.shape == (N_TIME,)
    assert (s >= 0).all() and (s <= 1).all()
    assert len(BAND_ROWS) > 0 and SEC_PER_BIN > 0
