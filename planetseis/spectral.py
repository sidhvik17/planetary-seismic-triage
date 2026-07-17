"""Time-frequency front end for the spectrogram U-Net (MQNet-style).

MarsQuakeNet (Dahmen et al. 2022, JGR Planets) operates on 256x256 STFTs of
27-min windows at 20 sps. This module is the equivalent geometry for the
project's common 6.625 Hz single-channel protocol: the existing 8192-sample
(~20.6 min) window maps to a fixed 128x128 complex STFT, real+imag stacked as
2 input channels. All shapes are locked here so training, evaluation and the
app can never disagree about the grid.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import istft, stft

from .config import DEFAULT as CFG

RATE_HZ = CFG.preproc.target_rate_hz          # 6.625
N_SAMPLES = CFG.window.n_samples              # 8192
NPERSEG = 128
NOVERLAP = 64                                 # hop = 64 samples = 9.66 s / time bin
NFFT = 254                                    # -> 128 frequency bins, df = 26.1 mHz
N_FREQ = NFFT // 2 + 1                        # 128
N_TIME = N_SAMPLES // (NPERSEG - NOVERLAP)    # 128 (after boundary trim)
SEC_PER_BIN = (NPERSEG - NOVERLAP) / RATE_HZ  # 9.66 s

_STFT_KW = dict(fs=RATE_HZ, nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
_FWD_KW = dict(_STFT_KW, boundary="zeros", padded=True)
_INV_KW = dict(_STFT_KW, boundary=True)

FREQS_HZ = np.fft.rfftfreq(NFFT, d=1.0 / RATE_HZ)   # [128]

# Frequency rows integrated for detection: the protocol band (0.5-3.0 Hz).
BAND_ROWS = np.where((FREQS_HZ >= CFG.preproc.band_hz[0])
                     & (FREQS_HZ <= CFG.preproc.band_hz[1]))[0]


def stft_window(x: np.ndarray) -> np.ndarray:
    """Complex STFT of one 8192-sample window -> [128 freq, 128 time].

    scipy's boundary padding yields 129 frames centered at k*64 samples,
    k = 0..128; the last frame (centered on the window edge) is dropped so
    time bin k maps exactly to k * 64 / rate seconds from window start.
    """
    if len(x) != N_SAMPLES:
        raise ValueError(f"expected {N_SAMPLES} samples, got {len(x)}")
    _, _, Z = stft(x, **_FWD_KW)
    Z = Z[:, :-1]
    if Z.shape != (N_FREQ, N_TIME):  # guard against scipy version drift
        raise AssertionError(f"STFT grid {Z.shape} != {(N_FREQ, N_TIME)}")
    return Z


def istft_window(Z: np.ndarray) -> np.ndarray:
    """Inverse of `stft_window` -> 8192 samples (used for denoising output)."""
    pad = np.zeros((N_FREQ, N_TIME + 1), dtype=complex)
    pad[:, :-1] = Z
    _, x = istft(pad, **_INV_KW)
    return x[:N_SAMPLES].astype(np.float32)


def normalize_stft(Z: np.ndarray, clip: float = 50.0) -> np.ndarray:
    """Complex STFT -> [2, F, T] float32 model input.

    MQNet recipe: robust-scale real and imag independently (median/IQR over
    the whole window — statistics from this window alone, no global stats),
    then hard-clip at +-clip so glitch energy cannot saturate the network.
    """
    out = np.empty((2, *Z.shape), dtype=np.float32)
    for c, part in enumerate((Z.real, Z.imag)):
        med = np.median(part)
        q25, q75 = np.percentile(part, (25, 75))
        iqr = q75 - q25
        if iqr < 1e-20:
            out[c] = 0.0
            continue
        out[c] = np.clip((part - med) / iqr, -clip, clip)
    return out


def ratio_mask(S_event: np.ndarray, S_noise: np.ndarray,
               floor_frac: float = 0.05) -> np.ndarray:
    """MQNet target: event energy fraction per T-F pixel, in [0, 1].

    A floor tied to the window's in-band noise level is added to the
    denominator: outside the passband both components are ~0 and a bare
    ratio degenerates to 0/0 noise (pixels claiming 'event' where there is
    no energy at all). With the floor, mask -> 0 wherever the event's
    absolute energy is negligible compared to real noise.
    """
    ev = np.abs(S_event)
    nz = np.abs(S_noise)
    floor = floor_frac * float(np.median(nz[BAND_ROWS, :]))
    return (ev / (ev + nz + floor + 1e-20)).astype(np.float32)


def band_score(mask: np.ndarray) -> np.ndarray:
    """Integrate an event mask over the protocol frequency band -> [T] curve.

    This is the detection statistic (MQNet sums mask energy over frequency);
    normalized to [0, 1] by the band height so thresholds are comparable
    across grids.
    """
    return mask[BAND_ROWS, :].mean(axis=0)
