"""Synthetic event injection: unlimited mask-labeled training data (MQNet).

The central obstacle of planetary ML is label scarcity (45 lunar training
events). MarsQuakeNet's answer, reproduced here: cut real event *templates*
out of the labeled traces, spectrally gate away their background noise, then
inject them into event-free *noise* windows at random amplitudes and offsets.
Because the event and noise components are known separately, the exact
per-pixel energy-ratio mask |S_ev| / (|S_ev| + |S_noise|) is computable —
turning 45 labels into an unbounded stream of supervised (input, mask) pairs
spanning SNRs far below anything the catalog contains.

File-level split discipline is preserved: a template bank and a noise pool
are built from ONE split's continuous traces only, so a val/test file never
contributes energy to a training sample.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import CODA_SEC, DATA_CACHE
from .spectral import N_SAMPLES, RATE_HZ, SEC_PER_BIN, normalize_stft, ratio_mask, stft_window

# Template geometry. Lunar picks are minute-quantized and onsets emergent, so
# the template starts PRE_SEC before the catalog pick to keep the true onset.
PRE_SEC = 120.0
PRE = int(PRE_SEC * RATE_HZ)

# Exclusion zone around each pick when harvesting noise windows: generous
# pre-event margin plus the body's full ring-down.
NOISE_GUARD_PRE_SEC = 900.0


@dataclass
class Template:
    """One cleaned event template plus bookkeeping."""
    data: np.ndarray        # float32 [N_SAMPLES], spectrally gated event
    onset: int              # sample index of catalog pick inside `data`
    rms: float              # RMS of the 10-min segment after onset
    source: str             # trace file stem (for provenance)


def despike(x: np.ndarray, scale_seg: np.ndarray | None = None,
            k: float = 10.0) -> np.ndarray:
    """Clip impulsive artifacts (Apollo thermal ticks, dropout edges).

    These spikes are broadband vertical stripes in the STFT; left inside a
    template they teach the network 'stripe = event' and it then fires on
    every real glitch. Emergent planetary events carry no single-sample
    energy anywhere near k x the robust scale of the event coda, so clipping
    there removes spikes while leaving event energy intact.
    """
    ref = scale_seg if scale_seg is not None and len(scale_seg) else x
    mad = np.median(np.abs(ref - np.median(ref)))
    lim = k * 1.4826 * mad
    if lim <= 0:
        return x
    return np.clip(x, -lim, lim)


def _spectral_gate(event_seg: np.ndarray, noise_seg: np.ndarray) -> np.ndarray:
    """Suppress background noise inside an event cutout.

    Estimate the per-frequency median noise magnitude from a pre-event
    segment of the same trace, subtract it (over-subtraction factor 1.5)
    from the event STFT magnitude, floor at zero, keep the event phase.
    The result is an approximately noise-free template — the residual it
    leaves behind is what lets the computed ratio mask stay honest.
    """
    S_ev = stft_window(event_seg)
    S_nz = stft_window(noise_seg)
    noise_prof = np.median(np.abs(S_nz), axis=1, keepdims=True)   # [F, 1]
    mag = np.maximum(np.abs(S_ev) - 1.5 * noise_prof, 0.0)
    from .spectral import istft_window
    return istft_window(mag * np.exp(1j * np.angle(S_ev)))


def build_template_bank(body: str, split: str = "train") -> list[Template]:
    """Extract + clean one template per catalog pick from a split's traces."""
    cont = DATA_CACHE / body / "continuous" / split
    bank: list[Template] = []
    for p in sorted(cont.glob("*.npz")):
        z = np.load(p)
        trace, rate = z["trace"], float(z["rate"])
        assert abs(rate - RATE_HZ) < 1e-6
        for pick in z["picks"]:
            start = int(pick * rate) - PRE
            if start < N_SAMPLES or start + N_SAMPLES > len(trace):
                # need a clean pre-event window of equal length for gating
                if start < 0 or start + N_SAMPLES > len(trace):
                    continue
                noise_seg = None
            else:
                noise_seg = trace[start - N_SAMPLES : start]
            event_seg = trace[start : start + N_SAMPLES].astype(np.float32)
            core = event_seg[PRE : PRE + int(600 * rate)]
            event_seg = despike(event_seg, scale_seg=core)
            if noise_seg is None:
                cleaned = event_seg - np.median(event_seg)
            else:
                noise_seg = despike(noise_seg.astype(np.float32))
                cleaned = _spectral_gate(event_seg, noise_seg)
            post = cleaned[PRE : PRE + int(600 * rate)]
            rms = float(np.sqrt(np.mean(post**2)) + 1e-20)
            if rms < 1e-12:
                continue
            bank.append(Template(cleaned.astype(np.float32), PRE, rms, p.stem))
    return bank


class NoisePool:
    """Random event-free window slices from a split's continuous traces.

    `screen` optionally supplies extra event times per file (stem -> seconds
    relative to trace start) that are guarded exactly like catalog picks but
    never become templates. It exists because the benchmark's Grade-A labels
    are a strict SUBSET of the Nakamura catalog, so guarding only the picks
    leaves ~5,300 real S12 events harvestable as "noise" — 8.4 % of eligible
    window starts on the lunar train split. Training those against a zero
    mask teaches the detector to suppress genuine moonquakes: positive-
    unlabeled contamination, not a nuisance. Build with
    scripts/build_nakamura_screen.py.
    """

    def __init__(self, body: str, split: str = "train",
                 screen: dict[str, list[float]] | None = None):
        cont = DATA_CACHE / body / "continuous" / split
        guard_post = CODA_SEC[body] + NOISE_GUARD_PRE_SEC
        self.traces: list[np.ndarray] = []
        self.valid_starts: list[np.ndarray] = []
        self.n_screened = 0        # window starts removed by `screen` alone
        for p in sorted(cont.glob("*.npz")):
            z = np.load(p)
            trace, rate, picks = z["trace"], float(z["rate"]), z["picks"]
            ok = np.ones(len(trace) - N_SAMPLES + 1, dtype=bool)

            def _guard(t: float):
                lo = int((t - NOISE_GUARD_PRE_SEC) * rate) - N_SAMPLES
                hi = int((t + guard_post) * rate)
                ok[max(lo, 0) : min(hi, len(ok))] = False

            for pick in picks:
                _guard(float(pick))
            if screen:
                before = int(ok.sum())
                for t in screen.get(p.stem, ()):
                    _guard(float(t))
                self.n_screened += before - int(ok.sum())
            starts = np.where(ok)[0]
            if len(starts):
                self.traces.append(trace.astype(np.float32))
                self.valid_starts.append(starts)
        if not self.traces:
            raise RuntimeError(f"no noise available for {body}/{split}")

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        i = rng.integers(len(self.traces))
        s = int(rng.choice(self.valid_starts[i]))
        return self.traces[i][s : s + N_SAMPLES].copy()


def inject(noise: np.ndarray, tpl: Template, snr: float, onset_frac: float
           ) -> tuple[np.ndarray, np.ndarray, float]:
    """Place a template into a noise window at the given SNR and position.

    SNR is defined as (scaled event RMS over first 10 min of coda) /
    (noise window RMS) — both already bandpassed by the shared preprocessing,
    so plain RMS is the in-band amplitude. Returns (event_component,
    noise_component, onset_sec); the caller sums them and builds the mask.
    """
    noise_rms = float(np.sqrt(np.mean(noise**2)) + 1e-20)
    scale = snr * noise_rms / tpl.rms
    onset_idx = int(onset_frac * N_SAMPLES)
    ev = np.zeros(N_SAMPLES, dtype=np.float32)
    src = tpl.data * scale
    # copy the template so its onset lands at onset_idx, truncated at edges
    lo = onset_idx - tpl.onset
    src_lo, dst_lo = (max(0, -lo), max(0, lo))
    n = min(N_SAMPLES - dst_lo, len(src) - src_lo)
    if n > 0:
        ev[dst_lo : dst_lo + n] = src[src_lo : src_lo + n]
    return ev, noise, onset_idx / RATE_HZ


def make_glitches(rng: np.random.Generator, noise_std: float) -> np.ndarray:
    """Synthetic impulsive artifacts, injected as part of the NOISE component.

    Apollo/InSight recordings are full of non-seismic transients (thermal
    ticks, glitches/donks) that appear as broadband stripes in the STFT —
    exactly what an injected-event-only model mistakes for events. Injecting
    them with a zero mask target teaches the network to suppress them
    (MQNet's noise set played the same role). Two families:
      impulses  1-4 samples, 5-40x noise std
      bursts    1-8 s decaying oscillation, 3-20x noise std
    """
    g = np.zeros(N_SAMPLES, dtype=np.float32)
    for _ in range(1 + rng.poisson(1.5)):
        pos = int(rng.uniform(0, N_SAMPLES - 60))
        if rng.random() < 0.6:  # impulse
            width = int(rng.integers(1, 5))
            amp = rng.uniform(5, 40) * noise_std * rng.choice((-1, 1))
            g[pos : pos + width] += amp
        else:  # decaying burst
            dur = int(rng.uniform(1.0, 8.0) * RATE_HZ)
            t = np.arange(dur) / RATE_HZ
            f = rng.uniform(0.5, 3.0)
            amp = rng.uniform(3, 20) * noise_std
            burst = amp * np.sin(2 * np.pi * f * t) * np.exp(-t / rng.uniform(0.3, 2.0))
            end = min(pos + dur, N_SAMPLES)
            g[pos:end] += burst[: end - pos].astype(np.float32)
    return g


class InjectionDataset(Dataset):
    """On-the-fly (input, mask) pairs for U-Net training.

    Sample mix per epoch (MQNet trains on event+noise and pure-noise samples):
      p_event      one injected event at log-uniform SNR
      p_double     two events (teaches separation of nearby arrivals)
      remainder    pure noise, all-zero mask
    Independently, with p_glitch the noise component gains synthetic
    impulsive artifacts (mask stays 0 there — they are noise).
    A fixed `seed` plus `epoch_len` makes the val variant deterministic.
    """

    def __init__(self, body: str, split: str, epoch_len: int = 12800,
                 snr_range: tuple[float, float] = (0.4, 12.0),
                 p_event: float = 0.6, p_double: float = 0.1,
                 p_glitch: float = 0.4, p_hardneg: float = 0.0,
                 hardneg_windows: np.ndarray | None = None,
                 seed: int | None = None,
                 screen: dict[str, list[float]] | None = None):
        self.bank = build_template_bank(body, split)
        self.pool = NoisePool(body, split, screen=screen)
        # mined false-positive windows used as extra 'noise' sources: energy
        # inside them is event-like but carries a zero mask target (they may
        # still receive an injected event on top, which IS labeled)
        self.hardneg = (hardneg_windows if hardneg_windows is not None
                        and len(hardneg_windows) else None)
        self.p_hardneg = p_hardneg if self.hardneg is not None else 0.0
        if not self.bank:
            raise RuntimeError(f"no templates for {body}/{split}")
        self.epoch_len = epoch_len
        self.snr_range = snr_range
        self.p_event = p_event
        self.p_double = p_double
        self.p_glitch = p_glitch
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        return self.epoch_len

    def _rng(self, idx: int) -> np.random.Generator:
        base = self.seed if self.seed is not None else np.random.SeedSequence().entropy
        return np.random.default_rng([base, self._epoch, idx])

    def _sample_snr(self, rng) -> float:
        lo, hi = self.snr_range
        return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))

    def __getitem__(self, idx: int):
        rng = self._rng(idx)
        if rng.random() < self.p_hardneg:
            noise = self.hardneg[rng.integers(len(self.hardneg))].copy()
        else:
            noise = self.pool.sample(rng)
            if rng.random() < 0.5:
                noise = -noise[::-1].copy()  # polarity+time-reversal augment
            if rng.random() < self.p_glitch:
                noise = noise + make_glitches(rng, float(noise.std()) + 1e-20)
        r = rng.random()
        ev = np.zeros(N_SAMPLES, dtype=np.float32)
        if r < self.p_event + self.p_double:
            tpl = self.bank[rng.integers(len(self.bank))]
            e1, _, _ = inject(noise, tpl, self._sample_snr(rng),
                              float(rng.uniform(0.02, 0.85)))
            ev += e1
            if r < self.p_double:
                tpl2 = self.bank[rng.integers(len(self.bank))]
                e2, _, _ = inject(noise, tpl2, self._sample_snr(rng),
                                  float(rng.uniform(0.02, 0.85)))
                ev += e2
        S_ev = stft_window(ev) if ev.any() else None
        S_nz = stft_window(noise)
        if S_ev is None:
            mask = np.zeros(S_nz.shape, dtype=np.float32)
            x = normalize_stft(S_nz)
        else:
            mask = ratio_mask(S_ev, S_nz)
            x = normalize_stft(S_ev + S_nz)
        return torch.from_numpy(x), torch.from_numpy(mask[None])
