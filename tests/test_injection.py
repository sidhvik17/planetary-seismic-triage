"""Injection engine invariants. Uses synthetic traces — no data cache needed."""
import numpy as np
import pytest
import torch

from planetseis.injection import Template, inject
from planetseis.spectral import N_SAMPLES, RATE_HZ


def _template(onset=800, dur=3000):
    data = np.zeros(N_SAMPLES, dtype=np.float32)
    t = np.arange(dur)
    data[onset : onset + dur] = (np.sin(2 * np.pi * 1.0 * t / RATE_HZ)
                                 * np.exp(-t / (dur / 3))).astype(np.float32)
    post = data[onset : onset + int(600 * RATE_HZ)]
    return Template(data, onset, float(np.sqrt(np.mean(post**2))), "synth")


def test_inject_snr_scaling():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1.0, N_SAMPLES).astype(np.float32)
    tpl = _template()
    for snr in (0.5, 2.0, 8.0):
        ev, nz, onset_sec = inject(noise, tpl, snr, 0.3)
        onset = int(onset_sec * RATE_HZ)
        post = ev[onset : onset + int(600 * RATE_HZ)]
        got = np.sqrt(np.mean(post**2)) / np.sqrt(np.mean(noise**2))
        assert got == pytest.approx(snr, rel=0.05)


def test_inject_onset_position():
    noise = np.random.default_rng(1).normal(0, 1, N_SAMPLES).astype(np.float32)
    tpl = _template()
    ev, _, onset_sec = inject(noise, tpl, 5.0, 0.5)
    onset = int(onset_sec * RATE_HZ)
    assert abs(onset - N_SAMPLES // 2) <= 1
    # energy before onset only from the template's pre-onset context
    assert np.abs(ev[: onset - tpl.onset]).max() == 0.0


def test_inject_edge_truncation():
    """Late onset: template truncated at the window edge, no wraparound."""
    noise = np.random.default_rng(2).normal(0, 1, N_SAMPLES).astype(np.float32)
    tpl = _template()
    ev, _, _ = inject(noise, tpl, 5.0, 0.95)
    assert len(ev) == N_SAMPLES
    assert np.isfinite(ev).all()


@pytest.mark.skipif(
    not (list(__import__("planetseis.config", fromlist=["DATA_CACHE"])
          .DATA_CACHE.glob("lunar/continuous/train/*.npz"))),
    reason="data cache not built",
)
def test_dataset_shapes_and_determinism():
    from planetseis.injection import InjectionDataset

    ds = InjectionDataset("lunar", "train", epoch_len=8, seed=123)
    x, m = ds[0]
    assert x.shape == (2, 128, 128) and m.shape == (1, 128, 128)
    assert x.dtype == torch.float32 and m.dtype == torch.float32
    assert 0.0 <= float(m.min()) and float(m.max()) <= 1.0
    x2, m2 = ds[0]
    assert torch.equal(x, x2) and torch.equal(m, m2)


def test_despike_clips_spikes_not_signal():
    from planetseis.injection import despike

    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, N_SAMPLES).astype(np.float32)
    x[100] = 500.0
    x[7000:7003] = -300.0
    d = despike(x)
    assert np.abs(d).max() < 50           # spikes gone
    quiet = np.abs(x) < 5
    assert np.allclose(d[quiet], x[quiet])  # ordinary samples untouched


def test_glitches_are_transient():
    from planetseis.injection import make_glitches

    rng = np.random.default_rng(1)
    g = make_glitches(rng, noise_std=1.0)
    assert g.shape == (N_SAMPLES,)
    assert np.abs(g).max() >= 3.0          # visible above noise
    assert (g != 0).sum() < N_SAMPLES // 8  # sparse, not a sustained signal
