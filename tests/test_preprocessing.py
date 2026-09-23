"""Input validation and clean-data compatibility for the shared front end."""
import io

import numpy as np
import obspy
import pytest

from planetseis.config import PreprocConfig
from planetseis.preprocessing import load_trace, preprocess


def load_csv(text):
    return load_trace(io.StringIO(text), filename="trace.csv")


@pytest.mark.parametrize("text", [
    "samples\n1\n2\n3\n",
    "1\n2\n3\n",
])
def test_sample_csv_preserves_every_sample_and_documents_native_rate(text):
    samples, rate, start = load_csv(text)
    np.testing.assert_array_equal(samples, [1.0, 2.0, 3.0])
    assert rate == 6.625 and start is None


def test_packet_csv_uses_relative_time_and_velocity():
    samples, rate, start = load_csv(
        "time_abs,time_rel(sec),velocity(m/s)\n"
        "2026-01-01,0.000000,1e-10\n"
        "2026-01-01,0.150943,2e-10\n"
        "2026-01-01,0.301887,3e-10\n"
    )
    np.testing.assert_array_equal(samples, [1e-10, 2e-10, 3e-10])
    assert rate == pytest.approx(6.625, rel=1e-5)
    assert start is None


def test_two_numeric_columns_infer_sampling_rate():
    samples, rate, _ = load_csv("seconds,counts\n0,3\n0.05,4\n0.1,5\n")
    assert rate == 20.0
    np.testing.assert_array_equal(samples, [3, 4, 5])


@pytest.mark.parametrize("text, message", [
    ("", "CSV must contain"),
    ("samples\n", "at least 1 sample"),
    ("time_rel,velocity\n", "no samples"),
    ("samples\nwords\n", "numeric"),
    ("label,description\none,two\n", "relative-time and amplitude"),
    ("samples\n1\nNaN\n3\n", "non-finite"),
    ("samples\n1\n\n3\n", "non-finite"),
    ("samples\n1\ninf\n3\n", "non-finite"),
    ("time_rel,velocity\n0,1\n1,noisy\n", "numeric"),
    ("time_rel,velocity\n0,1\n1,inf\n", "non-finite"),
    ("time_rel,velocity\n0,1\n1,NaN\n", "non-finite"),
    ("time_rel,velocity\n0,1\n", "at least two finite"),
    ("time_rel,velocity\n0,1\nNaN,2\n", "at least two finite"),
    ("time_rel,velocity\n0,1\ninf,2\n", "at least two finite"),
    ("time_rel,velocity\n0,1\nunknown,2\n", "numeric seconds"),
    ("time_rel,velocity\n0,1\n0,2\n", "strictly increasing"),
    ("time_rel,velocity\n1,1\n0,2\n", "strictly increasing"),
    ("time_rel,velocity\n0,1\n1,2\n3,3\n", "evenly spaced"),
])
def test_bad_csv_reports_actionable_value_error(text, message):
    with pytest.raises(ValueError, match=message):
        load_csv(text)


def mseed_buffer(traces):
    stream = obspy.Stream(traces)
    output = io.BytesIO()
    stream.write(output, format="MSEED")
    output.seek(0)
    return output


def make_trace(values, channel="MHZ", start=None):
    return obspy.Trace(np.asarray(values, dtype=np.float64), header={
        "network": "XA", "station": "S12", "channel": channel,
        "sampling_rate": 6.625,
        "starttime": start or obspy.UTCDateTime("2026-01-01"),
    })


def test_mseed_multiple_channels_are_rejected():
    source = mseed_buffer([make_trace([1, 2, 3]), make_trace([4, 5, 6], "MHE")])
    with pytest.raises(ValueError, match="multiple channels"):
        load_trace(source, filename="trace.mseed")


def test_mseed_single_channel_segments_still_merge_and_fill_gaps():
    first = make_trace([1, 2, 3, 4, 5])
    second = make_trace([6, 7, 8, 9, 10], start=first.stats.starttime + 10 / 6.625)
    samples, rate, start = load_trace(mseed_buffer([first, second]), filename="trace.mseed")
    np.testing.assert_array_equal(samples, [1, 2, 3, 4, 5, 0, 0, 0, 0, 0, 6, 7, 8, 9, 10])
    assert rate == 6.625 and start == first.stats.starttime


def test_mseed_gap_fill_span_is_bounded_before_allocation():
    first = make_trace([1, 2, 3, 4, 5])
    # Two tiny records a century apart would otherwise fill ~2e10 zeros.
    second = make_trace([6, 7, 8], start=first.stats.starttime + 100 * 365.25 * 86400)
    with pytest.raises(ValueError, match="spans more than"):
        load_trace(mseed_buffer([first, second]), filename="trace.mseed", max_samples=1000)
    samples, _, _ = load_trace(mseed_buffer([first]), filename="trace.mseed", max_samples=5)
    np.testing.assert_array_equal(samples, [1, 2, 3, 4, 5])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_mseed_nonfinite_amplitudes_are_rejected(invalid):
    source = mseed_buffer([make_trace([1, invalid, 3])])
    with pytest.raises(ValueError, match="non-finite"):
        load_trace(source, filename="trace.mseed")


@pytest.mark.parametrize("values, message", [
    ([], "at least 2 sample"),
    ([1], "at least 2 sample"),
    ([[1, 2], [3, 4]], "one channel"),
    ([1, np.nan], "non-finite"),
    ([1, np.inf], "non-finite"),
    ([1, "broken"], "numeric"),
    ([1, 2j], "real numbers"),
])
def test_preprocess_rejects_invalid_samples(values, message):
    with pytest.raises(ValueError, match=message):
        preprocess(values, 6.625, PreprocConfig())


@pytest.mark.parametrize("rate", [0, -1, np.nan, np.inf, "unknown"])
def test_preprocess_rejects_invalid_rates(rate):
    with pytest.raises(ValueError, match="Sampling rate must be a positive, finite"):
        preprocess(np.ones(256), rate, PreprocConfig())


@pytest.mark.parametrize("cfg, rate, message", [
    (PreprocConfig(target_rate_hz=0), 6.625, "Target sampling rate"),
    (PreprocConfig(target_rate_hz=np.nan), 6.625, "Target sampling rate"),
    (PreprocConfig(band_hz=(3.0, 0.5)), 6.625, "Bandpass must contain"),
    (PreprocConfig(band_hz=(0, 3.0)), 6.625, "Bandpass must contain"),
    (PreprocConfig(band_hz=(0.5, np.inf)), 6.625, "Bandpass must contain"),
    (PreprocConfig(band_hz=(0.5,)), 6.625, "Bandpass must contain"),
    (PreprocConfig(), 1.0, "too low"),
    (PreprocConfig(target_rate_hz=2), 20.0, "Nyquist"),
])
def test_preprocess_rejects_unusable_filter_configuration(cfg, rate, message):
    with pytest.raises(ValueError, match=message):
        preprocess(np.ones(256), rate, cfg)


@pytest.mark.parametrize("rate", [6.625, 20.0])
def test_clean_preprocessing_matches_existing_packet_pipeline(rate):
    """Validation must not change valid Apollo or InSight model inputs."""
    cfg = PreprocConfig()
    time = np.arange(4096) / rate
    raw = np.sin(2 * np.pi * time) + 0.1 * np.sin(2 * np.pi * 0.1 * time)
    expected = obspy.Trace(raw.copy())
    expected.stats.sampling_rate = rate
    expected.detrend("demean")
    expected.detrend("linear")
    expected.filter("bandpass", freqmin=0.5, freqmax=min(3.0, 0.45 * rate),
                    corners=4, zerophase=True)
    if rate != cfg.target_rate_hz:
        expected.resample(cfg.target_rate_hz, no_filter=True)
    actual, actual_rate = preprocess(raw.copy(), rate, cfg)
    np.testing.assert_array_equal(actual, expected.data.astype(np.float32))
    assert actual.dtype == np.float32 and actual_rate == cfg.target_rate_hz
