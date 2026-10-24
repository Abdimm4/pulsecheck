"""Synthetic-signal tests for pulsecheck.signals.processing (PRD section 15).

Frequency tolerance: for a clean or mildly noisy periodic signal of at least
10 s at 30 fps, the dominant-frequency estimate is within FREQ_TOL_HZ
(0.05 Hz, i.e. 3 bpm) of the true frequency.
"""

import math

import numpy as np
import pytest

from pulsecheck.signals import SignalConfig, bandpass, process_trace, resample_uniform
from pulsecheck.signals.processing import SignalRejected

FREQ_TOL_HZ = 0.05


def _sine(freq_hz, duration_s=15.0, fs=30.0, amp=1.0, noise=0.0, seed=0, offset=100.0):
    t = np.arange(0, duration_s, 1 / fs)
    rng = np.random.default_rng(seed)
    return t, offset + amp * np.sin(2 * np.pi * freq_hz * t) + noise * rng.normal(size=t.size)


def _rgb_pulse(freq_hz, duration_s=15.0, fs=30.0, noise=0.1, seed=0):
    """Skin-like RGB: a shared pulse, strongest in green, plus a common intensity drift."""
    t = np.arange(0, duration_s, 1 / fs)
    rng = np.random.default_rng(seed)
    pulse = np.sin(2 * np.pi * freq_hz * t)
    drift = 3.0 * np.sin(2 * np.pi * 0.1 * t)  # slow illumination change, below the band
    base = np.array([160.0, 120.0, 95.0])
    gain = np.array([0.3, 0.8, 0.2])
    rgb = base * (1 + drift[:, None] / 100) + pulse[:, None] * gain
    return t, rgb + noise * rng.normal(size=rgb.shape)


def _all_finite(sig):
    peak = sig.peak
    return all(
        math.isfinite(v) for v in (peak.frequency_hz, peak.prominence, peak.concentration)
    ) and math.isfinite(peak.snr_db)


@pytest.mark.parametrize("freq", [0.8, 1.0, 1.25, 1.7, 2.4, 2.9])
def test_green_estimates_known_frequency(freq):
    t, x = _sine(freq, noise=0.3, seed=int(freq * 10))
    sig = process_trace(t, x, "green")
    assert sig.ok, sig.detail
    assert sig.peak.frequency_hz == pytest.approx(freq, abs=FREQ_TOL_HZ)
    assert sig.settings["filter"]["applied"]
    assert _all_finite(sig)


@pytest.mark.parametrize("freq", [0.9, 1.3, 2.0])
def test_chrom_estimates_known_frequency_despite_intensity_drift(freq):
    t, rgb = _rgb_pulse(freq, seed=3)
    for method in ("chrom", "green"):
        sig = process_trace(t, rgb, method)
        assert sig.ok, sig.detail
        assert sig.peak.frequency_hz == pytest.approx(freq, abs=FREQ_TOL_HZ)


def test_clean_signal_has_concentrated_spectrum_and_stable_windows():
    t, x = _sine(1.2, duration_s=20)
    sig = process_trace(t, x, "green")
    assert sig.peak.concentration > 0.8
    assert sig.peak.prominence > 0.9
    assert sig.peak.snr_db > 20
    assert sig.window_freqs_hz.size >= 3
    np.testing.assert_allclose(sig.window_freqs_hz, 1.2, atol=FREQ_TOL_HZ)


def test_noise_has_lower_quality_than_pulse():
    t = np.arange(0, 15, 1 / 30)
    noise = 100 + np.random.default_rng(1).normal(size=t.size)
    pulse = process_trace(*_sine(1.2, noise=0.3), "green")
    rand = process_trace(t, noise, "green")
    assert rand.ok
    assert rand.peak.concentration < pulse.peak.concentration
    assert rand.peak.snr_db < pulse.peak.snr_db
    assert rand.peak.prominence < pulse.peak.prominence


def test_prominence_reflects_a_rival_peak():
    t = np.arange(0, 20, 1 / 30)
    x = 100 + np.sin(2 * np.pi * 1.0 * t) + 0.9 * np.sin(2 * np.pi * 2.0 * t)
    sig = process_trace(t, x, "green")
    assert sig.peak.frequency_hz == pytest.approx(1.0, abs=FREQ_TOL_HZ)
    # Power scales with amplitude squared: rival is about 0.81 of the peak.
    assert sig.peak.prominence == pytest.approx(0.19, abs=0.08)


def test_output_signal_is_unit_variance_on_uniform_grid():
    t, x = _sine(1.5)
    sig = process_trace(t, x, "green")
    assert np.allclose(np.diff(sig.t), 1 / sig.fs_hz)
    assert sig.signal.std() == pytest.approx(1.0)
    assert np.isfinite(sig.signal).all()


@pytest.mark.parametrize("fs", [5.0, 2.0])
def test_low_sample_rate_is_rejected(fs):
    t, x = _sine(1.0, duration_s=20, fs=fs)
    sig = process_trace(t, x, "green")
    assert not sig.ok
    assert sig.reason == "invalid_sample_rate"
    assert "below the minimum" in sig.detail
    assert sig.peak is None and sig.signal.size == 0


def test_too_short_signal_is_rejected():
    t, x = _sine(1.2, duration_s=4.0)
    sig = process_trace(t, x, "green")
    assert not sig.ok
    assert sig.reason == "too_short"
    assert sig.quality["duration_s"] == pytest.approx(4.0 - 1 / 30)


@pytest.mark.parametrize(
    ("t", "x", "reason"),
    [
        (np.array([]), np.array([]), "invalid_timestamps"),
        (np.array([1.0]), np.array([5.0]), "invalid_timestamps"),
        (np.full(300, np.nan), np.ones(300), "invalid_timestamps"),
        (np.zeros(300), np.ones(300), "invalid_timestamps"),  # all identical
    ],
)
def test_degenerate_timestamps_are_rejected(t, x, reason):
    sig = process_trace(t, x, "green")
    assert not sig.ok
    assert sig.reason == reason


def test_flat_signal_is_rejected():
    t = np.arange(0, 10, 1 / 30)
    sig = process_trace(t, np.full(t.size, 120.0), "green")
    assert not sig.ok
    assert sig.reason == "flat_signal"


def test_all_missing_is_rejected():
    t = np.arange(0, 10, 1 / 30)
    sig = process_trace(t, np.full((t.size, 3), np.nan), "chrom")
    assert not sig.ok
    assert sig.reason == "insufficient_valid_samples"


def test_scattered_missing_frames_are_interpolated():
    t, x = _sine(1.1, duration_s=15)
    rng = np.random.default_rng(4)
    x[rng.choice(t.size, size=t.size // 5, replace=False)] = np.nan  # 20% missing
    sig = process_trace(t, x, "green")
    assert sig.ok, sig.detail
    assert sig.peak.frequency_hz == pytest.approx(1.1, abs=FREQ_TOL_HZ)
    assert sig.quality["valid_fraction"] == pytest.approx(0.8, abs=0.01)
    assert np.isfinite(sig.signal).all()


def test_too_many_missing_frames_are_rejected():
    t, x = _sine(1.1, duration_s=15)
    x[::2] = np.nan
    x[1::4] = np.nan  # 75% missing
    sig = process_trace(t, x, "green")
    assert sig.reason == "insufficient_valid_samples"
    assert "25%" in sig.detail


def test_long_gap_is_rejected():
    t, x = _sine(1.1, duration_s=15)
    x[(t > 5) & (t < 8)] = np.nan  # 3 s gap, 80% still valid
    sig = process_trace(t, x, "green")
    assert sig.reason == "gap_too_long"
    assert sig.quality["longest_gap_s"] > 2.0


def test_nonuniform_sampling_is_resampled():
    rng = np.random.default_rng(7)
    # Jittered ~30 fps with occasional dropped frames.
    dt = 1 / 30 + rng.uniform(-0.008, 0.008, size=450)
    dt[rng.choice(450, size=20, replace=False)] += 1 / 30
    t = np.cumsum(dt)
    x = 100 + np.sin(2 * np.pi * 1.4 * t) + 0.2 * rng.normal(size=t.size)
    sig = process_trace(t, x, "green")
    assert sig.ok, sig.detail
    assert np.allclose(np.diff(sig.t), 1 / sig.fs_hz)
    assert sig.fs_hz == pytest.approx(30, rel=0.05)
    assert sig.peak.frequency_hz == pytest.approx(1.4, abs=FREQ_TOL_HZ)


def test_unsorted_and_duplicate_timestamps_are_handled():
    t, x = _sine(1.3)
    order = np.random.default_rng(2).permutation(t.size)
    t2 = np.concatenate([t[order], t[:5]])
    x2 = np.concatenate([x[order], x[:5]])
    sig = process_trace(t2, x2, "green")
    assert sig.ok
    assert sig.quality["dropped_timestamps"] == 5
    assert sig.peak.frequency_hz == pytest.approx(1.3, abs=FREQ_TOL_HZ)


def test_resample_shared_span_fills_edges():
    t, x = _sine(1.0, duration_s=12)
    x[t < 1.0] = np.nan  # ROI invalid for the first second
    u = resample_uniform(t, x, SignalConfig(), span_s=(0.0, float(t[-1])))
    assert u.t[0] == 0.0
    assert np.isfinite(u.values).all()
    assert u.quality["filled_fraction"] == pytest.approx(1 / 12, abs=0.02)


def test_resample_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        resample_uniform(np.arange(10.0), np.arange(9.0), SignalConfig())


def test_bandpass_attenuates_out_of_band_and_is_stable():
    fs = 30.0
    t = np.arange(0, 20, 1 / fs)
    inband = np.sin(2 * np.pi * 1.5 * t)
    low = np.sin(2 * np.pi * 0.1 * t)
    high = np.sin(2 * np.pi * 8.0 * t)
    y, info = bandpass(inband + low + high, fs, (0.7, 3.0))
    assert info["applied"] and info["zero_phase"]
    assert np.isfinite(y).all()
    core = slice(int(2 * fs), -int(2 * fs))  # away from the edges
    np.testing.assert_allclose(y[core], inband[core], atol=0.1)


def test_bandpass_is_skipped_with_reason_when_unsupported():
    x = np.sin(np.arange(20.0))
    _, info = bandpass(x, 30.0, (0.7, 3.0))
    assert not info["applied"]
    assert "fewer than" in info["reason"]
    _, info = bandpass(np.zeros(600), 6.0, (0.7, 3.0))
    assert not info["applied"]
    assert "Nyquist" in info["reason"]


def test_filter_skipped_still_records_settings():
    # A low minimum sample rate lets a 6.2 Hz signal through; 3 Hz is too close to Nyquist.
    config = SignalConfig(min_sample_rate_hz=5.0)
    t, x = _sine(1.0, duration_s=20, fs=6.2)
    sig = process_trace(t, x, "green", config)
    assert sig.ok
    assert sig.settings["filter"]["applied"] is False
    assert sig.settings["config"]["band_hz"] == [0.7, 3.0]


def test_configurable_band_restricts_peak_search():
    t = np.arange(0, 15, 1 / 30)
    x = 100 + np.sin(2 * np.pi * 2.5 * t) + 0.5 * np.sin(2 * np.pi * 1.0 * t)
    assert process_trace(t, x).peak.frequency_hz == pytest.approx(2.5, abs=FREQ_TOL_HZ)
    narrow = SignalConfig(band_hz=(0.7, 1.8))
    sig = process_trace(t, x, "green", narrow)
    assert sig.peak.frequency_hz == pytest.approx(1.0, abs=FREQ_TOL_HZ)
    assert sig.settings["filter"]["band_hz"] == [0.7, 1.8]


def test_config_validation_and_mapping():
    with pytest.raises(ValueError):
        SignalConfig(band_hz=(3.0, 0.7))
    with pytest.raises(ValueError):
        SignalConfig(detrend="quadratic")
    with pytest.raises(ValueError):
        SignalConfig.from_mapping({"bandpass": [1, 2]})
    config = SignalConfig.from_mapping({"bandpass_hz": [0.8, 2.5], "resample_hz": None})
    assert config.band_hz == (0.8, 2.5)


def test_unknown_method_raises():
    t, x = _sine(1.0)
    with pytest.raises(ValueError):
        process_trace(t, x, "pos")


def test_chrom_needs_rgb():
    t, x = _sine(1.0)
    with pytest.raises(ValueError):
        process_trace(t, x, "chrom")


def test_signal_rejected_carries_reason():
    exc = SignalRejected("too_short", "detail")
    assert exc.reason == "too_short" and str(exc) == "detail"
