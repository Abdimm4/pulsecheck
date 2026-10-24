import json
import math

import numpy as np
import pytest

from pulsecheck.config import load_config
from pulsecheck.face import ROI_NAMES
from pulsecheck.face.tracking import RoiTrackingStats, TrackingReport
from pulsecheck.features import FEATURE_NAMES, FeatureVector, QualityConfig, analyze_traces
from pulsecheck.models import HeuristicConfig, ResultState, classify
from pulsecheck.signals import RgbTraces, RoiTrace, SignalConfig

FS = 30.0


def _traces(per_roi_freq, duration_s=15.0, noise=0.1, missing=None, face_coverage=1.0, seed=0):
    """Synthetic RgbTraces; per_roi_freq[roi] is a pulse frequency, or None for pure noise."""
    t = np.arange(0, duration_s, 1 / FS)
    rng = np.random.default_rng(seed)
    rois = {}
    stats = {}
    for name in ROI_NAMES:
        freq = per_roi_freq[name]
        pulse = np.sin(2 * np.pi * freq * t) if freq is not None else np.zeros(t.size)
        rgb = np.array([160.0, 120.0, 95.0]) + pulse[:, None] * [0.3, 0.8, 0.2]
        rgb = rgb + noise * rng.normal(size=rgb.shape)
        valid = np.ones(t.size, dtype=bool)
        if missing and name in missing:
            valid = ~missing[name](t)
        rgb[~valid] = np.nan
        rois[name] = RoiTrace(name, rgb, valid, np.where(valid, 800, 0))
        fraction = float(valid.mean()) if valid.size else 0.0
        stats[name] = RoiTrackingStats(name, int(valid.sum()), fraction, 0.01)
    report = TrackingReport(
        total_frames=t.size,
        face_frames=int(face_coverage * t.size),
        face_coverage=face_coverage,
        all_rois_valid_fraction=min(s.valid_fraction for s in stats.values()),
        rois=stats,
        reliable=face_coverage >= 0.8,
    )
    return RgbTraces(t, np.arange(t.size), rois, report)


def _assert_clean(features):
    data = json.loads(json.dumps(features.to_dict(), allow_nan=False))  # raises on NaN/Inf
    assert set(data["values"]) == set(FEATURE_NAMES)
    for name, value in data["values"].items():
        if value is None:
            assert name in data["missing"]
        else:
            assert math.isfinite(value)
            assert name not in data["missing"]


def test_shared_pulse_gives_consistent_features():
    traces = _traces(dict.fromkeys(ROI_NAMES, 1.2))
    result = analyze_traces(traces)
    f = result.features
    _assert_clean(f)
    assert f.missing == {}
    for roi in ROI_NAMES:
        assert f.get(f"{roi}.dominant_hz") == pytest.approx(1.2, abs=0.05)
        assert f.get(f"{roi}.method_diff_hz") < 0.05
        assert f.get(f"{roi}.tracking_coverage") == 1.0
    assert f.get("usable_roi_count") == 3
    assert f.get("mean_cross_correlation") > 0.9
    assert f.get("max_freq_diff_hz") < 0.05
    assert 0.5 < f.get("signal_quality") <= 1.0
    # Shared grid across ROIs.
    sigs = [result.signals[roi]["chrom"] for roi in ROI_NAMES]
    assert all(np.array_equal(s.t, sigs[0].t) for s in sigs)


def test_mismatched_regions_lower_agreement():
    traces = _traces({"forehead": 1.0, "left_cheek": 1.8, "right_cheek": 2.5})
    f = analyze_traces(traces).features
    _assert_clean(f)
    assert f.get("max_freq_diff_hz") > 1.0
    assert abs(f.get("mean_cross_correlation")) < 0.3


def test_rejected_roi_is_explicitly_missing():
    # Left cheek invalid for most of the clip.
    traces = _traces(dict.fromkeys(ROI_NAMES, 1.2), missing={"left_cheek": lambda t: t < 10})
    result = analyze_traces(traces)
    f = result.features
    _assert_clean(f)
    assert result.signals["left_cheek"]["chrom"].reason == "insufficient_valid_samples"
    assert f.get("left_cheek.dominant_hz") is None
    assert f.missing["left_cheek.dominant_hz"] == "signal_rejected:insufficient_valid_samples"
    assert f.missing["forehead__left_cheek.correlation"] == "roi_unusable"
    assert f.get("left_cheek.quality") == 0.0
    assert f.get("usable_roi_count") == 2
    assert f.get("forehead__right_cheek.correlation") > 0.9


def test_too_short_video_has_no_nan_and_is_inconclusive():
    traces = _traces(dict.fromkeys(ROI_NAMES, 1.2), duration_s=3.0)
    result = analyze_traces(traces)
    _assert_clean(result.features)
    assert result.features.get("usable_roi_count") == 0
    assert result.features.get("signal_quality") == 0.0
    assert classify(result.features).state is ResultState.INCONCLUSIVE


def test_empty_traces_do_not_crash():
    traces = _traces(dict.fromkeys(ROI_NAMES, 1.2), duration_s=0.0)
    result = analyze_traces(traces)
    _assert_clean(result.features)
    assert result.signals["forehead"]["green"].reason == "invalid_timestamps"


def test_noise_only_has_low_quality():
    pulse = analyze_traces(_traces(dict.fromkeys(ROI_NAMES, 1.2))).features
    noise = analyze_traces(_traces(dict.fromkeys(ROI_NAMES), noise=1.0, seed=5)).features
    _assert_clean(noise)
    assert noise.get("signal_quality") < pulse.get("signal_quality")


def test_green_method_can_be_primary():
    f = analyze_traces(_traces(dict.fromkeys(ROI_NAMES, 1.5)), pulse_method="green").features
    assert f.pulse_method == "green"
    assert f.get("forehead.dominant_hz") == pytest.approx(1.5, abs=0.05)


def test_summary_is_json_without_nan():
    result = analyze_traces(_traces(dict.fromkeys(ROI_NAMES, 1.2)))
    json.dumps(result.summary(), allow_nan=False)


def test_heuristic_consistent_pulse_is_more_consistent_with_real():
    result = classify(analyze_traces(_traces(dict.fromkeys(ROI_NAMES, 1.2))).features)
    assert result.state is ResultState.REAL
    assert result.score >= HeuristicConfig().real_threshold
    assert result.calibrated is False
    data = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert data["label"] == "More consistent with real examples"
    assert "Not a probability" in data["score_semantics"]


def test_heuristic_inconsistent_regions_with_good_quality():
    features = analyze_traces(
        _traces({"forehead": 0.9, "left_cheek": 1.7, "right_cheek": 2.6}, noise=0.02)
    ).features
    result = classify(features)
    assert result.state is ResultState.MANIPULATED
    assert result.score <= 0.35


@pytest.mark.parametrize("seed", range(5))
def test_heuristic_noise_only_is_inconclusive_with_reasons(seed):
    features = analyze_traces(_traces(dict.fromkeys(ROI_NAMES), noise=1.0, seed=seed)).features
    assert features.get("signal_quality") < HeuristicConfig().min_signal_quality
    result = classify(features)
    assert result.state is ResultState.INCONCLUSIVE
    assert result.score is None
    assert any("Signal quality" in r for r in result.reasons)


def test_heuristic_poor_face_coverage_is_inconclusive():
    features = analyze_traces(_traces(dict.fromkeys(ROI_NAMES, 1.2), face_coverage=0.5)).features
    result = classify(features)
    assert result.state is ResultState.INCONCLUSIVE
    assert any("Face tracked" in r for r in result.reasons)


def test_heuristic_between_thresholds_is_inconclusive():
    values = dict.fromkeys(FEATURE_NAMES)
    values.update(
        signal_quality=0.8,
        usable_roi_count=3,
        face_coverage=1.0,
        mean_cross_correlation=0.5,
        max_freq_diff_hz=0.15,
        mean_window_freq_std_hz=0.15,
    )
    result = classify(FeatureVector(values))
    assert result.score == pytest.approx(0.5)
    assert result.state is ResultState.INCONCLUSIVE
    assert result.components == pytest.approx(
        {"correlation": 0.5, "frequency_agreement": 0.5, "stability": 0.5}
    )


def test_heuristic_config_validation():
    with pytest.raises(ValueError):
        HeuristicConfig(real_threshold=0.3, manipulated_threshold=0.6)
    with pytest.raises(ValueError):
        HeuristicConfig.from_mapping({"threshold": 0.5})


def test_default_yaml_loads_into_configs():
    config = load_config()
    signal = SignalConfig.from_mapping(config["signal"])
    assert signal.band_hz == (0.7, 3.0)
    assert signal == SignalConfig()
    assert HeuristicConfig.from_mapping(config["heuristic"]) == HeuristicConfig()
    assert QualityConfig.from_mapping(config["features"]["quality"]) == QualityConfig()
    assert config["heuristic"]["calibrated"] is False
    assert config["features"]["pulse_method"] in ("chrom", "green")
