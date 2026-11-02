"""One feature vector per video from per-ROI pulse signals (PRD FR-6).

Every feature has a fixed name (FEATURE_NAMES). A feature that cannot be
computed is None and its name maps to a reason in `missing`; NaN and Infinity
never appear in the output.

The signal-quality score is a transparent heuristic in [0, 1], not a
calibrated probability. Per estimate, spectral concentration and the SNR
proxy are each mapped onto 0-1 by a linear ramp (QualityConfig); their
geometric mean, so both must be high, is weighted by tracking coverage.

Schema 2 (Phase 5A) adds the optional whole-face ROI ("face.*", missing with
reason "roi_disabled" when color_signal.face_roi is off) and a primary pulse
estimate ("pulse_*"), read as configured by `hr_source`:

- "per_roi" (schema 1 behaviour): no primary estimate; `signal_quality` is
  the mean per-ROI quality over the regional ROIs (a rejected ROI counts 0);
- "fused": one estimate from the averaged unit-power spectra of the usable
  regional ROIs; quality weighted by their mean tracking coverage;
- "face": the whole-face ROI's estimate.

With "fused" or "face", `signal_quality` is the primary estimate's quality,
the number the injection benchmark calibrated (pulsecheck.benchmark).
`method_diff_hz` is the largest gap to the other pulse methods (green, CHROM,
POS since schema 2).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from itertools import combinations

import numpy as np

from pulsecheck.face.regions import ALL_ROI_NAMES, FACE_ROI, ROI_NAMES
from pulsecheck.signals.processing import (
    PULSE_METHODS,
    ProcessedSignal,
    SignalConfig,
    SignalRejected,
    dominant_peak,
    fuse_spectra,
    process_trace,
)
from pulsecheck.signals.rgb import RgbTraces

# 2: whole-face ROI, primary pulse estimate (pulse_*), POS as a third method.
FEATURE_SCHEMA_VERSION = 2
HR_SOURCES = ("per_roi", "fused", "face")
ROI_PAIRS = tuple(combinations(ROI_NAMES, 2))

PER_ROI_FEATURES = (
    "dominant_hz",  # strongest in-band frequency of the pulse signal
    "peak_prominence",  # margin over the runner-up peak, 0-1
    "spectral_concentration",  # in-band power fraction near the peak, 0-1
    "snr_db",  # peak over median in-band power
    "window_freq_std_hz",  # spread of per-window dominant frequencies (lower = stabler)
    "method_diff_hz",  # |green - CHROM| dominant frequency
    "tracking_coverage",  # fraction of frames the ROI was valid in
    "quality",  # per-ROI quality score, 0-1
)
PAIR_FEATURES = (
    "correlation",  # Pearson correlation of the two pulse signals
    "freq_diff_hz",  # absolute dominant-frequency difference
)
PULSE_FEATURES = (
    "pulse_hz",  # primary pulse frequency (hr_source)
    "pulse_prominence",
    "pulse_concentration",
    "pulse_snr_db",
    "pulse_quality",
)
VIDEO_FEATURES = (
    "face_coverage",
    "all_rois_valid_fraction",
    "usable_roi_count",
    "mean_cross_correlation",
    "max_freq_diff_hz",
    "mean_window_freq_std_hz",
    *PULSE_FEATURES,
    "signal_quality",
)


def _pair_name(a: str, b: str) -> str:
    return f"{a}__{b}"


FEATURE_NAMES = (
    *(f"{roi}.{name}" for roi in ALL_ROI_NAMES for name in PER_ROI_FEATURES),
    *(f"{_pair_name(a, b)}.{name}" for a, b in ROI_PAIRS for name in PAIR_FEATURES),
    *VIDEO_FEATURES,
)
# Features that measure face tracking rather than the pulse. They depend on
# resolution and compression, so a classifier could learn a source shortcut
# through them; they stay in the table for the quality gate but are left out of
# PULSE_CLASSIFIER_FEATURES, the trained classifiers' default candidates. The
# quality scores are weighted by tracking coverage and stay in; the confound
# audit (scripts/audit_confounds.py) shows how much media they carry.
TRACKING_FEATURES = (
    *(f"{roi}.tracking_coverage" for roi in ALL_ROI_NAMES),
    "face_coverage",
    "all_rois_valid_fraction",
    "usable_roi_count",
)
PULSE_CLASSIFIER_FEATURES = tuple(n for n in FEATURE_NAMES if n not in TRACKING_FEATURES)


@dataclass(frozen=True)
class QualityConfig:
    """Linear 0-1 ramps for the quality score (configs/default.yaml `features.quality`).

    The class defaults are the schema-1 values, set by hand: each lower end
    sat just above what white noise reached on 15 s synthetic clips, each
    upper end marked a clearly visible pulse. configs/default.yaml overrides
    them with ramps fitted on the injection benchmark's dev half (see that
    file and data/reports/benchmark/).
    """

    concentration_range: tuple[float, float] = (0.35, 0.75)
    snr_range_db: tuple[float, float] = (10.0, 20.0)

    def __post_init__(self) -> None:
        for lo, hi in (self.concentration_range, self.snr_range_db):
            if not lo < hi:
                raise ValueError("quality ranges must satisfy low < high")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> QualityConfig:
        values = dict(values or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown quality settings: {sorted(unknown)}")
        return cls(**{k: tuple(float(x) for x in v) for k, v in values.items()})


def _ramp(value: float, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _quality_from_peak(peak, coverage: float, config: QualityConfig) -> float:
    concentration = _ramp(peak.concentration, config.concentration_range)
    snr = _ramp(peak.snr_db, config.snr_range_db)
    return math.sqrt(concentration * snr) * float(np.clip(coverage, 0.0, 1.0))


@dataclass
class FeatureVector:
    values: dict[str, float | None]
    missing: dict[str, str] = field(default_factory=dict)
    pulse_method: str = "chrom"
    schema_version: int = FEATURE_SCHEMA_VERSION

    def get(self, name: str) -> float | None:
        return self.values[name]

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "pulse_method": self.pulse_method,
            "values": dict(self.values),
            "missing": dict(self.missing),
        }


class _Builder:
    def __init__(self) -> None:
        self.values: dict[str, float | None] = dict.fromkeys(FEATURE_NAMES)
        self.missing: dict[str, str] = {}

    def set(self, name: str, value: float | None, reason: str = "not_computed") -> None:
        if name not in self.values:
            raise KeyError(name)
        if value is None:
            self.values[name] = None
            self.missing[name] = reason
        elif not math.isfinite(float(value)):
            self.values[name] = None
            self.missing[name] = "non_finite"
        else:
            self.values[name] = float(value)
            self.missing.pop(name, None)

    def build(self, pulse_method: str) -> FeatureVector:
        for name, value in self.values.items():
            if value is None:
                self.missing.setdefault(name, "not_computed")
        return FeatureVector(dict(self.values), dict(self.missing), pulse_method)


def _roi_quality(sig: ProcessedSignal, coverage: float, config: QualityConfig) -> float:
    if not sig.ok or sig.peak is None:
        return 0.0
    return _quality_from_peak(sig.peak, coverage, config)


def build_features(
    signals: dict[str, dict[str, ProcessedSignal]],
    traces: RgbTraces,
    pulse_method: str = "chrom",
    quality_config: QualityConfig | None = None,
    hr_source: str = "per_roi",
    config: SignalConfig | None = None,
) -> FeatureVector:
    """Build the feature vector from processed signals (`signals[roi][method]`).

    `signals` may lack the face ROI (its features are then missing with
    reason "roi_disabled"); `config` supplies the peak settings of a fused
    estimate.
    """
    if pulse_method not in PULSE_METHODS:
        raise ValueError(f"unknown pulse method {pulse_method!r}")
    if hr_source not in HR_SOURCES:
        raise ValueError(f"hr_source must be one of {HR_SOURCES}, got {hr_source!r}")
    quality_config = quality_config or QualityConfig()
    config = config or SignalConfig()
    b = _Builder()
    tracking = traces.tracking
    usable = []
    qualities = []
    window_stds = []
    coverages = {}

    for roi in ALL_ROI_NAMES:
        if roi not in signals:
            for name in PER_ROI_FEATURES:
                b.set(f"{roi}.{name}", None, "roi_disabled")
            continue
        sig = signals[roi][pulse_method]
        stats = tracking.rois.get(roi)
        coverage = stats.valid_fraction if stats else 0.0
        coverages[roi] = coverage
        b.set(f"{roi}.tracking_coverage", coverage)
        quality = _roi_quality(sig, coverage, quality_config)
        if roi != FACE_ROI:
            qualities.append(quality)
        b.set(f"{roi}.quality", quality)
        if not sig.ok or sig.peak is None:
            reason = f"signal_rejected:{sig.reason}"
            for name in PER_ROI_FEATURES:
                if name not in ("tracking_coverage", "quality"):
                    b.set(f"{roi}.{name}", None, reason)
            continue
        if roi != FACE_ROI:
            usable.append(roi)
        b.set(f"{roi}.dominant_hz", sig.peak.frequency_hz)
        b.set(f"{roi}.peak_prominence", sig.peak.prominence)
        b.set(f"{roi}.spectral_concentration", sig.peak.concentration)
        b.set(f"{roi}.snr_db", sig.peak.snr_db)
        if sig.window_freqs_hz.size >= 3:
            std = float(np.std(sig.window_freqs_hz))
            if roi != FACE_ROI:
                window_stds.append(std)
            b.set(f"{roi}.window_freq_std_hz", std)
        else:
            b.set(f"{roi}.window_freq_std_hz", None, "fewer_than_3_windows")
        others = [signals[roi][m] for m in PULSE_METHODS if m != pulse_method and m in signals[roi]]
        if others and all(o.ok and o.peak for o in others):
            diff = max(abs(o.peak.frequency_hz - sig.peak.frequency_hz) for o in others)
            b.set(f"{roi}.method_diff_hz", diff)
        else:
            b.set(f"{roi}.method_diff_hz", None, "comparison_method_rejected")

    correlations = []
    freq_diffs = []
    for a, c in ROI_PAIRS:
        prefix = _pair_name(a, c)
        sa, sc = signals[a][pulse_method], signals[c][pulse_method]
        if not (a in usable and c in usable):
            b.set(f"{prefix}.correlation", None, "roi_unusable")
            b.set(f"{prefix}.freq_diff_hz", None, "roi_unusable")
            continue
        diff = abs(sa.peak.frequency_hz - sc.peak.frequency_hz)
        freq_diffs.append(diff)
        b.set(f"{prefix}.freq_diff_hz", diff)
        if sa.signal.shape == sc.signal.shape and np.allclose(sa.t, sc.t):
            corr = float(np.corrcoef(sa.signal, sc.signal)[0, 1])
            if math.isfinite(corr):
                correlations.append(corr)
            b.set(f"{prefix}.correlation", corr)
        else:
            b.set(f"{prefix}.correlation", None, "signals_not_on_shared_grid")

    b.set("face_coverage", tracking.face_coverage)
    b.set("all_rois_valid_fraction", tracking.all_rois_valid_fraction)
    b.set("usable_roi_count", len(usable))
    b.set(
        "mean_cross_correlation",
        float(np.mean(correlations)) if correlations else None,
        "no_usable_roi_pair",
    )
    b.set("max_freq_diff_hz", max(freq_diffs) if freq_diffs else None, "no_usable_roi_pair")
    b.set(
        "mean_window_freq_std_hz",
        float(np.mean(window_stds)) if window_stds else None,
        "no_roi_with_3_windows",
    )
    _set_primary(b, signals, usable, coverages, pulse_method, hr_source, config, quality_config)
    if hr_source == "per_roi":
        b.set("signal_quality", float(np.mean(qualities)) if qualities else 0.0)
    else:
        b.set("signal_quality", b.values["pulse_quality"] or 0.0)
    return b.build(pulse_method)


def _set_primary(b, signals, usable, coverages, method, hr_source, config, quality_config):
    """The pulse_* features from the configured primary estimate."""
    peak, coverage, reason = None, 0.0, None
    if hr_source == "per_roi":
        reason = "hr_source_per_roi"
    elif hr_source == "face":
        if FACE_ROI not in signals:
            reason = "roi_disabled"
        else:
            sig = signals[FACE_ROI][method]
            peak, coverage = sig.peak if sig.ok else None, coverages.get(FACE_ROI, 0.0)
            reason = None if peak else f"signal_rejected:{sig.reason}"
    else:
        spectra = [signals[r][method].spectrum for r in usable]
        if not spectra:
            reason = "no_usable_roi"
        else:
            try:
                peak = dominant_peak(
                    fuse_spectra(spectra),
                    config.peak_halfwidth_hz,
                    config.peak_picking,
                    config.harmonic_weight,
                )
                coverage = float(np.mean([coverages[r] for r in usable]))
            except (SignalRejected, ValueError) as exc:
                reason = f"fusion_failed:{getattr(exc, 'reason', type(exc).__name__)}"
    if peak is None:
        for name in PULSE_FEATURES:
            b.set(name, None, reason or "not_computed")
        if hr_source != "per_roi":
            b.set("pulse_quality", 0.0)
        return
    b.set("pulse_hz", peak.frequency_hz)
    b.set("pulse_prominence", peak.prominence)
    b.set("pulse_concentration", peak.concentration)
    b.set("pulse_snr_db", peak.snr_db)
    b.set("pulse_quality", _quality_from_peak(peak, coverage, quality_config))


@dataclass
class VideoSignals:
    """Processed pulse signals for every ROI and method, plus the feature vector."""

    signals: dict[str, dict[str, ProcessedSignal]]
    features: FeatureVector
    settings: dict

    def summary(self) -> dict:
        return {
            "signals": {
                roi: {method: sig.summary() for method, sig in by_method.items()}
                for roi, by_method in self.signals.items()
            },
            "features": self.features.to_dict(),
            "settings": self.settings,
        }


def analyze_traces(
    traces: RgbTraces,
    config: SignalConfig | None = None,
    pulse_method: str = "chrom",
    quality_config: QualityConfig | None = None,
    hr_source: str = "per_roi",
) -> VideoSignals:
    """Process every ROI trace with every pulse method and build the features.

    All ROIs share one time grid spanning the video's first to last frame, so
    their pulse signals can be correlated sample by sample.
    """
    config = config or SignalConfig()
    quality_config = quality_config or QualityConfig()
    t = traces.timestamps_s
    finite = t[np.isfinite(t)] if t.size else t
    span = (float(finite.min()), float(finite.max())) if finite.size >= 2 else None
    signals = {
        roi: {
            method: process_trace(t, traces.rois[roi].rgb, method, config, span)
            for method in PULSE_METHODS
        }
        for roi in ALL_ROI_NAMES
        if roi in traces.rois
    }
    features = build_features(signals, traces, pulse_method, quality_config, hr_source, config)
    settings = {
        "signal": config.to_dict(),
        "pulse_method": pulse_method,
        "hr_source": hr_source,
        "quality": {k: list(v) for k, v in asdict(quality_config).items()},
    }
    return VideoSignals(signals=signals, features=features, settings=settings)
