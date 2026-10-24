"""Transparent heuristic baseline (PRD FR-7, item 1).

The idea under test: a real face's skin regions share one pulse, so their
pulse signals should agree in frequency, correlate with each other and stay
stable over time. The baseline scores that cross-region consistency in [0, 1]
and maps it onto three result states.

The thresholds (configs/default.yaml, `heuristic:`) are UNCALIBRATED
placeholders chosen by hand, not fitted or evaluated on any dataset. The
score is not a probability, and the states describe consistency with the
assumption above, not whether a video is authentic. The result is
inconclusive whenever signal quality or tracking is too low to judge.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum

import numpy as np

from pulsecheck.features.vector import FeatureVector

MODEL_NAME = "heuristic-consistency"
MODEL_VERSION = "0.1.0"


class ResultState(StrEnum):
    REAL = "more_consistent_with_real"
    MANIPULATED = "more_consistent_with_manipulated"
    INCONCLUSIVE = "inconclusive"

    @property
    def label(self) -> str:
        return {
            ResultState.REAL: "More consistent with real examples",
            ResultState.MANIPULATED: "More consistent with manipulated examples",
            ResultState.INCONCLUSIVE: "Inconclusive",
        }[self]


@dataclass(frozen=True)
class HeuristicConfig:
    # Quality gates: below any of these the result is inconclusive.
    min_signal_quality: float = 0.35
    min_usable_rois: int = 2
    min_face_coverage: float = 0.8
    # Consistency score at or above real_threshold -> REAL, at or below
    # manipulated_threshold -> MANIPULATED, otherwise inconclusive.
    real_threshold: float = 0.65
    manipulated_threshold: float = 0.35
    # Scales mapping raw features onto 0-1 component scores.
    max_freq_diff_hz: float = 0.3  # regional frequency gap that scores 0
    max_window_std_hz: float = 0.3  # window-frequency spread that scores 0

    def __post_init__(self) -> None:
        if not 0 <= self.manipulated_threshold <= self.real_threshold <= 1:
            raise ValueError("need 0 <= manipulated_threshold <= real_threshold <= 1")
        if self.max_freq_diff_hz <= 0 or self.max_window_std_hz <= 0:
            raise ValueError("score scales must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> HeuristicConfig:
        values = dict(values or {})
        values.pop("calibrated", None)  # documentation flag in the YAML
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown heuristic settings: {sorted(unknown)}")
        return cls(**values)


@dataclass
class HeuristicResult:
    state: ResultState
    # Cross-region consistency in [0, 1]; None when the quality gates failed.
    # Not a probability.
    score: float | None
    components: dict[str, float | None]
    reasons: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)
    calibrated: bool = False
    model: str = MODEL_NAME
    model_version: str = MODEL_VERSION

    @property
    def label(self) -> str:
        return self.state.label

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "label": self.label,
            "score": self.score,
            "score_semantics": (
                "Cross-region pulse consistency in [0, 1] from hand-set, uncalibrated "
                "thresholds. Not a probability of authenticity or manipulation."
            ),
            "components": self.components,
            "reasons": self.reasons,
            "config": self.config,
            "calibrated": self.calibrated,
            "model": self.model,
            "model_version": self.model_version,
        }


def _scaled_inverse(value: float | None, scale: float) -> float | None:
    """1 at value 0, falling linearly to 0 at `scale`."""
    if value is None:
        return None
    return float(np.clip(1.0 - value / scale, 0.0, 1.0))


def classify(features: FeatureVector, config: HeuristicConfig | None = None) -> HeuristicResult:
    config = config or HeuristicConfig()
    v = features.values
    reasons: list[str] = []

    quality = v["signal_quality"] or 0.0
    usable = int(v["usable_roi_count"] or 0)
    coverage = v["face_coverage"] or 0.0
    if quality < config.min_signal_quality:
        reasons.append(
            f"Signal quality {quality:.2f} is below the minimum {config.min_signal_quality:.2f}"
        )
    if usable < config.min_usable_rois:
        reasons.append(
            f"Only {usable} face region(s) gave a usable signal; "
            f"{config.min_usable_rois} are needed"
        )
    if coverage < config.min_face_coverage:
        reasons.append(
            f"Face tracked in {coverage:.0%} of frames; {config.min_face_coverage:.0%} is needed"
        )

    corr = v["mean_cross_correlation"]
    components = {
        "correlation": None if corr is None else float(np.clip(corr, 0.0, 1.0)),
        "frequency_agreement": _scaled_inverse(v["max_freq_diff_hz"], config.max_freq_diff_hz),
        "stability": _scaled_inverse(v["mean_window_freq_std_hz"], config.max_window_std_hz),
    }
    available = [c for c in components.values() if c is not None]
    if not available:
        reasons.append("No cross-region consistency measure could be computed")

    if reasons:
        return HeuristicResult(ResultState.INCONCLUSIVE, None, components, reasons, asdict(config))

    score = float(np.mean(available))
    if score >= config.real_threshold:
        state = ResultState.REAL
        reasons.append(
            f"Consistency {score:.2f} is at or above the uncalibrated threshold "
            f"{config.real_threshold:.2f}"
        )
    elif score <= config.manipulated_threshold:
        state = ResultState.MANIPULATED
        reasons.append(
            f"Consistency {score:.2f} is at or below the uncalibrated threshold "
            f"{config.manipulated_threshold:.2f}"
        )
    else:
        state = ResultState.INCONCLUSIVE
        reasons.append(f"Consistency {score:.2f} falls between the thresholds")
    return HeuristicResult(state, score, components, reasons, asdict(config))
