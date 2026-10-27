"""The analysis a worker runs on one uploaded video.

decode (bounded, re-validated) -> face landmarks and ROIs -> RGB traces ->
pulse signals -> features -> classifier. The classifier is the heuristic
baseline unless `analysis.model_path` names a trained model. Only summaries
leave this module: no frames, landmarks or ROI geometry (PRD UP-7).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from pulsecheck.dataset.table import format_row
from pulsecheck.face import FaceLandmarkerDetector, LandmarkerUnavailableError, TrackingReport
from pulsecheck.features import FeatureVector, QualityConfig, analyze_traces
from pulsecheck.jobs.validation import (
    CORRUPT_VIDEO,
    UploadLimits,
    UploadRejected,
    check_metadata,
)
from pulsecheck.models import MODEL_NAME, MODEL_VERSION, HeuristicConfig, ResultState, classify
from pulsecheck.signals import SignalConfig, extract_rgb_traces
from pulsecheck.video import VideoDecodeError, VideoMetadata, iter_frames, probe_video

ANALYSIS_UNAVAILABLE = "analysis_unavailable"


class AnalysisUnavailable(Exception):
    """The service cannot analyse anything right now (e.g. landmarker model missing)."""


@dataclass
class AnalysisOutput:
    video: VideoMetadata
    tracking: TrackingReport | None
    features: FeatureVector | None
    signal_summary: dict | None  # per ROI and method, without sample arrays
    result: dict  # state, label, score, score_semantics, reasons, calibrated, model...
    settings: dict = field(default_factory=dict)


class Analyzer(Protocol):
    model_name: str
    model_version: str

    def __call__(self, path: Path) -> AnalysisOutput: ...


@dataclass
class Classifier:
    """Heuristic baseline, or a trained model loaded from `model_path`."""

    heuristic: HeuristicConfig = field(default_factory=HeuristicConfig)
    model_path: str | None = None

    def __post_init__(self) -> None:
        self._model = None
        if self.model_path:
            from pulsecheck.models.training import load_model

            self._model = load_model(self.model_path)

    @property
    def name(self) -> str:
        return self._model.sidecar["model_name"] if self._model else MODEL_NAME

    @property
    def version(self) -> str:
        return self._model.sidecar["model_version"] if self._model else MODEL_VERSION

    def __call__(self, features: FeatureVector) -> dict:
        if self._model is None:
            return classify(features, self.heuristic).to_dict()
        from pulsecheck.models.training import SCORE_SEMANTICS

        row = format_row({}, {}, features)
        score = float(self._model.scores([row])[0])
        state, reason = self._model.decide(row, score)
        return {
            "state": state,
            "label": ResultState(state).label,
            "score": score,
            "score_semantics": SCORE_SEMANTICS,
            "reasons": [reason] if reason else [],
            "calibrated": False,
            "model": self.name,
            "model_version": self.version,
        }


@dataclass
class PipelineAnalyzer:
    limits: UploadLimits = field(default_factory=UploadLimits)
    signal: SignalConfig = field(default_factory=SignalConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    pulse_method: str = "chrom"
    classifier: Classifier = field(default_factory=Classifier)
    landmarker_path: str | None = None

    @classmethod
    def from_config(
        cls, config: Mapping, limits: UploadLimits, model_path: str | None = None
    ) -> PipelineAnalyzer:
        features = config.get("features") or {}
        return cls(
            limits=limits,
            signal=SignalConfig.from_mapping(config.get("signal")),
            quality=QualityConfig.from_mapping(features.get("quality")),
            pulse_method=features.get("pulse_method", "chrom"),
            classifier=Classifier(
                HeuristicConfig.from_mapping(config.get("heuristic")), model_path
            ),
        )

    @property
    def model_name(self) -> str:
        return self.classifier.name

    @property
    def model_version(self) -> str:
        return self.classifier.version

    def __call__(self, path: Path) -> AnalysisOutput:
        try:
            # One frame past the limit is enough to know the video is too long.
            meta = probe_video(path, max_frames=self.limits.max_frames + 1)
        except VideoDecodeError:
            raise UploadRejected(CORRUPT_VIDEO, "The video could not be decoded.") from None
        check_metadata(meta, self.limits)
        try:
            with FaceLandmarkerDetector(self.landmarker_path) as detector:
                frames = iter_frames(path, max_frames=self.limits.max_frames)
                traces = extract_rgb_traces(frames, detector)
        except LandmarkerUnavailableError as exc:
            raise AnalysisUnavailable(str(exc)) from exc
        except VideoDecodeError:
            raise UploadRejected(CORRUPT_VIDEO, "The video could not be decoded.") from None
        signals = analyze_traces(traces, self.signal, self.pulse_method, self.quality)
        return AnalysisOutput(
            video=meta,
            tracking=traces.tracking,
            features=signals.features,
            signal_summary=signals.summary()["signals"],
            result=self.classifier(signals.features),
            settings={"color_signal": traces.settings, **signals.settings},
        )
