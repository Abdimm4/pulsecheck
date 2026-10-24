"""Picklable stub analyzers for batch-extraction tests (importable by worker processes)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pulsecheck.dataset.extraction import ExtractionFailure, FatalExtractionError, VideoResult
from pulsecheck.features import FEATURE_NAMES, FeatureVector


@dataclass
class StubAnalyzer:
    """Returns fixed features; fails for stems listed in `fail`, crashes for `crash`."""

    fail: tuple[str, ...] = ()
    crash: tuple[str, ...] = ()
    fatal: tuple[str, ...] = ()
    log_dir: str | None = None  # writes one marker file per call, to count calls

    def __call__(self, path: Path) -> VideoResult:
        if self.log_dir:
            marker = Path(self.log_dir) / f"{path.parent.parent.parent.name}-{path.stem}"
            marker.write_text(str(int(marker.read_text()) + 1) if marker.exists() else "1")
        if path.stem in self.fatal:
            raise FatalExtractionError("landmarker model missing")
        if path.stem in self.fail:
            raise ExtractionFailure("video_decode_error", "stub decode failure")
        if path.stem in self.crash:
            raise RuntimeError("unexpected stub crash")
        values = dict.fromkeys(FEATURE_NAMES, 0.5)
        values["usable_roi_count"] = 3.0
        values["forehead.window_freq_std_hz"] = None
        features = FeatureVector(values, {"forehead.window_freq_std_hz": "fewer_than_3_windows"})
        return VideoResult(features, n_frames=300, duration_s=10.0, fps=30.0)
