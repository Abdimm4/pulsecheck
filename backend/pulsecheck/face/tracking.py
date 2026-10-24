"""Per-video face tracking coverage and ROI stability (PRD FR-3).

Only scalar summaries are kept: the monitor remembers each ROI's previous
centroid to measure frame-to-frame jitter, but never stores landmark or ROI
geometry for the whole video (PRD UP-7).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field

import numpy as np

from pulsecheck.face.regions import ROI_NAMES, Roi


@dataclass(frozen=True)
class TrackingThresholds:
    min_face_coverage: float = 0.8  # fraction of frames with a detected face
    min_roi_valid_fraction: float = 0.8  # fraction of frames each ROI is valid in
    max_median_jitter: float = 0.05  # median centroid step per frame, in IOD


@dataclass
class RoiTrackingStats:
    name: str
    valid_frames: int
    valid_fraction: float
    # Median frame-to-frame centroid displacement over consecutive valid
    # frames, normalised by interocular distance; None if never measurable.
    median_jitter: float | None
    invalid_reasons: dict[str, int] = field(default_factory=dict)


@dataclass
class TrackingReport:
    total_frames: int
    face_frames: int
    face_coverage: float
    all_rois_valid_fraction: float
    rois: dict[str, RoiTrackingStats]
    reliable: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class TrackingMonitor:
    def __init__(self, thresholds: TrackingThresholds | None = None) -> None:
        self.thresholds = thresholds or TrackingThresholds()
        self._total = 0
        self._faces = 0
        self._all_valid = 0
        self._valid: Counter[str] = Counter()
        self._reasons: dict[str, Counter[str]] = {name: Counter() for name in ROI_NAMES}
        self._jitter: dict[str, list[float]] = {name: [] for name in ROI_NAMES}
        self._prev_centroid: dict[str, tuple[float, float] | None] = dict.fromkeys(ROI_NAMES)

    def update(self, rois: dict[str, Roi], face_detected: bool, interocular: float | None) -> None:
        self._total += 1
        self._faces += int(face_detected)
        self._all_valid += int(all(roi.valid for roi in rois.values()))
        for name, roi in rois.items():
            centroid = roi.centroid() if roi.valid else None
            if roi.valid:
                self._valid[name] += 1
            else:
                self._reasons[name][roi.reason or "unknown"] += 1
            prev = self._prev_centroid[name]
            if centroid is not None and prev is not None and interocular:
                step = float(np.hypot(centroid[0] - prev[0], centroid[1] - prev[1]))
                self._jitter[name].append(step / interocular)
            self._prev_centroid[name] = centroid

    def report(self) -> TrackingReport:
        t = self.thresholds
        total = self._total
        coverage = self._faces / total if total else 0.0
        warnings: list[str] = []
        if total == 0:
            warnings.append("No frames were processed")
        elif coverage < t.min_face_coverage:
            warnings.append(f"Face detected in only {coverage:.0%} of frames")

        rois = {}
        for name in ROI_NAMES:
            valid = self._valid[name]
            fraction = valid / total if total else 0.0
            jitter = float(np.median(self._jitter[name])) if self._jitter[name] else None
            rois[name] = RoiTrackingStats(
                name=name,
                valid_frames=valid,
                valid_fraction=fraction,
                median_jitter=jitter,
                invalid_reasons=dict(self._reasons[name]),
            )
            if total and fraction < t.min_roi_valid_fraction:
                warnings.append(f"Region {name} is valid in only {fraction:.0%} of frames")
            if jitter is not None and jitter > t.max_median_jitter:
                warnings.append(f"Region {name} is unstable (median jitter {jitter:.3f} IOD/frame)")

        reliable = (
            total > 0
            and coverage >= t.min_face_coverage
            and all(r.valid_fraction >= t.min_roi_valid_fraction for r in rois.values())
        )
        return TrackingReport(
            total_frames=total,
            face_frames=self._faces,
            face_coverage=coverage,
            all_rois_valid_fraction=self._all_valid / total if total else 0.0,
            rois=rois,
            reliable=reliable,
            warnings=warnings,
        )
