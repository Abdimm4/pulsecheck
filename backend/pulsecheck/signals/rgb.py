"""Per-ROI RGB mean time series (PRD FR-4).

For every frame and ROI we keep the mean R, G and B of the masked skin pixels,
the frame timestamp, the pixel count and the ROI's validity. Frames where an
ROI is invalid hold NaN so the series stay aligned with the timestamps. No
landmarks, masks or pixels are retained (PRD UP-7).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from pulsecheck.face.landmarker import FaceLandmarkerDetector, LandmarkDetector
from pulsecheck.face.regions import ROI_NAMES, Roi, RoiConfig, compute_rois, face_frame
from pulsecheck.face.tracking import TrackingMonitor, TrackingReport, TrackingThresholds
from pulsecheck.video.reader import Frame, iter_frames

# Pixels at or beyond these 8-bit levels carry no usable colour variation.
CLIP_LOW = 5
CLIP_HIGH = 250


@dataclass
class RoiTrace:
    name: str
    rgb: np.ndarray  # (N, 3) float64 mean R, G, B; NaN where invalid
    valid: np.ndarray  # (N,) bool
    pixel_count: np.ndarray  # (N,) int

    @property
    def green(self) -> np.ndarray:
        """Green channel, the simple rPPG baseline."""
        return self.rgb[:, 1]


@dataclass
class RgbTraces:
    timestamps_s: np.ndarray  # (N,) per-frame timestamps from the decoder
    frame_indices: np.ndarray  # (N,) decoded frame indices
    rois: dict[str, RoiTrace]
    tracking: TrackingReport
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON-serialisable form; NaN becomes None."""

        def clean(values: np.ndarray) -> list:
            return [None if isinstance(v, float) and math.isnan(v) else v for v in values.tolist()]

        return {
            "timestamps_s": clean(self.timestamps_s),
            "frame_indices": self.frame_indices.tolist(),
            "rois": {
                name: {
                    "rgb": [clean(row) for row in trace.rgb],
                    "valid": trace.valid.tolist(),
                    "pixel_count": trace.pixel_count.tolist(),
                }
                for name, trace in self.rois.items()
            },
            "tracking": self.tracking.to_dict(),
            "settings": self.settings,
        }


def roi_mean_rgb(image_bgr: np.ndarray, roi: Roi) -> tuple[np.ndarray, float]:
    """Mean RGB over the ROI mask and the fraction of clipped pixels."""
    x0, y0, x1, y1 = roi.bbox
    pixels = image_bgr[y0:y1, x0:x1][roi.mask].astype(np.float64)
    if pixels.size == 0:
        return np.full(3, np.nan), 0.0
    brightest = pixels.max(axis=1)
    clipped = float(np.mean((brightest >= CLIP_HIGH) | (brightest <= CLIP_LOW)))
    return pixels.mean(axis=0)[::-1], clipped


def extract_rgb_traces(
    frames: Iterable[Frame],
    detector: LandmarkDetector,
    roi_config: RoiConfig | None = None,
    thresholds: TrackingThresholds | None = None,
    max_clipped_fraction: float = 0.25,
) -> RgbTraces:
    """Run detection and ROI extraction over frames, e.g. from iter_frames()."""
    roi_config = roi_config or RoiConfig()
    monitor = TrackingMonitor(thresholds)
    timestamps: list[float] = []
    indices: list[int] = []
    rgb = {name: [] for name in ROI_NAMES}
    valid = {name: [] for name in ROI_NAMES}
    counts = {name: [] for name in ROI_NAMES}

    for frame in frames:
        landmarks = detector.detect(frame.image, frame.timestamp_s)
        rois = compute_rois(landmarks, frame.image.shape, roi_config)
        for name, roi in rois.items():
            mean = np.full(3, np.nan)
            if roi.valid:
                mean, clipped = roi_mean_rgb(frame.image, roi)
                if clipped > max_clipped_fraction:
                    roi = rois[name] = replace(roi, valid=False, reason="clipped_exposure")
                    mean = np.full(3, np.nan)
            rgb[name].append(mean)
            valid[name].append(roi.valid)
            counts[name].append(roi.pixel_count)
        interocular = face_frame(landmarks).interocular if landmarks is not None else None
        monitor.update(rois, landmarks is not None, interocular)
        timestamps.append(frame.timestamp_s)
        indices.append(frame.index)

    traces = {
        name: RoiTrace(
            name=name,
            rgb=np.array(rgb[name], dtype=np.float64).reshape(-1, 3),
            valid=np.array(valid[name], dtype=bool),
            pixel_count=np.array(counts[name], dtype=np.int64),
        )
        for name in ROI_NAMES
    }
    return RgbTraces(
        timestamps_s=np.array(timestamps, dtype=np.float64),
        frame_indices=np.array(indices, dtype=np.int64),
        rois=traces,
        tracking=monitor.report(),
        settings={
            "statistic": "mean",
            "color_order": "RGB",
            "roi_config": asdict(roi_config),
            "max_clipped_fraction": max_clipped_fraction,
        },
    )


def extract_video_traces(
    path: str | Path,
    model_path: str | Path | None = None,
    max_frames: int | None = None,
    **kwargs,
) -> RgbTraces:
    """Decode a video and extract RGB traces with the MediaPipe landmarker."""
    with FaceLandmarkerDetector(model_path) as detector:
        return extract_rgb_traces(iter_frames(path, max_frames=max_frames), detector, **kwargs)
