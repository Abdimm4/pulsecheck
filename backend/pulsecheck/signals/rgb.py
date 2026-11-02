"""Per-ROI RGB mean time series (PRD FR-4).

For every frame and ROI we keep the mean R, G and B of the masked skin pixels,
the frame timestamp, the pixel count and the ROI's validity. Frames where an
ROI is invalid hold NaN so the series stay aligned with the timestamps. No
landmarks, masks or pixels are retained (PRD UP-7).

Options (ColorSignalConfig, configs/default.yaml `color_signal:`):

- `statistic`: "mean" averages every masked pixel; "robust" first drops
  clipped pixels and pixels whose luma is more than ROBUST_K robust standard
  deviations (1.4826 x MAD, at least one 8-bit level) from the ROI median, e.g. specular highlights,
  stray hair or shadow at the mask edge.
- `landmark_smoothing_frames`: a centred moving average of the landmarks over
  this many frames (0 or 1 = off) before the ROIs are built, to reduce
  frame-to-frame ROI jitter. Frames are buffered for half a window, so the
  video is still decoded once.
- `roi_layout` and `face_roi`: see pulsecheck.face.regions.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import numpy as np

from pulsecheck.face.landmarker import FaceLandmarkerDetector, LandmarkDetector
from pulsecheck.face.regions import Roi, RoiConfig, compute_rois, face_frame
from pulsecheck.face.tracking import TrackingMonitor, TrackingReport, TrackingThresholds
from pulsecheck.video.reader import Frame, iter_frames

# Pixels at or beyond these 8-bit levels carry no usable colour variation.
CLIP_LOW = 5
CLIP_HIGH = 250
STATISTICS = ("mean", "robust")
ROBUST_K = 2.5  # "robust" statistic: luma outlier cut, in robust standard deviations
LUMA_BGR = np.array([0.114, 0.587, 0.299])


@dataclass(frozen=True)
class ColorSignalConfig:
    """Colour-trace settings (configs/default.yaml `color_signal:`)."""

    statistic: str = "mean"
    max_clipped_fraction: float = 0.25
    roi_layout: str = "standard"
    face_roi: bool = False
    landmark_smoothing_frames: int = 0

    def __post_init__(self) -> None:
        if self.statistic not in STATISTICS:
            raise ValueError(f"statistic must be one of {STATISTICS}, got {self.statistic!r}")
        if self.landmark_smoothing_frames < 0:
            raise ValueError("landmark_smoothing_frames must be >= 0")
        RoiConfig(layout=self.roi_layout)  # validates the layout

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> ColorSignalConfig:
        """Build from a config section (configs/default.yaml `color_signal:`)."""
        values = dict(values or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown color_signal settings: {sorted(unknown)}")
        return cls(**values)

    def roi_config(self) -> RoiConfig:
        return RoiConfig(layout=self.roi_layout, face_roi=self.face_roi)

    def extract_kwargs(self) -> dict:
        """Keyword arguments for extract_rgb_traces()."""
        return {
            "roi_config": self.roi_config(),
            "max_clipped_fraction": self.max_clipped_fraction,
            "statistic": self.statistic,
            "landmark_smoothing_frames": self.landmark_smoothing_frames,
        }

    def to_dict(self) -> dict:
        return asdict(self)


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


def roi_mean_rgb(
    image_bgr: np.ndarray, roi: Roi, statistic: str = "mean"
) -> tuple[np.ndarray, float]:
    """Mean RGB over the ROI mask (see `statistic`) and the fraction of clipped pixels."""
    x0, y0, x1, y1 = roi.bbox
    pixels = image_bgr[y0:y1, x0:x1][roi.mask].astype(np.float64)
    if pixels.size == 0:
        return np.full(3, np.nan), 0.0
    brightest = pixels.max(axis=1)
    clipped_rows = (brightest >= CLIP_HIGH) | (brightest <= CLIP_LOW)
    clipped = float(np.mean(clipped_rows))
    if statistic == "robust":
        kept = pixels[~clipped_rows]
        if len(kept) >= 10:
            luma = kept @ LUMA_BGR
            median = float(np.median(luma))
            # Robust SD, floored at one 8-bit level so flat regions still drop outliers.
            spread = max(1.4826 * float(np.median(np.abs(luma - median))), 1.0)
            kept = kept[np.abs(luma - median) <= ROBUST_K * spread]
        if len(kept):
            pixels = kept
    return pixels.mean(axis=0)[::-1], clipped


def frame_roi_means(
    image_bgr: np.ndarray,
    landmarks: np.ndarray | None,
    roi_config: RoiConfig,
    statistic: str = "mean",
    max_clipped_fraction: float = 0.25,
) -> tuple[dict[str, Roi], dict[str, np.ndarray]]:
    """ROIs of one frame and their mean RGB (NaN where invalid or overexposed)."""
    rois = compute_rois(landmarks, image_bgr.shape, roi_config)
    means = {}
    for name, roi in rois.items():
        mean = np.full(3, np.nan)
        if roi.valid:
            mean, clipped = roi_mean_rgb(image_bgr, roi, statistic)
            if clipped > max_clipped_fraction:
                rois[name] = replace(roi, valid=False, reason="clipped_exposure")
                mean = np.full(3, np.nan)
        means[name] = mean
    return rois, means


def _window_mean(window: Iterable[np.ndarray | None]) -> np.ndarray:
    return np.mean([lm for lm in window if lm is not None], axis=0).astype(np.float32)


def smooth_landmarks(landmarks: list[np.ndarray | None], window: int) -> list[np.ndarray | None]:
    """Centred moving average over `window` frames, skipping frames without a face.

    A frame without a face stays None; near the ends the window is truncated.
    """
    half = window // 2 if window > 1 else 0
    if half == 0:
        return list(landmarks)
    return [
        None if lm is None else _window_mean(landmarks[max(0, i - half) : i + half + 1])
        for i, lm in enumerate(landmarks)
    ]


def smoothed_landmark_stream(
    frames: Iterable[Frame], detector: LandmarkDetector, window: int = 0
) -> Iterator[tuple[Frame, np.ndarray | None]]:
    """(frame, landmarks) in order, with landmarks as smooth_landmarks() would give.

    Holds at most `window // 2 + 1` decoded frames, so memory stays bounded.
    """
    half = window // 2 if window > 1 else 0
    frames_held: deque[Frame] = deque()
    marks: deque[np.ndarray | None] = deque()  # landmarks from index `first` onwards
    first = 0  # absolute index of marks[0]
    emitted = 0  # absolute index of the next frame to emit

    def emit() -> tuple[Frame, np.ndarray | None]:
        nonlocal emitted, first
        frame = frames_held.popleft()
        raw = marks[emitted - first]
        lo = max(0, emitted - half) - first
        hi = emitted + half + 1 - first
        smoothed = None if raw is None else _window_mean(list(marks)[lo:hi])
        emitted += 1
        while emitted - half > first:  # drop landmarks no window needs any more
            marks.popleft()
            first += 1
        return frame, smoothed

    for frame in frames:
        marks.append(detector.detect(frame.image, frame.timestamp_s))
        frames_held.append(frame)
        if len(frames_held) > half:
            yield emit()
    while frames_held:
        yield emit()


def extract_rgb_traces(
    frames: Iterable[Frame],
    detector: LandmarkDetector,
    roi_config: RoiConfig | None = None,
    thresholds: TrackingThresholds | None = None,
    max_clipped_fraction: float = 0.25,
    statistic: str = "mean",
    landmark_smoothing_frames: int = 0,
) -> RgbTraces:
    """Run detection and ROI extraction over frames, e.g. from iter_frames()."""
    roi_config = roi_config or RoiConfig()
    names = roi_config.roi_names
    monitor = TrackingMonitor(thresholds)
    timestamps: list[float] = []
    indices: list[int] = []
    rgb = {name: [] for name in names}
    valid = {name: [] for name in names}
    counts = {name: [] for name in names}

    for frame, landmarks in smoothed_landmark_stream(frames, detector, landmark_smoothing_frames):
        rois, means = frame_roi_means(
            frame.image, landmarks, roi_config, statistic, max_clipped_fraction
        )
        for name, roi in rois.items():
            rgb[name].append(means[name])
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
        for name in names
    }
    return RgbTraces(
        timestamps_s=np.array(timestamps, dtype=np.float64),
        frame_indices=np.array(indices, dtype=np.int64),
        rois=traces,
        tracking=monitor.report(),
        settings={
            "statistic": statistic,
            "color_order": "RGB",
            "roi_config": asdict(roi_config),
            "max_clipped_fraction": max_clipped_fraction,
            "landmark_smoothing_frames": landmark_smoothing_frames,
        },
    )


def extract_video_traces(
    path: str | Path,
    model_path: str | Path | None = None,
    max_frames: int | None = None,
    color_config: ColorSignalConfig | None = None,
    **kwargs,
) -> RgbTraces:
    """Decode a video and extract RGB traces with the MediaPipe landmarker.

    `color_config` supplies the ROI and trace settings; extra keyword arguments
    go to extract_rgb_traces() and override it.
    """
    if color_config is not None:
        kwargs = {**color_config.extract_kwargs(), **kwargs}
    with FaceLandmarkerDetector(model_path) as detector:
        return extract_rgb_traces(iter_frames(path, max_frames=max_frames), detector, **kwargs)
