"""Video decoding and metadata inspection (PRD FR-2).

Metadata is measured by decoding every frame rather than trusting the
container header, because header frame counts and FPS are often wrong and
variable-frame-rate videos are common. Each frame keeps its own timestamp.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np


class VideoDecodeError(Exception):
    """Raised when a file cannot be opened or decoded as video."""


@dataclass(frozen=True)
class Frame:
    index: int
    timestamp_s: float
    image: np.ndarray  # BGR, as returned by OpenCV


@dataclass
class VideoMetadata:
    path: str
    width: int
    height: int
    header_fps: float
    header_frame_count: int
    decoded_frame_count: int
    duration_s: float
    estimated_fps: float
    timestamps_monotonic: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _open(path: str | Path) -> cv2.VideoCapture:
    path = Path(path)
    if not path.is_file():
        raise VideoDecodeError(f"File not found: {path}")
    if path.stat().st_size == 0:
        raise VideoDecodeError("File is empty")
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoDecodeError("File could not be opened as video")
    return cap


def iter_frames(path: str | Path, max_frames: int | None = None) -> Iterator[Frame]:
    """Yield decoded frames with timestamps in seconds.

    Uses the decoder's per-frame position when available and falls back to
    index / header FPS when the container does not report timestamps.
    """
    cap = _open(path)
    try:
        header_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        index = 0
        while max_frames is None or index < max_frames:
            ok, image = cap.read()
            if not ok:
                break
            pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_ms > 0 or index == 0:
                timestamp_s = pos_ms / 1000.0
            elif header_fps > 0:
                timestamp_s = index / header_fps
            else:
                timestamp_s = float("nan")
            yield Frame(index=index, timestamp_s=timestamp_s, image=image)
            index += 1
    finally:
        cap.release()


def probe_video(path: str | Path) -> VideoMetadata:
    """Decode the whole video and report measured metadata."""
    cap = _open(path)
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        header_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        header_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()

    timestamps = np.array([f.timestamp_s for f in iter_frames(path)], dtype=float)
    warnings: list[str] = []

    if timestamps.size == 0:
        raise VideoDecodeError("No frames could be decoded")

    if np.isnan(timestamps).any():
        warnings.append("Some frame timestamps are unavailable")

    diffs = np.diff(timestamps[~np.isnan(timestamps)])
    monotonic = bool(np.all(diffs > 0)) if diffs.size else True
    if not monotonic:
        warnings.append("Frame timestamps are not strictly increasing")

    if diffs.size and monotonic:
        median_dt = float(np.median(diffs))
        estimated_fps = 1.0 / median_dt
        # Duration covers the last frame's display interval, not just its start.
        duration_s = float(timestamps[-1] - timestamps[0]) + median_dt
        if diffs.size > 1 and float(np.std(diffs)) > 0.1 * median_dt:
            warnings.append("Frame intervals vary; video may have a variable frame rate")
    else:
        estimated_fps = header_fps
        duration_s = timestamps.size / header_fps if header_fps > 0 else 0.0

    if header_frame_count and abs(header_frame_count - timestamps.size) > 1:
        warnings.append(
            f"Header reports {header_frame_count} frames but {timestamps.size} were decoded"
        )

    return VideoMetadata(
        path=str(path),
        width=width,
        height=height,
        header_fps=header_fps,
        header_frame_count=header_frame_count,
        decoded_frame_count=int(timestamps.size),
        duration_s=duration_s,
        estimated_fps=estimated_fps,
        timestamps_monotonic=monotonic,
        warnings=warnings,
    )
