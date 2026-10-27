"""Upload validation (PRD UP-1).

Nothing about an upload is trusted: not its name, not a client MIME type, not
its container header. Validation runs in two steps:

1. check_upload(), when the client confirms an upload: size, container type
   from the file's first bytes (ISO base media / QuickTime boxes), and a
   decode of the header and first frame with the project's video reader.
   Cheap, so it can run inside the API request.
2. check_metadata(), in the worker: the whole video is decoded (bounded by
   max_frames) and the measured duration, frame count and resolution are
   checked again, since container headers can be wrong or forged.

Rejections carry a stable code and a message that is safe to show users.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path

from pulsecheck.video import VideoDecodeError, VideoHeader, VideoMetadata, read_header

SUPPORTED_FORMATS = ("mp4", "mov")
BYTES_PER_MB = 1_000_000
# Measured duration includes the last frame's display interval and container
# timestamps are rounded, so allow a little over the limit.
DURATION_TOLERANCE_S = 0.5

EMPTY_FILE = "empty_file"
FILE_TOO_LARGE = "file_too_large"
UNSUPPORTED_FORMAT = "unsupported_format"
CORRUPT_VIDEO = "corrupt_video"
RESOLUTION_TOO_HIGH = "resolution_too_high"
VIDEO_TOO_LONG = "video_too_long"
TOO_MANY_FRAMES = "too_many_frames"

# Top-level boxes that can open a QuickTime file written without an ftyp box.
_QUICKTIME_FIRST_BOXES = {b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"}


class UploadRejected(Exception):
    """The upload breaks a limit or is not a supported, decodable video."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class UploadLimits:
    allowed_formats: tuple[str, ...] = SUPPORTED_FORMATS
    max_size_mb: float = 100
    max_duration_s: float = 60
    max_resolution: tuple[int, int] = (1920, 1080)  # long side, short side
    max_frames: int = 3600

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_formats", tuple(self.allowed_formats))
        object.__setattr__(self, "max_resolution", tuple(self.max_resolution))
        unknown = set(self.allowed_formats) - set(SUPPORTED_FORMATS)
        if unknown or not self.allowed_formats:
            raise ValueError(f"allowed_formats must be a subset of {SUPPORTED_FORMATS}")
        if len(self.max_resolution) != 2 or self.max_resolution[0] < self.max_resolution[1]:
            raise ValueError("max_resolution must be [long side, short side]")
        if min(self.max_size_mb, self.max_duration_s, self.max_frames, *self.max_resolution) <= 0:
            raise ValueError("upload limits must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> UploadLimits:
        values = dict(values or {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown upload settings: {sorted(unknown)}")
        return cls(**values)

    @property
    def max_bytes(self) -> int:
        return int(self.max_size_mb * BYTES_PER_MB)

    def to_public(self) -> dict:
        """The limits to show users before they upload (UP-1)."""
        return {
            "allowed_formats": list(self.allowed_formats),
            "max_size_bytes": self.max_bytes,
            "max_duration_s": self.max_duration_s,
            "max_resolution": list(self.max_resolution),
            "max_frames": self.max_frames,
        }


@dataclass(frozen=True)
class ValidatedUpload:
    container: str
    size_bytes: int
    header: VideoHeader


def sniff_container(path: str | Path) -> str | None:
    """'mp4', 'mov' or None, from the first box of the file."""
    with Path(path).open("rb") as handle:
        head = handle.read(12)
    if len(head) < 8:
        return None
    box = head[4:8]
    if box == b"ftyp":
        return "mov" if head[8:12] == b"qt  " else "mp4"
    if box in _QUICKTIME_FIRST_BOXES:
        return "mov"
    return None


def _check_resolution(width: int, height: int, limits: UploadLimits) -> None:
    if width <= 0 or height <= 0:
        raise UploadRejected(CORRUPT_VIDEO, "The video's frame size could not be read.")
    long_side, short_side = max(width, height), min(width, height)
    max_long, max_short = limits.max_resolution
    if long_side > max_long or short_side > max_short:
        raise UploadRejected(
            RESOLUTION_TOO_HIGH,
            f"The video is {width}x{height}; the maximum is {max_long}x{max_short}.",
        )


def _check_duration(duration_s: float | None, limits: UploadLimits) -> None:
    if duration_s is not None and duration_s > limits.max_duration_s + DURATION_TOLERANCE_S:
        raise UploadRejected(
            VIDEO_TOO_LONG,
            f"The video is {duration_s:.1f} s long; the maximum is {limits.max_duration_s:g} s.",
        )


def _check_frames(frame_count: int, limits: UploadLimits) -> None:
    if frame_count > limits.max_frames:
        raise UploadRejected(
            TOO_MANY_FRAMES, f"The video has more than {limits.max_frames} frames."
        )


def check_size(size_bytes: int, limits: UploadLimits) -> None:
    if size_bytes <= 0:
        raise UploadRejected(EMPTY_FILE, "The uploaded file is empty.")
    if size_bytes > limits.max_bytes:
        raise UploadRejected(FILE_TOO_LARGE, f"The file is larger than {limits.max_size_mb:g} MB.")


def check_upload(path: str | Path, limits: UploadLimits) -> ValidatedUpload:
    """Fast checks before queueing; raises UploadRejected."""
    path = Path(path)
    size = path.stat().st_size
    check_size(size, limits)
    container = sniff_container(path)
    if container is None or container not in limits.allowed_formats:
        allowed = ", ".join(f.upper() for f in limits.allowed_formats)
        raise UploadRejected(UNSUPPORTED_FORMAT, f"Only {allowed} videos are supported.")
    try:
        header = read_header(path)
    except VideoDecodeError:
        raise UploadRejected(CORRUPT_VIDEO, "The file could not be opened as a video.") from None
    if not header.first_frame_decoded:
        raise UploadRejected(
            CORRUPT_VIDEO,
            "The video could not be decoded; it may be corrupt or use an unsupported codec.",
        )
    _check_resolution(header.width, header.height, limits)
    _check_frames(header.header_frame_count, limits)
    _check_duration(header.header_duration_s, limits)
    return ValidatedUpload(container, size, header)


def check_metadata(meta: VideoMetadata, limits: UploadLimits) -> None:
    """Checks on the measured metadata from a full (bounded) decode; raises UploadRejected.

    Pass max_frames = limits.max_frames + 1 to probe_video() so an over-long
    video is detected without decoding all of it.
    """
    _check_resolution(meta.width, meta.height, limits)
    _check_frames(meta.decoded_frame_count, limits)
    _check_duration(meta.duration_s, limits)
