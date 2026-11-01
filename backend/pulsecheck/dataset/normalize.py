"""Normalised copies of dataset videos, to remove source/compression shortcuts.

When a dataset's real and fake clips come from different sources (NFI-rPPG:
real clips <= 500 px tall at ~1 Mbit/s, fakes mostly 720p at ~2 Mbit/s, with
different durations and brightness), a classifier can separate them on file
properties alone. `normalize_video` gives every clip the same format:

- a face-centred square crop: the face box (landmark extent) is measured on
  every frame of the analysed span, its median over the clip is enlarged by
  `crop_scale`, clamped to the frame and kept fixed for the whole clip (no
  per-frame jitter), then resized to `size` x `size`;
- a constant frame rate and length: output frame k shows the source frame
  nearest to k / fps (frames are repeated or dropped, never blended), for the
  first `duration_s` seconds. A clip shorter than that, or one where the face
  is found in too few frames, fails with a reason instead of being written;
- one constant gain per clip so the mean grey level of the face region (the
  central 1 / crop_scale of the crop) hits `target_luma`, capped so that the
  face's 99th-percentile channel value stays below `max_face_level` (a gain
  set on the whole frame saturated bright skin in front of dark backgrounds,
  and the ROI code rejects clipped pixels). The rPPG pipeline divides each
  colour trace by its temporal mean, which cancels a constant gain, so the
  pulse is unchanged except where pixels clip at 255 (the clipped fraction
  and whether the cap applied are recorded per clip);
- identical encoder settings: libx264 at a fixed CRF, yuv420p, no audio, no
  metadata, through the ffmpeg binary bundled with imageio-ffmpeg. Without it
  the OpenCV VideoWriter (mp4v) is used and the settings record that.

What this cannot remove: content differences (people, scenes, motion), the
generator's own artefacts, and the source's earlier compression, which the
re-encode leaves baked into the pixels. A 24 fps source still repeats every
fourth frame at 30 fps. Only summary numbers per clip are kept besides the
videos; everything stays under data/ (ignored by Git).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

TOO_SHORT = "too_short"
NO_FACE = "no_face"
DECODE_ERROR = "video_decode_error"
ENCODE_ERROR = "encode_error"


class NormalizeFailure(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class NormalizeConfig:
    size: int = 256  # output width = height, px
    fps: float = 30.0
    duration_s: float = 8.0
    crop_scale: float = 2.0  # crop side / larger side of the median face box
    min_face_fraction: float = 0.5  # of analysed frames with a detected face
    target_luma: float = 110.0  # mean grey level (0-255) of the face region after the gain
    max_face_level: float = 235.0  # gain cap: 99th-percentile face pixel stays below this
    codec: str = "libx264"
    crf: int = 23
    preset: str = "medium"
    pix_fmt: str = "yuv420p"

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.fps))

    def encoder(self) -> dict:
        try:
            import imageio_ffmpeg

            return {
                "writer": "imageio-ffmpeg",
                "ffmpeg": Path(imageio_ffmpeg.get_ffmpeg_exe()).name,
                "codec": self.codec,
                "crf": self.crf,
                "preset": self.preset,
                "pix_fmt": self.pix_fmt,
                "threads": 1,
                "audio": None,
            }
        except ImportError:
            return {"writer": "opencv", "codec": "mp4v", "audio": None}


def _frames(path: Path, end_s: float):
    """(timestamp, BGR image) for frames up to `end_s`, plus the first one past it."""
    from pulsecheck.video import iter_frames

    for frame in iter_frames(path):
        yield frame.timestamp_s, frame.image
        if frame.timestamp_s > end_s:
            return


def face_box(points: np.ndarray) -> tuple[float, float, float, float]:
    """(centre x, centre y, width, height) of a landmark set."""
    lo, hi = points.min(axis=0), points.max(axis=0)
    centre = (lo + hi) / 2
    return float(centre[0]), float(centre[1]), float(hi[0] - lo[0]), float(hi[1] - lo[1])


def square_crop(
    boxes: np.ndarray, frame_w: int, frame_h: int, scale: float
) -> tuple[int, int, int]:
    """(x0, y0, side) of a fixed square around the median face box, inside the frame."""
    cx, cy, w, h = np.median(boxes, axis=0)
    side = int(round(min(scale * max(w, h), frame_w, frame_h)))
    x0 = int(round(np.clip(cx - side / 2, 0, frame_w - side)))
    y0 = int(round(np.clip(cy - side / 2, 0, frame_h - side)))
    return x0, y0, side


def resample_indices(timestamps: np.ndarray, fps: float, n_frames: int) -> np.ndarray:
    """Index of the source frame nearest to each output time k / fps."""
    times = np.arange(n_frames) / fps + timestamps[0]
    right = np.clip(np.searchsorted(timestamps, times), 1, len(timestamps) - 1)
    left = right - 1
    nearer_left = times - timestamps[left] <= timestamps[right] - times
    return np.where(nearer_left, left, right)


def luma(image_bgr: np.ndarray) -> float:
    b, g, r = (image_bgr[..., i].astype(np.float64).mean() for i in range(3))
    return 0.114 * b + 0.587 * g + 0.299 * r


def _write(path: Path, frames: list[np.ndarray], config: NormalizeConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".part" + path.suffix)
    try:
        import imageio_ffmpeg
    except ImportError:
        imageio_ffmpeg = None
    try:
        if imageio_ffmpeg is not None:
            writer = imageio_ffmpeg.write_frames(
                str(tmp),
                (config.size, config.size),
                pix_fmt_in="bgr24",
                pix_fmt_out=config.pix_fmt,
                fps=config.fps,
                quality=None,
                codec=config.codec,
                macro_block_size=1,
                ffmpeg_log_level="error",
                output_params=[
                    "-crf", str(config.crf),
                    "-preset", config.preset,
                    "-threads", "1",
                    "-an",
                    "-map_metadata", "-1",
                ],
            )
            writer.send(None)
            for image in frames:
                writer.send(np.ascontiguousarray(image))
            writer.close()
        else:  # pragma: no cover - fallback when imageio-ffmpeg is missing
            import cv2

            out = cv2.VideoWriter(
                str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), config.fps, (config.size,) * 2
            )
            for image in frames:
                out.write(image)
            out.release()
        tmp.replace(path)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise NormalizeFailure(ENCODE_ERROR, f"{type(exc).__name__}: {exc}"[:200]) from exc


def normalize_video(
    src: str | Path, dst: str | Path, config: NormalizeConfig | None = None, model_path=None
) -> dict:
    """Write the normalised copy of `src` to `dst`; returns per-clip summary numbers.

    Raises NormalizeFailure (TOO_SHORT, NO_FACE, DECODE_ERROR, ENCODE_ERROR).
    """
    import cv2

    from pulsecheck.face import FaceLandmarkerDetector
    from pulsecheck.video import VideoDecodeError

    config = config or NormalizeConfig()
    src, dst = Path(src), Path(dst)
    end_s = config.duration_s
    # Pass 1: face boxes over the analysed span.
    boxes, timestamps, shape = [], [], None
    try:
        with FaceLandmarkerDetector(model_path) as detector:
            for t, image in _frames(src, end_s):
                timestamps.append(t)
                shape = image.shape
                points = detector.detect(image, t)
                if points is not None:
                    boxes.append(face_box(points))
    except VideoDecodeError as exc:
        raise NormalizeFailure(DECODE_ERROR, str(exc)) from None
    ts = np.asarray(timestamps, dtype=float)
    if ts.size < 2 or not np.all(np.isfinite(ts)):
        raise NormalizeFailure(DECODE_ERROR, "no usable frame timestamps")
    source_fps = (ts.size - 1) / (ts[-1] - ts[0]) if ts[-1] > ts[0] else math.nan
    # The clip must cover the last output frame's time.
    last_needed = ts[0] + (config.n_frames - 1) / config.fps
    if ts[-1] + 0.5 / source_fps < last_needed:
        raise NormalizeFailure(TOO_SHORT, f"{ts[-1] - ts[0]:.2f} s decoded, need {end_s} s")
    face_fraction = len(boxes) / ts.size
    if face_fraction < config.min_face_fraction:
        raise NormalizeFailure(NO_FACE, f"face found in {face_fraction:.0%} of frames")
    height, width = shape[:2]
    x0, y0, side = square_crop(np.array(boxes), width, height, config.crop_scale)

    # Pass 2: fixed crop, resize, constant-rate resampling, one gain.
    pick = resample_indices(ts, config.fps, config.n_frames)
    wanted = set(pick.tolist())
    crops: dict[int, np.ndarray] = {}
    interp = cv2.INTER_AREA if side >= config.size else cv2.INTER_CUBIC
    try:
        for i, (_, image) in enumerate(_frames(src, end_s)):
            if i in wanted:
                crop = image[y0 : y0 + side, x0 : x0 + side]
                crops[i] = cv2.resize(crop, (config.size, config.size), interpolation=interp)
    except VideoDecodeError as exc:
        raise NormalizeFailure(DECODE_ERROR, str(exc)) from None
    if len(crops) < len(wanted):
        raise NormalizeFailure(DECODE_ERROR, "second decode returned fewer frames")
    frames = [crops[i] for i in pick]
    # The face sits in the central 1 / crop_scale of the crop; measure brightness there.
    half = max(1, int(round(config.size / config.crop_scale / 2)))
    mid = config.size // 2
    faces = np.stack([f[mid - half : mid + half, mid - half : mid + half] for f in frames])
    mean_luma = float(np.mean([luma(f) for f in faces]))
    gain = config.target_luma / mean_luma if mean_luma > 0 else 1.0
    # Cap the gain so bright skin does not saturate (the ROI code rejects clipped pixels).
    bright = float(np.percentile(faces.max(axis=-1), 99))
    max_gain = config.max_face_level / bright if bright > 0 else gain
    capped = gain > max_gain
    gain = min(gain, max_gain)
    clipped = 0
    out = []
    for f in frames:
        scaled = f.astype(np.float32) * gain
        clipped += int(np.sum(scaled > 255))
        out.append(np.clip(np.round(scaled), 0, 255).astype(np.uint8))
    _write(dst, out, config)
    return {
        "source_width": int(width),
        "source_height": int(height),
        "source_fps": float(source_fps),
        "face_fraction": float(face_fraction),
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_side": side,
        "upscaled": bool(side < config.size),
        "source_face_luma": mean_luma,
        "gain": float(gain),
        "gain_capped": bool(capped),
        "clipped_fraction": clipped / (len(out) * config.size * config.size * 3),
        "frames_written": len(out),
        "unique_source_frames": len(wanted),
    }


def settings(config: NormalizeConfig) -> dict:
    return {**asdict(config), "n_frames": config.n_frames, "encoder": config.encoder()}
