"""Build the per-clip trace cache of the injection benchmark (PRD FR-12).

For one source clip:

1. Decode the first `max_duration_s` seconds, detect landmarks on every frame
   and fix a face-centred square crop (the median face box x CROP_SCALE, as in
   pulsecheck.dataset.normalize; NFI's normalised 256 px clips are kept
   whole). Cropping keeps the face's pixels and codec behaviour but avoids
   encoding and decoding the background 35 times.
2. For each amplitude (a = 0 first) the cropped frames get the pulse
   (pulsecheck.benchmark.injection) and are encoded once per degradation
   (libx264 CRF 18/23/28/35, and CRF 23 at half resolution) with the
   normalisation encoder settings. The measured bitrate is recorded.
3. Each encoded file is decoded again and landmarks are detected afresh, as
   the production pipeline would, and per-frame ROI mean RGB is computed for
   every spatial variant: ROI layout (standard, large, plus the whole-face
   ROI), statistic (mean, robust) and landmark smoothing (off, SMOOTHING
   frames). These traces go to `<cache>/<clip>.npz`, with a JSON sidecar of
   per-clip diagnostics: skin pixels per ROI, motion (median and 90th
   percentile landmark displacement per frame, in interocular distances),
   bitrate, face detection rate, the injected pulse and the native HR.
4. The native HR is estimated at a = 0 / CRF 18 six ways (whole-face and
   fused regional ROIs x green, CHROM, POS), and the injected HR is drawn at
   least 15 bpm from all of them (see injection.choose_target_hr for what
   happens when no such rate exists).

Encoded videos are temporary (deleted after tracing). The cache holds only
per-frame colour means and summary numbers, but it is still derived
per-video data: it stays under data/ (ignored by Git) and is not shared.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pulsecheck.benchmark.injection import (
    AMPLITUDES,
    choose_target_hr,
    fill_missing,
    inject,
    random_pulse,
    skin_weight_map,
)
from pulsecheck.benchmark.sources import BenchmarkClip
from pulsecheck.face.regions import ROI_NAMES, RoiConfig, compute_rois, face_frame
from pulsecheck.signals.rgb import STATISTICS, roi_mean_rgb, smooth_landmarks

log = logging.getLogger("pulsecheck.benchmark")

CROP_SCALE = 2.0
SMOOTHING = (0, 7)  # landmark moving-average windows, frames
MIN_FACE_FRACTION = 0.5
MAX_CLIPPED_FRACTION = 0.25
NATIVE_CONDITION = (0.0, "crf18")
# Spatial ROI sets traced per frame: (prefix, RoiConfig). Keys in the cache are
# "s<smoothing>/<statistic>/<prefix><roi>", e.g. "s0/mean/face", "s7/robust/large:forehead".
ROI_SETS = (
    ("", RoiConfig(layout="standard", face_roi=True)),
    ("large:", RoiConfig(layout="large", face_roi=False)),
)


@dataclass(frozen=True)
class Degradation:
    name: str
    crf: int
    scale: float = 1.0  # resize factor before encoding


DEGRADATIONS = (
    Degradation("crf18", 18),
    Degradation("crf23", 23),
    Degradation("crf28", 28),
    Degradation("crf35", 35),
    Degradation("crf23_half", 23, 0.5),
)
ENCODER = {"codec": "libx264", "preset": "medium", "pix_fmt": "yuv420p", "threads": 1}


def condition_key(amplitude: float, degradation: str) -> str:
    return f"{amplitude:g}@{degradation}"


def trace_keys() -> list[str]:
    keys = []
    for smooth in SMOOTHING:
        for stat in STATISTICS:
            for prefix, config in ROI_SETS:
                keys += [f"s{smooth}/{stat}/{prefix}{roi}" for roi in config.roi_names]
    return keys


def cache_name(clip_id: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in clip_id)
    return f"{safe}-{hashlib.sha1(clip_id.encode()).hexdigest()[:6]}"


def _seed(clip_id: str, salt: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{clip_id}|{salt}".encode()).digest()[:8], "little")


def _detect_all(path: Path, model_path, max_s: float | None = None):
    """Timestamps, landmarks (None where no face) and frame shape of a video."""
    from pulsecheck.face import FaceLandmarkerDetector
    from pulsecheck.video import iter_frames

    times, marks, shape = [], [], None
    with FaceLandmarkerDetector(model_path) as detector:
        for frame in iter_frames(path):
            if max_s is not None and times and frame.timestamp_s - times[0] >= max_s:
                break
            times.append(frame.timestamp_s)
            shape = frame.image.shape
            marks.append(detector.detect(frame.image, frame.timestamp_s))
    return np.asarray(times, dtype=np.float64), marks, shape


def _source_frames(path: Path, n: int, crop: tuple[int, int, int]):
    from pulsecheck.video import iter_frames

    x0, y0, side = crop
    for i, frame in enumerate(iter_frames(path)):
        if i >= n:
            return
        yield frame.image[y0 : y0 + side, x0 : x0 + side]


def motion_stats(marks: list[np.ndarray | None]) -> dict:
    """Median and 90th-percentile mean landmark displacement per frame, in IOD."""
    steps = []
    for prev, cur in zip(marks, marks[1:], strict=False):
        if prev is None or cur is None:
            continue
        iod = face_frame(cur).interocular
        if iod > 0:
            steps.append(float(np.mean(np.linalg.norm(cur - prev, axis=1))) / iod)
    if not steps:
        return {"median_iod_per_frame": None, "p90_iod_per_frame": None}
    return {
        "median_iod_per_frame": float(np.median(steps)),
        "p90_iod_per_frame": float(np.percentile(steps, 90)),
    }


def trace_video(path: Path, model_path=None) -> tuple[dict[str, np.ndarray], dict]:
    """Per-frame ROI means of every spatial variant of one encoded video, plus diagnostics."""
    from pulsecheck.video import iter_frames

    times, raw, _ = _detect_all(path, model_path)
    variants = {smooth: smooth_landmarks(raw, smooth) for smooth in SMOOTHING}
    keys = trace_keys()
    rgb = {k: np.full((len(times), 3), np.nan) for k in keys}
    counts = {k: np.zeros(len(times), np.int32) for k in keys}
    for i, frame in enumerate(iter_frames(path)):
        if i >= len(times):
            break
        for smooth, marks in variants.items():
            for prefix, config in ROI_SETS:
                rois = compute_rois(marks[i], frame.image.shape, config)
                for name, roi in rois.items():
                    for stat in STATISTICS:
                        key = f"s{smooth}/{stat}/{prefix}{name}"
                        counts[key][i] = roi.pixel_count
                        if not roi.valid:
                            continue
                        mean, clipped = roi_mean_rgb(frame.image, roi, stat)
                        if clipped <= MAX_CLIPPED_FRACTION:
                            rgb[key][i] = mean
    arrays = {"t": times - (times[0] if len(times) else 0.0)}
    for key in keys:
        arrays[f"{key}|rgb"] = rgb[key]
        arrays[f"{key}|n"] = counts[key]
    face = float(np.mean([m is not None for m in raw])) if raw else 0.0
    pixels = {}
    for prefix, config in ROI_SETS:
        for name in config.roi_names:
            c = counts[f"s0/mean/{prefix}{name}"]
            pixels[f"{prefix}{name}"] = float(np.median(c[c > 0])) if (c > 0).any() else 0.0
    return arrays, {"face_fraction": face, "median_skin_pixels": pixels, "frames": len(times)}


def native_hr_estimates(arrays: dict[str, np.ndarray], cond: str) -> dict[str, float | None]:
    """Native HR estimates of one condition: whole-face and fused regional ROIs x
    green, CHROM and POS (default signal settings)."""
    from pulsecheck.benchmark.variants import Variant, estimate

    t = arrays[f"{cond}|t"]
    out = {}
    for method in ("green", "chrom", "pos"):
        for source in ("face", "fused"):
            variant = Variant(f"native-{source}-{method}", method=method, hr_source=source)
            traces = {key: arrays[f"{cond}|{key}|rgb"] for _, key in variant.roi_keys()}
            out[f"{source}-{method}"] = estimate(variant, t, traces)[0].hr_bpm
    return out


def _encode_all(frames, fps, side, tmp: Path, amplitude, marks, pulse_t, rng) -> dict[str, Path]:
    from pulsecheck.dataset.normalize import x264_writer

    writers, paths, sizes = {}, {}, {}
    for deg in DEGRADATIONS:
        size = int(round(side * deg.scale)) // 2 * 2
        paths[deg.name] = tmp / f"{amplitude:g}_{deg.name}.mp4"
        sizes[deg.name] = size
        writers[deg.name] = x264_writer(
            paths[deg.name], (size, size), fps, deg.crf, ENCODER["preset"], ENCODER["pix_fmt"]
        )
    try:
        for k, image in enumerate(frames):
            weight = None
            if amplitude and marks[k] is not None:
                weight = skin_weight_map(marks[k], (side, side, 3))
            out = inject(image, weight, amplitude, float(pulse_t[k]), rng)
            out = np.ascontiguousarray(out)
            for deg in DEGRADATIONS:
                size = sizes[deg.name]
                img = out
                if size != side:
                    img = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA)
                writers[deg.name].send(img)
    finally:
        for writer in writers.values():
            writer.close()
    return paths


def build_clip(task: tuple[BenchmarkClip, str, str | None]) -> dict:
    """Build one clip's cache (skipped if present). Returns a status record."""
    clip, cache_dir, model_path = task
    cache_dir = Path(cache_dir)
    out = cache_dir / f"{cache_name(clip.clip_id)}.npz"
    meta_path = out.with_suffix(".json")
    start = time.perf_counter()
    if out.is_file() and meta_path.is_file():
        return {"clip_id": clip.clip_id, "status": "cached", "runtime_s": 0.0}
    tmp = cache_dir / "tmp" / cache_name(clip.clip_id)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        meta = _build(clip, out, tmp, model_path)
        meta["runtime_s"] = round(time.perf_counter() - start, 1)
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return {"clip_id": clip.clip_id, "status": "ok", "runtime_s": meta["runtime_s"]}
    except _ClipFailure as exc:
        meta = {
            "clip": clip.to_dict(),
            "status": "failed",
            "reason": exc.code,
            "detail": exc.detail,
        }
        meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
        np.savez_compressed(out, failed=np.zeros(0))
        return {"clip_id": clip.clip_id, "status": "failed", "reason": exc.code}
    except Exception as exc:  # noqa: BLE001 - reported, the run continues
        return {
            "clip_id": clip.clip_id,
            "status": "error",
            "reason": f"{type(exc).__name__}: {exc}"[:300],
            "traceback": traceback.format_exc()[-2000:],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


class _ClipFailure(Exception):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(code)
        self.code, self.detail = code, detail


def _build(clip: BenchmarkClip, out: Path, tmp: Path, model_path) -> dict:
    from pulsecheck.dataset.normalize import face_box, square_crop

    src = Path(clip.path)
    times, marks, shape = _detect_all(src, model_path, clip.max_duration_s)
    n = len(times)
    if n < 2:
        raise _ClipFailure("no_frames")
    duration = float(times[-1] - times[0])
    fps = (n - 1) / duration if duration > 0 else math.nan
    if not math.isfinite(fps) or duration + 1.0 / fps < 0.95 * clip.max_duration_s:
        raise _ClipFailure("too_short", f"{duration:.2f} s")
    face_fraction = float(np.mean([m is not None for m in marks]))
    if face_fraction < MIN_FACE_FRACTION:
        raise _ClipFailure("no_face", f"face in {face_fraction:.0%} of frames")
    height, width = shape[:2]
    boxes = np.array([face_box(m) for m in marks if m is not None])
    x0, y0, side = square_crop(boxes, width, height, CROP_SCALE)
    side -= side % 2
    crop = (x0, y0, side)
    filled = fill_missing(marks)
    local = [None if m is None else (m - np.array([x0, y0], np.float32)) for m in filled]
    t_rel = np.arange(n) / fps  # the encoded timeline
    arrays: dict[str, np.ndarray] = {}
    conditions: dict[str, dict] = {}
    pulse = None
    natives: dict = {}
    target_separated = None
    zeros = np.zeros(n)
    for amplitude in AMPLITUDES:
        if amplitude == 0:
            pulse_t = zeros
        else:
            if pulse is None:
                rng = np.random.default_rng(_seed(clip.clip_id, "hr"))
                natives = native_hr_estimates(arrays, condition_key(*NATIVE_CONDITION))
                hr, target_separated = choose_target_hr(list(natives.values()), rng)
                pulse = random_pulse(rng, hr)
            pulse_t = pulse.intensity(t_rel)
        rng = np.random.default_rng(_seed(clip.clip_id, f"dither{amplitude:g}"))
        paths = _encode_all(
            _source_frames(src, n, crop), fps, side, tmp, amplitude, local, pulse_t, rng
        )
        for deg in DEGRADATIONS:
            path = paths[deg.name]
            cond = condition_key(amplitude, deg.name)
            traced, diag = trace_video(path, model_path)
            for key, value in traced.items():
                arrays[f"{cond}|{key}"] = value
            diag["bitrate_kbps"] = path.stat().st_size * 8 / 1000 / (n / fps)
            diag["encoded_side_px"] = int(round(side * deg.scale)) // 2 * 2
            conditions[cond] = diag
            path.unlink(missing_ok=True)
    part = out.with_name(out.stem + ".part.npz")
    np.savez_compressed(part, **arrays)
    part.replace(out)
    return {
        "clip": clip.to_dict(),
        "status": "ok",
        "source": {
            "width": int(width),
            "height": int(height),
            "fps": fps,
            "frames": n,
            "duration_s": duration,
            "face_fraction": face_fraction,
            "crop": {"x0": x0, "y0": y0, "side": side, "scale": CROP_SCALE},
            "interocular_px_median": float(
                np.median([face_frame(m).interocular for m in marks if m is not None])
            ),
        },
        "motion": motion_stats(marks),
        "native_hr_bpm": natives,
        "pulse": pulse.to_dict() if pulse else None,
        "true_hr_bpm": pulse.mean_hr(t_rel) if pulse else None,
        "target_separated": target_separated,
        "conditions": conditions,
        "amplitudes": list(AMPLITUDES),
        "degradations": [d.__dict__ for d in DEGRADATIONS],
        "encoder": ENCODER,
        "smoothing_frames": list(SMOOTHING),
        "roi_names": list(ROI_NAMES),
    }


def load_clip(cache_dir: str | Path, clip_id: str) -> tuple[dict, dict[str, np.ndarray]] | None:
    """(meta, arrays) of a built clip, or None if it is missing or failed."""
    out = Path(cache_dir) / f"{cache_name(clip_id)}.npz"
    meta_path = out.with_suffix(".json")
    if not out.is_file() or not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("status") != "ok":
        return None
    with np.load(out) as data:
        arrays = {k: data[k] for k in data.files}
    return meta, arrays
