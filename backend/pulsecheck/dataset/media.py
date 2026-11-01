"""Non-physiological per-video properties for shortcut checks.

Resolution, container frame rate and duration, bitrate and mean frame
brightness say nothing about a pulse, but they can differ systematically
between the real and fake sources of a dataset. A classifier that separates
real from fake on these alone has found a source/compression shortcut, so
pulse-feature results should be read against it. Only summary numbers are
stored; no frames.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Iterable, Mapping
from pathlib import Path

MEDIA_COLUMNS = (
    "video_id",
    "width",
    "height",
    "container_fps",
    "container_duration_s",
    "bitrate_kbps",
    "mean_brightness",
    "probe_error",
)
# Shortcut feature set: media columns plus table columns that carry no pulse
# information (fps and duration from extraction, face tracking coverage).
SHORTCUT_FEATURES = (
    "width",
    "height",
    "bitrate_kbps",
    "mean_brightness",
    "fps",
    "duration_s",
    "face_coverage",
)


def probe_media(path: str | Path, samples: int = 24) -> dict[str, float | str]:
    """Container properties and the mean grey level of `samples` evenly spaced frames."""
    import cv2

    path = Path(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"probe_error": "cannot_open"}
    try:
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = count / fps if fps > 0 and count > 0 else math.nan
        step = max(1, count // samples) if count > 0 else 1
        levels = []
        index = 0
        while len(levels) < samples:
            ok = cap.grab()
            if not ok:
                break
            if index % step == 0:
                ok, frame = cap.retrieve()
                if ok:
                    levels.append(float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean()))
            index += 1
    finally:
        cap.release()
    size_bits = path.stat().st_size * 8
    return {
        # Report the long side first so portrait and landscape clips compare.
        "width": float(max(width, height)),
        "height": float(min(width, height)),
        "container_fps": float(fps),
        "container_duration_s": float(duration),
        "bitrate_kbps": size_bits / duration / 1000 if duration and duration > 0 else math.nan,
        "mean_brightness": sum(levels) / len(levels) if levels else math.nan,
        "probe_error": "" if levels else "no_frames",
    }


def write_media_table(path: str | Path, rows: Iterable[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MEDIA_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    c: (
                        ""
                        if isinstance(row.get(c), float) and not math.isfinite(row[c])
                        else row.get(c, "")
                    )
                    for c in MEDIA_COLUMNS
                }
            )
