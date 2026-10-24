"""Draw forehead/cheek ROIs on sample frames and plot the raw RGB traces.

Usage:
    python scripts/visualize_rois.py path/to/video.mp4
    python scripts/visualize_rois.py path/to/video.mp4 --out data/roi_debug --samples 6

Writes annotated PNG frames, traces.png and tracking.json to --out (default
data/roi_debug, which Git ignores). The PNGs contain the face, so treat them
as you would the source video and do not commit or share them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

from pulsecheck.face import FaceLandmarkerDetector, LandmarkerUnavailableError, compute_rois
from pulsecheck.face.regions import exclusion_hulls
from pulsecheck.signals import RgbTraces, extract_rgb_traces
from pulsecheck.video import VideoDecodeError, iter_frames, probe_video

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROI_COLORS_BGR = {
    "forehead": (0, 215, 255),
    "left_cheek": (80, 200, 60),
    "right_cheek": (230, 140, 30),
}
CHANNEL_COLORS = {"R": "#d1495b", "G": "#2e933c", "B": "#3066be"}


def annotate(image: np.ndarray, landmarks: np.ndarray | None) -> np.ndarray:
    out = image.copy()
    if landmarks is None:
        cv2.putText(out, "no face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return out
    for hull in exclusion_hulls(landmarks):
        cv2.polylines(out, [hull.astype(np.int32).reshape(-1, 1, 2)], True, (60, 60, 230), 1)
    for name, roi in compute_rois(landmarks, image.shape).items():
        color = ROI_COLORS_BGR[name]
        if roi.pixel_count:
            x0, y0, x1, y1 = roi.bbox
            region = out[y0:y1, x0:x1]
            region[roi.mask] = (0.55 * region[roi.mask] + 0.45 * np.array(color)).astype(np.uint8)
        poly = roi.polygon.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [poly], True, color if roi.valid else (0, 0, 255), 1)
    return out


class SamplingDetector:
    """Wraps a detector and saves annotated copies of selected frames as it goes.

    Frames arrive in decode order, so the call count is the frame index.
    """

    def __init__(self, inner: FaceLandmarkerDetector, wanted: set[int], out_dir: Path) -> None:
        self.inner = inner
        self.wanted = wanted
        self.out_dir = out_dir
        self.index = 0
        self.saved = 0

    def detect(self, image: np.ndarray, timestamp_s: float) -> np.ndarray | None:
        landmarks = self.inner.detect(image, timestamp_s)
        if self.index in self.wanted:
            name = f"rois_{self.index:05d}_{timestamp_s:.2f}s.png"
            cv2.imwrite(str(self.out_dir / name), annotate(image, landmarks))
            self.saved += 1
        self.index += 1
        return landmarks


def plot_traces(traces: RgbTraces, path: Path) -> None:
    t = traces.timestamps_s
    fig, axes = plt.subplots(len(traces.rois), 1, figsize=(10, 2.6 * len(traces.rois)), sharex=True)
    for ax, (name, trace) in zip(np.atleast_1d(axes), traces.rois.items(), strict=True):
        for i, channel in enumerate("RGB"):
            ax.plot(t, trace.rgb[:, i], color=CHANNEL_COLORS[channel], lw=1, label=channel)
        invalid = ~trace.valid
        if invalid.any():
            ax.fill_between(
                t,
                0,
                1,
                where=invalid,
                transform=ax.get_xaxis_transform(),
                color="0.85",
                step="mid",
                label="invalid",
            )
        stats = traces.tracking.rois[name]
        ax.set_title(f"{name} — valid in {stats.valid_fraction:.0%} of frames", loc="left")
        ax.set_ylabel("mean value (0–255)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="upper right", ncol=4, frameon=False)
    np.atleast_1d(axes)[-1].set_xlabel("time (s)")
    fig.suptitle("Raw ROI colour traces (unfiltered)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("video", type=Path)
    parser.add_argument("--out", type=Path, default=Path("data/roi_debug"))
    parser.add_argument("--samples", type=int, default=6, help="annotated frames to save")
    parser.add_argument("--model", type=Path, help="face_landmarker.task path")
    parser.add_argument("--max-frames", type=int, help="only process the first N frames")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        total = probe_video(args.video).decoded_frame_count
        if args.max_frames:
            total = min(total, args.max_frames)
        wanted = np.linspace(0, total - 1, num=min(args.samples, total)).round().astype(int)
        with FaceLandmarkerDetector(args.model) as inner:
            detector = SamplingDetector(inner, set(wanted.tolist()), args.out)
            frames = iter_frames(args.video, max_frames=args.max_frames)
            traces = extract_rgb_traces(frames, detector)
    except (VideoDecodeError, LandmarkerUnavailableError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    plot_traces(traces, args.out / "traces.png")
    report = traces.tracking.to_dict()
    (args.out / "tracking.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(
        f"saved {detector.saved} annotated frames, traces.png and tracking.json to {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
