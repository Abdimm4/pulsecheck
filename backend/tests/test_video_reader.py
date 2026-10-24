from itertools import pairwise
from pathlib import Path

import cv2
import numpy as np
import pytest

from pulsecheck.video import VideoDecodeError, iter_frames, probe_video


def _write_video(path: Path, n_frames: int, fps: float = 30.0, size=(64, 48)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV could not create a test video"
    for i in range(n_frames):
        frame = np.full((size[1], size[0], 3), i % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def sample_video(tmp_path: Path) -> Path:
    return _write_video(tmp_path / "sample.mp4", n_frames=60, fps=30.0)


def test_probe_reports_measured_metadata(sample_video: Path):
    meta = probe_video(sample_video)
    assert (meta.width, meta.height) == (64, 48)
    assert meta.decoded_frame_count == 60
    assert meta.estimated_fps == pytest.approx(30.0, rel=0.02)
    assert meta.duration_s == pytest.approx(2.0, rel=0.05)
    assert meta.timestamps_monotonic


def test_iter_frames_timestamps_increase(sample_video: Path):
    frames = list(iter_frames(sample_video, max_frames=10))
    assert [f.index for f in frames] == list(range(10))
    assert frames[0].timestamp_s == pytest.approx(0.0, abs=1e-6)
    assert all(b.timestamp_s > a.timestamp_s for a, b in pairwise(frames))
    assert frames[0].image.shape == (48, 64, 3)


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(VideoDecodeError, match="not found"):
        probe_video(tmp_path / "nope.mp4")


def test_empty_file_raises(tmp_path: Path):
    empty = tmp_path / "empty.mp4"
    empty.touch()
    with pytest.raises(VideoDecodeError, match="empty"):
        probe_video(empty)


def test_non_video_file_raises(tmp_path: Path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"this is not a video" * 100)
    with pytest.raises(VideoDecodeError):
        probe_video(junk)
