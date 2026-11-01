import cv2
import numpy as np
import pytest

import pulsecheck.face
from pulsecheck.dataset.media import probe_media
from pulsecheck.dataset.normalize import (
    NO_FACE,
    TOO_SHORT,
    NormalizeConfig,
    NormalizeFailure,
    luma,
    normalize_video,
    resample_indices,
    square_crop,
)
from pulsecheck.video import iter_frames


def test_square_crop_is_median_box_enlarged_and_clamped():
    boxes = np.array([[100, 80, 40, 50], [102, 80, 40, 50], [300, 80, 40, 50]], dtype=float)
    # Median centre (102, 80), larger side 50, x2 -> 100 px square.
    assert square_crop(boxes, 640, 480, 2.0) == (52, 30, 100)
    # Near the top-left corner the square shifts into the frame.
    assert square_crop(np.array([[10.0, 10.0, 40.0, 40.0]]), 640, 480, 2.0) == (0, 0, 80)
    # Never larger than the frame's short side.
    assert square_crop(np.array([[320.0, 240.0, 400.0, 400.0]]), 640, 480, 2.0) == (80, 0, 480)


def test_resample_indices_to_constant_rate():
    ts = np.arange(48) / 24.0  # 2 s at 24 fps
    idx = resample_indices(ts, 30.0, 60)
    assert len(idx) == 60 and idx[0] == 0
    assert np.all(np.diff(idx) >= 0)
    assert np.max(np.abs(ts[idx] - np.arange(60) / 30.0)) <= 0.5 / 24 + 1e-9
    # 24 -> 30 fps repeats one frame in five output frames.
    assert len(np.unique(idx)) == 48
    # Same rate: identity.
    assert np.array_equal(resample_indices(np.arange(30) / 30.0, 30.0, 30), np.arange(30))


def test_luma_weights():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[..., 1] = 100  # green only (BGR)
    assert luma(image) == pytest.approx(58.7)


class _StubDetector:
    """Returns a fixed face box (two corner landmarks) regardless of the frame."""

    box = ((260.0, 160.0), (380.0, 320.0))
    detect_every = 1

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def detect(self, image, timestamp_s):
        self.calls += 1
        if self.calls % self.detect_every:
            return None
        return np.array(self.box, dtype=np.float32)


def _clip(path, seconds, fps=24.0, size=(640, 480), level=60):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(int(seconds * fps)):
        image = np.full((size[1], size[0], 3), level, dtype=np.uint8)
        image[:, :, 2] = level + 10 * np.sin(2 * np.pi * 1.2 * i / fps)  # a pulse-like red
        writer.write(image)
    writer.release()
    return path


@pytest.fixture
def stub_detector(monkeypatch):
    monkeypatch.setattr(pulsecheck.face, "FaceLandmarkerDetector", _StubDetector)
    return _StubDetector


def test_normalize_video_end_to_end(tmp_path, stub_detector):
    src = _clip(tmp_path / "src.mp4", seconds=3.0)
    config = NormalizeConfig(size=128, duration_s=2.0, target_luma=110.0)
    stats = normalize_video(src, tmp_path / "out" / "dst.mp4", config)
    # Box 120 x 160 centred at (320, 240): crop side 2 x 160 = 320.
    assert (stats["crop_x0"], stats["crop_y0"], stats["crop_side"]) == (160, 80, 320)
    assert stats["frames_written"] == 60
    assert stats["gain"] == pytest.approx(110.0 / stats["source_face_luma"])
    assert not stats["gain_capped"]
    assert abs(stats["source_fps"] - 24.0) < 0.5
    media = probe_media(tmp_path / "out" / "dst.mp4")
    assert (media["width"], media["height"]) == (128.0, 128.0)
    assert media["container_fps"] == pytest.approx(30.0)
    assert media["container_duration_s"] == pytest.approx(2.0)
    frames = list(iter_frames(tmp_path / "out" / "dst.mp4"))
    assert len(frames) == 60
    mean = np.mean([luma(f.image) for f in frames])
    assert mean == pytest.approx(110.0, abs=3.0)
    assert not list((tmp_path / "out").glob("*.part*"))


def test_normalize_rejects_short_clips_and_missing_faces(tmp_path, stub_detector, monkeypatch):
    src = _clip(tmp_path / "short.mp4", seconds=1.0)
    with pytest.raises(NormalizeFailure) as err:
        normalize_video(src, tmp_path / "x.mp4", NormalizeConfig(duration_s=2.0))
    assert err.value.code == TOO_SHORT
    assert not (tmp_path / "x.mp4").exists()

    monkeypatch.setattr(stub_detector, "detect_every", 3)  # face in a third of the frames
    src = _clip(tmp_path / "long.mp4", seconds=3.0)
    with pytest.raises(NormalizeFailure) as err:
        normalize_video(src, tmp_path / "y.mp4", NormalizeConfig(duration_s=2.0))
    assert err.value.code == NO_FACE
