from pathlib import Path

import numpy as np
import pytest

from pulsecheck.face import FaceLandmarkerDetector, LandmarkerUnavailableError, default_model_path
from pulsecheck.face.landmarker import MODEL_ENV_VAR, _MonotonicClock


def test_clock_converts_seconds_to_increasing_ms():
    clock = _MonotonicClock()
    assert [clock(t) for t in (0.0, 0.0333, 0.0667, 0.1)] == [0, 33, 67, 100]


def test_clock_repairs_missing_and_repeated_timestamps():
    clock = _MonotonicClock()
    timestamps = (float("nan"), 0.0, 0.0, 0.05, 0.04, float("nan"))
    assert [clock(t) for t in timestamps] == [0, 1, 2, 50, 51, 52]


def test_model_path_can_be_overridden(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(MODEL_ENV_VAR, str(tmp_path / "custom.task"))
    assert default_model_path() == tmp_path / "custom.task"
    monkeypatch.delenv(MODEL_ENV_VAR)
    assert default_model_path().parts[-2:] == ("models", "face_landmarker.task")


def test_missing_model_raises_clear_error(tmp_path: Path):
    with pytest.raises(LandmarkerUnavailableError, match="download_models.py"):
        FaceLandmarkerDetector(tmp_path / "missing.task")


@pytest.mark.skipif(not default_model_path().is_file(), reason="landmarker model not downloaded")
def test_real_landmarker_finds_no_face_in_blank_frames():
    pytest.importorskip("mediapipe")
    with FaceLandmarkerDetector() as detector:
        for i in range(3):
            assert detector.detect(np.full((240, 320, 3), 128, np.uint8), i / 30) is None
