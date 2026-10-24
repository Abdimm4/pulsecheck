"""Facial landmark detection with MediaPipe Face Landmarker (PRD FR-3).

Only the 2D pixel positions of the 478 mesh landmarks are returned. Depth,
blendshapes and transformation matrices are never requested, and nothing here
computes identity embeddings (PRD UP-7).
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from types import TracebackType
from typing import Protocol

import cv2
import numpy as np

NUM_LANDMARKS = 478

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"
MODEL_ENV_VAR = "PULSECHECK_FACE_MODEL"


class LandmarkerUnavailableError(Exception):
    """Raised when MediaPipe or the landmarker model file cannot be loaded."""


class LandmarkDetector(Protocol):
    """Anything that maps a BGR frame to (478, 2) pixel landmarks or None."""

    def detect(self, image_bgr: np.ndarray, timestamp_s: float) -> np.ndarray | None: ...


def default_model_path() -> Path:
    """Model location: $PULSECHECK_FACE_MODEL, else <repo>/models/face_landmarker.task."""
    override = os.environ.get(MODEL_ENV_VAR)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "models" / "face_landmarker.task"


class _MonotonicClock:
    """Converts frame timestamps to the strictly increasing ms MediaPipe VIDEO mode needs.

    Missing (NaN) or non-increasing timestamps are nudged forward by 1 ms so a
    bad container timestamp cannot crash tracking.
    """

    def __init__(self) -> None:
        self._last_ms: int | None = None

    def __call__(self, timestamp_s: float) -> int:
        ms = None if math.isnan(timestamp_s) else int(round(timestamp_s * 1000.0))
        if self._last_ms is not None and (ms is None or ms <= self._last_ms):
            ms = self._last_ms + 1
        elif ms is None:
            ms = 0
        self._last_ms = ms
        return ms


class FaceLandmarkerDetector:
    """Tracks one face across consecutive video frames.

    Use as a context manager, one instance per video, and feed frames in
    order: VIDEO mode carries tracking state between calls.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = Path(model_path) if model_path else default_model_path()
        if not model_path.is_file():
            raise LandmarkerUnavailableError(
                f"Face landmarker model not found at {model_path}; "
                "run scripts/download_models.py or set " + MODEL_ENV_VAR
            )
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                FaceLandmarker,
                FaceLandmarkerOptions,
                RunningMode,
            )
        except ImportError as exc:  # pragma: no cover - depends on install
            raise LandmarkerUnavailableError(f"MediaPipe is not installed: {exc}") from exc

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._mp = mp
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._clock = _MonotonicClock()

    def detect(self, image_bgr: np.ndarray, timestamp_s: float) -> np.ndarray | None:
        height, width = image_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(mp_image, self._clock(timestamp_s))
        if not result.face_landmarks:
            return None
        face = result.face_landmarks[0]
        return np.array([(lm.x * width, lm.y * height) for lm in face], dtype=np.float32)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> FaceLandmarkerDetector:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
