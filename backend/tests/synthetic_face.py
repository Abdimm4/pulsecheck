"""A synthetic, roughly frontal face mesh for tests that need no real face.

Only the landmark groups the ROI code uses are placed geometrically (eyes,
brows, lips, face oval); every other landmark sits at the nose. Frames are
rendered with flat colours per feature so contamination of an ROI by eyes,
brows, lips or background is easy to detect.
"""

from __future__ import annotations

import numpy as np

from pulsecheck.face.landmarker import NUM_LANDMARKS
from pulsecheck.face.regions import (
    FACE_OVAL,
    LEFT_EYE,
    LEFT_EYEBROW,
    LIPS_OUTER,
    RIGHT_EYE,
    RIGHT_EYEBROW,
)

FRAME_SHAPE = (480, 640, 3)
BACKGROUND_BGR = (40, 160, 40)
FEATURE_BGR = (20, 20, 200)


def _ellipse(center, rx, ry, n, start=-np.pi / 2):
    angles = start + np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([center[0] + rx * np.cos(angles), center[1] + ry * np.sin(angles)], axis=1)


def _eye(center):
    # Groups list the lower lid from one corner to the other, then the upper lid.
    lower = [
        (center[0] - 22 + 44 * t, center[1] + 8 * np.sin(np.pi * t)) for t in np.linspace(0, 1, 9)
    ]
    upper = [
        (center[0] - 18 + 36 * t, center[1] - 8 * np.sin(np.pi * (0.1 + 0.8 * t)))
        for t in np.linspace(0, 1, 7)
    ]
    return np.array(lower + upper)


def _brow(center, outer_sign):
    # Five lower then five upper points, each from the outer end to the inner end.
    xs = center[0] + outer_sign * np.linspace(30, -30, 5)
    lower = np.stack([xs, np.full(5, center[1] + 5)], axis=1)
    upper = np.stack([xs, np.full(5, center[1] - 5)], axis=1)
    return np.concatenate([lower, upper])


def synthetic_landmarks() -> np.ndarray:
    """478 x 2 pixel landmarks: IOD 100 px, eye line y=220, mouth line y=330."""
    lm = np.tile(np.array([[320.0, 275.0]]), (NUM_LANDMARKS, 1))
    # Subject's right eye/brow appear on the image left in an unmirrored frame.
    lm[list(RIGHT_EYE)] = _eye((270, 220))
    lm[list(LEFT_EYE)] = _eye((370, 220)) * [-1, 1] + [740, 0]
    lm[list(RIGHT_EYEBROW)] = _brow((270, 190), outer_sign=-1)
    lm[list(LEFT_EYEBROW)] = _brow((370, 190), outer_sign=1)
    lips = _ellipse((320, 330), 45, 15, len(LIPS_OUTER), start=np.pi)  # starts at corner 61
    lm[list(LIPS_OUTER)] = lips
    lm[list(FACE_OVAL)] = _ellipse((320, 240), 130, 170, len(FACE_OVAL))  # starts at top (10)
    return lm.astype(np.float32)


def transform(landmarks: np.ndarray, angle_deg=0.0, scale=1.0, shift=(0.0, 0.0)) -> np.ndarray:
    """Rotate and scale about the face centre, then translate."""
    center = np.array([320.0, 240.0])
    a = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    out = (landmarks - center) @ rot.T * scale + center + np.asarray(shift)
    return out.astype(np.float32)


def render_face(landmarks: np.ndarray, skin_bgr, shape=FRAME_SHAPE) -> np.ndarray:
    """Flat-colour frame: background, skin inside the oval, dark features."""
    import cv2

    from pulsecheck.face.regions import exclusion_hulls

    image = np.zeros(shape, np.uint8)
    image[:] = BACKGROUND_BGR
    oval = np.round(landmarks[list(FACE_OVAL)]).astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [oval], tuple(int(c) for c in skin_bgr))
    for hull in exclusion_hulls(landmarks):
        cv2.fillPoly(image, [np.round(hull).astype(np.int32).reshape(-1, 1, 2)], FEATURE_BGR)
    return image
