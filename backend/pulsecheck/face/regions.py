"""Forehead and cheek regions of interest from face mesh landmarks (PRD FR-3).

ROIs are placed in a face-aligned affine frame: the x axis runs from the
subject's right eye centre to the left eye centre (one interocular distance,
IOD), and the y axis runs from the eye midpoint to the mouth midpoint. This
follows roll, scale and pitch without a 3D head model.

Each ROI mask is then clipped to the face oval (removing background and most
hair) and has dilated hulls of the eyes, eyebrows and lips subtracted, so those
features are excluded even if the template polygons are tuned badly.

"Left" and "right" follow MediaPipe's convention: they mean the subject's left
and right, so in an unmirrored video the left cheek appears on the image right.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pulsecheck.face.landmarker import NUM_LANDMARKS

ROI_NAMES = ("forehead", "left_cheek", "right_cheek")

# Landmark indices of the MediaPipe 478-point face mesh. The groups match
# mediapipe.tasks.python.vision.FaceLandmarksConnections (checked in tests).
FACE_OVAL = (
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
    152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
)  # fmt: skip
RIGHT_EYE = (33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173)
LEFT_EYE = (263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398)
RIGHT_EYEBROW = (46, 53, 52, 65, 55, 70, 63, 105, 66, 107)
LEFT_EYEBROW = (276, 283, 282, 295, 285, 300, 293, 334, 296, 336)
LIPS_OUTER = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
)  # fmt: skip
MOUTH_CORNERS = (61, 291)
# Upper brow contour from the subject's right to left, without the outermost
# points that run into the temples and hair.
BROW_TOPS = (63, 105, 66, 107, 336, 296, 334, 293)

# Subject's right cheek in face-frame units (u: IOD from the eye midpoint, with
# the right eye at u=-0.5; v: 0 at the eye line, 1 at the mouth line). The left
# cheek mirrors it. Chosen to sit below the eye, lateral to the nose and above
# the mouth corner.
RIGHT_CHEEK_TEMPLATE = np.array(
    [(-0.30, 0.28), (-0.72, 0.28), (-0.76, 0.58), (-0.50, 0.72), (-0.36, 0.55)],
    dtype=np.float32,
)


@dataclass(frozen=True)
class RoiConfig:
    # Forehead band above the brow line, as fractions of the eye-to-mouth vector.
    forehead_gap: float = 0.12
    forehead_height: float = 0.45
    # Margin added around eyes, brows and lips before subtracting them, in IOD.
    exclusion_margin: float = 0.06
    # Validity thresholds.
    min_interocular_px: float = 20.0
    min_pixels: int = 30
    min_area_fraction: float = 0.5


@dataclass(frozen=True)
class Roi:
    name: str
    polygon: np.ndarray  # (K, 2) float32 nominal vertices in pixels, before clipping
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive), clipped to the frame
    mask: np.ndarray  # bool, shape (y1 - y0, x1 - x0)
    pixel_count: int
    valid: bool
    reason: str | None = None  # why the ROI is invalid, if it is

    def centroid(self) -> tuple[float, float] | None:
        if self.pixel_count == 0:
            return None
        ys, xs = np.nonzero(self.mask)
        return float(xs.mean() + self.bbox[0]), float(ys.mean() + self.bbox[1])


@dataclass(frozen=True)
class FaceFrame:
    """Affine face-aligned coordinate frame derived from landmarks."""

    origin: np.ndarray  # eye midpoint
    x_axis: np.ndarray  # right eye centre -> left eye centre
    y_axis: np.ndarray  # eye midpoint -> mouth midpoint

    @property
    def interocular(self) -> float:
        return float(np.linalg.norm(self.x_axis))

    def to_pixels(self, uv: np.ndarray) -> np.ndarray:
        uv = np.asarray(uv, dtype=np.float32)
        return self.origin + uv[:, :1] * self.x_axis + uv[:, 1:2] * self.y_axis


def face_frame(landmarks: np.ndarray) -> FaceFrame:
    right_eye = landmarks[list(RIGHT_EYE)].mean(axis=0)
    left_eye = landmarks[list(LEFT_EYE)].mean(axis=0)
    origin = (right_eye + left_eye) / 2.0
    mouth = landmarks[list(MOUTH_CORNERS)].mean(axis=0)
    return FaceFrame(origin=origin, x_axis=left_eye - right_eye, y_axis=mouth - origin)


def roi_polygons(landmarks: np.ndarray, config: RoiConfig | None = None) -> dict[str, np.ndarray]:
    """Nominal ROI polygons in pixel coordinates, before clipping and exclusion."""
    config = config or RoiConfig()
    frame = face_frame(landmarks)

    brow = landmarks[list(BROW_TOPS)].astype(np.float32)
    up = -frame.y_axis
    forehead = np.concatenate(
        [brow + config.forehead_gap * up, (brow + config.forehead_height * up)[::-1]]
    )

    mirrored = RIGHT_CHEEK_TEMPLATE * np.array([-1.0, 1.0], dtype=np.float32)
    return {
        "forehead": forehead.astype(np.float32),
        "left_cheek": frame.to_pixels(mirrored[::-1]).astype(np.float32),
        "right_cheek": frame.to_pixels(RIGHT_CHEEK_TEMPLATE).astype(np.float32),
    }


def exclusion_hulls(landmarks: np.ndarray) -> list[np.ndarray]:
    """Convex hulls of the eyes, eyebrows and lips."""
    groups = (RIGHT_EYE, LEFT_EYE, RIGHT_EYEBROW, LEFT_EYEBROW, LIPS_OUTER)
    return [cv2.convexHull(landmarks[list(g)].astype(np.float32)).reshape(-1, 2) for g in groups]


def _invalid(name: str, reason: str, polygon: np.ndarray | None = None) -> Roi:
    polygon = np.empty((0, 2), np.float32) if polygon is None else polygon
    return Roi(name, polygon, (0, 0, 0, 0), np.zeros((0, 0), bool), 0, False, reason)


def _local_points(points: np.ndarray, x0: int, y0: int) -> np.ndarray:
    return np.round(points - (x0, y0)).astype(np.int32).reshape(-1, 1, 2)


def compute_rois(
    landmarks: np.ndarray | None,
    frame_shape: tuple[int, ...],
    config: RoiConfig | None = None,
) -> dict[str, Roi]:
    """Build forehead and cheek masks for one frame and judge their validity."""
    config = config or RoiConfig()
    if landmarks is None:
        return {name: _invalid(name, "no_face") for name in ROI_NAMES}
    if landmarks.shape != (NUM_LANDMARKS, 2):
        raise ValueError(f"Expected landmarks of shape ({NUM_LANDMARKS}, 2), got {landmarks.shape}")

    landmarks = landmarks.astype(np.float32)
    height, width = frame_shape[:2]
    frame = face_frame(landmarks)
    if frame.interocular < config.min_interocular_px:
        return {name: _invalid(name, "face_too_small") for name in ROI_NAMES}

    oval = landmarks[list(FACE_OVAL)]
    hulls = exclusion_hulls(landmarks)
    margin = max(1, int(round(config.exclusion_margin * frame.interocular)))

    rois = {}
    for name, polygon in roi_polygons(landmarks, config).items():
        x0, y0 = (int(v) for v in np.floor(polygon.min(axis=0)))
        x1, y1 = (int(v) + 1 for v in np.ceil(polygon.max(axis=0)))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            rois[name] = _invalid(name, "out_of_frame", polygon)
            continue

        shape = (y1 - y0, x1 - x0)
        roi_mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(roi_mask, [_local_points(polygon, x0, y0)], 1)
        oval_mask = np.zeros(shape, np.uint8)
        cv2.fillPoly(oval_mask, [_local_points(oval, x0, y0)], 1)
        excluded = np.zeros(shape, np.uint8)
        for hull in hulls:
            pts = _local_points(hull, x0, y0)
            cv2.fillPoly(excluded, [pts], 1)
            cv2.polylines(excluded, [pts], isClosed=True, color=1, thickness=2 * margin)

        mask = (roi_mask & oval_mask & (1 - excluded)).astype(bool)
        count = int(mask.sum())
        nominal_area = max(float(cv2.contourArea(polygon)), 1.0)

        reason = None
        if count < config.min_pixels:
            reason = "too_few_pixels"
        elif count / nominal_area < config.min_area_fraction:
            reason = "occluded_or_out_of_frame"
        rois[name] = Roi(name, polygon, (x0, y0, x1, y1), mask, count, reason is None, reason)
    return rois
