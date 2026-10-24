import cv2
import numpy as np
import pytest
from synthetic_face import FRAME_SHAPE, render_face, synthetic_landmarks, transform

from pulsecheck.face import ROI_NAMES, RoiConfig, compute_rois, face_frame
from pulsecheck.face import regions as R
from pulsecheck.face.regions import exclusion_hulls

SKIN_BGR = (120, 150, 200)


def _full_mask(roi, shape=FRAME_SHAPE):
    mask = np.zeros(shape[:2], bool)
    x0, y0, x1, y1 = roi.bbox
    mask[y0:y1, x0:x1] = roi.mask
    return mask


def _poly_mask(points, shape=FRAME_SHAPE):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round(points).astype(np.int32).reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def test_landmark_groups_match_mediapipe():
    vision = pytest.importorskip("mediapipe.tasks.python.vision")
    conns = vision.FaceLandmarksConnections

    def points(connections):
        return {p for c in connections for p in (c.start, c.end)}

    assert set(R.FACE_OVAL) == points(conns.FACE_LANDMARKS_FACE_OVAL)
    assert set(R.RIGHT_EYE) == points(conns.FACE_LANDMARKS_RIGHT_EYE)
    assert set(R.LEFT_EYE) == points(conns.FACE_LANDMARKS_LEFT_EYE)
    assert set(R.RIGHT_EYEBROW) == points(conns.FACE_LANDMARKS_RIGHT_EYEBROW)
    assert set(R.LEFT_EYEBROW) == points(conns.FACE_LANDMARKS_LEFT_EYEBROW)
    assert set(R.LIPS_OUTER) <= points(conns.FACE_LANDMARKS_LIPS)
    assert set(R.BROW_TOPS) <= set(R.RIGHT_EYEBROW) | set(R.LEFT_EYEBROW)


def test_frontal_face_gives_three_valid_rois():
    rois = compute_rois(synthetic_landmarks(), FRAME_SHAPE)
    assert set(rois) == set(ROI_NAMES)
    for roi in rois.values():
        assert roi.valid, roi.reason
        assert roi.pixel_count > 500
        assert roi.mask.shape == (roi.bbox[3] - roi.bbox[1], roi.bbox[2] - roi.bbox[0])


def test_rois_exclude_eyes_brows_lips_and_background():
    landmarks = synthetic_landmarks()
    rois = compute_rois(landmarks, FRAME_SHAPE)
    excluded = np.zeros(FRAME_SHAPE[:2], bool)
    for hull in exclusion_hulls(landmarks):
        excluded |= _poly_mask(hull)
    inside_oval = _poly_mask(landmarks[list(R.FACE_OVAL)])
    for roi in rois.values():
        mask = _full_mask(roi)
        assert not (mask & excluded).any(), roi.name
        assert not (mask & ~inside_oval).any(), roi.name


def test_roi_pixels_are_pure_skin_on_rendered_face():
    landmarks = synthetic_landmarks()
    image = render_face(landmarks, SKIN_BGR)
    for roi in compute_rois(landmarks, image.shape).values():
        x0, y0, x1, y1 = roi.bbox
        pixels = image[y0:y1, x0:x1][roi.mask]
        assert (pixels == SKIN_BGR).all(), roi.name


def test_roi_layout_follows_subject_left_right_convention():
    rois = compute_rois(synthetic_landmarks(), FRAME_SHAPE)
    forehead, left, right = (rois[n].centroid() for n in ROI_NAMES)
    # Unmirrored video: the subject's left cheek is on the image right.
    assert left[0] > right[0]
    assert forehead[1] < 190 < left[1] and forehead[1] < right[1]
    assert left[1] == pytest.approx(right[1], abs=1)


def test_rois_follow_rotation_scale_and_translation():
    base = synthetic_landmarks()
    moved = transform(base, angle_deg=20, scale=0.7, shift=(15, -10))
    base_rois = compute_rois(base, FRAME_SHAPE)
    moved_rois = compute_rois(moved, FRAME_SHAPE)
    for name in ROI_NAMES:
        assert moved_rois[name].valid, moved_rois[name].reason
        expected = transform(np.array([base_rois[name].centroid()]), 20, 0.7, (15, -10))[0]
        assert moved_rois[name].centroid() == pytest.approx(tuple(expected), abs=2.0)
        area_ratio = moved_rois[name].pixel_count / base_rois[name].pixel_count
        assert area_ratio == pytest.approx(0.49, rel=0.15)


def test_no_face_marks_all_rois_invalid():
    rois = compute_rois(None, FRAME_SHAPE)
    assert all(not r.valid and r.reason == "no_face" and r.pixel_count == 0 for r in rois.values())
    assert all(r.centroid() is None for r in rois.values())


def test_tiny_face_is_invalid():
    tiny = transform(synthetic_landmarks(), scale=0.15)
    rois = compute_rois(tiny, FRAME_SHAPE)
    assert all(r.reason == "face_too_small" for r in rois.values())


def test_face_partly_out_of_frame_invalidates_clipped_rois():
    base = compute_rois(synthetic_landmarks(), FRAME_SHAPE)
    shifted = transform(synthetic_landmarks(), shift=(-300, 0))  # image-left side cut off
    rois = compute_rois(shifted, FRAME_SHAPE)
    assert rois["right_cheek"].reason == "out_of_frame"
    assert rois["forehead"].pixel_count < 0.8 * base["forehead"].pixel_count
    assert rois["forehead"].bbox[0] == 0
    assert rois["left_cheek"].valid

    much_further = transform(synthetic_landmarks(), shift=(-340, 0))
    assert compute_rois(much_further, FRAME_SHAPE)["forehead"].reason == (
        "occluded_or_out_of_frame"
    )


def test_turned_head_invalidates_hidden_cheek():
    # Approximate yaw: the image-left side of the face oval moves towards the nose.
    landmarks = synthetic_landmarks()
    oval_idx = np.array(R.FACE_OVAL)
    left_side = oval_idx[landmarks[oval_idx, 0] < 320]
    landmarks[left_side, 0] = 320 - (320 - landmarks[left_side, 0]) * 0.3
    rois = compute_rois(landmarks, FRAME_SHAPE)
    assert rois["right_cheek"].reason == "occluded_or_out_of_frame"
    assert rois["left_cheek"].valid


def test_config_thresholds_are_applied():
    strict = RoiConfig(min_pixels=10**6)
    rois = compute_rois(synthetic_landmarks(), FRAME_SHAPE, strict)
    assert all(r.reason == "too_few_pixels" for r in rois.values())


def test_wrong_landmark_shape_raises():
    with pytest.raises(ValueError, match="shape"):
        compute_rois(np.zeros((68, 2), np.float32), FRAME_SHAPE)


def test_face_frame_measures_interocular_distance():
    frame = face_frame(transform(synthetic_landmarks(), angle_deg=30, scale=2.0))
    assert frame.interocular == pytest.approx(200.0, rel=1e-3)
