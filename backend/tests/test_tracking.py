import json

import pytest
from synthetic_face import FRAME_SHAPE, synthetic_landmarks, transform

from pulsecheck.face import ROI_NAMES, TrackingMonitor, TrackingThresholds, compute_rois


def _feed(monitor, landmarks):
    rois = compute_rois(landmarks, FRAME_SHAPE)
    monitor.update(rois, landmarks is not None, 100.0 if landmarks is not None else None)


def test_steady_face_is_reliable():
    monitor = TrackingMonitor()
    for _ in range(30):
        _feed(monitor, synthetic_landmarks())
    report = monitor.report()
    assert report.total_frames == report.face_frames == 30
    assert report.face_coverage == 1.0
    assert report.all_rois_valid_fraction == 1.0
    assert report.reliable and not report.warnings
    for name in ROI_NAMES:
        assert report.rois[name].valid_fraction == 1.0
        assert report.rois[name].median_jitter == pytest.approx(0.0, abs=1e-6)


def test_missing_face_lowers_coverage_and_warns():
    monitor = TrackingMonitor()
    for i in range(20):
        _feed(monitor, synthetic_landmarks() if i % 2 == 0 else None)
    report = monitor.report()
    assert report.face_coverage == 0.5
    assert not report.reliable
    assert any("Face detected in only 50%" in w for w in report.warnings)
    assert report.rois["forehead"].invalid_reasons == {"no_face": 10}
    # Jitter needs consecutive valid frames, which never happen here.
    assert report.rois["forehead"].median_jitter is None


def test_jumpy_regions_are_flagged_unstable():
    monitor = TrackingMonitor(TrackingThresholds(max_median_jitter=0.05))
    for i in range(10):
        _feed(monitor, transform(synthetic_landmarks(), shift=(10 * (-1) ** i, 0)))
    report = monitor.report()
    assert report.rois["left_cheek"].median_jitter == pytest.approx(0.2, abs=0.02)
    assert any("unstable" in w for w in report.warnings)
    assert report.reliable  # jitter is reported as a warning, not a hard failure


def test_empty_video_is_not_reliable():
    report = TrackingMonitor().report()
    assert report.total_frames == 0
    assert not report.reliable
    assert report.warnings == ["No frames were processed"]


def test_report_is_json_serialisable():
    monitor = TrackingMonitor()
    _feed(monitor, synthetic_landmarks())
    _feed(monitor, None)
    data = json.loads(json.dumps(monitor.report().to_dict()))
    assert data["rois"]["forehead"]["valid_frames"] == 1
