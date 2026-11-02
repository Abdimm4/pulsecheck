import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from synthetic_face import render_face, synthetic_landmarks

from pulsecheck.face import ROI_NAMES
from pulsecheck.signals import extract_rgb_traces
from pulsecheck.video import Frame, iter_frames


class FakeDetector:
    """Returns the synthetic landmarks, or None for frame indices in `missing`."""

    def __init__(self, missing=()):
        self.missing = set(missing)
        self.calls = []

    def detect(self, image_bgr, timestamp_s):
        self.calls.append(timestamp_s)
        return None if len(self.calls) - 1 in self.missing else synthetic_landmarks()


def _skin_bgr(i):
    # Each channel varies differently so a BGR/RGB mix-up is caught.
    return (100 + i, 150 + 2 * (i % 5), 200 - i)


def _frames(n, fps=30.0):
    landmarks = synthetic_landmarks()
    return [Frame(i, i / fps, render_face(landmarks, _skin_bgr(i))) for i in range(n)]


def test_traces_hold_per_roi_rgb_means_and_timestamps():
    frames = _frames(12)
    traces = extract_rgb_traces(frames, FakeDetector())
    np.testing.assert_allclose(traces.timestamps_s, [f.timestamp_s for f in frames])
    np.testing.assert_array_equal(traces.frame_indices, np.arange(12))
    expected_rgb = np.array([_skin_bgr(i)[::-1] for i in range(12)], dtype=float)
    for name in ROI_NAMES:
        trace = traces.rois[name]
        assert trace.rgb.shape == (12, 3)
        assert trace.valid.all()
        assert (trace.pixel_count > 500).all()
        np.testing.assert_allclose(trace.rgb, expected_rgb)
        np.testing.assert_allclose(trace.green, expected_rgb[:, 1])
    assert traces.tracking.reliable
    assert traces.settings["color_order"] == "RGB"


def test_frames_without_a_face_are_nan_and_invalid():
    traces = extract_rgb_traces(_frames(10), FakeDetector(missing={2, 3, 4}))
    for name in ROI_NAMES:
        trace = traces.rois[name]
        assert trace.valid.tolist() == [i not in {2, 3, 4} for i in range(10)]
        assert np.isnan(trace.rgb[2:5]).all()
        assert not np.isnan(trace.rgb[trace.valid]).any()
        assert (trace.pixel_count[2:5] == 0).all()
    assert traces.tracking.face_coverage == pytest.approx(0.7)
    assert traces.tracking.rois["forehead"].invalid_reasons == {"no_face": 3}
    assert not traces.tracking.reliable


def test_saturated_skin_is_marked_invalid():
    frames = [
        Frame(i, i / 30, render_face(synthetic_landmarks(), (255, 255, 255))) for i in range(3)
    ]
    traces = extract_rgb_traces(frames, FakeDetector())
    for name in ROI_NAMES:
        assert not traces.rois[name].valid.any()
        assert np.isnan(traces.rois[name].rgb).all()
        assert traces.tracking.rois[name].invalid_reasons == {"clipped_exposure": 3}


def test_detector_receives_frame_timestamps():
    image = render_face(synthetic_landmarks(), (90, 120, 160))
    frames = [Frame(i, t, image) for i, t in enumerate([0.0, 0.031, 0.07, 0.1])]
    detector = FakeDetector()
    traces = extract_rgb_traces(frames, detector)
    assert detector.calls == [0.0, 0.031, 0.07, 0.1]
    np.testing.assert_allclose(traces.timestamps_s, detector.calls)


def test_empty_input_gives_empty_traces():
    traces = extract_rgb_traces([], FakeDetector())
    assert traces.timestamps_s.shape == (0,)
    assert traces.rois["forehead"].rgb.shape == (0, 3)
    assert not traces.tracking.reliable


def test_to_dict_is_json_and_holds_no_geometry():
    traces = extract_rgb_traces(_frames(4), FakeDetector(missing={1}))
    data = json.loads(json.dumps(traces.to_dict()))
    assert data["rois"]["forehead"]["rgb"][1] == [None, None, None]
    assert data["rois"]["forehead"]["rgb"][0] == [200.0, 150.0, 100.0]
    # PRD UP-7: only colour statistics and validity leave the extractor.
    assert set(data["rois"]["forehead"]) == {"rgb", "valid", "pixel_count"}
    assert "landmarks" not in json.dumps(data)


def test_works_end_to_end_with_iter_frames(tmp_path: Path):
    path = tmp_path / "face.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (640, 480))
    assert writer.isOpened()
    landmarks = synthetic_landmarks()
    for i in range(30):
        writer.write(render_face(landmarks, _skin_bgr(i)))
    writer.release()

    frames = list(iter_frames(path))
    traces = extract_rgb_traces(iter_frames(path), FakeDetector())
    np.testing.assert_allclose(traces.timestamps_s, [f.timestamp_s for f in frames])
    assert traces.timestamps_s[-1] == pytest.approx(29 / 30, abs=1e-3)
    expected = np.array([_skin_bgr(i)[::-1] for i in range(30)], dtype=float)
    for name in ROI_NAMES:
        assert traces.rois[name].valid.all()
        # Lossy compression shifts flat colours slightly.
        np.testing.assert_allclose(traces.rois[name].rgb, expected, atol=4)


def test_landmark_smoothing_stream_matches_offline_smoothing():
    from pulsecheck.signals.rgb import smooth_landmarks, smoothed_landmark_stream

    rng = np.random.default_rng(0)
    base = synthetic_landmarks()
    marks = [base + rng.normal(0, 2, base.shape).astype(np.float32) for _ in range(20)]
    marks[3] = marks[11] = None

    class Replay:
        def __init__(self):
            self.i = -1

        def detect(self, image, timestamp_s):
            self.i += 1
            return marks[self.i]

    frames = [Frame(i, i / 30, np.zeros((4, 4, 3), np.uint8)) for i in range(20)]
    for window in (0, 5, 7):
        streamed = list(smoothed_landmark_stream(frames, Replay(), window))
        offline = smooth_landmarks(marks, window)
        assert [f.index for f, _ in streamed] == list(range(20))
        for (_, got), want in zip(streamed, offline, strict=True):
            if want is None:
                assert got is None
            else:
                np.testing.assert_allclose(got, want, atol=1e-5)
    smoothed = smooth_landmarks(marks, 7)
    jitter = np.mean([np.abs(a - base).mean() for a in marks if a is not None])
    assert np.mean([np.abs(a - base).mean() for a in smoothed if a is not None]) < jitter


def test_robust_statistic_ignores_outlier_pixels():
    from pulsecheck.face import RoiConfig, compute_rois
    from pulsecheck.signals.rgb import roi_mean_rgb

    landmarks = synthetic_landmarks()
    image = render_face(landmarks, (100, 150, 200))
    roi = compute_rois(landmarks, image.shape, RoiConfig())["forehead"]
    x0, y0, x1, y1 = roi.bbox
    ys, xs = np.nonzero(roi.mask)
    image[y0 + ys[:40], x0 + xs[:40]] = (20, 20, 20)  # e.g. stray hair
    plain, _ = roi_mean_rgb(image, roi, "mean")
    robust, _ = roi_mean_rgb(image, roi, "robust")
    np.testing.assert_allclose(robust, (200, 150, 100))
    assert plain[0] < 199


def test_color_config_builds_face_roi_traces():
    from pulsecheck.signals import ColorSignalConfig

    config = ColorSignalConfig.from_mapping(
        {"face_roi": True, "roi_layout": "large", "landmark_smoothing_frames": 5}
    )
    traces = extract_rgb_traces(_frames(12), FakeDetector(), **config.extract_kwargs())
    assert set(traces.rois) == {*ROI_NAMES, "face"}
    assert traces.rois["face"].valid.all()
    assert "face" in traces.tracking.rois
    assert traces.settings["landmark_smoothing_frames"] == 5
    with pytest.raises(ValueError):
        ColorSignalConfig.from_mapping({"statistic": "median"})
    with pytest.raises(ValueError):
        ColorSignalConfig.from_mapping({"face": True})
