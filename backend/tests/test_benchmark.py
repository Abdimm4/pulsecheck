"""Tests for the pulse-injection benchmark (PRD FR-12) and the UBFC-rPPG loader.

The tiny end-to-end benchmark injects a known pulse into a few seconds of the
synthetic face (backend/tests/synthetic_face.py), optionally re-encodes it
with libx264, and checks that the whole-face POS variant recovers the heart
rate while the null condition (a = 0) gives no estimate near it.
"""

import math

import numpy as np
import pytest
from synthetic_face import FRAME_SHAPE, render_face, synthetic_landmarks

from pulsecheck.benchmark.injection import (
    BVP_SIGNATURE_RGB,
    PulseSpec,
    bvp_waveform,
    choose_target_hr,
    fill_missing,
    inject,
    skin_weight_map,
)
from pulsecheck.benchmark.metrics import GroupBootstrap, roc_auc
from pulsecheck.benchmark.ubfc import (
    find_subjects,
    parse_ground_truth,
    reference_hr,
    windows,
)
from pulsecheck.benchmark.variants import Variant, estimate, score_estimate
from pulsecheck.face.regions import FACE_OVAL, RoiConfig, exclusion_hulls
from pulsecheck.signals.rgb import frame_roi_means

SKIN_BGR = (110, 140, 190)


def test_bvp_waveform_has_unit_range_zero_mean_and_a_dicrotic_notch():
    phase = np.linspace(0, 1, 2000, endpoint=False)
    wave = bvp_waveform(phase)
    assert wave.mean() == pytest.approx(0.0, abs=1e-6)
    assert wave.max() - wave.min() == pytest.approx(1.0, abs=1e-3)
    peaks = np.flatnonzero((wave > np.roll(wave, 1)) & (wave > np.roll(wave, -1)))
    assert len(peaks) == 2  # systolic and diastolic peak, with the notch between them
    spectrum = np.abs(np.fft.rfft(np.tile(wave, 4)))
    assert spectrum[8] > 0.2 * spectrum[4]  # a real second harmonic


def test_pulse_phase_integrates_the_instantaneous_heart_rate():
    spec = PulseSpec(hr_bpm=72.0, variability_hz=0.1, variability_phase=0.3)
    t = np.linspace(0, 10, 10001)
    rate = np.gradient(spec.cycles(t), t) * 60
    np.testing.assert_allclose(rate, spec.instantaneous_hr(t), atol=0.01)
    assert abs(spec.mean_hr(t) - 72.0) <= 3.0
    p = spec.intensity(t)
    assert p.max() - p.min() == pytest.approx(1.0, abs=0.01)


def test_target_hr_keeps_its_distance_from_the_native_rate():
    rng = np.random.default_rng(0)
    for _ in range(50):
        hr, separated = choose_target_hr([70.0, 75.0], rng)
        assert separated and 50 <= hr <= 120
        assert min(abs(hr - 70), abs(hr - 75)) >= 15
    hr, separated = choose_target_hr([60.0, 85.0, 110.0], rng, min_separation_bpm=40)
    assert not separated  # nothing is 40 bpm away from all three: farthest rate used
    assert hr in (50.0, 120.0) or 50 <= hr <= 120


def test_skin_weight_map_covers_skin_only_and_favours_cheeks_and_forehead():
    lm = synthetic_landmarks()
    weight = skin_weight_map(lm, FRAME_SHAPE)
    assert weight.shape == FRAME_SHAPE[:2]
    assert weight.min() >= 0 and weight.max() <= 1.0 + 1e-6
    assert weight[5, 5] == 0  # background
    hull = exclusion_hulls(lm)[0].mean(axis=0).astype(int)
    assert weight[hull[1], hull[0]] == 0  # inside an eye
    # Cheek centre (subject's right cheek, image left) vs nose bridge region.
    assert weight[285, 255] == pytest.approx(1.0, abs=0.05)
    assert 0.3 < weight[260, 320] < 0.8
    oval = lm[list(FACE_OVAL)]
    assert weight[int(oval[:, 1].max()) + 5, 320] == 0


def test_injection_is_unbiased_after_stochastic_rounding():
    rng = np.random.default_rng(1)
    image = np.full((200, 200, 3), 150, np.uint8)
    weight = np.ones((200, 200), np.float32)
    out = inject(image, weight, amplitude=0.001, p_t=0.5, rng=rng).astype(float)
    # A 0.05 % change (0.075 levels) would round away without dithering.
    expected = 150 * (1 + 0.0005 * np.array(BVP_SIGNATURE_RGB[::-1]))
    np.testing.assert_allclose(out.reshape(-1, 3).mean(axis=0), expected, atol=0.01)
    assert inject(image, weight, 0.0, 0.5, rng) is image


def test_fill_missing_uses_the_nearest_detection():
    a, b = np.zeros((2, 2)), np.ones((2, 2))
    filled = fill_missing([None, a, None, None, b])
    assert filled[0] is a and filled[2] is a and filled[3] is b


def _synthetic_clip(hr_bpm, amplitude, seconds=10.0, fps=30.0, seed=0):
    """Frames of the synthetic face with sensor noise and an injected pulse."""
    rng = np.random.default_rng(seed)
    lm = synthetic_landmarks()
    base = render_face(lm, SKIN_BGR).astype(np.float32)
    weight = skin_weight_map(lm, FRAME_SHAPE)
    spec = PulseSpec(hr_bpm=hr_bpm, variability_bpm=0.0)
    t = np.arange(int(seconds * fps)) / fps
    p = spec.intensity(t)
    frames = []
    for k in range(len(t)):
        noisy = np.clip(base + rng.normal(0, 3.0, base.shape), 0, 255).astype(np.uint8)
        frames.append(inject(noisy, weight, amplitude, float(p[k]), rng))
    return t, frames, lm


def _face_traces(frames, lm):
    config = RoiConfig(face_roi=True)
    rgb = [frame_roi_means(f, lm, config)[1]["face"] for f in frames]
    return np.array(rgb)


def test_tiny_benchmark_recovers_the_injected_rate_and_rejects_the_null():
    face = Variant("face-pos", method="pos", hr_source="face")
    t, frames, lm = _synthetic_clip(84.0, amplitude=0.005)
    est = estimate(face, t, {"s0/mean/face": _face_traces(frames, lm)})
    scored = score_estimate(est[0], 84.0)
    assert scored["detected"], scored
    assert scored["snr_true_db"] > 0

    t, frames, lm = _synthetic_clip(84.0, amplitude=0.0, seed=2)
    null = score_estimate(estimate(face, t, {"s0/mean/face": _face_traces(frames, lm)})[0], 84.0)
    assert not null["detected"]


def test_tiny_benchmark_survives_x264_round_trip(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    from pulsecheck.dataset.normalize import x264_writer
    from pulsecheck.video import iter_frames

    t, frames, lm = _synthetic_clip(66.0, amplitude=0.01, seconds=8.0)
    path = tmp_path / "clip.mp4"
    writer = x264_writer(path, (FRAME_SHAPE[1], FRAME_SHAPE[0]), 30.0, crf=23)
    for frame in frames:
        writer.send(np.ascontiguousarray(frame))
    writer.close()
    decoded = [f.image for f in iter_frames(path)]
    assert len(decoded) == len(frames)
    face = Variant("face-pos", method="pos", hr_source="face")
    est = estimate(face, t, {"s0/mean/face": _face_traces(decoded, lm)})
    assert score_estimate(est[0], 66.0)["detected"]


def test_fused_and_per_roi_variants_return_one_and_three_estimates():
    t, frames, lm = _synthetic_clip(78.0, amplitude=0.01, seconds=8.0)
    config = RoiConfig(face_roi=False)
    means = [frame_roi_means(f, lm, config)[1] for f in frames]
    traces = {f"s0/mean/{roi}": np.array([m[roi] for m in means]) for roi in means[0]}
    per_roi = estimate(Variant("chrom", method="chrom"), t, traces)
    fused = estimate(Variant("fused", method="pos", hr_source="fused"), t, traces)
    assert [e.roi for e in per_roi] == ["forehead", "left_cheek", "right_cheek"]
    assert len(fused) == 1 and fused[0].roi == "fused"
    assert score_estimate(fused[0], 78.0)["detected"]


def test_transfer_stats_recover_gain_and_correlation_of_the_injected_change():
    from pulsecheck.benchmark.run import transfer_stats
    from pulsecheck.face.regions import ROI_NAMES

    fps, n = 30.0, 300
    spec = PulseSpec(hr_bpm=72.0, variability_bpm=0.0)
    p = spec.intensity(np.arange(n) / fps)
    rng = np.random.default_rng(0)
    arrays = {}
    for deg in ("crf18", "crf23", "crf28", "crf35", "crf23_half"):
        base = 120.0 + rng.normal(0, 0.05, n)
        for amp in (0.0, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02):
            green = base * (1 + 0.5 * amp * p)  # the codec keeps half of the change
            rgb = np.stack([green, green, green], axis=1)
            for roi in ROI_NAMES:
                arrays[f"{amp:g}@{deg}|s0/mean/large:{roi}|rgb"] = rgb
    meta = {"pulse": spec.to_dict(), "source": {"fps": fps}}
    rows = transfer_stats(meta, arrays)
    assert len(rows) == 5 * 6
    for row in rows:
        assert row["gain"] == pytest.approx(0.5, rel=1e-3)
        assert row["r"] > 0.999
    assert transfer_stats({"pulse": None}, arrays) == []


def test_variant_signal_overrides_drop_defaults():
    v = Variant("x").with_signal("y", filter_order=4, welch_window_s=6.0)
    assert dict(v.signal) == {"welch_window_s": 6.0}
    assert v.signal_config().welch_window_s == 6.0
    with pytest.raises(ValueError):
        Variant("bad", hr_source="nose")


def test_group_bootstrap_summary_and_paired_delta():
    rows_a = [
        {"group": g, "detected": d, "error": e, "snr_true_db": 1.0}
        for g, d, e in [
            ("g1", True, 1.0),
            ("g1", False, 9.0),
            ("g2", True, -2.0),
            ("g3", False, None),
        ]
    ]
    boot = GroupBootstrap(["g1", "g2", "g3"], n_boot=200, seed=0)
    s = boot.summarize(rows_a)
    assert s["n"] == 4 and s["n_groups"] == 3
    assert s["detection_rate"] == pytest.approx(0.5)
    assert s["estimate_rate"] == pytest.approx(0.75)
    assert s["mae_bpm"] == pytest.approx(4.0)
    assert s["rmse_bpm"] == pytest.approx(math.sqrt((1 + 81 + 4) / 3))
    lo, hi = s["detection_rate_ci"]
    assert 0 <= lo <= 0.5 <= hi <= 1
    same = boot.paired_delta(rows_a, rows_a, "detection_rate")
    assert same["delta"] == 0 and same["ci"] == [0.0, 0.0]


def test_roc_auc_matches_sklearn_with_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(3)
    scores = np.round(rng.normal(size=200), 1)
    labels = rng.random(200) < 0.4
    assert roc_auc(scores, labels) == pytest.approx(roc_auc_score(labels, scores))
    assert roc_auc([1, 2], [True, True]) is None


# --------------------------------------------------------------------------- UBFC


def _write_gt(path, hr_bpm=72.0, seconds=30.0, fs=60.0):
    t = np.arange(0, seconds, 1 / fs)
    ppg = np.sin(2 * np.pi * hr_bpm / 60 * t) + 0.3 * np.sin(4 * np.pi * hr_bpm / 60 * t)
    hr = np.full_like(t, hr_bpm)
    lines = [" ".join(f"{v:.6e}" for v in row) for row in (ppg, hr, t)]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_ubfc_ground_truth_parser_and_reference_hr(tmp_path):
    gt_path = tmp_path / "subject1" / "ground_truth.txt"
    _write_gt(gt_path, hr_bpm=72.0)
    gt = parse_ground_truth(gt_path)
    assert len(gt.ppg) == len(gt.hr_bpm) == len(gt.t) == 1800
    assert gt.hr_bpm[0] == pytest.approx(72.0)
    assert reference_hr(gt, 0.0, 10.0) == pytest.approx(72.0, abs=1.0)
    assert reference_hr(gt, 100.0, 110.0) is None  # outside the recording
    assert windows(30.0) == [(0.0, 10.0), (5.0, 15.0), (10.0, 20.0), (15.0, 25.0), (20.0, 30.0)]


def test_ubfc_parser_rejects_malformed_files(tmp_path):
    bad = tmp_path / "gt.txt"
    bad.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="3 lines"):
        parse_ground_truth(bad)
    bad.write_text("1 2 3\n4 5\n0 1 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lengths"):
        parse_ground_truth(bad)
    bad.write_text("1 2 3\n4 5 6\n0 2 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="increasing"):
        parse_ground_truth(bad)


def test_ubfc_subjects_in_natural_order(tmp_path):
    for name in ("subject10", "subject2", "subject1", "notes"):
        folder = tmp_path / name
        folder.mkdir()
        if name != "notes":
            (folder / "vid.avi").write_bytes(b"")
            _write_gt(folder / "ground_truth.txt", seconds=1.0)
    (tmp_path / "subject3").mkdir()  # no files: skipped
    assert [s.subject for s in find_subjects(tmp_path)] == ["subject1", "subject2", "subject10"]
