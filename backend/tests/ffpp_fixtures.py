"""Tiny fake FaceForensics++ trees and synthetic feature tables for tests (no real data)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pulsecheck.dataset.table import STATUS_FAILED, STATUS_OK, append_rows, format_row
from pulsecheck.features import FEATURE_NAMES, FeatureVector

SPLIT_PAIRS = {
    "train": [["000", "001"], ["002", "003"], ["004", "005"]],
    "val": [["006", "007"]],
    "test": [["008", "009"]],
}


def make_ffpp_tree(
    root: Path,
    methods=("Deepfakes", "Face2Face"),
    compression="c23",
    split_pairs=SPLIT_PAIRS,
    reverse_fakes=(),
    write_splits=True,
) -> Path:
    """Stub .mp4 files in the official layout plus split JSONs; returns the splits dir.

    For every pair (a, b) both a_b and b_a fakes exist, as in FF++. Pairs in
    `reverse_fakes` only get b_a, to test reversed-pair lookup.
    """
    originals = root / "original_sequences" / "youtube" / compression / "videos"
    originals.mkdir(parents=True)
    for pairs in split_pairs.values():
        for a, b in pairs:
            for vid in (a, b):
                (originals / f"{vid}.mp4").write_bytes(b"stub")
    for method in methods:
        folder = root / "manipulated_sequences" / method / compression / "videos"
        folder.mkdir(parents=True)
        for pairs in split_pairs.values():
            for a, b in pairs:
                names = (
                    [f"{b}_{a}"]
                    if [a, b] in [list(p) for p in reverse_fakes]
                    else [
                        f"{a}_{b}",
                        f"{b}_{a}",
                    ]
                )
                for name in names:
                    (folder / f"{name}.mp4").write_bytes(b"stub")
    splits_dir = root / "splits"
    if write_splits:
        splits_dir.mkdir()
        for split, pairs in split_pairs.items():
            (splits_dir / f"{split}.json").write_text(json.dumps(pairs))
    return splits_dir


def _features(rng: np.random.Generator, fake: bool, missing_rate: float) -> FeatureVector:
    values: dict[str, float | None] = {}
    for name in FEATURE_NAMES:
        values[name] = float(rng.normal())
    # A separable signal for the classifiers, plus noise.
    shift = -1.0 if fake else 1.0
    corr = 0.5 + 0.3 * shift + 0.15 * rng.normal()
    values["mean_cross_correlation"] = float(np.clip(corr, -1, 1))
    values["max_freq_diff_hz"] = float(abs(0.3 - 0.2 * shift + 0.1 * rng.normal()))
    values["mean_window_freq_std_hz"] = float(abs(0.15 - 0.08 * shift + 0.05 * rng.normal()))
    values["signal_quality"] = float(rng.uniform(0.5, 1.0))
    values["usable_roi_count"] = 3.0
    values["face_coverage"] = 1.0
    missing = {}
    for name in FEATURE_NAMES:
        if name.startswith("forehead__") and rng.random() < missing_rate:
            values[name] = None
            missing[name] = "roi_unusable"
    # One feature that is always missing, to test dropping.
    values["right_cheek.method_diff_hz"] = None
    missing["right_cheek.method_diff_hz"] = "comparison_method_rejected"
    return FeatureVector(values, missing)


def make_feature_table(
    path: Path,
    pairs_per_split=None,
    methods=("Deepfakes", "Face2Face"),
    seed=0,
    missing_rate=0.2,
    failed_ids=(),
) -> Path:
    """A synthetic feature table with FF++-style groups (originals and both fakes per pair)."""
    pairs_per_split = pairs_per_split or {"train": 30, "val": 10, "test": 10}
    rng = np.random.default_rng(seed)
    rows = []
    counter = 0
    for split, n_pairs in pairs_per_split.items():
        for _ in range(n_pairs):
            a, b = f"{counter:03d}", f"{counter + 1:03d}"
            counter += 2
            group = f"pair:{a}_{b}"
            entries = [
                dict(
                    video_id=f"original_{v}",
                    label="real",
                    manipulation_type="none",
                    subject_id=v,
                    original_video_id=v,
                    source_video_id="",
                )
                for v in (a, b)
            ]
            for method in methods:
                for t, s in ((a, b), (b, a)):
                    entries.append(
                        dict(
                            video_id=f"{method}_{t}_{s}",
                            label="fake",
                            manipulation_type=method,
                            subject_id=s,
                            original_video_id=t,
                            source_video_id=s,
                        )
                    )
            for meta in entries:
                meta.update(split=split, group_id=group, source_dataset="FaceForensics++")
                if meta["video_id"] in failed_ids:
                    status = {
                        "status": STATUS_FAILED,
                        "failure_reason": "video_decode_error",
                        "runtime_s": 0.01,
                    }
                    rows.append(format_row(meta, status, None))
                    continue
                feats = _features(rng, meta["label"] == "fake", missing_rate)
                status = {
                    "status": STATUS_OK,
                    "runtime_s": float(rng.uniform(1, 3)),
                    "n_frames": 300,
                    "duration_s": 10.0,
                    "fps": 30.0,
                    "feature_schema_version": 1,
                    "pulse_method": "chrom",
                }
                rows.append(format_row(meta, status, feats))
    append_rows(path, rows)
    sidecar = {
        "manifest_sha256": "0" * 64,
        "dataset": {
            "dataset": "FaceForensics++",
            "compression": "c23",
            "methods": list(methods),
            "split_source": "synthetic",
            "license_notes": "synthetic test data",
            "non_commercial_use_only": True,
        },
        "feature_schema_version": 1,
        "pipeline": {"pulse_method": "chrom"},
    }
    path.with_name(path.name + ".json").write_text(json.dumps(sidecar))
    return path
