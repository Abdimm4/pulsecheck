import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
from ffpp_fixtures import SPLIT_PAIRS, make_ffpp_tree

from pulsecheck.config import REPO_ROOT
from pulsecheck.dataset import (
    MANIFEST_FIELDS,
    UNASSIGNED,
    LeakageError,
    check_leakage,
    read_manifest,
    read_sidecar,
    write_manifest,
)
from pulsecheck.dataset.ffpp import (
    LICENSE_NOTES,
    ManifestOptions,
    build_manifest,
    load_official_splits,
)

PRD_FIELDS = (
    "video_id",
    "path",
    "label",
    "source_dataset",
    "subject_id",
    "manipulation_type",
    "original_video_id",
    "split",
    "license_notes",
)


def _by_id(rows):
    return {r.video_id: r for r in rows}


def test_manifest_fields_and_parsing(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    result = build_manifest(tmp_path / "ffpp", ManifestOptions(splits_dir=splits))
    rows = _by_id(result.rows)
    assert set(PRD_FIELDS) <= set(MANIFEST_FIELDS)
    # 10 originals + 2 methods x 5 pairs x 2 directions.
    assert len(rows) == 10 + 2 * 5 * 2
    real = rows["original_000"]
    assert (real.label, real.manipulation_type, real.original_video_id) == ("real", "none", "000")
    assert real.path == "original_sequences/youtube/c23/videos/000.mp4"
    swap = rows["Deepfakes_000_001"]
    assert (swap.label, swap.original_video_id, swap.source_video_id) == ("fake", "000", "001")
    assert swap.subject_id == "001"  # face swap shows the source identity
    assert rows["Face2Face_000_001"].subject_id == "000"  # reenactment keeps the target face
    assert all(r.source_dataset == "FaceForensics++" for r in rows.values())
    assert all(r.license_notes == LICENSE_NOTES for r in rows.values())
    assert "non-commercial" in LICENSE_NOTES
    assert all(r.compression == "c23" for r in rows.values())
    assert result.meta["non_commercial_use_only"] is True


def test_official_splits_keep_pairs_together(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    rows = build_manifest(tmp_path / "ffpp", ManifestOptions(splits_dir=splits)).rows
    expected = {vid: split for split, pairs in SPLIT_PAIRS.items() for p in pairs for vid in p}
    for row in rows:
        assert row.split == expected[row.original_video_id]
        if row.source_video_id:
            assert row.split == expected[row.source_video_id]
    assert Counter(r.split for r in rows if r.label == "real") == {"train": 6, "val": 2, "test": 2}
    check_leakage(rows)


def test_reversed_pair_is_assigned_to_the_pair_split(tmp_path):
    # Only 009_008 exists for the test pair ["008", "009"].
    splits = make_ffpp_tree(tmp_path / "ffpp", reverse_fakes=[["008", "009"]])
    rows = _by_id(build_manifest(tmp_path / "ffpp", ManifestOptions(splits_dir=splits)).rows)
    assert "Deepfakes_008_009" not in rows
    reversed_fake = rows["Deepfakes_009_008"]
    assert reversed_fake.split == "test"
    assert reversed_fake.group_id == "pair:008_009"
    assert rows["original_008"].group_id == "pair:008_009"


def test_overlapping_split_files_fail_loudly(tmp_path):
    bad = {**SPLIT_PAIRS, "test": [["008", "009"], ["000", "010"]]}  # 000 also in train
    splits = make_ffpp_tree(tmp_path / "ffpp", split_pairs=bad)
    with pytest.raises(LeakageError, match="000"):
        load_official_splits(splits)


def test_fake_linking_two_splits_fails_loudly(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    # A manipulation mixing a train original with a test original.
    folder = tmp_path / "ffpp" / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    (folder / "000_008.mp4").write_bytes(b"stub")
    with pytest.raises(LeakageError, match="spans splits"):
        build_manifest(tmp_path / "ffpp", ManifestOptions(splits_dir=splits))


def test_check_leakage_detects_group_and_duplicate(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    rows = build_manifest(tmp_path / "ffpp", ManifestOptions(splits_dir=splits)).rows
    moved = [r if r.video_id != "Face2Face_001_000" else _with(r, split="val") for r in rows]
    with pytest.raises(LeakageError, match="spans splits"):
        check_leakage(moved)
    dup = [*rows, _with(rows[0], split="test" if rows[0].split != "test" else "val")]
    with pytest.raises(LeakageError, match="appears in splits"):
        check_leakage(dup)


def _with(row, **kw):
    from dataclasses import replace

    return replace(row, **kw)


def test_hash_split_without_split_files_is_group_aware(tmp_path):
    make_ffpp_tree(tmp_path / "ffpp", write_splits=False)
    a = build_manifest(tmp_path / "ffpp", ManifestOptions(seed=3)).rows
    b = build_manifest(tmp_path / "ffpp", ManifestOptions(seed=3)).rows
    assert [(r.video_id, r.split) for r in a] == [(r.video_id, r.split) for r in b]
    assert all(r.split != UNASSIGNED for r in a)
    check_leakage(a)
    by_pair = {}
    for r in a:
        key = frozenset((r.original_video_id, r.source_video_id or r.original_video_id))
        by_pair.setdefault(key, set()).add(r.split)
    assert all(len(s) == 1 for s in by_pair.values())


def test_videos_missing_from_split_files_are_unassigned(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    originals = tmp_path / "ffpp" / "original_sequences" / "youtube" / "c23" / "videos"
    (originals / "500.mp4").write_bytes(b"stub")
    (originals / "notes.mp4").write_bytes(b"stub")
    result = build_manifest(tmp_path / "ffpp", ManifestOptions(splits_dir=splits))
    assert _by_id(result.rows)["original_500"].split == UNASSIGNED
    assert any("notes.mp4" in w for w in result.warnings)
    assert any("original_500" in w for w in result.warnings)


def test_limit_and_balance(tmp_path):
    methods = ("Deepfakes", "Face2Face", "FaceSwap")
    splits = make_ffpp_tree(tmp_path / "ffpp", methods=methods)
    opts = ManifestOptions(methods=methods, splits_dir=splits, balance=True, seed=1)
    rows = build_manifest(tmp_path / "ffpp", opts).rows
    for split in ("train", "val", "test"):
        counts = Counter(r.label for r in rows if r.split == split)
        assert counts["fake"] == counts["real"]
    train_fakes = Counter(
        r.manipulation_type for r in rows if r.split == "train" and r.label == "fake"
    )
    assert max(train_fakes.values()) - min(train_fakes.values()) <= 1  # spread across methods

    opts = ManifestOptions(methods=methods, splits_dir=splits, limit_per_class=1, seed=1)
    rows = build_manifest(tmp_path / "ffpp", opts).rows
    assert Counter((r.split, r.label) for r in rows) == {
        (s, label): 1 for s in ("train", "val", "test") for label in ("real", "fake")
    }
    check_leakage(rows)


def test_methods_filter_and_dfd_flag(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    dfd_real = tmp_path / "ffpp" / "original_sequences" / "actors" / "c23" / "videos"
    dfd_fake = tmp_path / "ffpp" / "manipulated_sequences" / "DeepFakeDetection" / "c23" / "videos"
    dfd_real.mkdir(parents=True)
    dfd_fake.mkdir(parents=True)
    (dfd_real / "01__walking_outside.mp4").write_bytes(b"stub")
    (dfd_real / "02__walking_outside.mp4").write_bytes(b"stub")
    (dfd_fake / "01_02__walking_outside__ABC123XY.mp4").write_bytes(b"stub")

    rows = build_manifest(
        tmp_path / "ffpp", ManifestOptions(methods=("Face2Face",), splits_dir=splits)
    ).rows
    assert {r.manipulation_type for r in rows} == {"none", "Face2Face"}  # DFD skipped

    rows = build_manifest(
        tmp_path / "ffpp", ManifestOptions(splits_dir=splits, include_dfd=True)
    ).rows
    dfd = [r for r in rows if "dfd" in r.video_id or r.manipulation_type == "DeepFakeDetection"]
    assert len(dfd) == 3
    assert len({r.split for r in dfd}) == 1  # actors 01 and 02 are linked by the fake
    check_leakage(rows)


def test_write_and_read_round_trip(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    result = build_manifest(tmp_path / "ffpp", ManifestOptions(splits_dir=splits))
    out = tmp_path / "manifest.csv"
    write_manifest(result.rows, out, result.meta)
    assert read_manifest(out) == result.rows
    meta = read_sidecar(out)
    assert meta["license_notes"] == LICENSE_NOTES
    assert meta["counts"]["train"]["real"] == 6
    assert len(meta["manifest_sha256"]) == 64


def test_cli_builds_manifest_and_exits_2_on_leakage(tmp_path):
    splits = make_ffpp_tree(tmp_path / "ffpp")
    script = REPO_ROOT / "scripts" / "build_manifest.py"
    out = tmp_path / "m.csv"
    cmd = [sys.executable, str(script), str(tmp_path / "ffpp"), "--splits", str(splits)]
    done = subprocess.run([*cmd, "--out", str(out)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert len(read_manifest(out)) == 30
    assert json.loads(Path(str(out) + ".json").read_text())["compression"] == "c23"

    folder = tmp_path / "ffpp" / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
    (folder / "000_008.mp4").write_bytes(b"stub")
    done = subprocess.run([*cmd, "--out", str(tmp_path / "m2.csv")], capture_output=True, text=True)
    assert done.returncode == 2
    assert "leakage" in done.stderr
