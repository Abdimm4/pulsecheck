import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from batch_stubs import StubAnalyzer
from ffpp_fixtures import make_ffpp_tree

from pulsecheck.dataset.extraction import (
    FILE_NOT_FOUND,
    PIPELINE_ERROR,
    VIDEO_DECODE_ERROR,
    FatalExtractionError,
    PipelineAnalyzer,
    run_extraction,
)
from pulsecheck.dataset.ffpp import ManifestOptions, build_manifest
from pulsecheck.dataset.table import TABLE_COLUMNS, read_table, row_features
from pulsecheck.face import default_model_path
from pulsecheck.features import FEATURE_NAMES


@pytest.fixture
def manifest(tmp_path):
    root = tmp_path / "ffpp"
    splits = make_ffpp_tree(root, methods=("Deepfakes",))
    options = ManifestOptions(methods=("Deepfakes",), splits_dir=splits)
    return root, build_manifest(root, options).rows


def _calls(log_dir: Path) -> int:
    return sum(int(p.read_text()) for p in log_dir.iterdir())


def test_extraction_writes_one_row_per_video(tmp_path, manifest):
    root, rows = manifest
    out = tmp_path / "features.csv"
    summary = run_extraction(rows, root, out, StubAnalyzer())
    assert (summary.ok, summary.failed, summary.skipped) == (len(rows), 0, 0)
    table = read_table(out)
    assert [r["video_id"] for r in table] == [r.video_id for r in rows]
    first = table[0]
    assert tuple(first) == TABLE_COLUMNS
    assert first["status"] == "ok" and float(first["runtime_s"]) >= 0
    assert first["split"] == rows[0].split and first["group_id"] == rows[0].group_id
    # Missing features round-trip explicitly as empty cells plus a reason.
    assert first["forehead.window_freq_std_hz"] == ""
    fv = row_features(first)
    assert fv.values["forehead.window_freq_std_hz"] is None
    assert fv.missing["forehead.window_freq_std_hz"] == "fewer_than_3_windows"
    assert fv.values["usable_roi_count"] == 3.0
    # No geometry or frame columns.
    assert not any(c for c in first if "landmark" in c or "frame_" in c or c == "path")


def test_failures_are_recorded_not_dropped(tmp_path, manifest):
    root, rows = manifest
    (root / rows[0].path).unlink()
    out = tmp_path / "features.csv"
    analyzer = StubAnalyzer(fail=("001",), crash=("002",))
    summary = run_extraction(rows, root, out, analyzer)
    table = {r["video_id"]: r for r in read_table(out)}
    assert len(table) == len(rows)
    assert summary.failed == 3
    assert table["original_001"]["failure_reason"] == VIDEO_DECODE_ERROR
    assert table["original_002"]["failure_reason"] == PIPELINE_ERROR
    assert "RuntimeError" in table["original_002"]["failure_detail"]
    for row in table.values():
        if row["status"] == "failed":
            assert all(row[name] == "" for name in FEATURE_NAMES)
    assert summary.failure_reasons == {VIDEO_DECODE_ERROR: 1, PIPELINE_ERROR: 1, FILE_NOT_FOUND: 1}


def test_missing_file_is_file_not_found(tmp_path, manifest):
    root, rows = manifest
    (root / rows[0].path).unlink()
    out = tmp_path / "features.csv"
    run_extraction(rows[:1], root, out, PipelineAnalyzer())
    assert read_table(out)[0]["failure_reason"] == FILE_NOT_FOUND


def test_resume_skips_done_and_retries_failed_on_request(tmp_path, manifest):
    root, rows = manifest
    out = tmp_path / "features.csv"
    calls = tmp_path / "calls"
    calls.mkdir()
    # Interrupted run: only the first five videos were processed.
    run_extraction(rows[:5], root, out, StubAnalyzer(fail=("000",), log_dir=str(calls)))
    assert _calls(calls) == 5

    summary = run_extraction(rows, root, out, StubAnalyzer(log_dir=str(calls)))
    assert summary.skipped == 5
    assert summary.processed == len(rows) - 5
    assert _calls(calls) == len(rows)
    assert read_table(out)[0]["status"] == "failed"  # kept, not retried

    summary = run_extraction(rows, root, out, StubAnalyzer(), retry_failed=True)
    assert summary.processed == 1
    table = read_table(out)
    assert len(table) == len(rows)
    assert {r["status"] for r in table} == {"ok"}  # the retry row replaces the failure


def test_fatal_error_aborts_without_recording(tmp_path, manifest):
    root, rows = manifest
    out = tmp_path / "features.csv"
    with pytest.raises(FatalExtractionError):
        run_extraction(rows, root, out, StubAnalyzer(fatal=("002",)))
    done = {r["video_id"] for r in read_table(out)}
    assert "original_002" not in done  # will be processed on the next run


def test_multiple_workers(tmp_path, manifest):
    root, rows = manifest
    out = tmp_path / "features.csv"
    summary = run_extraction(rows, root, out, StubAnalyzer(fail=("003",)), workers=2)
    assert summary.processed == len(rows)
    table = read_table(out)
    assert sorted(r["video_id"] for r in table) == sorted(r.video_id for r in rows)
    assert sum(r["status"] == "failed" for r in table) == 1


def test_changed_table_layout_is_rejected(tmp_path, manifest):
    root, rows = manifest
    out = tmp_path / "features.csv"
    out.write_text("video_id,status\nx,ok\n")
    with pytest.raises(ValueError, match="column layout"):
        run_extraction(rows[:1], root, out, StubAnalyzer())


def test_pipeline_records_decode_error_for_stub_file(tmp_path, manifest):
    root, rows = manifest
    out = tmp_path / "features.csv"
    # The fixture files are 4-byte stubs, not videos.
    run_extraction(rows[:1], root, out, PipelineAnalyzer())
    row = read_table(out)[0]
    assert row["status"] == "failed"
    assert row["failure_reason"] in (VIDEO_DECODE_ERROR, "no_frames")


@pytest.mark.skipif(not default_model_path().is_file(), reason="landmarker model not downloaded")
def test_pipeline_runs_on_a_faceless_video(tmp_path, manifest):
    root, rows = manifest
    path = root / rows[0].path
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for i in range(30):
        writer.write(np.full((48, 64, 3), i, dtype=np.uint8))
    writer.release()
    out = tmp_path / "features.csv"
    run_extraction(rows[:1], root, out, PipelineAnalyzer())
    row = read_table(out)[0]
    assert row["status"] == "ok", row["failure_detail"]
    assert int(row["n_frames"]) == 30
    assert float(row["face_coverage"]) == 0.0
    assert json.loads(row["missing_json"])  # most features are missing, with reasons


@pytest.mark.skipif(not default_model_path().is_file(), reason="landmarker model not downloaded")
def test_extract_features_cli_resumes_and_writes_sidecar(tmp_path):
    import subprocess
    import sys

    from pulsecheck.config import REPO_ROOT
    from pulsecheck.dataset import write_manifest

    root = tmp_path / "ffpp"
    splits = make_ffpp_tree(root, methods=("Deepfakes",))
    result = build_manifest(root, ManifestOptions(methods=("Deepfakes",), splits_dir=splits))
    manifest = tmp_path / "m.csv"
    write_manifest(result.rows, manifest, result.meta)
    out = tmp_path / "features.csv"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "extract_features.py"),
        str(manifest),
        "--out",
        str(out),
        "--workers",
        "2",
    ]
    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    table = read_table(out)
    assert len(table) == len(result.rows)
    assert {r["status"] for r in table} == {"failed"}  # stub files are not videos
    side = json.loads((tmp_path / "features.csv.json").read_text())
    assert side["dataset"]["license_notes"] == result.meta["license_notes"]
    assert side["rows"] == {"failed": len(result.rows)}
    assert (tmp_path / "features.log").is_file()

    done = subprocess.run(cmd, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "already in" in done.stderr
    assert len(read_table(out)) == len(result.rows)
