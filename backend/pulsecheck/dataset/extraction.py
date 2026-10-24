"""Batch feature extraction over a manifest (PRD FR-11).

Runs the per-video pipeline (decode -> landmarks -> ROI traces -> pulse
signals -> features) for every manifest row and appends one row per video to
the feature table as soon as it finishes, so an interrupted run resumes where
it stopped. Videos that fail are recorded with a reason code. A missing
landmarker model would fail every video, so it aborts the run instead.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
import traceback
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from itertools import chain
from pathlib import Path

import numpy as np

from pulsecheck.dataset.manifest import ManifestRow
from pulsecheck.dataset.table import (
    STATUS_FAILED,
    STATUS_OK,
    append_rows,
    check_header,
    format_row,
    read_table,
)
from pulsecheck.features import FEATURE_SCHEMA_VERSION, FeatureVector, QualityConfig, analyze_traces
from pulsecheck.signals import SignalConfig

log = logging.getLogger("pulsecheck.extraction")

# Failure reason codes recorded in the feature table.
FILE_NOT_FOUND = "file_not_found"
VIDEO_DECODE_ERROR = "video_decode_error"
NO_FRAMES = "no_frames"
PIPELINE_ERROR = "pipeline_error"


class ExtractionFailure(Exception):
    """A per-video failure with a reason code; recorded, the run continues."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


class FatalExtractionError(Exception):
    """A failure that would hit every video (e.g. no landmarker model); aborts the run."""


@dataclass
class VideoResult:
    features: FeatureVector
    n_frames: int
    duration_s: float | None
    fps: float | None


Analyzer = Callable[[Path], VideoResult]


@dataclass
class PipelineAnalyzer:
    """The production pipeline as a picklable callable (one landmarker per video)."""

    model_path: str | None = None
    max_frames: int | None = None
    signal: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    pulse_method: str = "chrom"

    def settings(self) -> dict:
        return {
            "signal": SignalConfig.from_mapping(self.signal).to_dict(),
            "quality": {
                k: list(v) for k, v in asdict(QualityConfig.from_mapping(self.quality)).items()
            },
            "pulse_method": self.pulse_method,
            "max_frames": self.max_frames,
        }

    def __call__(self, path: Path) -> VideoResult:
        from pulsecheck.face import FaceLandmarkerDetector, LandmarkerUnavailableError
        from pulsecheck.signals import extract_rgb_traces
        from pulsecheck.video import VideoDecodeError, iter_frames

        frames = iter_frames(path, max_frames=self.max_frames)
        try:
            # Open and decode the first frame before loading the landmarker, so
            # an unreadable file fails fast with a decode error.
            first = next(frames)
        except StopIteration:
            raise ExtractionFailure(NO_FRAMES, "no frame could be decoded") from None
        except VideoDecodeError as exc:
            raise ExtractionFailure(VIDEO_DECODE_ERROR, str(exc)) from None
        try:
            with FaceLandmarkerDetector(self.model_path) as detector:
                traces = extract_rgb_traces(chain([first], frames), detector)
        except LandmarkerUnavailableError as exc:
            raise FatalExtractionError(str(exc)) from exc
        except VideoDecodeError as exc:
            raise ExtractionFailure(VIDEO_DECODE_ERROR, str(exc)) from None
        analysis = analyze_traces(
            traces,
            SignalConfig.from_mapping(self.signal),
            self.pulse_method,
            QualityConfig.from_mapping(self.quality),
        )
        t = traces.timestamps_s[np.isfinite(traces.timestamps_s)]
        duration = float(t[-1] - t[0]) if t.size >= 2 else None
        fps = (t.size - 1) / duration if duration else None
        return VideoResult(analysis.features, int(traces.timestamps_s.size), duration, fps)


def _meta(row: ManifestRow) -> dict:
    return asdict(row)


def process_one(task: tuple[ManifestRow, Path, Analyzer]) -> dict[str, str]:
    """Run one video and return its table row; FatalExtractionError propagates."""
    row, path, analyzer = task
    start = time.perf_counter()
    status: dict = {"feature_schema_version": FEATURE_SCHEMA_VERSION}
    features = None
    try:
        if not path.is_file():
            raise ExtractionFailure(FILE_NOT_FOUND, str(path))
        result = analyzer(path)
        features = result.features
        status.update(
            status=STATUS_OK,
            n_frames=result.n_frames,
            duration_s=result.duration_s,
            fps=result.fps,
            pulse_method=features.pulse_method,
        )
    except FatalExtractionError:
        raise
    except ExtractionFailure as exc:
        status.update(status=STATUS_FAILED, failure_reason=exc.code, failure_detail=exc.detail)
    except Exception as exc:  # noqa: BLE001 - recorded with its type, never dropped
        detail = f"{type(exc).__name__}: {exc}"
        log.debug("pipeline error on %s\n%s", row.video_id, traceback.format_exc())
        status.update(status=STATUS_FAILED, failure_reason=PIPELINE_ERROR, failure_detail=detail)
    status["runtime_s"] = round(time.perf_counter() - start, 3)
    if status.get("failure_detail"):
        status["failure_detail"] = str(status["failure_detail"])[:300]
    return format_row(_meta(row), status, features)


@dataclass
class ExtractionSummary:
    total: int
    skipped: int
    processed: int
    ok: int
    failed: int
    failure_reasons: dict[str, int]
    elapsed_s: float


def pending_rows(
    rows: Iterable[ManifestRow], output: Path, retry_failed: bool = False
) -> tuple[list[ManifestRow], int]:
    """Rows still to process given an existing output table, and how many are skipped."""
    rows = list(rows)
    if not output.is_file() or output.stat().st_size == 0:
        return rows, 0
    check_header(output)
    done = {
        r["video_id"]
        for r in read_table(output)
        if r["status"] == STATUS_OK or (r["status"] == STATUS_FAILED and not retry_failed)
    }
    todo = [r for r in rows if r.video_id not in done]
    return todo, len(rows) - len(todo)


def run_extraction(
    rows: Iterable[ManifestRow],
    root: str | Path,
    output: str | Path,
    analyzer: Analyzer,
    workers: int = 1,
    retry_failed: bool = False,
) -> ExtractionSummary:
    """Process every pending row, appending results to `output` as they finish."""
    rows = list(rows)
    root, output = Path(root), Path(output)
    todo, skipped = pending_rows(rows, output, retry_failed)
    if skipped:
        log.info("resuming: %d of %d videos already in %s", skipped, len(rows), output)
    tasks = [(row, root / row.path, analyzer) for row in todo]
    reasons: Counter[str] = Counter()
    ok = failed = 0
    start = time.perf_counter()

    def record(result: dict[str, str], index: int) -> None:
        nonlocal ok, failed
        append_rows(output, [result])
        if result["status"] == STATUS_OK:
            ok += 1
        else:
            failed += 1
            reasons[result["failure_reason"]] += 1
        elapsed = time.perf_counter() - start
        eta = elapsed / index * (len(tasks) - index)
        log.info(
            "[%d/%d] %s %s%s %.1fs (elapsed %.0fs, eta %.0fs)",
            index,
            len(tasks),
            result["video_id"],
            result["status"],
            f" ({result['failure_reason']})" if result["failure_reason"] else "",
            float(result["runtime_s"]),
            elapsed,
            eta,
        )

    if workers <= 1 or len(tasks) <= 1:
        for i, task in enumerate(tasks, 1):
            record(process_one(task), i)
    else:
        ctx = multiprocessing.get_context("spawn")  # same behaviour on every OS
        with ctx.Pool(workers) as pool:
            for i, result in enumerate(pool.imap_unordered(process_one, tasks), 1):
                record(result, i)

    return ExtractionSummary(
        total=len(rows),
        skipped=skipped,
        processed=ok + failed,
        ok=ok,
        failed=failed,
        failure_reasons=dict(reasons),
        elapsed_s=round(time.perf_counter() - start, 3),
    )
