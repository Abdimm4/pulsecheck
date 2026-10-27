"""The queued analysis task, run by the RQ worker (worker/entrypoint.py).

queued -> processing, run the pipeline on the stored upload, store the
result and FR-10 report, -> completed (or failed with a reason code), then
delete the upload. If the job was deleted or expired while it ran, the
result is discarded. Logs carry job IDs, never tokens.
"""

from __future__ import annotations

import logging

from rq.timeouts import JobTimeoutException

from pulsecheck.jobs.models import JobStatus
from pulsecheck.jobs.pipeline import ANALYSIS_UNAVAILABLE, AnalysisOutput, AnalysisUnavailable
from pulsecheck.jobs.runtime import Runtime
from pulsecheck.jobs.service import JobService
from pulsecheck.jobs.settings import Settings
from pulsecheck.jobs.store import ConcurrentUpdate
from pulsecheck.jobs.validation import UploadRejected
from pulsecheck.reporting import build_report

log = logging.getLogger("pulsecheck.worker")

S = JobStatus
TIMED_OUT = "timed_out"
UPLOAD_MISSING = "upload_missing"
PIPELINE_ERROR = "pipeline_error"

_runtime: Runtime | None = None


def set_runtime(runtime: Runtime | None) -> None:
    """Use this runtime for tasks in this process (worker startup, tests)."""
    global _runtime
    _runtime = runtime


def get_runtime() -> Runtime:
    global _runtime
    if _runtime is None:
        _runtime = Runtime.from_settings(Settings.load())
    return _runtime


def retention_policy(runtime: Runtime) -> str:
    r = runtime.settings.retention
    return (
        "The uploaded video is deleted as soon as analysis finishes or fails. Uploads not "
        f"submitted for analysis are deleted after {r.abandoned_upload_hours:g} h. Results, "
        f"reports and job metadata expire {r.job_metadata_hours:g} h after the job was created."
    )


def result_summary(output: AnalysisOutput) -> dict:
    """The part of the result shown in job status responses."""
    values = output.features.values if output.features else {}
    result = output.result
    return {
        "state": result["state"],
        "label": result["label"],
        "score": result.get("score"),
        "score_semantics": result.get("score_semantics"),
        "calibrated": bool(result.get("calibrated", False)),
        "reasons": list(result.get("reasons", [])),
        "signal_quality": values.get("signal_quality"),
        "face_coverage": values.get("face_coverage"),
        "usable_roi_count": values.get("usable_roi_count"),
        "duration_s": output.video.duration_s,
        "decoded_frame_count": output.video.decoded_frame_count,
    }


def analyze_job(job_id: str) -> str:
    """Run one analysis; returns the final status (or 'skipped')."""
    rt = get_runtime()
    service = JobService(rt)
    job = rt.store.get(job_id)
    if job is None or job.job_status != S.QUEUED:
        return "skipped"
    try:
        job = rt.store.transition(job_id, S.QUEUED, S.PROCESSING, started_at=rt.clock())
    except ConcurrentUpdate:
        return "skipped"
    log.info("job %s: processing", job_id)

    analyzer = rt.analyzer
    try:
        with rt.storage.local_copy(job.upload_key) as path:
            output = analyzer(path)
    except UploadRejected as exc:
        service.fail(job_id, S.PROCESSING, exc.code, exc.message)
        return S.FAILED.value
    except AnalysisUnavailable as exc:
        log.error("job %s: analysis unavailable: %s", job_id, exc)
        service.fail(
            job_id, S.PROCESSING, ANALYSIS_UNAVAILABLE, "Analysis is temporarily unavailable."
        )
        return S.FAILED.value
    except JobTimeoutException:
        log.warning("job %s: timed out", job_id)
        service.fail(job_id, S.PROCESSING, TIMED_OUT, "The analysis took too long and was stopped.")
        return S.FAILED.value
    except FileNotFoundError:
        service.fail(job_id, S.PROCESSING, UPLOAD_MISSING, "The uploaded video was not found.")
        return S.FAILED.value
    except Exception:  # noqa: BLE001 - recorded as a failure; details only in server logs
        log.exception("job %s: pipeline error", job_id)
        service.fail(job_id, S.PROCESSING, PIPELINE_ERROR, "The analysis failed unexpectedly.")
        return S.FAILED.value

    finished = rt.clock()
    report = build_report(
        job_id=job_id,
        created_at=job.created_at,
        completed_at=finished,
        expires_at=job.expires_at,
        container=job.container,
        output=output,
        model_name=analyzer.model_name,
        model_version=analyzer.model_version,
        upload_limits=rt.settings.upload.to_public(),
        retention_policy=retention_policy(rt),
    )
    try:
        rt.store.transition(
            job_id,
            S.PROCESSING,
            S.COMPLETED,
            finished_at=finished,
            result=result_summary(output),
            report=report,
            model_name=analyzer.model_name,
            model_version=analyzer.model_version,
        )
        status = S.COMPLETED.value
    except ConcurrentUpdate:
        status = "discarded"  # deleted or expired while processing
    service.delete_upload(job_id)
    log.info("job %s: %s", job_id, status)
    return status
