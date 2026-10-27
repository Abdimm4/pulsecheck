"""Scheduled expiry and deletion (PRD UP-4).

One sweep, safe to run repeatedly and concurrently with the API and worker:

1. unfinished uploads past their upload expiry -> expired (abandoned);
2. analyses processing longer than the job timeout plus a grace period
   (e.g. the worker died) -> failed (timed_out);
3. jobs past their expiry time -> expired (result, report and client hash dropped);
4. uploads of finished jobs whose deletion failed earlier -> deleted again;
5. stored files no unfinished job refers to, older than the upload expiry -> deleted;
6. data-free expired records older than `expired_record_hours` -> purged.

Run it with `python worker/entrypoint.py cleanup` (add `--loop SECONDS` to repeat).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import timedelta

from pulsecheck.jobs.models import (
    ACTIVE_STATUSES,
    TERMINAL_STATUSES,
    ExpiryReason,
    Job,
    JobStatus,
)
from pulsecheck.jobs.runtime import Runtime
from pulsecheck.jobs.service import JobService
from pulsecheck.jobs.tasks import TIMED_OUT

log = logging.getLogger("pulsecheck.cleanup")

S = JobStatus
PROCESSING_GRACE_S = 120


@dataclass
class CleanupSummary:
    abandoned: int = 0
    timed_out: int = 0
    expired: int = 0
    uploads_deleted: int = 0
    orphans_deleted: int = 0
    purged: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def run_cleanup(rt: Runtime) -> CleanupSummary:
    service = JobService(rt)
    store = rt.store
    now = rt.clock()
    retention = rt.settings.retention
    summary = CleanupSummary()

    for job in store.find(
        Job.status.in_([S.CREATED.value, S.UPLOADING.value]), Job.upload_expires_at <= now
    ):
        service.expire(job, ExpiryReason.ABANDONED)
        summary.abandoned += 1

    stale = now - timedelta(seconds=rt.settings.capacity.job_timeout_s + PROCESSING_GRACE_S)
    for job in store.find(Job.status == S.PROCESSING.value, Job.started_at <= stale):
        if service.fail(job.id, S.PROCESSING, TIMED_OUT, "The analysis did not finish in time."):
            summary.timed_out += 1

    for job in store.find(Job.status != S.EXPIRED.value, Job.expires_at <= now):
        service.expire(job, ExpiryReason.RETENTION_ELAPSED)
        summary.expired += 1

    for job in store.find(
        Job.status.in_([s.value for s in TERMINAL_STATUSES]),
        Job.upload_key.is_not(None),
        Job.upload_deleted_at.is_(None),
    ):
        service.delete_upload(job.id)
        if store.get(job.id).upload_deleted_at is not None:
            summary.uploads_deleted += 1

    in_use = {
        job.upload_key
        for job in store.find(Job.status.in_([s.value for s in ACTIVE_STATUSES]))
        if job.upload_key
    }
    cutoff = now - timedelta(hours=retention.abandoned_upload_hours)
    for key in list(rt.storage.stale_keys(cutoff)):
        if key not in in_use and rt.storage.delete(key):
            summary.orphans_deleted += 1

    purge_before = now - timedelta(hours=retention.expired_record_hours)
    old = store.find(Job.status == S.EXPIRED.value, Job.expired_at <= purge_before)
    summary.purged = store.purge(job.id for job in old)

    log.info("cleanup: %s", summary.to_dict())
    return summary
