"""Job lifecycle operations shared by the API, the worker and cleanup.

Authorization rule: a missing token, a wrong token, another job's token, a
malformed job ID and an unknown (or purged) job all raise the same
JobNotFound, which the API returns as one 404 body. The token is checked in
constant time, against a dummy hash when the job does not exist.

Expiry is enforced on read as well as by the cleanup sweep, so a job past
its expiry time never returns data even if cleanup has not run yet.
"""

from __future__ import annotations

import logging
from datetime import timedelta

from pulsecheck.jobs.auth import (
    JOB_ID_PATTERN,
    hash_token,
    new_access_token,
    new_job_id,
    token_matches,
)
from pulsecheck.jobs.models import (
    DeletionStatus,
    ExpiryReason,
    Job,
    JobStatus,
)
from pulsecheck.jobs.queue import QueueUnavailable
from pulsecheck.jobs.runtime import Runtime
from pulsecheck.jobs.store import ConcurrentUpdate
from pulsecheck.jobs.validation import (
    FILE_TOO_LARGE,
    UploadRejected,
    check_size,
    check_upload,
)
from pulsecheck.storage import new_upload_key

log = logging.getLogger("pulsecheck.jobs")

S = JobStatus
QUEUE_UNAVAILABLE = "queue_unavailable"
UPLOAD_INCOMPLETE = "upload_incomplete"


class JobNotFound(Exception):
    """Unknown job or failed authorization; deliberately indistinguishable."""


class ServiceBusy(Exception):
    """Global capacity is exhausted (UP-2)."""


class TooManyActiveJobs(Exception):
    """This client already has the maximum number of unfinished jobs."""


class JobConflict(Exception):
    """The request does not fit the job's current status."""

    def __init__(self, code: str, message: str, status: JobStatus) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status


class JobService:
    def __init__(self, runtime: Runtime) -> None:
        self.rt = runtime
        self.store = runtime.store
        self.storage = runtime.storage
        self.settings = runtime.settings

    def now(self):
        return self.rt.clock()

    # --- creation and access ---------------------------------------------

    def check_capacity(self) -> None:
        """Raise ServiceBusy when the queue or the global active-job cap is full."""
        cap = self.settings.capacity
        try:
            depth = self.rt.queue.depth()
        except QueueUnavailable:
            raise ServiceBusy() from None
        if depth >= cap.max_queue_length:
            raise ServiceBusy()
        if self.store.count_active() >= cap.max_active_jobs:
            raise ServiceBusy()

    def create(self, client_hash: str) -> tuple[Job, str]:
        """Create a job; returns it with its access token (shown once, never stored)."""
        self.check_capacity()
        per_client = self.settings.capacity.max_active_jobs_per_client
        if self.store.count_active(client_hash) >= per_client:
            raise TooManyActiveJobs()
        now = self.now()
        retention = self.settings.retention
        token = new_access_token()
        job = Job(
            id=new_job_id(),
            token_hash=hash_token(token),
            status=S.CREATED.value,
            client_hash=client_hash,
            created_at=now,
            updated_at=now,
            upload_expires_at=now + timedelta(hours=retention.abandoned_upload_hours),
            expires_at=now + timedelta(hours=retention.job_metadata_hours),
            upload_key=new_upload_key(),
            deletion_status=DeletionStatus.NOTHING_STORED.value,
        )
        return self.store.add(job), token

    def authorize(self, job_id: str, token: str | None) -> Job:
        """The job if `token` is its access token; expires it first if it is overdue."""
        job = self.store.get(job_id) if JOB_ID_PATTERN.match(job_id or "") else None
        if not token_matches(token, job.token_hash if job else None):
            raise JobNotFound()
        return self.expire_if_due(job)

    def expire_if_due(self, job: Job) -> Job:
        now = self.now()
        if job.job_status in (S.CREATED, S.UPLOADING) and now >= job.upload_expires_at:
            return self.expire(job, ExpiryReason.ABANDONED)
        if job.job_status != S.EXPIRED and now >= job.expires_at:
            return self.expire(job, ExpiryReason.RETENTION_ELAPSED)
        return job

    # --- upload --------------------------------------------------------------

    def begin_upload(self, job: Job, declared_size: int | None) -> Job:
        """created -> uploading before a direct upload streams in."""
        if job.job_status != S.CREATED:
            raise JobConflict(
                "upload_not_allowed", "This job does not accept an upload now.", job.job_status
            )
        if declared_size is not None and declared_size > self.settings.upload.max_bytes:
            self.fail(job.id, S.CREATED, *self._too_large())
            raise UploadRejected(*self._too_large())
        return self._transition(job, S.UPLOADING, upload_started_at=self.now())

    def _too_large(self) -> tuple[str, str]:
        mb = self.settings.upload.max_size_mb
        return FILE_TOO_LARGE, f"The file is larger than {mb:g} MB."

    def upload_too_large(self, job: Job) -> UploadRejected:
        self.fail(job.id, S.UPLOADING, *self._too_large())
        return UploadRejected(*self._too_large())

    def upload_interrupted(self, job: Job) -> None:
        self.fail(job.id, S.UPLOADING, UPLOAD_INCOMPLETE, "The upload did not finish.")

    def finish_upload(self, job: Job, size: int) -> Job:
        self.store.update(
            job.id, upload_size_bytes=size, deletion_status=DeletionStatus.UPLOAD_STORED.value
        )
        return self.store.get(job.id)

    def complete_upload(self, job: Job) -> Job:
        """Validate the stored upload and queue the analysis (uploading -> queued)."""
        status = job.job_status
        if status not in (S.CREATED, S.UPLOADING):
            raise JobConflict(
                "upload_already_completed" if status != S.EXPIRED else "job_expired",
                "The upload for this job has already been completed or closed.",
                status,
            )
        size = self.storage.size(job.upload_key)
        if size is None:
            raise JobConflict(
                "upload_missing", "No complete upload was found for this job.", status
            )
        if status == S.CREATED:  # presigned uploads never pass through the API
            job = self._transition(job, S.UPLOADING, upload_started_at=self.now())
        try:
            check_size(size, self.settings.upload)
            with self.storage.local_copy(job.upload_key) as path:
                validated = check_upload(path, self.settings.upload)
        except UploadRejected as exc:
            self.fail(job.id, S.UPLOADING, exc.code, exc.message)
            raise
        # The upload stays valid until its expiry, so a client told "busy" can retry.
        self.check_capacity()
        job = self._transition(
            job,
            S.QUEUED,
            queued_at=self.now(),
            container=validated.container,
            upload_size_bytes=size,
            deletion_status=DeletionStatus.UPLOAD_STORED.value,
        )
        try:
            self.rt.queue.enqueue(job.id)
        except QueueUnavailable:
            log.error("could not enqueue job %s: queue unavailable", job.id)
            self.fail(job.id, S.QUEUED, QUEUE_UNAVAILABLE, "The analysis queue is unavailable.")
            raise ServiceBusy() from None
        return job

    # --- failure, expiry and deletion -----------------------------------

    def _transition(self, job: Job, new: JobStatus, **fields) -> Job:
        try:
            return self.store.transition(job.id, job.job_status, new, **fields)
        except ConcurrentUpdate:
            current = self.store.get(job.id)
            raise JobConflict(
                "job_changed", "The job changed status; fetch it again.", current.job_status
            ) from None

    def fail(self, job_id: str, current: JobStatus, code: str, message: str) -> Job | None:
        """Move an unfinished job to failed and delete its upload (UP-4)."""
        try:
            job = self.store.transition(
                job_id,
                current,
                S.FAILED,
                finished_at=self.now(),
                error_code=code,
                error_message=message,
            )
        except ConcurrentUpdate:
            job = None  # expired or deleted meanwhile; its data is handled there
        self.delete_upload(job_id)
        return job

    def delete_upload(self, job_id: str) -> None:
        """Delete a finished job's upload if it may still exist; record the outcome."""
        job = self.store.get(job_id)
        if job is None or job.job_status in (S.UPLOADING, S.QUEUED, S.PROCESSING):
            return  # gone, or the upload is still needed
        already = job.upload_deleted_at is not None or job.upload_key is None
        gone = already or self.storage.delete(job.upload_key)
        if not gone:
            status = DeletionStatus.UPLOAD_DELETE_PENDING
        elif job.job_status == S.EXPIRED:
            status = DeletionStatus.ALL_DELETED
        elif job.deletion_status == DeletionStatus.NOTHING_STORED:
            status = DeletionStatus.NOTHING_STORED
        else:
            status = DeletionStatus.UPLOAD_DELETED
        fields = {"deletion_status": status.value}
        if gone and not already:
            fields["upload_deleted_at"] = self.now()
        if status.value != job.deletion_status or len(fields) > 1:
            self.store.update(job_id, **fields)

    def expire(self, job: Job, reason: ExpiryReason) -> Job:
        """Expire a job: drop its result, report and client hash and delete its upload.

        A data-free record remains so its owner sees status "expired" until
        cleanup purges it.
        """
        for _ in range(3):
            if job.job_status == S.EXPIRED:
                break
            try:
                self.store.transition(
                    job.id,
                    job.job_status,
                    S.EXPIRED,
                    expired_at=self.now(),
                    expiry_reason=reason.value,
                    result=None,
                    report=None,
                    client_hash=None,
                    error_message=None,
                )
                break
            except ConcurrentUpdate:
                job = self.store.get(job.id)
        self.delete_upload(job.id)
        return self.store.get(job.id)
