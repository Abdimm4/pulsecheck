"""Job records and the job state machine (PRD FR-9, Section 9).

A job row holds status, timestamps, expiry, result metadata and deletion
status. It never holds video bytes, and the access token only as a SHA-256
hash. Every status change goes through `ALLOWED_TRANSITIONS`, checked in
`check_transition()` and applied atomically by `JobStore.transition()`.

    created -> uploading -> queued -> processing -> completed | failed
    any unfinished state -> failed (rejected upload, queue error, timeout)
    any state except expired -> expired (retention elapsed, abandoned, or deleted on request)
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum

from sqlalchemy import JSON, DateTime, Integer, String, Text, TypeDecorator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class JobStatus(StrEnum):
    CREATED = "created"
    UPLOADING = "uploading"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


S = JobStatus
ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    S.CREATED: frozenset({S.UPLOADING, S.FAILED, S.EXPIRED}),
    S.UPLOADING: frozenset({S.QUEUED, S.FAILED, S.EXPIRED}),
    S.QUEUED: frozenset({S.PROCESSING, S.FAILED, S.EXPIRED}),
    S.PROCESSING: frozenset({S.COMPLETED, S.FAILED, S.EXPIRED}),
    S.COMPLETED: frozenset({S.EXPIRED}),
    S.FAILED: frozenset({S.EXPIRED}),
    S.EXPIRED: frozenset(),
}
# Statuses that count against the active-job caps.
ACTIVE_STATUSES = (S.CREATED, S.UPLOADING, S.QUEUED, S.PROCESSING)
TERMINAL_STATUSES = (S.COMPLETED, S.FAILED, S.EXPIRED)


class InvalidTransition(Exception):
    def __init__(self, current: JobStatus | str, new: JobStatus | str) -> None:
        super().__init__(f"cannot move a job from {current} to {new}")
        self.current = JobStatus(current)
        self.new = JobStatus(new)


def check_transition(current: JobStatus | str, new: JobStatus | str) -> None:
    """Raise InvalidTransition unless current -> new is allowed."""
    if JobStatus(new) not in ALLOWED_TRANSITIONS[JobStatus(current)]:
        raise InvalidTransition(current, new)


class DeletionStatus(StrEnum):
    """What has happened to the job's stored data (recorded, never the data itself)."""

    NOTHING_STORED = "nothing_stored"  # no upload yet
    UPLOAD_STORED = "upload_stored"  # video in private storage
    UPLOAD_DELETED = "upload_deleted"  # video gone; result/report may remain until expiry
    UPLOAD_DELETE_PENDING = "upload_delete_pending"  # delete failed; cleanup retries
    ALL_DELETED = "all_deleted"  # video, result and report gone (job expired)


class ExpiryReason(StrEnum):
    RETENTION_ELAPSED = "retention_elapsed"
    ABANDONED = "abandoned"
    DELETED_ON_REQUEST = "deleted_on_request"


def utcnow() -> datetime:
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator):
    """Timezone-aware UTC datetimes on every backend (SQLite drops the zone)."""

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            raise ValueError("naive datetime; use timezone-aware UTC")
        value = value.astimezone(UTC)
        return value.replace(tzinfo=None) if dialect.name == "sqlite" else value

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


class Base(DeclarativeBase):
    pass


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    token_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), index=True)
    # Salted hash of the client IP for rate limits and per-client caps;
    # cleared when the job expires.
    client_hash: Mapped[str | None] = mapped_column(String(64), index=True)

    created_at: Mapped[datetime] = mapped_column(UTCDateTime())
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime())
    upload_started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    queued_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    started_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    # Unfinished uploads expire at upload_expires_at, everything at expires_at.
    upload_expires_at: Mapped[datetime] = mapped_column(UTCDateTime())
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime(), index=True)

    # Server-generated storage key; never chosen by the client.
    upload_key: Mapped[str | None] = mapped_column(String(128))
    upload_size_bytes: Mapped[int | None] = mapped_column(Integer)
    container: Mapped[str | None] = mapped_column(String(8))
    deletion_status: Mapped[str] = mapped_column(String(32))
    upload_deleted_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    expired_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    expiry_reason: Mapped[str | None] = mapped_column(String(32))

    error_code: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    model_name: Mapped[str | None] = mapped_column(String(64))
    model_version: Mapped[str | None] = mapped_column(String(32))
    # Result summary (state, label, score, quality) and the FR-10 report.
    result: Mapped[dict | None] = mapped_column(JSON)
    report: Mapped[dict | None] = mapped_column(JSON)

    @property
    def job_status(self) -> JobStatus:
        return JobStatus(self.status)
