"""Database access for job records.

Works with any SQLAlchemy backend; PostgreSQL in deployment, SQLite for
local development and tests. Status changes are compare-and-set UPDATEs, so
the API, worker and cleanup can race without overwriting each other.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, delete, func, select, update
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker

from pulsecheck.jobs.models import (
    ACTIVE_STATUSES,
    Base,
    Job,
    JobStatus,
    check_transition,
    utcnow,
)

Clock = Callable[[], datetime]


class ConcurrentUpdate(Exception):
    """The job changed status between reading and updating it."""


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    kwargs: dict = {"pool_pre_ping": True}
    if parsed.get_backend_name() == "sqlite":
        # The API serves requests from a thread pool.
        kwargs["connect_args"] = {"check_same_thread": False}
        if parsed.database and parsed.database != ":memory:":
            Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url, **kwargs)


def create_tables(engine: Engine) -> None:
    """Create the schema directly (tests, quick local runs); deployments use Alembic."""
    Base.metadata.create_all(engine)


class JobStore:
    def __init__(self, engine: Engine, clock: Clock = utcnow) -> None:
        self.engine = engine
        self.sessions = sessionmaker(engine, expire_on_commit=False)
        self.clock = clock

    def add(self, job: Job) -> Job:
        with self.sessions.begin() as session:
            session.add(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self.sessions() as session:
            return session.get(Job, job_id)

    def transition(self, job_id: str, current: JobStatus | str, new: JobStatus, **fields) -> Job:
        """Move a job from `current` to `new`, setting `fields` in the same UPDATE.

        Raises InvalidTransition if the state machine forbids it and
        ConcurrentUpdate if the job is no longer in `current`.
        """
        check_transition(current, new)
        values = {"status": new.value, "updated_at": self.clock(), **fields}
        with self.sessions.begin() as session:
            done = session.execute(
                update(Job)
                .where(Job.id == job_id, Job.status == JobStatus(current).value)
                .values(**values)
            )
            if done.rowcount != 1:
                raise ConcurrentUpdate(job_id)
            return session.get(Job, job_id)

    def update(self, job_id: str, **fields) -> None:
        """Set non-status fields."""
        if "status" in fields:
            raise ValueError("use transition() to change status")
        with self.sessions.begin() as session:
            session.execute(
                update(Job).where(Job.id == job_id).values(updated_at=self.clock(), **fields)
            )

    def count_active(self, client_hash: str | None = None) -> int:
        query = (
            select(func.count())
            .select_from(Job)
            .where(Job.status.in_([s.value for s in ACTIVE_STATUSES]))
        )
        if client_hash is not None:
            query = query.where(Job.client_hash == client_hash)
        with self.sessions() as session:
            return int(session.execute(query).scalar_one())

    def find(self, *conditions) -> Sequence[Job]:
        with self.sessions() as session:
            return session.execute(select(Job).where(*conditions)).scalars().all()

    def upload_keys(self) -> set[str]:
        with self.sessions() as session:
            rows = session.execute(select(Job.upload_key).where(Job.upload_key.is_not(None)))
            return {key for (key,) in rows}

    def purge(self, job_ids: Iterable[str]) -> int:
        ids = list(job_ids)
        if not ids:
            return 0
        with self.sessions.begin() as session:
            return session.execute(delete(Job).where(Job.id.in_(ids))).rowcount
