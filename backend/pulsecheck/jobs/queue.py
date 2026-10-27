"""Redis-backed analysis queue (RQ).

The worker is RQ's SimpleWorker, which runs jobs in its own process instead
of forking one per job: RQ's default Worker needs os.fork() and does not run
on Windows. On platforms without SIGALRM (Windows) RQ enforces job timeouts
with TimerDeathPenalty, a timer thread that raises inside the job. A single
SimpleWorker processes one analysis at a time; run more worker processes to
add concurrency (the global caps in `capacity:` still apply).
"""

from __future__ import annotations

from typing import Protocol

from redis import Redis
from redis.exceptions import RedisError
from rq import Queue, SimpleWorker

QUEUE_NAME = "pulsecheck-analysis"
TASK = "pulsecheck.jobs.tasks.analyze_job"
FAILURE_TTL_S = 24 * 3600


class QueueUnavailable(Exception):
    """Redis could not be reached."""


class JobQueue(Protocol):
    def enqueue(self, job_id: str) -> None: ...

    def depth(self) -> int:
        """Number of analyses waiting to start."""
        ...


class RQJobQueue:
    def __init__(self, redis: Redis, job_timeout_s: int, name: str = QUEUE_NAME) -> None:
        self.redis = redis
        self.job_timeout_s = job_timeout_s
        self.queue = Queue(name, connection=redis)

    def enqueue(self, job_id: str) -> None:
        try:
            self.queue.enqueue_call(
                TASK,
                args=(job_id,),
                job_id=f"analysis-{job_id}",
                timeout=self.job_timeout_s,
                result_ttl=0,  # results live in the database, not in Redis
                failure_ttl=FAILURE_TTL_S,
                description="pulsecheck analysis",
            )
        except RedisError as exc:
            raise QueueUnavailable(type(exc).__name__) from None

    def depth(self) -> int:
        try:
            return self.queue.count
        except RedisError as exc:
            raise QueueUnavailable(type(exc).__name__) from None


def make_worker(redis: Redis, name: str = QUEUE_NAME) -> SimpleWorker:
    return SimpleWorker([Queue(name, connection=redis)], connection=redis)
