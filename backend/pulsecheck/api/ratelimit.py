"""Fixed-window rate limit on job creation, per client hash, stored in Redis (PRD UP-2)."""

from __future__ import annotations

import time
from collections.abc import Callable

from redis import Redis
from redis.exceptions import RedisError

from pulsecheck.jobs.queue import QueueUnavailable

PREFIX = "pulsecheck:rate:create"


class RateLimiter:
    def __init__(
        self,
        redis: Redis,
        limit: int,
        window_s: int,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.redis = redis
        self.limit = limit
        self.window_s = window_s
        self.clock = clock

    def hit(self, client_hash: str) -> int | None:
        """Count one request; None if allowed, else seconds until the window resets."""
        now = self.clock()
        bucket = int(now // self.window_s)
        key = f"{PREFIX}:{client_hash}:{bucket}"
        try:
            pipe = self.redis.pipeline()
            pipe.incr(key)
            pipe.expire(key, self.window_s)
            count, _ = pipe.execute()
        except RedisError as exc:
            raise QueueUnavailable(type(exc).__name__) from None
        if int(count) > self.limit:
            return max(1, int((bucket + 1) * self.window_s - now))
        return None
