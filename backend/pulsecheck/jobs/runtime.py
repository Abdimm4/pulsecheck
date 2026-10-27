"""Shared service objects for one process (API, worker or cleanup)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from redis import Redis

from pulsecheck.jobs.models import utcnow
from pulsecheck.jobs.pipeline import Analyzer, PipelineAnalyzer
from pulsecheck.jobs.queue import JobQueue, RQJobQueue
from pulsecheck.jobs.settings import Settings
from pulsecheck.jobs.store import Clock, JobStore, make_engine
from pulsecheck.models import MODEL_NAME, MODEL_VERSION
from pulsecheck.storage import LocalStorage, Storage


def describe_model(model_path: str | None) -> tuple[str, str]:
    """(name, version) of the configured classifier, from a trained model's sidecar."""
    if not model_path:
        return MODEL_NAME, MODEL_VERSION
    sidecar = json.loads(Path(model_path).with_suffix(".json").read_text(encoding="utf-8"))
    return sidecar["model_name"], sidecar["model_version"]


@dataclass
class Runtime:
    settings: Settings
    store: JobStore
    storage: Storage
    queue: JobQueue
    redis: Redis
    clock: Clock = utcnow
    _analyzer: Analyzer | None = field(default=None, repr=False)

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        redis: Redis | None = None,
        clock: Clock = utcnow,
        analyzer: Analyzer | None = None,
    ) -> Runtime:
        redis = redis if redis is not None else Redis.from_url(settings.redis_url)
        return cls(
            settings=settings,
            store=JobStore(make_engine(settings.database_url), clock),
            storage=LocalStorage(settings.storage.resolved_dir()),
            queue=RQJobQueue(redis, settings.capacity.job_timeout_s),
            redis=redis,
            clock=clock,
            _analyzer=analyzer,
        )

    @property
    def analyzer(self) -> Analyzer:
        """The analysis pipeline, built on first use (only workers need it)."""
        if self._analyzer is None:
            self._analyzer = PipelineAnalyzer.from_config(
                self.settings.pipeline_config,
                self.settings.upload,
                self.settings.analysis.model_path,
            )
        return self._analyzer

    def model(self) -> tuple[str, str]:
        if self._analyzer is not None:
            return self._analyzer.model_name, self._analyzer.model_version
        return describe_model(self.settings.analysis.model_path)
