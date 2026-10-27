"""Service settings: YAML sections (configs/default.yaml) plus environment variables.

Limits and retention come from the YAML so they can be reviewed in one place;
connection strings and secrets come only from the environment (.env.example):

- DATABASE_URL: SQLAlchemy URL, PostgreSQL in deployment (default: SQLite under data/).
- REDIS_URL: queue and rate-limit store (default redis://localhost:6379/0).
- PULSECHECK_SECRET_KEY: salts the client hash used for rate limits and per-client caps.
- PULSECHECK_STORAGE_DIR: overrides storage.local_dir.
- PULSECHECK_MODEL_PATH: overrides analysis.model_path.
- PULSECHECK_CORS_ORIGINS: comma-separated frontend origins (none by default).
- PULSECHECK_TRUST_PROXY: "1" to take the client IP from X-Forwarded-For (only behind
  a proxy that sets it).
"""

from __future__ import annotations

import logging
import os
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path

from pulsecheck.config import REPO_ROOT, load_config
from pulsecheck.jobs.validation import UploadLimits

log = logging.getLogger("pulsecheck.settings")

DEFAULT_DATABASE_URL = f"sqlite:///{(REPO_ROOT / 'data' / 'pulsecheck.db').as_posix()}"
DEFAULT_REDIS_URL = "redis://localhost:6379/0"


def _from_mapping(cls, values: Mapping | None, section: str):
    values = dict(values or {})
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {section} settings: {sorted(unknown)}")
    return cls(**values)


@dataclass(frozen=True)
class CapacityConfig:
    """Queue bounds, active-job caps and the job-creation rate limit (PRD UP-2)."""

    max_queue_length: int = 20
    max_active_jobs: int = 50
    max_active_jobs_per_client: int = 2
    job_creations_per_window: int = 10
    rate_window_s: int = 3600
    job_timeout_s: int = 300
    retry_after_s: int = 60

    def __post_init__(self) -> None:
        if min(asdict_values(self)) <= 0:
            raise ValueError("capacity settings must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> CapacityConfig:
        return _from_mapping(cls, values, "capacity")


@dataclass(frozen=True)
class RetentionConfig:
    """Expiry periods (PRD UP-4)."""

    abandoned_upload_hours: float = 1.0
    job_metadata_hours: float = 24.0
    expired_record_hours: float = 168.0

    def __post_init__(self) -> None:
        if min(asdict_values(self)) <= 0:
            raise ValueError("retention periods must be positive")
        if self.abandoned_upload_hours > self.job_metadata_hours:
            raise ValueError("abandoned_upload_hours must not exceed job_metadata_hours")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> RetentionConfig:
        return _from_mapping(cls, values, "retention")


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "local"
    local_dir: str = "data/storage"

    def __post_init__(self) -> None:
        if self.backend != "local":
            raise ValueError("only the 'local' storage backend exists so far")

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> StorageConfig:
        return _from_mapping(cls, values, "storage")

    def resolved_dir(self) -> Path:
        path = Path(self.local_dir)
        return path if path.is_absolute() else REPO_ROOT / path


@dataclass(frozen=True)
class AnalysisConfig:
    model_path: str | None = None

    @classmethod
    def from_mapping(cls, values: Mapping | None) -> AnalysisConfig:
        return _from_mapping(cls, values, "analysis")


def asdict_values(obj) -> list[float]:
    return [getattr(obj, f.name) for f in fields(obj)]


@dataclass(frozen=True)
class Settings:
    database_url: str = DEFAULT_DATABASE_URL
    redis_url: str = DEFAULT_REDIS_URL
    secret_key: str = field(default="", repr=False)
    cors_origins: tuple[str, ...] = ()
    trust_proxy: bool = False
    upload: UploadLimits = field(default_factory=UploadLimits)
    capacity: CapacityConfig = field(default_factory=CapacityConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    # The full parsed YAML, for the pipeline sections (signal, features, heuristic).
    pipeline_config: dict = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, config: Mapping | None = None, env: Mapping[str, str] | None = None) -> Settings:
        """Build settings from a parsed config (default: load_config()) and the environment."""
        config = dict(load_config() if config is None else config)
        env = os.environ if env is None else env
        storage = StorageConfig.from_mapping(config.get("storage"))
        if env.get("PULSECHECK_STORAGE_DIR"):
            storage = StorageConfig(storage.backend, env["PULSECHECK_STORAGE_DIR"])
        analysis = AnalysisConfig.from_mapping(config.get("analysis"))
        if env.get("PULSECHECK_MODEL_PATH"):
            analysis = AnalysisConfig(env["PULSECHECK_MODEL_PATH"])
        secret = env.get("PULSECHECK_SECRET_KEY", "")
        if not secret:
            # Per-process salt: fine for development, but rate limits and
            # per-client caps then differ between API processes and restarts.
            log.warning("PULSECHECK_SECRET_KEY is not set; using a random per-process key")
            secret = secrets.token_urlsafe(32)
        origins = tuple(
            o.strip() for o in env.get("PULSECHECK_CORS_ORIGINS", "").split(",") if o.strip()
        )
        return cls(
            database_url=env.get("DATABASE_URL") or DEFAULT_DATABASE_URL,
            redis_url=env.get("REDIS_URL") or DEFAULT_REDIS_URL,
            secret_key=secret,
            cors_origins=origins,
            trust_proxy=env.get("PULSECHECK_TRUST_PROXY", "") == "1",
            upload=UploadLimits.from_mapping(config.get("upload")),
            capacity=CapacityConfig.from_mapping(config.get("capacity")),
            retention=RetentionConfig.from_mapping(config.get("retention")),
            storage=storage,
            analysis=analysis,
            pipeline_config=config,
        )
