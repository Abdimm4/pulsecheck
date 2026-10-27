"""Helpers for API/job tests: a fake clock, a stub analyzer, synthetic videos and a runtime.

Everything runs in-process: SQLite for the database, fakeredis for the queue
and rate limits, and RQ's SimpleWorker in burst mode for the worker.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import fakeredis
import numpy as np

from pulsecheck.config import load_config
from pulsecheck.features import FEATURE_NAMES, FeatureVector
from pulsecheck.jobs import tasks
from pulsecheck.jobs.pipeline import AnalysisOutput
from pulsecheck.jobs.queue import make_worker
from pulsecheck.jobs.runtime import Runtime
from pulsecheck.jobs.settings import Settings
from pulsecheck.jobs.store import create_tables
from pulsecheck.models import classify
from pulsecheck.video import probe_video


class FakeClock:
    def __init__(self) -> None:
        self.now = datetime.now(UTC).replace(microsecond=0)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **kwargs) -> None:
        self.now += timedelta(**kwargs)


@dataclass
class StubAnalyzer:
    """Decodes the video for metadata, then returns fixed, decidable features."""

    error: BaseException | None = None
    calls: list[Path] = field(default_factory=list)
    model_name: str = "stub-model"
    model_version: str = "0.0.1"

    def __call__(self, path: Path) -> AnalysisOutput:
        self.calls.append(path)
        if self.error is not None:
            raise self.error
        meta = probe_video(path)
        values = dict.fromkeys(FEATURE_NAMES, 0.5)
        values.update(usable_roi_count=3.0, face_coverage=1.0, signal_quality=0.9)
        values.update(mean_cross_correlation=0.9, max_freq_diff_hz=0.0)
        values["mean_window_freq_std_hz"] = 0.0
        features = FeatureVector(values, {})
        return AnalysisOutput(
            video=meta,
            tracking=None,
            features=features,
            signal_summary=None,
            result=classify(features).to_dict(),
            settings={"stub": True},
        )


def write_video(path: Path, n_frames: int = 30, fps: float = 30.0, size=(64, 48)) -> Path:
    fourcc = "MJPG" if path.suffix == ".avi" else "mp4v"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
    assert writer.isOpened(), "OpenCV could not create a test video"
    for i in range(n_frames):
        writer.write(np.full((size[1], size[0], 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path


def make_settings(tmp_path: Path, **sections) -> Settings:
    """Settings from configs/default.yaml with `sections` merged in, isolated under tmp_path."""
    config = copy.deepcopy(load_config())
    for name, values in sections.items():
        config[name] = {**config.get(name, {}), **values}
    env = {
        "DATABASE_URL": f"sqlite:///{(tmp_path / 'jobs.db').as_posix()}",
        "PULSECHECK_SECRET_KEY": "test-secret-key",
        "PULSECHECK_STORAGE_DIR": str(tmp_path / "storage"),
        "PULSECHECK_TRUST_PROXY": "1",
    }
    return Settings.load(config, env)


def make_runtime(tmp_path: Path, analyzer=None, clock=None, **sections) -> Runtime:
    settings = make_settings(tmp_path, **sections)
    runtime = Runtime.from_settings(
        settings,
        redis=fakeredis.FakeRedis(),
        clock=clock or FakeClock(),
        analyzer=analyzer,
    )
    create_tables(runtime.store.engine)
    tasks.set_runtime(runtime)
    return runtime


def run_worker(runtime: Runtime) -> None:
    """Process everything queued, in this process."""
    make_worker(runtime.redis).work(burst=True, with_scheduler=False)


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def stored_files(runtime: Runtime) -> list[Path]:
    root = runtime.settings.storage.resolved_dir()
    return [p for p in root.rglob("*") if p.is_file()]
