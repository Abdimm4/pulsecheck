import asyncio
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from api_support import bearer, make_settings, run_worker
from sqlalchemy import inspect
from test_api_jobs import create, submit

from pulsecheck.api.schemas import Report
from pulsecheck.config import REPO_ROOT
from pulsecheck.face import default_model_path
from pulsecheck.jobs.models import Base
from pulsecheck.jobs.settings import CapacityConfig, RetentionConfig, Settings, StorageConfig
from pulsecheck.jobs.store import make_engine
from pulsecheck.storage import InvalidKey, LocalStorage, ObjectTooLarge, new_upload_key


async def _chunks(data: bytes, size: int = 100):
    for i in range(0, len(data), size):
        yield data[i : i + size]


# --- safe errors ------------------------------------------------------------


def test_unexpected_errors_return_a_safe_body(api, video, monkeypatch):
    runtime, client = api
    created = create(client).json()

    def explode(self, job):
        raise RuntimeError(f"secret detail at {runtime.settings.storage.resolved_dir()}")

    monkeypatch.setattr("pulsecheck.jobs.service.JobService.complete_upload", explode)
    response = client.post(
        f"/v1/jobs/{created['job_id']}/complete-upload", headers=bearer(created["access_token"])
    )
    assert response.status_code == 500
    assert response.json() == {
        "error": {"code": "internal_error", "message": "Something went wrong. Please try again."}
    }
    assert "secret" not in response.text and "Traceback" not in response.text


def test_validation_errors_do_not_echo_input(api):
    _, client = api
    response = client.post("/v1/jobs", json={"size_bytes": "<script>alert(1)</script>"})
    assert response.status_code == 422
    body = response.json()["error"]
    assert body["code"] == "invalid_request"
    assert body["fields"][0]["loc"] == ["body", "size_bytes"]
    assert "<script>" not in response.text
    extra = client.post("/v1/jobs", json={"filename": "../../etc/passwd"})
    assert extra.status_code == 422  # clients cannot name files or paths
    assert "passwd" not in extra.text


def test_large_or_chunked_json_bodies_are_refused(api):
    _, client = api
    big = client.post("/v1/jobs", content=b"{" + b" " * 20_000 + b"}")
    assert big.status_code == 413 and big.json()["error"]["code"] == "request_too_large"
    chunked = client.post("/v1/jobs", content=iter([b"{}"]))
    assert chunked.status_code == 411


def test_unknown_routes_and_methods_use_the_error_shape(api):
    _, client = api
    assert client.get("/nope").json()["error"]["code"] == "not_found"
    assert client.patch("/v1/jobs").json()["error"]["code"] == "method_not_allowed"


def test_cors_is_off_by_default_and_never_wildcard(make_api):
    from fastapi.testclient import TestClient

    from pulsecheck.api.app import create_app

    runtime, client = make_api()
    preflight = {"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    response = client.options("/v1/jobs", headers=preflight)
    assert "access-control-allow-origin" not in response.headers

    runtime.settings = replace(runtime.settings, cors_origins=("https://a.example",))
    with TestClient(create_app(runtime)) as cors_client:
        ok = cors_client.options(
            "/v1/jobs",
            headers={"Origin": "https://a.example", "Access-Control-Request-Method": "POST"},
        )
        assert ok.headers["access-control-allow-origin"] == "https://a.example"
        bad = cors_client.options("/v1/jobs", headers=preflight)
        assert "access-control-allow-origin" not in bad.headers


# --- storage -------------------------------------------------------------------


def test_storage_keys_are_server_generated_and_validated(tmp_path):
    storage = LocalStorage(tmp_path / "store")
    key = new_upload_key()
    assert key.startswith("uploads/") and len(key) == len("uploads/") + 32
    for bad in ("../x", "uploads/../../x", "/etc/passwd", "uploads/abc", "C:\\x", key + "/.."):
        with pytest.raises(InvalidKey):
            storage.size(bad)
        with pytest.raises(InvalidKey):
            storage.delete(bad)


def test_storage_stream_cap_leaves_nothing_behind(tmp_path):
    storage = LocalStorage(tmp_path / "store")
    key = new_upload_key()
    with pytest.raises(ObjectTooLarge):
        asyncio.run(storage.receive(key, _chunks(b"x" * 1000), max_bytes=999))
    assert list((tmp_path / "store").rglob("*.*")) == []
    assert storage.size(key) is None
    assert asyncio.run(storage.receive(key, _chunks(b"x" * 999), max_bytes=999)) == 999
    assert storage.size(key) == 999
    with storage.local_copy(key) as path:
        assert path.read_bytes() == b"x" * 999
    assert storage.delete(key) and storage.size(key) is None
    assert storage.delete(key)  # deleting a missing object is fine


def test_storage_stale_keys(tmp_path):
    storage = LocalStorage(tmp_path / "store")
    key = new_upload_key()
    asyncio.run(storage.receive(key, _chunks(b"abc"), max_bytes=10))
    (tmp_path / "store" / "uploads" / "not-a-key.txt").write_text("ignored")
    future = datetime.now(UTC) + timedelta(minutes=1)
    assert list(storage.stale_keys(future)) == [key]
    assert list(storage.stale_keys(future - timedelta(hours=1))) == []


def test_upload_target_points_at_the_api(tmp_path):
    target = LocalStorage(tmp_path).upload_target(
        new_upload_key(), "a" * 32, datetime.now(UTC), 100
    )
    assert target.method == "PUT" and target.url == f"/v1/jobs/{'a' * 32}/upload"
    assert target.requires_access_token


# --- settings ------------------------------------------------------------------


def test_settings_from_yaml_and_env(tmp_path):
    settings = make_settings(tmp_path, capacity={"max_queue_length": 7})
    assert settings.capacity.max_queue_length == 7
    assert settings.upload.max_bytes == 100_000_000
    assert settings.retention.abandoned_upload_hours == 1
    assert settings.retention.job_metadata_hours == 24
    assert settings.storage.resolved_dir() == tmp_path / "storage"
    assert settings.trust_proxy
    assert "secret" not in repr(settings)
    defaults = Settings.load({}, env={})
    assert defaults.database_url.startswith("sqlite:///")
    assert defaults.redis_url == "redis://localhost:6379/0"
    assert len(defaults.secret_key) >= 32 and not defaults.trust_proxy
    assert StorageConfig().resolved_dir() == REPO_ROOT / "data" / "storage"


def test_settings_reject_bad_values():
    with pytest.raises(ValueError, match="unknown capacity"):
        CapacityConfig.from_mapping({"max_jobs": 1})
    with pytest.raises(ValueError, match="positive"):
        CapacityConfig(max_queue_length=0)
    with pytest.raises(ValueError, match="exceed"):
        RetentionConfig(abandoned_upload_hours=48, job_metadata_hours=24)
    with pytest.raises(ValueError, match="local"):
        StorageConfig(backend="s3")


# --- database migration -----------------------------------------------------------


def test_alembic_migration_matches_the_models(tmp_path):
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    url = f"sqlite:///{(tmp_path / 'migrated.db').as_posix()}"
    config = Config(str(REPO_ROOT / "backend" / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "backend/pulsecheck/jobs/migrations"))
    config.attributes["database_url"] = url
    config.attributes["configure_logger"] = False
    command.upgrade(config, "head")
    engine = make_engine(url)
    with engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []
    assert "jobs" in inspect(engine).get_table_names()
    command.downgrade(config, "base")
    assert "jobs" not in inspect(engine).get_table_names()
    engine.dispose()


# --- the real pipeline ----------------------------------------------------------


@pytest.mark.skipif(not default_model_path().is_file(), reason="landmarker model not downloaded")
def test_real_pipeline_end_to_end(make_api, tmp_path):
    from api_support import write_video

    runtime, client = make_api(analyzer=None)  # PipelineAnalyzer from configs/default.yaml
    video = write_video(tmp_path / "faceless.mp4", n_frames=45)
    job_id, token, done = submit(client, video)
    assert done.status_code == 202, done.text
    run_worker(runtime)
    job = client.get(f"/v1/jobs/{job_id}", headers=bearer(token)).json()
    assert job["status"] == "completed", job
    assert job["model"] == {"name": "heuristic-consistency", "version": "0.1.0"}
    # No face: the honest answer is inconclusive, with reasons.
    assert job["result"]["state"] == "inconclusive"
    assert job["result"]["reasons"]
    assert job["result"]["face_coverage"] == 0.0
    report = Report.model_validate(
        client.get(f"/v1/jobs/{job_id}/report", headers=bearer(token)).json()
    )
    assert report.quality_checks["tracking"]["total_frames"] == 45
    assert set(report.quality_checks["signals"]) == {"forehead", "left_cheek", "right_cheek"}
    assert report.features["missing"]
    assert report.configuration["signal"]["band_hz"] == [0.7, 3.0]
    assert report.configuration["upload_limits"]["max_duration_s"] == 60
    assert report.warnings
    assert client.get("/version").json()["model"]["name"] == "heuristic-consistency"
