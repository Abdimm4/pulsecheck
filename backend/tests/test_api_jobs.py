import hashlib
import json
import logging

from api_support import bearer, run_worker, stored_files

from pulsecheck.api.schemas import Report
from pulsecheck.jobs.models import Job
from pulsecheck.reporting import REPORT_SCHEMA_VERSION


def create(client, ip="203.0.113.1", **body):
    return client.post("/v1/jobs", json=body or None, headers={"X-Forwarded-For": ip})


def submit(client, path, ip="203.0.113.1"):
    """Create, upload and complete; returns (job_id, token, complete-upload response)."""
    created = create(client, ip)
    assert created.status_code == 201, created.text
    job_id, token = created.json()["job_id"], created.json()["access_token"]
    put = client.put(f"/v1/jobs/{job_id}/upload", content=path.read_bytes(), headers=bearer(token))
    assert put.status_code == 200, put.text
    done = client.post(f"/v1/jobs/{job_id}/complete-upload", headers=bearer(token))
    return job_id, token, done


def test_health_version_and_openapi(api):
    _, client = api
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json() == {"status": "ok", "database": "ok", "queue": "ok"}
    version = client.get("/version").json()
    assert version["version"] == "0.1.0" and version["api_version"] == "v1"
    assert version["model"] == {"name": "stub-model", "version": "0.0.1"}
    assert version["report_schema_version"] == REPORT_SCHEMA_VERSION
    paths = client.get("/openapi.json").json()["paths"]
    for path in (
        "/health",
        "/version",
        "/v1/jobs",
        "/v1/jobs/{job_id}",
        "/v1/jobs/{job_id}/upload",
        "/v1/jobs/{job_id}/complete-upload",
        "/v1/jobs/{job_id}/report",
    ):
        assert path in paths
    assert set(paths["/v1/jobs/{job_id}"]) == {"get", "delete"}


def test_full_flow_submit_process_and_retrieve(api, video, caplog):
    caplog.set_level(logging.DEBUG)
    runtime, client = api
    created = create(client)
    assert created.status_code == 201
    body = created.json()
    job_id, token = body["job_id"], body["access_token"]
    assert body["status"] == "created"
    assert len(token) >= 43  # 32 random bytes, URL-safe base64
    assert body["upload"]["method"] == "PUT"
    assert body["upload"]["url"] == f"/v1/jobs/{job_id}/upload"
    assert body["upload"]["requires_access_token"] is True
    assert body["limits"]["max_size_bytes"] == 100_000_000
    assert body["limits"]["allowed_formats"] == ["mp4", "mov"]

    put = client.put(f"/v1/jobs/{job_id}/upload", content=video.read_bytes(), headers=bearer(token))
    assert put.status_code == 200
    assert put.json()["status"] == "uploading"
    assert put.json()["deletion_status"] == "upload_stored"

    done = client.post(f"/v1/jobs/{job_id}/complete-upload", headers=bearer(token))
    assert done.status_code == 202, done.text
    assert done.json()["status"] == "queued"
    assert client.get(f"/v1/jobs/{job_id}", headers=bearer(token)).json()["status"] == "queued"
    assert runtime.queue.depth() == 1

    run_worker(runtime)
    status = client.get(f"/v1/jobs/{job_id}", headers=bearer(token))
    assert status.status_code == 200
    job = status.json()
    assert job["status"] == "completed"
    assert job["result"]["state"] == "more_consistent_with_real"
    assert job["result"]["decoded_frame_count"] == 30
    assert job["model"] == {"name": "stub-model", "version": "0.0.1"}
    assert job["report_url"] == f"/v1/jobs/{job_id}/report"
    assert job["deletion_status"] == "upload_deleted"
    assert job["error"] is None
    assert stored_files(runtime) == []  # the video is deleted right after analysis
    assert status.headers["cache-control"] == "no-store"
    assert status.headers["referrer-policy"] == "no-referrer"

    report = client.get(job["report_url"], headers=bearer(token))
    assert report.status_code == 200
    data = Report.model_validate(report.json())
    assert data.report_schema_version == REPORT_SCHEMA_VERSION
    assert data.app == {"name": "PulseCheck", "version": "0.1.0"}
    assert data.model.name == "stub-model"
    assert data.expiry.expires_at == job["expires_at"].replace("Z", "+00:00")
    assert data.video["decoded_frame_count"] == 30 and data.video["container"] == "mp4"
    assert data.result["state"] == "more_consistent_with_real"
    assert data.caveats

    # The token is stored only as its hash and never logged.
    stored = runtime.store.get(job_id)
    assert stored.token_hash == hashlib.sha256(token.encode()).hexdigest()
    row = {c.name: getattr(stored, c.name) for c in Job.__table__.columns}
    assert token not in json.dumps(row, default=str)
    assert token not in caplog.text


def test_report_excludes_tokens_video_and_client_data(api, video):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    stored = runtime.store.get(job_id)
    run_worker(runtime)
    report = client.get(f"/v1/jobs/{job_id}/report", headers=bearer(token)).json()
    text = json.dumps(report, allow_nan=False)  # also proves there is no NaN/Infinity
    assert token not in text
    assert stored.token_hash not in text
    assert stored.client_hash not in text and "203.0.113.1" not in text
    assert stored.upload_key not in text
    assert str(runtime.settings.storage.resolved_dir()) not in text
    assert "path" not in report["video"]
    assert len(text) < 100_000  # summaries only, no frames or video bytes
    assert set(report) == set(Report.model_fields)


def test_job_ids_and_tokens_are_unique(api):
    _, client = api
    jobs = [create(client, ip=f"198.51.100.{i}").json() for i in range(5)]
    assert len({j["job_id"] for j in jobs}) == 5
    assert len({j["access_token"] for j in jobs}) == 5
