import os
import time
from datetime import timedelta

import pytest
from api_support import StubAnalyzer, bearer, run_worker, stored_files
from rq.timeouts import JobTimeoutException
from test_api_jobs import create, submit

from pulsecheck.jobs.cleanup import run_cleanup
from pulsecheck.jobs.models import (
    ALLOWED_TRANSITIONS,
    InvalidTransition,
    JobStatus,
    check_transition,
)
from pulsecheck.jobs.pipeline import AnalysisUnavailable
from pulsecheck.jobs.queue import QueueUnavailable
from pulsecheck.jobs.store import ConcurrentUpdate
from pulsecheck.storage import new_upload_key

S = JobStatus


def status_of(client, job_id, token):
    return client.get(f"/v1/jobs/{job_id}", headers=bearer(token)).json()


# --- state machine ---------------------------------------------------------


def test_transition_table():
    check_transition(S.CREATED, S.UPLOADING)
    check_transition(S.UPLOADING, S.QUEUED)
    check_transition(S.QUEUED, S.PROCESSING)
    check_transition(S.PROCESSING, S.COMPLETED)
    check_transition(S.PROCESSING, S.FAILED)
    for status in S:
        if status != S.EXPIRED:
            check_transition(status, S.EXPIRED)
    forbidden = [(a, b) for a in S for b in S if b not in ALLOWED_TRANSITIONS[a]]
    assert (S.CREATED, S.QUEUED) in forbidden  # no queueing without an upload
    assert (S.COMPLETED, S.PROCESSING) in forbidden
    assert (S.EXPIRED, S.COMPLETED) in forbidden
    for a, b in forbidden:
        with pytest.raises(InvalidTransition):
            check_transition(a, b)


def test_store_rejects_invalid_and_stale_transitions(api):
    runtime, client = api
    job_id = create(client).json()["job_id"]
    with pytest.raises(InvalidTransition):
        runtime.store.transition(job_id, S.CREATED, S.COMPLETED)
    with pytest.raises(ConcurrentUpdate):  # the job is not in the status the caller assumed
        runtime.store.transition(job_id, S.UPLOADING, S.QUEUED)
    with pytest.raises(ValueError):
        runtime.store.update(job_id, status="completed")
    assert runtime.store.get(job_id).status == "created"


def test_out_of_order_requests_are_conflicts(api, video):
    runtime, client = api
    created = create(client).json()
    job_id, token = created["job_id"], created["access_token"]
    url = f"/v1/jobs/{job_id}"
    early = client.post(f"{url}/complete-upload", headers=bearer(token))
    assert early.status_code == 409 and early.json()["error"]["code"] == "upload_missing"
    report = client.get(f"{url}/report", headers=bearer(token))
    assert report.status_code == 409 and report.json()["error"]["code"] == "report_not_ready"

    assert client.put(f"{url}/upload", content=video.read_bytes(), headers=bearer(token)).is_success
    again = client.put(f"{url}/upload", content=video.read_bytes(), headers=bearer(token))
    assert again.status_code == 409 and again.json()["error"]["code"] == "upload_not_allowed"
    assert client.post(f"{url}/complete-upload", headers=bearer(token)).status_code == 202
    twice = client.post(f"{url}/complete-upload", headers=bearer(token))
    assert twice.status_code == 409
    assert twice.json()["error"]["code"] == "upload_already_completed"
    run_worker(runtime)
    late = client.put(f"{url}/upload", content=b"x", headers=bearer(token))
    assert late.status_code == 409
    assert status_of(client, job_id, token)["status"] == "completed"


def test_presigned_style_upload_goes_created_to_queued(api, video):
    """A backend upload that bypasses PUT (as presigned URLs will) is picked up on complete."""
    runtime, client = api
    created = create(client).json()
    key = runtime.store.get(created["job_id"]).upload_key
    path = runtime.settings.storage.resolved_dir() / key
    path.write_bytes(video.read_bytes())
    done = client.post(
        f"/v1/jobs/{created['job_id']}/complete-upload", headers=bearer(created["access_token"])
    )
    assert done.status_code == 202 and done.json()["status"] == "queued"


# --- capacity and rate limits ------------------------------------------------


def test_queue_full_returns_service_busy(make_api, video):
    runtime, client = make_api(capacity={"max_queue_length": 1})
    submit(client, video, ip="198.51.100.1")
    assert runtime.queue.depth() == 1
    busy = create(client, ip="198.51.100.2")
    assert busy.status_code == 503
    assert busy.json()["error"]["code"] == "service_busy"
    assert busy.headers["retry-after"] == "60"
    run_worker(runtime)
    assert create(client, ip="198.51.100.2").status_code == 201


def test_queue_full_at_complete_keeps_the_upload(make_api, video):
    runtime, client = make_api(capacity={"max_queue_length": 1})
    first = create(client, ip="198.51.100.1").json()
    second = create(client, ip="198.51.100.2").json()
    for job in (first, second):
        put = client.put(
            f"/v1/jobs/{job['job_id']}/upload",
            content=video.read_bytes(),
            headers=bearer(job["access_token"]),
        )
        assert put.is_success
    url = "/v1/jobs/{}/complete-upload"
    ok = client.post(url.format(first["job_id"]), headers=bearer(first["access_token"]))
    assert ok.status_code == 202
    busy = client.post(url.format(second["job_id"]), headers=bearer(second["access_token"]))
    assert busy.status_code == 503
    assert status_of(client, second["job_id"], second["access_token"])["status"] == "uploading"
    run_worker(runtime)
    retry = client.post(url.format(second["job_id"]), headers=bearer(second["access_token"]))
    assert retry.status_code == 202


def test_global_active_job_cap(make_api):
    _, client = make_api(capacity={"max_active_jobs": 2})
    assert create(client, ip="198.51.100.1").status_code == 201
    assert create(client, ip="198.51.100.2").status_code == 201
    busy = create(client, ip="198.51.100.3")
    assert busy.status_code == 503 and busy.json()["error"]["code"] == "service_busy"


def test_per_client_active_job_cap(make_api, video):
    runtime, client = make_api(capacity={"max_active_jobs_per_client": 1})
    job_id, token, _ = submit(client, video, ip="198.51.100.1")
    second = create(client, ip="198.51.100.1")
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "too_many_active_jobs"
    assert create(client, ip="198.51.100.9").status_code == 201  # other visitors unaffected
    run_worker(runtime)  # the first job finishes, freeing the slot
    assert create(client, ip="198.51.100.1").status_code == 201


def test_job_creation_rate_limit(make_api):
    _, client = make_api(capacity={"job_creations_per_window": 2, "max_active_jobs_per_client": 10})
    assert create(client, ip="198.51.100.1").status_code == 201
    assert create(client, ip="198.51.100.1").status_code == 201
    limited = create(client, ip="198.51.100.1")
    assert limited.status_code == 429
    assert limited.json()["error"]["code"] == "rate_limited"
    assert 1 <= int(limited.headers["retry-after"]) <= 3600
    assert create(client, ip="198.51.100.2").status_code == 201


def test_rate_limit_ignores_client_supplied_forwarded_entries(make_api):
    _, client = make_api(capacity={"job_creations_per_window": 1, "max_active_jobs_per_client": 10})
    assert create(client, ip="10.0.0.1, 198.51.100.1").status_code == 201
    # A forged leading entry does not make this a new client; the proxy's last entry decides.
    assert create(client, ip="10.0.0.2, 198.51.100.1").status_code == 429


def test_queue_unavailable_fails_job_cleanly(api, video, monkeypatch):
    runtime, client = api

    def broken(job_id):
        raise QueueUnavailable("ConnectionError")

    monkeypatch.setattr(runtime.queue, "enqueue", broken)
    job_id, token, done = submit(client, video)
    assert done.status_code == 503
    status = status_of(client, job_id, token)
    assert status["status"] == "failed" and status["error"]["code"] == "queue_unavailable"
    assert stored_files(runtime) == []


def test_redis_down_is_service_busy(api, monkeypatch):
    runtime, client = api
    monkeypatch.setattr(
        runtime.queue, "depth", lambda: (_ for _ in ()).throw(QueueUnavailable("x"))
    )
    assert create(client).status_code == 503
    health = client.get("/health")
    assert health.status_code == 503 and health.json()["queue"] == "unavailable"


# --- worker outcomes -----------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RuntimeError("boom in C:\\secret\\path"), "pipeline_error"),
        (AnalysisUnavailable("landmarker model missing at C:\\models"), "analysis_unavailable"),
        (JobTimeoutException("Task exceeded maximum timeout value (300 seconds)"), "timed_out"),
    ],
)
def test_worker_failures_are_recorded_safely(make_api, video, error, code):
    runtime, client = make_api(analyzer=StubAnalyzer(error=error))
    job_id, token, _ = submit(client, video)
    run_worker(runtime)
    status = status_of(client, job_id, token)
    assert status["status"] == "failed"
    assert status["error"]["code"] == code
    assert "secret" not in status["error"]["message"] and "C:\\" not in status["error"]["message"]
    assert status["result"] is None and status["report_url"] is None
    assert status["deletion_status"] == "upload_deleted"
    assert stored_files(runtime) == []
    report = client.get(f"/v1/jobs/{job_id}/report", headers=bearer(token))
    assert report.status_code == 409 and report.json()["error"]["code"] == "report_unavailable"


def test_upload_deleted_before_processing(api, video):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    for path in stored_files(runtime):
        path.unlink()
    run_worker(runtime)
    status = status_of(client, job_id, token)
    assert status["status"] == "failed" and status["error"]["code"] == "upload_missing"


def test_worker_skips_jobs_deleted_while_queued(api, video, analyzer):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    assert client.delete(f"/v1/jobs/{job_id}", headers=bearer(token)).status_code == 204
    run_worker(runtime)
    assert analyzer.calls == []
    assert status_of(client, job_id, token)["status"] == "expired"


def test_result_discarded_if_job_deleted_while_processing(api, video, analyzer):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    original = analyzer.__call__

    class DeletingAnalyzer:
        model_name, model_version = analyzer.model_name, analyzer.model_version

        def __call__(self, path):
            output = original(path)
            client.delete(f"/v1/jobs/{job_id}", headers=bearer(token))
            return output

    runtime._analyzer = DeletingAnalyzer()
    run_worker(runtime)
    status = status_of(client, job_id, token)
    assert status["status"] == "expired" and status["result"] is None
    assert runtime.store.get(job_id).report is None
    assert stored_files(runtime) == []


# --- deletion -----------------------------------------------------------------


def test_delete_removes_upload_and_results(api, video):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    run_worker(runtime)
    url = f"/v1/jobs/{job_id}"
    assert client.get(f"{url}/report", headers=bearer(token)).status_code == 200
    deleted = client.delete(url, headers=bearer(token))
    assert deleted.status_code == 204 and deleted.content == b""
    status = status_of(client, job_id, token)
    assert status["status"] == "expired"
    assert status["expiry_reason"] == "deleted_on_request"
    assert status["deletion_status"] == "all_deleted"
    assert status["result"] is None
    stored = runtime.store.get(job_id)
    assert stored.result is None and stored.report is None and stored.client_hash is None
    gone = client.get(f"{url}/report", headers=bearer(token))
    assert gone.status_code == 410 and gone.json()["error"]["code"] == "job_expired"
    assert client.delete(url, headers=bearer(token)).status_code == 204  # idempotent


def test_delete_before_analysis_removes_the_file(api, video):
    runtime, client = api
    created = create(client).json()
    job_id, token = created["job_id"], created["access_token"]
    client.put(f"/v1/jobs/{job_id}/upload", content=video.read_bytes(), headers=bearer(token))
    assert len(stored_files(runtime)) == 1
    assert client.delete(f"/v1/jobs/{job_id}", headers=bearer("wrong")).status_code == 404
    assert len(stored_files(runtime)) == 1  # a failed attempt deletes nothing
    assert client.delete(f"/v1/jobs/{job_id}", headers=bearer(token)).status_code == 204
    assert stored_files(runtime) == []
    after = client.post(f"/v1/jobs/{job_id}/complete-upload", headers=bearer(token))
    assert after.status_code == 409 and after.json()["error"]["code"] == "job_expired"


# --- expiry and cleanup -------------------------------------------------------


def test_completed_job_expires_after_retention(api, video, clock):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    run_worker(runtime)
    clock.advance(hours=23, minutes=59)
    assert run_cleanup(runtime).expired == 0
    assert status_of(client, job_id, token)["status"] == "completed"
    clock.advance(minutes=2)
    summary = run_cleanup(runtime)
    assert summary.expired == 1
    status = status_of(client, job_id, token)
    assert status["status"] == "expired"
    assert status["expiry_reason"] == "retention_elapsed"
    assert status["deletion_status"] == "all_deleted"
    assert status["result"] is None and status["model"] is not None
    assert client.get(f"/v1/jobs/{job_id}/report", headers=bearer(token)).status_code == 410
    stored = runtime.store.get(job_id)
    assert stored.report is None and stored.result is None and stored.client_hash is None


def test_expiry_is_enforced_on_read_without_cleanup(api, video, clock):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    run_worker(runtime)
    clock.advance(hours=25)
    report = client.get(f"/v1/jobs/{job_id}/report", headers=bearer(token))
    assert report.status_code == 410
    assert runtime.store.get(job_id).report is None


def test_failed_job_expires_and_its_upload_is_gone_at_once(api, clock):
    runtime, client = api
    created = create(client).json()
    job_id, token = created["job_id"], created["access_token"]
    client.put(f"/v1/jobs/{job_id}/upload", content=b"not a video", headers=bearer(token))
    client.post(f"/v1/jobs/{job_id}/complete-upload", headers=bearer(token))
    assert status_of(client, job_id, token)["status"] == "failed"
    assert stored_files(runtime) == []
    clock.advance(hours=25)
    assert run_cleanup(runtime).expired == 1
    status = status_of(client, job_id, token)
    assert status["status"] == "expired" and status["error"] is None


def test_abandoned_uploads_expire_after_an_hour(api, video, clock):
    runtime, client = api
    never = create(client, ip="198.51.100.1").json()
    partial = create(client, ip="198.51.100.2").json()
    client.put(
        f"/v1/jobs/{partial['job_id']}/upload",
        content=video.read_bytes(),
        headers=bearer(partial["access_token"]),
    )
    assert len(stored_files(runtime)) == 1
    clock.advance(minutes=59)
    assert run_cleanup(runtime).abandoned == 0
    clock.advance(minutes=2)
    assert run_cleanup(runtime).abandoned == 2
    assert stored_files(runtime) == []
    for job in (never, partial):
        status = status_of(client, job["job_id"], job["access_token"])
        assert status["status"] == "expired" and status["expiry_reason"] == "abandoned"
        assert status["deletion_status"] == "all_deleted"
    # An abandoned upload cannot be completed later.
    late = client.post(
        f"/v1/jobs/{partial['job_id']}/complete-upload", headers=bearer(partial["access_token"])
    )
    assert late.status_code == 409


def test_stuck_processing_job_times_out(api, video, clock):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    runtime.store.transition(job_id, S.QUEUED, S.PROCESSING, started_at=clock())
    clock.advance(seconds=runtime.settings.capacity.job_timeout_s + 60)
    assert run_cleanup(runtime).timed_out == 0
    clock.advance(minutes=2)
    assert run_cleanup(runtime).timed_out == 1
    status = status_of(client, job_id, token)
    assert status["status"] == "failed" and status["error"]["code"] == "timed_out"
    assert stored_files(runtime) == []


def test_cleanup_retries_failed_upload_deletes(api, video, monkeypatch):
    runtime, client = api
    real_delete = runtime.storage.delete
    monkeypatch.setattr(runtime.storage, "delete", lambda key: False)  # e.g. file locked
    job_id, token, _ = submit(client, video)
    run_worker(runtime)
    assert status_of(client, job_id, token)["deletion_status"] == "upload_delete_pending"
    assert len(stored_files(runtime)) == 1
    monkeypatch.setattr(runtime.storage, "delete", real_delete)
    assert run_cleanup(runtime).uploads_deleted == 1
    assert status_of(client, job_id, token)["deletion_status"] == "upload_deleted"
    assert stored_files(runtime) == []


def test_cleanup_deletes_old_orphan_files_only(api, clock):
    runtime, client = api
    root = runtime.settings.storage.resolved_dir()
    old, recent = root / new_upload_key(), root / new_upload_key()
    old.write_bytes(b"x")
    recent.write_bytes(b"y")
    (root / (new_upload_key() + ".part")).write_bytes(b"z")
    stamp = time.time() - 2 * 3600
    for path in root.joinpath("uploads").iterdir():
        if path != recent:
            os.utime(path, (stamp, stamp))
    assert run_cleanup(runtime).orphans_deleted == 2
    assert stored_files(runtime) == [recent]


def test_expired_records_are_purged_after_the_record_period(api, video, clock):
    runtime, client = api
    job_id, token, _ = submit(client, video)
    run_worker(runtime)
    clock.advance(hours=25)
    run_cleanup(runtime)
    clock.advance(hours=runtime.settings.retention.expired_record_hours - 1)
    assert run_cleanup(runtime).purged == 0
    assert status_of(client, job_id, token)["status"] == "expired"
    clock.advance(hours=2)
    assert run_cleanup(runtime).purged == 1
    assert runtime.store.get(job_id) is None
    gone = client.get(f"/v1/jobs/{job_id}", headers=bearer(token))
    assert gone.status_code == 404


def test_cleanup_is_idempotent(api, video, clock):
    runtime, client = api
    submit(client, video)
    run_worker(runtime)
    clock.advance(hours=25)
    first = run_cleanup(runtime)
    assert first.expired == 1
    assert run_cleanup(runtime).to_dict() == dict.fromkeys(first.to_dict(), 0)


def test_expiry_times_follow_the_config(make_api, clock):
    runtime, client = make_api(retention={"abandoned_upload_hours": 0.5, "job_metadata_hours": 2})
    created = create(client).json()
    assert created["expires_at"].startswith((clock() + timedelta(hours=2)).strftime("%Y-%m-%dT%H"))
    assert created["upload"]["expires_at"] < created["expires_at"]
