import pytest
from api_support import bearer, run_worker
from test_api_jobs import create, submit

from pulsecheck.jobs.auth import (
    bearer_token,
    hash_token,
    new_access_token,
    new_job_id,
    token_matches,
)


def test_token_helpers():
    token = new_access_token()
    assert len(token) >= 43
    assert token_matches(token, hash_token(token))
    assert not token_matches(token + "x", hash_token(token))
    assert not token_matches(None, hash_token(token))
    assert not token_matches("", hash_token(""))  # an empty token never matches
    assert not token_matches(token, None)
    assert bearer_token(f"Bearer {token}") == token
    assert bearer_token(f"bearer  {token}") == token
    for header in (None, "", "Bearer", "Bearer ", f"Basic {token}", f"Token {token}"):
        assert bearer_token(header) is None
    assert len(new_job_id()) == 32


@pytest.fixture
def two_jobs(api, video):
    runtime, client = api
    a = submit(client, video, ip="198.51.100.1")
    b = submit(client, video, ip="198.51.100.2")
    run_worker(runtime)
    return client, a[:2], b[:2]


ENDPOINTS = [
    ("get", "/v1/jobs/{id}"),
    ("get", "/v1/jobs/{id}/report"),
    ("post", "/v1/jobs/{id}/complete-upload"),
    ("put", "/v1/jobs/{id}/upload"),
    ("delete", "/v1/jobs/{id}"),
]


@pytest.mark.parametrize(("method", "path"), ENDPOINTS)
def test_missing_wrong_and_foreign_tokens_look_like_unknown_jobs(two_jobs, method, path):
    client, (job_a, token_a), (job_b, token_b) = two_jobs
    url = path.format(id=job_a)
    attempts = {
        "missing": client.request(method, url),
        "wrong": client.request(method, url, headers=bearer(token_a[:-2] + "xx")),
        "other job's token": client.request(method, url, headers=bearer(token_b)),
        "basic scheme": client.request(method, url, headers={"Authorization": f"Basic {token_a}"}),
        "token in query": client.request(method, f"{url}?access_token={token_a}"),
        "unknown job": client.request(method, path.format(id="0" * 32), headers=bearer(token_a)),
        "malformed id": client.request(
            method, path.format(id="Not-A-Job"), headers=bearer(token_a)
        ),
    }
    bodies = set()
    for name, response in attempts.items():
        assert response.status_code == 404, (name, response.text)
        bodies.add(response.text)
    assert len(bodies) == 1  # indistinguishable
    assert "job_not_found" in bodies.pop()
    # The owner still has access and nothing was changed by the failed attempts.
    own = client.get(f"/v1/jobs/{job_a}", headers=bearer(token_a))
    assert own.status_code == 200 and own.json()["status"] == "completed"


def test_each_token_opens_only_its_own_job(two_jobs):
    client, (job_a, token_a), (job_b, token_b) = two_jobs
    assert client.get(f"/v1/jobs/{job_a}", headers=bearer(token_a)).json()["job_id"] == job_a
    assert client.get(f"/v1/jobs/{job_b}", headers=bearer(token_b)).json()["job_id"] == job_b
    report = client.get(f"/v1/jobs/{job_b}/report", headers=bearer(token_b)).json()
    assert report["job"]["id"] == job_b


def test_creation_response_is_the_only_place_the_token_appears(api, video):
    _, client = api
    created = create(client).json()
    token = created["access_token"]
    status = client.get(f"/v1/jobs/{created['job_id']}", headers=bearer(token))
    assert token not in status.text
    assert "access_token" not in status.json()
