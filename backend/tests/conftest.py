import pytest
from api_support import FakeClock, StubAnalyzer, make_runtime, write_video

from pulsecheck.jobs import tasks


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def analyzer():
    return StubAnalyzer()


@pytest.fixture
def make_api(tmp_path, clock, analyzer):
    """Factory: (runtime, TestClient) with config sections overridden by keyword."""
    from fastapi.testclient import TestClient

    from pulsecheck.api.app import create_app

    clients = []

    def factory(analyzer=analyzer, **sections):
        runtime = make_runtime(tmp_path, analyzer=analyzer, clock=clock, **sections)
        client = TestClient(create_app(runtime), raise_server_exceptions=False)
        clients.append(client)
        return runtime, client

    yield factory
    for client in clients:
        client.close()
    tasks.set_runtime(None)


@pytest.fixture
def api(make_api):
    return make_api()


@pytest.fixture
def video(tmp_path):
    return write_video(tmp_path / "clip.mp4")
