import struct

import pytest
from api_support import bearer, stored_files, write_video
from test_api_jobs import create

from pulsecheck.jobs.validation import (
    UploadLimits,
    UploadRejected,
    check_metadata,
    sniff_container,
)
from pulsecheck.video import probe_video


def upload(client, data: bytes, **headers):
    created = create(client).json()
    job_id, token = created["job_id"], created["access_token"]
    put = client.put(
        f"/v1/jobs/{job_id}/upload", content=data, headers={**bearer(token), **headers}
    )
    return job_id, token, put


def upload_and_complete(client, data: bytes):
    job_id, token, put = upload(client, data)
    assert put.status_code == 200, put.text
    done = client.post(f"/v1/jobs/{job_id}/complete-upload", headers=bearer(token))
    status = client.get(f"/v1/jobs/{job_id}", headers=bearer(token)).json()
    return done, status


def assert_rejected(runtime, done, status, code, http=422):
    assert done.status_code == http, done.text
    assert done.json()["error"]["code"] == code
    assert status["status"] == "failed"
    assert status["error"]["code"] == code
    assert status["deletion_status"] == "upload_deleted"
    assert stored_files(runtime) == []
    assert runtime.queue.depth() == 0


def test_sniff_container(tmp_path):
    def box(kind: bytes, brand: bytes = b"") -> bytes:
        return struct.pack(">I", 8 + len(brand)) + kind + brand

    cases = {
        box(b"ftyp", b"isom0000"): "mp4",
        box(b"ftyp", b"mp42"): "mp4",
        box(b"ftyp", b"qt  "): "mov",
        box(b"moov"): "mov",
        box(b"wide") + box(b"mdat"): "mov",
        b"RIFF\x00\x00\x00\x00AVI LIST": None,
        b"\x1aE\xdf\xa3 matroska/webm": None,
        b"GIF89a....": None,
        b"abc": None,
    }
    for i, (data, expected) in enumerate(cases.items()):
        path = tmp_path / f"f{i}.bin"
        path.write_bytes(data)
        assert sniff_container(path) == expected, data


def test_real_mp4_and_mov_are_sniffed(tmp_path):
    assert sniff_container(write_video(tmp_path / "a.mp4")) == "mp4"
    assert sniff_container(write_video(tmp_path / "a.mov")) == "mov"


def test_mov_upload_is_accepted(api, tmp_path):
    runtime, client = api
    done, status = upload_and_complete(client, write_video(tmp_path / "a.mov").read_bytes())
    assert done.status_code == 202, done.text
    assert status["status"] == "queued"
    assert runtime.store.get(status["job_id"]).container == "mov"


def test_extension_and_mime_type_are_ignored(api, tmp_path):
    runtime, client = api
    avi = write_video(tmp_path / "clip.avi").read_bytes()  # a real video, wrong container
    job_id, token, put = upload(client, avi, **{"Content-Type": "video/mp4"})
    done = client.post(f"/v1/jobs/{job_id}/complete-upload", headers=bearer(token))
    status = client.get(f"/v1/jobs/{job_id}", headers=bearer(token)).json()
    assert_rejected(runtime, done, status, "unsupported_format")


def test_disallowed_format_is_rejected(make_api, tmp_path):
    runtime, client = make_api(upload={"allowed_formats": ["mp4"]})
    done, status = upload_and_complete(client, write_video(tmp_path / "a.mov").read_bytes())
    assert_rejected(runtime, done, status, "unsupported_format")
    assert "MP4" in done.json()["error"]["message"]


@pytest.mark.parametrize(
    ("data", "code"),
    [
        (b"", "empty_file"),
        (b"this is plain text, not a video" * 50, "unsupported_format"),
        (struct.pack(">I", 16) + b"ftypisom" + b"\x00" * 4 + b"\x13" * 4000, "corrupt_video"),
    ],
)
def test_invalid_uploads_fail_the_job_and_delete_the_file(api, data, code):
    runtime, client = api
    done, status = upload_and_complete(client, data)
    assert_rejected(runtime, done, status, code)


def test_truncated_video_is_corrupt(api, video):
    runtime, client = api
    done, status = upload_and_complete(client, video.read_bytes()[: video.stat().st_size // 2])
    assert_rejected(runtime, done, status, "corrupt_video")


def test_too_long_video(make_api, tmp_path):
    runtime, client = make_api(upload={"max_duration_s": 1})
    done, status = upload_and_complete(
        client, write_video(tmp_path / "long.mp4", n_frames=60).read_bytes()
    )
    assert_rejected(runtime, done, status, "video_too_long")


def test_too_many_frames(make_api, video):
    runtime, client = make_api(upload={"max_frames": 10})
    done, status = upload_and_complete(client, video.read_bytes())
    assert_rejected(runtime, done, status, "too_many_frames")


def test_resolution_too_high(make_api, video):
    runtime, client = make_api(upload={"max_resolution": [32, 24]})
    done, status = upload_and_complete(client, video.read_bytes())
    assert_rejected(runtime, done, status, "resolution_too_high")
    assert "64x48" in done.json()["error"]["message"]


def test_oversize_stream_is_cut_off(make_api, video):
    runtime, client = make_api(upload={"max_size_mb": 0.001})  # 1000 bytes
    assert video.stat().st_size > 1000

    def chunks():  # no Content-Length: the stream itself must be capped
        data = video.read_bytes()
        for i in range(0, len(data), 256):
            yield data[i : i + 256]

    created = create(client).json()
    job_id, token = created["job_id"], created["access_token"]
    put = client.put(f"/v1/jobs/{job_id}/upload", content=chunks(), headers=bearer(token))
    assert put.status_code == 413
    assert put.json()["error"]["code"] == "file_too_large"
    status = client.get(f"/v1/jobs/{job_id}", headers=bearer(token)).json()
    assert status["status"] == "failed" and status["error"]["code"] == "file_too_large"
    assert stored_files(runtime) == []  # the partial file is gone too


def test_oversize_content_length_is_rejected_before_streaming(make_api, video):
    runtime, client = make_api(upload={"max_size_mb": 0.001})
    job_id, token, put = upload(client, video.read_bytes())
    assert put.status_code == 413
    assert stored_files(runtime) == []
    assert runtime.store.get(job_id).status == "failed"


def test_oversize_declared_at_creation(api):
    _, client = api
    response = create(client, size_bytes=100_000_001)
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "file_too_large"


def test_check_metadata_uses_measured_values(video):
    meta = probe_video(video)
    check_metadata(meta, UploadLimits())
    meta.duration_s = 61.0  # e.g. a header that understated the duration
    with pytest.raises(UploadRejected) as err:
        check_metadata(meta, UploadLimits())
    assert err.value.code == "video_too_long"


def test_bounded_probe_stops_at_the_frame_limit(video):
    meta = probe_video(video, max_frames=11)
    assert meta.decoded_frame_count == 11
    with pytest.raises(UploadRejected) as err:
        check_metadata(meta, UploadLimits(max_frames=10))
    assert err.value.code == "too_many_frames"


def test_upload_limits_config_validation():
    assert UploadLimits.from_mapping({"max_size_mb": 5}).max_bytes == 5_000_000
    with pytest.raises(ValueError, match="unknown"):
        UploadLimits.from_mapping({"max_size": 5})
    with pytest.raises(ValueError, match="subset"):
        UploadLimits(allowed_formats=("avi",))
    with pytest.raises(ValueError, match="long side"):
        UploadLimits(max_resolution=(720, 1280))
