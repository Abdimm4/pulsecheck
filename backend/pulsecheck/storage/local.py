"""Local-filesystem storage backend for development and single-host use.

Files live in a private directory (default data/storage, ignored by Git)
that the API never serves. Uploads are written to `<key>.part` and renamed
into place only when complete, so a half-written file is never analysed.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pulsecheck.storage.base import (
    UPLOAD_PREFIX,
    InvalidKey,
    ObjectTooLarge,
    UploadTarget,
    check_key,
)

log = logging.getLogger("pulsecheck.storage")
PART_SUFFIX = ".part"


class LocalStorage:
    supports_direct_upload = True

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        (self.root / UPLOAD_PREFIX).mkdir(parents=True, exist_ok=True, mode=0o700)

    def _path(self, key: str) -> Path:
        path = (self.root / check_key(key)).resolve()
        if not path.is_relative_to(self.root):  # defence in depth; the key pattern forbids it
            raise InvalidKey("invalid storage key")
        return path

    def upload_target(
        self, key: str, job_id: str, expires_at: datetime, max_bytes: int
    ) -> UploadTarget:
        check_key(key)
        return UploadTarget(
            method="PUT",
            url=f"/v1/jobs/{job_id}/upload",
            expires_at=expires_at,
            max_bytes=max_bytes,
            headers={"Content-Type": "application/octet-stream"},
            requires_access_token=True,
        )

    async def receive(self, key: str, chunks: AsyncIterator[bytes], max_bytes: int) -> int:
        path = self._path(key)
        part = path.with_name(path.name + PART_SUFFIX)
        size = 0
        try:
            with part.open("wb") as handle:
                async for chunk in chunks:
                    size += len(chunk)
                    if size > max_bytes:
                        raise ObjectTooLarge(f"upload exceeds {max_bytes} bytes")
                    handle.write(chunk)
            os.replace(part, path)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        return size

    def size(self, key: str) -> int | None:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    @contextlib.contextmanager
    def local_copy(self, key: str) -> Iterator[Path]:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError("stored object not found")
        yield path

    def delete(self, key: str) -> bool:
        path = self._path(key)
        gone = True
        for candidate in (path, path.with_name(path.name + PART_SUFFIX)):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as exc:  # e.g. still open by a reader on Windows
                log.warning("could not delete a stored upload yet: %s", type(exc).__name__)
                gone = False
        return gone

    def stale_keys(self, older_than: datetime) -> Iterator[str]:
        cutoff = older_than.astimezone(UTC).timestamp()
        for path in (self.root / UPLOAD_PREFIX).iterdir():
            name = path.name.removesuffix(PART_SUFFIX)
            key = f"{UPLOAD_PREFIX}/{name}"
            try:
                check_key(key)
                stale = path.stat().st_mtime < cutoff
            except (InvalidKey, OSError):
                continue
            if stale:
                yield key
