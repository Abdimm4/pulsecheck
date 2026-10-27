"""Job identifiers, access tokens and client hashes (PRD Section 9, UP-5).

A job ID is an opaque handle and grants nothing on its own. Each job gets a
separate access token (32 random bytes, URL-safe base64) that is returned
once at creation and stored only as its SHA-256 hash; requests present it in
an `Authorization: Bearer` header. A fast unsalted hash is enough here
because the token is high-entropy random data, not a password.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets

TOKEN_BYTES = 32
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
# Compared against when the job does not exist, so unknown jobs and wrong
# tokens take the same code path.
_DUMMY_HASH = hashlib.sha256(b"pulsecheck-no-such-job").hexdigest()


def new_job_id() -> str:
    return secrets.token_hex(16)


def new_access_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def token_matches(token: str | None, stored_hash: str | None) -> bool:
    """Constant-time check of a presented token against a stored hash."""
    presented = hash_token(token or "")
    ok = hmac.compare_digest(presented, stored_hash or _DUMMY_HASH)
    return ok and bool(token) and stored_hash is not None


def bearer_token(header: str | None) -> str | None:
    """The token from an `Authorization: Bearer <token>` header, else None."""
    if not header:
        return None
    scheme, _, value = header.partition(" ")
    value = value.strip()
    if scheme.lower() != "bearer" or not value or len(value) > 512:
        return None
    return value


def client_hash(client_ip: str, secret_key: str) -> str:
    """Keyed hash of the client IP: lets limits work without storing the address."""
    return hmac.new(secret_key.encode(), client_ip.encode(), hashlib.sha256).hexdigest()
