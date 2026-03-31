"""
test_auth.py — Unit tests for JWT auth (apps.gateway.auth).

Tests: token generation/decode, expired token rejection, wrong secret
rejection, missing required claims rejection.
"""

import os
import time
import pytest
import jwt

# Use the same secret as test_gateway.py to avoid cross-file env-var collisions.
os.environ["MAESTRO_JWT_SECRET"] = "test-secret-key-for-testing"
os.environ["USE_MOCK_ADAPTER"] = "true"
os.environ["MAESTRO_LAUNCH_MODE"] = "private_full"
os.environ["RATE_LIMIT_RPM"] = "20"  # match test_gateway.py so the limiter is consistent

from apps.gateway.auth import ALGORITHM, verify_auth
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


SECRET = "test-secret-key-for-testing"


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_token(claims: dict, secret: str = SECRET) -> str:
    return jwt.encode(claims, secret, algorithm=ALGORITHM)


def _make_credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


# ──────────────────────────────────────────────────────────────────────────────
# Happy-path
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_valid_token_returns_context():
    token = _make_token({"sub": "user-1", "tenant_id": "tenant-A"})
    ctx = await verify_auth(_make_credentials(token))
    assert ctx["user_id"] == "user-1"
    assert ctx["tenant_id"] == "tenant-A"


@pytest.mark.anyio
async def test_valid_token_with_extra_claims():
    """Extra claims (iat, exp, session_id) don't cause failures."""
    now = int(time.time())
    token = _make_token({
        "sub": "user-2",
        "tenant_id": "tenant-B",
        "session_id": "sess-xyz",
        "iat": now,
        "exp": now + 3600,
    })
    ctx = await verify_auth(_make_credentials(token))
    assert ctx["user_id"] == "user-2"
    assert ctx["tenant_id"] == "tenant-B"


# ──────────────────────────────────────────────────────────────────────────────
# Expiry
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_expired_token_rejected():
    past = int(time.time()) - 3600
    token = _make_token({"sub": "user-3", "tenant_id": "t", "exp": past})
    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(_make_credentials(token))
    assert exc_info.value.status_code == 401


# ──────────────────────────────────────────────────────────────────────────────
# Wrong secret
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_wrong_secret_rejected():
    token = _make_token({"sub": "user-4", "tenant_id": "t"}, secret="wrong-secret")
    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(_make_credentials(token))
    assert exc_info.value.status_code == 401


# ──────────────────────────────────────────────────────────────────────────────
# Missing required claims
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_missing_sub_rejected():
    token = _make_token({"tenant_id": "t"})
    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(_make_credentials(token))
    assert exc_info.value.status_code == 401
    assert "sub" in exc_info.value.detail or "user_id" in exc_info.value.detail


@pytest.mark.anyio
async def test_missing_tenant_id_rejected():
    token = _make_token({"sub": "user-5"})
    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(_make_credentials(token))
    assert exc_info.value.status_code == 401
    assert "tenant_id" in exc_info.value.detail


@pytest.mark.anyio
async def test_completely_empty_claims_rejected():
    token = _make_token({})
    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(_make_credentials(token))
    assert exc_info.value.status_code == 401


# ──────────────────────────────────────────────────────────────────────────────
# Malformed token string
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_garbage_token_rejected():
    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(_make_credentials("this-is-not-a-jwt"))
    assert exc_info.value.status_code == 401


@pytest.mark.anyio
async def test_empty_token_rejected():
    with pytest.raises(HTTPException) as exc_info:
        await verify_auth(_make_credentials(""))
    assert exc_info.value.status_code == 401


# ──────────────────────────────────────────────────────────────────────────────
# Endpoint-level smoke test via TestClient
# ──────────────────────────────────────────────────────────────────────────────

def test_chat_no_auth_returns_401_or_403():
    from fastapi.testclient import TestClient
    from apps.gateway.main import app
    client = TestClient(app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "maestro-default", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code in (401, 403)


def test_chat_with_valid_token_passes_auth():
    from fastapi.testclient import TestClient
    from apps.gateway.main import app
    client = TestClient(app)
    token = _make_token({"sub": "u-auth-test", "tenant_id": "t-auth-test"})
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "maestro-default", "messages": [{"role": "user", "content": "hi"}]},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200
