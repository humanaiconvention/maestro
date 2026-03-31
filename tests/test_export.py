"""
test_export.py — Tests for the /v1/export/kernels endpoint.

Tests: missing API key → 401, wrong API key → 403, valid key returns data.
Uses FastAPI TestClient and patches apps.gateway.export._EXPORT_API_KEY
so the lazy-loader returns a controlled value without touching os.environ.

Receipt data is seeded by writing directly into _receipt_store (bypassing
the async store_receipt to keep fixtures synchronous).
"""

import os
import pytest

# Use the same JWT secret as test_gateway.py to avoid cross-file env-var
# collisions (all test files are imported before any test function runs).
os.environ["MAESTRO_JWT_SECRET"] = "test-secret-key-for-testing"
os.environ["USE_MOCK_ADAPTER"] = "true"
os.environ["MAESTRO_LAUNCH_MODE"] = "private_full"

from fastapi.testclient import TestClient
import apps.gateway.export as _export_mod
from apps.gateway.main import app
from apps.gateway.participation import (
    ParticipationReceipt,
    _receipt_store,
)

client = TestClient(app)

VALID_KEY = "super-secret-export-key"


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _patch_export_key():
    """Override the module-level _EXPORT_API_KEY for each test."""
    original = _export_mod._EXPORT_API_KEY
    _export_mod._EXPORT_API_KEY = VALID_KEY
    yield
    _export_mod._EXPORT_API_KEY = original


@pytest.fixture(autouse=True)
def _clear_receipt_store():
    _receipt_store.clear()
    yield
    _receipt_store.clear()


def _seed_receipt(merkle_root: str = "mr-001", session_id: str = "s-001",
                  created_at: str = "2025-06-01T12:00:00") -> ParticipationReceipt:
    """Write a receipt directly into the in-memory store (sync, no DB)."""
    r = ParticipationReceipt(
        merkle_root=merkle_root,
        session_id=session_id,
        turn_count=2,
        node_count=4,
        consent_summary={"transcript": "Used verbatim for AI training"},
        created_at=created_at,
        qr_payload=f"HAIC:v1:{merkle_root}:{session_id}:1748779200",
        qr_data_url=None,
    )
    _receipt_store[merkle_root] = r
    return r


# ──────────────────────────────────────────────────────────────────────────────
# Auth failures
# ──────────────────────────────────────────────────────────────────────────────

def test_missing_api_key_returns_401():
    resp = client.get("/v1/export/kernels")
    assert resp.status_code == 401


def test_wrong_bearer_key_returns_403():
    resp = client.get(
        "/v1/export/kernels",
        headers={"Authorization": "Bearer wrong-key"},
    )
    assert resp.status_code == 403


def test_wrong_query_key_returns_403():
    resp = client.get("/v1/export/kernels", params={"key": "wrong-key"})
    assert resp.status_code == 403


def test_export_not_configured_returns_503():
    """When _EXPORT_API_KEY is empty, endpoint should 503."""
    _export_mod._EXPORT_API_KEY = ""
    resp = client.get("/v1/export/kernels")
    assert resp.status_code == 503
    _export_mod._EXPORT_API_KEY = VALID_KEY  # restore


# ──────────────────────────────────────────────────────────────────────────────
# Valid key — CSV format (default)
# ──────────────────────────────────────────────────────────────────────────────

def test_valid_bearer_returns_csv():
    _seed_receipt()
    resp = client.get(
        "/v1/export/kernels",
        headers={"Authorization": f"Bearer {VALID_KEY}"},
    )
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_valid_query_key_returns_csv():
    _seed_receipt()
    resp = client.get("/v1/export/kernels", params={"key": VALID_KEY})
    assert resp.status_code == 200
    assert "text/csv" in resp.headers["content-type"]


def test_csv_contains_header_row():
    _seed_receipt()
    resp = client.get("/v1/export/kernels", params={"key": VALID_KEY})
    assert "receipt_id" in resp.text
    assert "session_id" in resp.text
    assert "merkle_root" in resp.text


def test_csv_contains_receipt_data():
    _seed_receipt(merkle_root="csv-root-001")
    resp = client.get("/v1/export/kernels", params={"key": VALID_KEY})
    assert "csv-root-001" in resp.text


def test_csv_empty_store_returns_header_only():
    resp = client.get("/v1/export/kernels", params={"key": VALID_KEY})
    assert resp.status_code == 200
    lines = [l for l in resp.text.strip().splitlines() if l]
    assert len(lines) == 1  # header only


# ──────────────────────────────────────────────────────────────────────────────
# Valid key — JSON format
# ──────────────────────────────────────────────────────────────────────────────

def test_json_format_returns_json():
    _seed_receipt()
    resp = client.get(
        "/v1/export/kernels",
        params={"key": VALID_KEY, "format": "json"},
    )
    assert resp.status_code == 200
    assert "application/json" in resp.headers["content-type"]
    data = resp.json()
    assert "count" in data
    assert "kernels" in data


def test_json_format_count_matches_store():
    _seed_receipt(merkle_root="j1")
    _seed_receipt(merkle_root="j2")
    resp = client.get(
        "/v1/export/kernels",
        params={"key": VALID_KEY, "format": "json"},
    )
    data = resp.json()
    assert data["count"] == 2
    assert len(data["kernels"]) == 2


def test_json_kernel_has_required_fields():
    _seed_receipt(merkle_root="field-check")
    resp = client.get(
        "/v1/export/kernels",
        params={"key": VALID_KEY, "format": "json"},
    )
    kernel = resp.json()["kernels"][0]
    for field in ("receipt_id", "session_id", "merkle_root", "consent_level",
                  "timestamp", "payload_hash"):
        assert field in kernel, f"Missing field: {field}"


def test_payload_hash_is_sha256_hex():
    _seed_receipt()
    resp = client.get(
        "/v1/export/kernels",
        params={"key": VALID_KEY, "format": "json"},
    )
    h = resp.json()["kernels"][0]["payload_hash"]
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


# ──────────────────────────────────────────────────────────────────────────────
# since= filter
# ──────────────────────────────────────────────────────────────────────────────

def test_since_filter_excludes_old_receipts():
    _seed_receipt(merkle_root="old-r", created_at="2024-01-01T00:00:00")
    _seed_receipt(merkle_root="new-r", created_at="2025-06-01T00:00:00")

    resp = client.get(
        "/v1/export/kernels",
        params={"key": VALID_KEY, "format": "json", "since": "2025-01-01T00:00:00"},
    )
    roots = {k["merkle_root"] for k in resp.json()["kernels"]}
    assert "new-r" in roots
    assert "old-r" not in roots
