"""
test_participation.py — Unit tests for participation receipt generation.

Tests: build_receipt structure, consent_summary mapping, Merkle root
determinism, and in-memory store/retrieve cycle.
"""

import os
import pytest

# Ensure no DB is used — in-memory fallback
os.environ.pop("DATABASE_URL", None)

from apps.gateway.participation import (
    ParticipationReceipt,
    build_receipt,
    store_receipt,
    get_receipt,
    list_receipts,
    _consent_to_summary,
    _build_qr_payload,
    _receipt_store,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _minimal_input(
    merkle_root: str = "abc123",
    session_id: str = "sess-001",
    turn_count: int = 3,
    node_count: int = 6,
    created_at: str = "2025-01-15T10:00:00",
    consent: dict = None,
) -> dict:
    return {
        "merkle_root": merkle_root,
        "session_id": session_id,
        "turn_count": turn_count,
        "node_count": node_count,
        "created_at": created_at,
        "consent": consent or {
            "transcript": "train",
            "felt_state": "denied",
            "gfs_activations": "improve_model",
            "training_signal": "sft_dpo",
            "retention": "6mo",
            "agreed_at": "2025-01-15T09:59:00",
        },
    }


@pytest.fixture(autouse=True)
def _clear_store():
    """Wipe in-memory receipt store before each test."""
    _receipt_store.clear()
    yield
    _receipt_store.clear()


# ──────────────────────────────────────────────────────────────────────────────
# build_receipt  (sync — no DB I/O)
# ──────────────────────────────────────────────────────────────────────────────

def test_build_receipt_returns_dataclass():
    r = build_receipt(_minimal_input())
    assert isinstance(r, ParticipationReceipt)


def test_build_receipt_fields_populated():
    inp = _minimal_input(merkle_root="deadbeef", session_id="s-42", turn_count=5, node_count=10)
    r = build_receipt(inp)
    assert r.merkle_root == "deadbeef"
    assert r.session_id == "s-42"
    assert r.turn_count == 5
    assert r.node_count == 10
    assert r.created_at == "2025-01-15T10:00:00"


def test_build_receipt_qr_payload_format():
    r = build_receipt(_minimal_input(merkle_root="aabbcc", session_id="s-99"))
    assert r.qr_payload.startswith("HAIC:v1:aabbcc:s-99:")


def test_build_receipt_to_dict_is_serializable():
    r = build_receipt(_minimal_input())
    d = r.to_dict()
    assert isinstance(d, dict)
    for field in ("merkle_root", "session_id", "turn_count", "node_count",
                  "consent_summary", "created_at", "qr_payload"):
        assert field in d


def test_build_receipt_defaults_zero_counts():
    """Missing turn_count / node_count should default to 0, not raise."""
    inp = {"merkle_root": "x", "session_id": "y", "created_at": ""}
    r = build_receipt(inp)
    assert r.turn_count == 0
    assert r.node_count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Consent summary  (sync helper)
# ──────────────────────────────────────────────────────────────────────────────

def test_consent_summary_known_values():
    consent = {
        "transcript": "train",
        "felt_state": "train_abstracted",
        "gfs_activations": "improve_model",
        "training_signal": "sft_dpo",
        "retention": "6mo",
        "agreed_at": "2025-01-01T00:00:00",
    }
    summary = _consent_to_summary(consent)
    assert summary["transcript"] == "Used verbatim for AI training"
    assert summary["felt_state"] == "Abstracted emotional signal allowed for training"
    assert summary["gfs_activations"] == "Grounding scores used for model improvement"
    assert summary["training_signal"] == "Session used for training signal (SFT/DPO)"
    assert summary["retention"] == "Deleted after 6 months"


def test_consent_summary_denied_values():
    consent = {
        "transcript": "denied",
        "felt_state": "denied",
        "gfs_activations": "denied",
        "training_signal": "denied",
        "retention": "2yr",
        "agreed_at": "",
    }
    summary = _consent_to_summary(consent)
    assert summary["transcript"] == "Not retained"
    assert summary["felt_state"] == "Not recorded"
    assert summary["gfs_activations"] == "Grounding scores not computed"
    assert summary["training_signal"] == "Not used for training signal"
    assert summary["retention"] == "Retained for up to 2 years"


def test_consent_summary_unknown_value_passthrough():
    """Unrecognised values are passed through as-is."""
    consent = {"transcript": "some_unknown_value"}
    summary = _consent_to_summary(consent)
    assert summary["transcript"] == "some_unknown_value"


def test_consent_summary_none_consent():
    summary = _consent_to_summary(None)
    assert "note" in summary


# ──────────────────────────────────────────────────────────────────────────────
# QR payload determinism  (sync helper)
# ──────────────────────────────────────────────────────────────────────────────

def test_qr_payload_deterministic():
    """Same inputs always produce the same QR payload."""
    p1 = _build_qr_payload("root123", "sessA", "2025-06-01T12:00:00")
    p2 = _build_qr_payload("root123", "sessA", "2025-06-01T12:00:00")
    assert p1 == p2


def test_qr_payload_differs_on_different_roots():
    p1 = _build_qr_payload("rootAAA", "sessA", "2025-06-01T12:00:00")
    p2 = _build_qr_payload("rootBBB", "sessA", "2025-06-01T12:00:00")
    assert p1 != p2


# ──────────────────────────────────────────────────────────────────────────────
# In-memory store / retrieve  (async)
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.anyio
async def test_store_and_get_receipt():
    r = build_receipt(_minimal_input(merkle_root="storetest"))
    await store_receipt(r)
    retrieved = await get_receipt("storetest")
    assert retrieved is not None
    assert retrieved.merkle_root == "storetest"
    assert retrieved.session_id == r.session_id


@pytest.mark.anyio
async def test_get_receipt_missing_returns_none():
    assert await get_receipt("nonexistent-root") is None


@pytest.mark.anyio
async def test_list_receipts_returns_stored():
    r1 = build_receipt(_minimal_input(merkle_root="r1", session_id="s1", created_at="2025-03-01T10:00:00"))
    r2 = build_receipt(_minimal_input(merkle_root="r2", session_id="s2", created_at="2025-04-01T10:00:00"))
    await store_receipt(r1)
    await store_receipt(r2)
    all_receipts = await list_receipts()
    roots = {r.merkle_root for r in all_receipts}
    assert "r1" in roots
    assert "r2" in roots


@pytest.mark.anyio
async def test_list_receipts_since_filter():
    r1 = build_receipt(_minimal_input(merkle_root="old", created_at="2025-01-01T00:00:00"))
    r2 = build_receipt(_minimal_input(merkle_root="new", created_at="2025-06-01T00:00:00"))
    await store_receipt(r1)
    await store_receipt(r2)
    filtered = await list_receipts(since="2025-03-01T00:00:00")
    roots = {r.merkle_root for r in filtered}
    assert "new" in roots
    assert "old" not in roots
