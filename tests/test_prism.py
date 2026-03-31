"""
test_prism.py — Tests for PRISM API endpoints.

Covers:
  - GET /v1/prism/status         — tool list, model track, record count
  - GET /v1/prism/runs           — empty state, normalized records, required fields
  - GET /v1/prism/runs/summary   — lightweight PrismRunSummary shapes
  - GET /v1/prism/runs/{run_id}  — lookup by logical id, by filename stem, 404
  - Path traversal guard          — invalid/overlong run_id → 404
  - Normalization                 — records with missing fields get required defaults
  - Rate limiting                 — 21st request to /v1/prism/runs → 429
"""

import itertools
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import jwt
import pytest
from fastapi.testclient import TestClient

# env must be set before app import
os.environ.setdefault("MAESTRO_JWT_SECRET", "test-secret-key-for-testing")
os.environ.setdefault("USE_MOCK_ADAPTER", "true")
os.environ.setdefault("MAESTRO_LAUNCH_MODE", "private_full")
os.environ.setdefault("RATE_LIMIT_RPM", "20")
os.environ.setdefault("TRUST_PROXY_HEADERS", "true")

from apps.gateway.main import app  # noqa: E402

client = TestClient(app)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_MINIMAL_RECORD = {
    "id": "run-abc-123",
    "base_model_ref": "example-model",
    "run_kind": "sft",
    "status": "completed",
    "source_axis": "emotional_grounding",
    "updated_at": "2026-03-28T08:00:00Z",
}

_FULL_RECORD = {
    **_MINIMAL_RECORD,
    "id": "run-full-001",
    "title": "v7 grounding run",
    "objective": "Improve hedging handling",
    "hypothesis_or_goal": "Constrained mode echoes feeling words",
    "model_or_agent_version": "v7",
    "tuned_model_ref": "example-model-v1",
    "entropy_snapshot_before": {
        "model_name": "example-model",
        "mean_spectral_entropy": 3.14,
        "mean_effective_dimension": 128.0,
        "mean_viability_score": 0.72,
        "mean_phase_coherence": 0.88,
        "n_samples": 10,
        "layer_profiles": [
            {"layer_idx": 0, "spectral_entropy": 3.1, "effective_dimension": 125.0, "viability_score": 0.70, "fisher_curvature": 0.01}
        ],
    },
    "entropy_snapshot_after": {
        "model_name": "example-model-v1",
        "mean_spectral_entropy": 2.80,
        "mean_effective_dimension": 131.0,
        "mean_viability_score": 0.78,
        "mean_phase_coherence": 0.91,
        "n_samples": 10,
        "layer_profiles": [
            {"layer_idx": 0, "spectral_entropy": 2.78, "effective_dimension": 130.0, "viability_score": 0.77, "fisher_curvature": 0.01}
        ],
    },
    "entropy_delta_proof": {
        "delta_spectral_entropy": -0.34,
        "delta_effective_dimension": 3.0,
        "delta_viability_score": 0.06,
        "delta_phase_coherence": 0.03,
        "epsilon_threshold": 0.05,
        "reduction_verified": True,
        "cka_drift": 0.12,
    },
}


_TENANT_COUNTER = itertools.count(start=1000)

_SECRET    = "test-secret-key-for-testing"
_ALGORITHM = "HS256"


def _make_token(tenant_id: str) -> str:
    payload = {"sub": "prism-test", "tenant_id": tenant_id, "exp": int(time.time()) + 3600}
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


@pytest.fixture()
def auth_headers():
    """
    Return Authorization headers backed by a unique tenant_id so each test
    gets its own rate-limit bucket and never collides with other tests.
    Prism endpoints are public (no auth check), so passing a JWT only affects
    the rate-limiter key.
    """
    tenant_id = f"prism-test-{next(_TENANT_COUNTER)}"
    return {"Authorization": f"Bearer {_make_token(tenant_id)}"}


@pytest.fixture()
def records_dir(monkeypatch):
    """
    Point _RECORDS_DIR at a fresh temp directory and yield it.
    Uses tempfile.mkdtemp() directly to avoid Windows tmp_path permission issues.
    Each test starts with an empty records directory.
    """
    import apps.gateway.prism as prism_module
    tmpdir = Path(tempfile.mkdtemp(prefix="prism_test_"))
    monkeypatch.setattr(prism_module, "_RECORDS_DIR", tmpdir)
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


def _write_record(directory: Path, stem: str, data: dict) -> Path:
    path = directory / f"{stem}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ── /v1/prism/status ──────────────────────────────────────────────────────────

def test_prism_status_shape(records_dir, auth_headers):
    resp = client.get("/v1/prism/status", headers=auth_headers)
    assert resp.status_code == 200
    body = resp.json()
    assert "record_count" in body
    assert "model_track" in body
    assert "tools" in body
    assert isinstance(body["tools"], list)
    assert len(body["tools"]) > 0


def test_prism_status_tool_names(records_dir, auth_headers):
    resp = client.get("/v1/prism/status", headers=auth_headers)
    names = {t["name"] for t in resp.json()["tools"]}
    assert "TransformerLens" in names
    assert "NNsight" in names
    assert "Neuronpedia" in names


def test_prism_status_empty_dir_record_count(records_dir, auth_headers):
    resp = client.get("/v1/prism/status", headers=auth_headers)
    assert resp.json()["record_count"] == 0


def test_prism_status_record_count_increments(records_dir, auth_headers):
    _write_record(records_dir, "run-1", _MINIMAL_RECORD)
    _write_record(records_dir, "run-2", _MINIMAL_RECORD)
    resp = client.get("/v1/prism/status", headers=auth_headers)
    assert resp.json()["record_count"] == 2


# ── /v1/prism/runs (list) ─────────────────────────────────────────────────────

def test_prism_runs_empty(records_dir, auth_headers):
    resp = client.get("/v1/prism/runs", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_prism_runs_returns_list(records_dir, auth_headers):
    _write_record(records_dir, "run-a", _MINIMAL_RECORD)
    _write_record(records_dir, "run-b", {**_MINIMAL_RECORD, "id": "run-b"})
    resp = client.get("/v1/prism/runs", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_prism_runs_required_fields_present(records_dir, auth_headers):
    """Every record in /runs must have the six required LearningRun fields."""
    _write_record(records_dir, "run-req", _MINIMAL_RECORD)
    resp = client.get("/v1/prism/runs", headers=auth_headers)
    run = resp.json()[0]
    for field in ("id", "base_model_ref", "run_kind", "status", "source_axis", "updated_at"):
        assert field in run, f"missing required field: {field}"


def test_prism_runs_normalization_fills_missing_fields(records_dir, auth_headers):
    """Records that omit required fields must get safe string defaults."""
    _write_record(records_dir, "sparse", {"model_name": "mymodel"})
    resp = client.get("/v1/prism/runs", headers=auth_headers)
    run = resp.json()[0]
    assert run["id"] == "sparse"            # falls back to filename stem
    assert run["base_model_ref"] == "mymodel"
    assert run["run_kind"] == "probe"       # default
    assert run["status"] == "unknown"       # default
    assert run["source_axis"] == "unknown"  # default
    assert isinstance(run["updated_at"], str)


def test_prism_runs_skips_corrupt_json(records_dir, auth_headers):
    (records_dir / "bad.json").write_text("{not valid json", encoding="utf-8")
    _write_record(records_dir, "good", _MINIMAL_RECORD)
    resp = client.get("/v1/prism/runs", headers=auth_headers)
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_prism_runs_full_record_entropy_fields(records_dir, auth_headers):
    _write_record(records_dir, "full", _FULL_RECORD)
    resp = client.get("/v1/prism/runs", headers=auth_headers)
    run = resp.json()[0]
    assert run["entropy_delta_proof"] is not None
    assert run["entropy_delta_proof"]["reduction_verified"] is True
    assert run["entropy_snapshot_before"] is not None
    assert run["entropy_snapshot_after"] is not None


# ── /v1/prism/runs/summary ────────────────────────────────────────────────────

def test_prism_runs_summary_empty(records_dir, auth_headers):
    resp = client.get("/v1/prism/runs/summary", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json() == []


def test_prism_runs_summary_shape(records_dir, auth_headers):
    _write_record(records_dir, "s1", _FULL_RECORD)
    resp = client.get("/v1/prism/runs/summary", headers=auth_headers)
    assert resp.status_code == 200
    item = resp.json()[0]
    assert "id" in item
    assert "has_entropy_proof" in item
    assert "has_layer_profiles" in item
    assert item["has_entropy_proof"] is True
    assert item["has_layer_profiles"] is True


def test_prism_runs_summary_no_proof_flags(records_dir, auth_headers):
    _write_record(records_dir, "nopr", _MINIMAL_RECORD)
    resp = client.get("/v1/prism/runs/summary", headers=auth_headers)
    item = resp.json()[0]
    assert item["has_entropy_proof"] is False
    assert item["has_layer_profiles"] is False


# ── /v1/prism/runs/{run_id} ───────────────────────────────────────────────────

def test_prism_run_found_by_id_field(records_dir, auth_headers):
    _write_record(records_dir, "file-stem", _MINIMAL_RECORD)  # id field = "run-abc-123"
    resp = client.get("/v1/prism/runs/run-abc-123", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == "run-abc-123"


def test_prism_run_found_by_filename_stem(records_dir, auth_headers):
    """A record whose logical id differs from the filename stem is still findable by stem."""
    data = {**_MINIMAL_RECORD, "id": "logical-id-99"}
    _write_record(records_dir, "filestem99", data)
    resp = client.get("/v1/prism/runs/filestem99", headers=auth_headers)
    assert resp.status_code == 200
    assert resp.json()["id"] == "logical-id-99"


def test_prism_run_not_found(records_dir, auth_headers):
    resp = client.get("/v1/prism/runs/nonexistent-run", headers=auth_headers)
    assert resp.status_code == 404


def test_prism_run_required_fields_in_detail(records_dir, auth_headers):
    _write_record(records_dir, "detail", _MINIMAL_RECORD)
    resp = client.get("/v1/prism/runs/run-abc-123", headers=auth_headers)
    run = resp.json()
    for field in ("id", "base_model_ref", "run_kind", "status", "source_axis", "updated_at"):
        assert field in run, f"detail endpoint missing required field: {field}"


def test_prism_run_list_and_detail_consistent(records_dir, auth_headers):
    """GET /runs and GET /runs/{id} must return the same normalized shape."""
    _write_record(records_dir, "c1", _FULL_RECORD)
    list_runs = client.get("/v1/prism/runs", headers=auth_headers).json()
    detail = client.get(f"/v1/prism/runs/{_FULL_RECORD['id']}", headers=auth_headers).json()
    assert list_runs[0]["id"] == detail["id"]
    assert list_runs[0]["base_model_ref"] == detail["base_model_ref"]
    assert list_runs[0]["status"] == detail["status"]


# ── Path traversal guard ──────────────────────────────────────────────────────

@pytest.mark.parametrize("bad_id", [
    "../etc/passwd",
    "../../secrets",
    "run id with spaces",
    "run\x00null",
    "a" * 129,          # exceeds 128-char limit
    "a" * 256,
    "run?foo=bar",
    "run#anchor",
])
def test_prism_run_invalid_id_rejected(records_dir, auth_headers, bad_id):
    """Malformed or dangerous run IDs must return 404, not 500 or a file read."""
    from urllib.parse import quote
    resp = client.get(f"/v1/prism/runs/{quote(bad_id, safe='')}", headers=auth_headers)
    assert resp.status_code == 404


def test_prism_run_valid_boundary_id(records_dir, auth_headers):
    """A 128-char alphanumeric id (max allowed length) must pass the regex guard."""
    max_id = "a" * 128
    resp = client.get(f"/v1/prism/runs/{max_id}", headers=auth_headers)
    # No record exists — should be 404 from the not-found path, not the guard
    assert resp.status_code == 404


# ── Rate limiting ──────────────────────────────────────────────────────────────

def test_prism_runs_rate_limited(records_dir):
    """After RATE_LIMIT_RPM (20) requests the 21st should be 429.
    Uses a dedicated tenant so the bucket is independent of other tests.
    """
    tenant_id = f"prism-test-{next(_TENANT_COUNTER)}"
    headers = {"Authorization": f"Bearer {_make_token(tenant_id)}"}
    for _ in range(20):
        resp = client.get("/v1/prism/runs", headers=headers)
        assert resp.status_code == 200
    resp21 = client.get("/v1/prism/runs", headers=headers)
    assert resp21.status_code == 429
    assert resp21.json()["error"]["type"] == "rate_limited"
    assert "Retry-After" in resp21.headers


def test_prism_status_rate_limited(records_dir):
    """The /v1/prism/status endpoint is also rate-limited."""
    tenant_id = f"prism-test-{next(_TENANT_COUNTER)}"
    headers = {"Authorization": f"Bearer {_make_token(tenant_id)}"}
    for _ in range(20):
        resp = client.get("/v1/prism/status", headers=headers)
        assert resp.status_code == 200
    resp21 = client.get("/v1/prism/status", headers=headers)
    assert resp21.status_code == 429
