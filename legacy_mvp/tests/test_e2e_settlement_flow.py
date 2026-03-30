import os
import pytest
from fastapi.testclient import TestClient

# Set TESTING environment variable BEFORE importing the app
os.environ["TESTING"] = "true"

from app.main import app
from app.db import Base, engine

client = TestClient(app)

API_HEADERS = {"X-Maestro-API-Key": "dev-insecure-key-change-me"}

@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

def test_full_settlement_flow():
    # 1. Register an agent (PROTECTED)
    res = client.post("/api/agents", json={"display_name": "Agent_Alpha"}, headers=API_HEADERS)
    assert res.status_code == 200
    agent_id = res.json()["id"]

    # 2. Register a human (OPEN)
    res = client.post("/api/humans", json={"display_name": "Human_Contributor_1"})
    assert res.status_code == 200
    human_id = res.json()["id"]

    # 3. Set consent profile (OPEN)
    consent = {
        "allowed_axes": ["causal_reasoning"],
        "max_task_duration_minutes": 60,
        "privacy_level": "standard",
        "allow_model_training": True,
        "allow_public_aggregate": False,
        "min_hourly_equivalent_usd": 50.0,
        "payout_threshold_usd": 25.0,
        "payout_method": "crypto",
        "daily_max_contributions": 10,
        "weekly_max_hours": 8,
        "require_full_provenance": True,
        "revocation_window_days": 90
    }
    res = client.post(f"/api/consent-profiles/{human_id}", json=consent)
    assert res.status_code == 200

    # 4. Create grounding request (PROTECTED)
    req = {
        "agent_id": agent_id,
        "title": "Retail Observation",
        "description": "Observe retail shoppers to ground baseline priors.",
        "lived_experience_axis": "causal_reasoning",
        "requested_modality": "text",
        "protocol_instructions": "Observe 30 customers and log if they request help.",
        "estimated_duration_minutes": 10,
        "compensation_amount": 10.0,
        "effective_hourly_estimate": 60.0,
        "entropy_reduction_hypothesis": "Will reduce uncertainty bits",
        "phase_synchronization_hypothesis": "Latent space coupling",
        "evidence_artifact_description": "JSON payload",
        "contributions_required": 1,
        "viability_policy_snapshot": {}
    }
    res = client.post("/api/grounding-requests", json=req, headers=API_HEADERS)
    assert res.status_code == 200
    req_id = res.json()["id"]

    # 5. Evaluate request viability (PROTECTED)
    res = client.post(f"/api/grounding-requests/{req_id}/evaluate", headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "PASS"

    # 6. Submit contribution (PROTECTED)
    contrib = {
        "human_id": human_id,
        "request_id": req_id,
        "content_payload": {"observed_rate": 0.35},
        "integrity_hash": "mock_sha256_hash",
        "creation_context": {"humanity_score": 0.98, "typing_cadence_std": 65}
    }
    res = client.post("/api/contributions", json=contrib, headers=API_HEADERS)
    assert res.status_code == 200
    contrib_id = res.json()["id"]

    # 7. Evaluate contribution viability (PROTECTED)
    res = client.post(f"/api/contributions/{contrib_id}/evaluate", headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "PASS"

    # 8. Attempt settlement (PROTECTED)
    settlement = {
        "prior_distribution": {"rate_0_20": 0.2},
        "posterior_distribution": {"rate_0_20": 0.05},
        "observation_count": 1,
        "claimed_entropy_reduction": 0.42,
        "total_payout": 10.0
    }
    res = client.post(f"/api/settlements/{req_id}/attempt", json=settlement, headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "COMPLETED"

    # 9. Verify audit trail (OPEN)
    res = client.get("/api/audit-events")
    assert res.status_code == 200
    assert len(res.json()) >= 5

def test_settlement_blocked_no_entropy():
    # Setup happy path until contribution evaluation
    res = client.post("/api/agents", json={"display_name": "Agent_Alpha"}, headers=API_HEADERS)
    agent_id = res.json()["id"]
    res = client.post("/api/humans", json={"display_name": "Human_Contributor_1"})
    human_id = res.json()["id"]
    consent = {
        "allowed_axes": ["causal_reasoning"],
        "max_task_duration_minutes": 60,
        "privacy_level": "standard",
        "allow_model_training": True,
        "allow_public_aggregate": False,
        "min_hourly_equivalent_usd": 50.0,
        "payout_threshold_usd": 25.0,
        "payout_method": "crypto",
        "daily_max_contributions": 10,
        "weekly_max_hours": 8,
        "require_full_provenance": True,
        "revocation_window_days": 90
    }
    client.post(f"/api/consent-profiles/{human_id}", json=consent)
    req = {
        "agent_id": agent_id,
        "title": "T1", "description": "D1", "lived_experience_axis": "causal_reasoning",
        "requested_modality": "text", "protocol_instructions": "P1", "estimated_duration_minutes": 10,
        "compensation_amount": 10.0, "effective_hourly_estimate": 60.0,
        "entropy_reduction_hypothesis": "H1", "phase_synchronization_hypothesis": "H2",
        "evidence_artifact_description": "E1", "contributions_required": 1, "viability_policy_snapshot": {}
    }
    res = client.post("/api/grounding-requests", json=req, headers=API_HEADERS)
    req_id = res.json()["id"]
    client.post(f"/api/grounding-requests/{req_id}/evaluate", headers=API_HEADERS)
    
    contrib = {
        "human_id": human_id, "request_id": req_id, "content_payload": {"v": 1},
        "integrity_hash": "h1", "creation_context": {"humanity_score": 0.98}
    }
    res = client.post("/api/contributions", json=contrib, headers=API_HEADERS)
    contrib_id = res.json()["id"]
    client.post(f"/api/contributions/{contrib_id}/evaluate", headers=API_HEADERS)

    # Attempt settlement with ZERO entropy reduction
    settlement = {
        "prior_distribution": {"a": 0.5}, "posterior_distribution": {"a": 0.5},
        "observation_count": 1, "claimed_entropy_reduction": 0.0, "total_payout": 10.0
    }
    res = client.post(f"/api/settlements/{req_id}/attempt", json=settlement, headers=API_HEADERS)
    assert res.status_code == 200
    data = res.json()
    assert data["evaluation"]["status"] == "FAIL"
    assert data["status"] != "COMPLETED"

def test_contribution_rejected_bad_provenance():
    # Setup
    res = client.post("/api/agents", json={"display_name": "Agent_Alpha"}, headers=API_HEADERS)
    agent_id = res.json()["id"]
    res = client.post("/api/humans", json={"display_name": "Human_Contributor_1"})
    human_id = res.json()["id"]
    consent = {
        "allowed_axes": ["causal_reasoning"], "max_task_duration_minutes": 60, "privacy_level": "standard",
        "allow_model_training": True, "allow_public_aggregate": False, "min_hourly_equivalent_usd": 50.0,
        "payout_threshold_usd": 25.0, "payout_method": "crypto", "daily_max_contributions": 10,
        "weekly_max_hours": 8, "require_full_provenance": True, "revocation_window_days": 90
    }
    client.post(f"/api/consent-profiles/{human_id}", json=consent)
    req = {
        "agent_id": agent_id, "title": "T1", "description": "D1", "lived_experience_axis": "causal_reasoning",
        "requested_modality": "text", "protocol_instructions": "P1", "estimated_duration_minutes": 10,
        "compensation_amount": 10.0, "effective_hourly_estimate": 60.0,
        "entropy_reduction_hypothesis": "H1", "phase_synchronization_hypothesis": "H2",
        "evidence_artifact_description": "E1", "contributions_required": 1, "viability_policy_snapshot": {}
    }
    res = client.post("/api/grounding-requests", json=req, headers=API_HEADERS)
    req_id = res.json()["id"]
    client.post(f"/api/grounding-requests/{req_id}/evaluate", headers=API_HEADERS)

    # Submit with low humanity_score (0.5 < 0.85 threshold)
    contrib = {
        "human_id": human_id, "request_id": req_id, "content_payload": {"v": 1},
        "integrity_hash": "h1", "creation_context": {"humanity_score": 0.5}
    }
    res = client.post("/api/contributions", json=contrib, headers=API_HEADERS)
    contrib_id = res.json()["id"]
    
    res = client.post(f"/api/contributions/{contrib_id}/evaluate", headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "FAIL"

def test_revocation_within_window():
    # Setup
    res = client.post("/api/agents", json={"display_name": "Agent_Alpha"}, headers=API_HEADERS)
    agent_id = res.json()["id"]
    res = client.post("/api/humans", json={"display_name": "Human_Contributor_1"})
    human_id = res.json()["id"]
    consent = {
        "allowed_axes": ["causal_reasoning"], "max_task_duration_minutes": 60, "privacy_level": "standard",
        "allow_model_training": True, "allow_public_aggregate": False, "min_hourly_equivalent_usd": 50.0,
        "payout_threshold_usd": 25.0, "payout_method": "crypto", "daily_max_contributions": 10,
        "weekly_max_hours": 8, "require_full_provenance": True, "revocation_window_days": 90
    }
    client.post(f"/api/consent-profiles/{human_id}", json=consent)
    req = {
        "agent_id": agent_id, "title": "T1", "description": "D1", "lived_experience_axis": "causal_reasoning",
        "requested_modality": "text", "protocol_instructions": "P1", "estimated_duration_minutes": 10,
        "compensation_amount": 10.0, "effective_hourly_estimate": 60.0,
        "entropy_reduction_hypothesis": "H1", "phase_synchronization_hypothesis": "H2",
        "evidence_artifact_description": "E1", "contributions_required": 1, "viability_policy_snapshot": {}
    }
    res = client.post("/api/grounding-requests", json=req, headers=API_HEADERS)
    req_id = res.json()["id"]
    
    contrib = {
        "human_id": human_id, "request_id": req_id, "content_payload": {"v": 1},
        "integrity_hash": "h1", "creation_context": {"humanity_score": 0.98}
    }
    res = client.post("/api/contributions", json=contrib, headers=API_HEADERS)
    contrib_id = res.json()["id"]

    # Revoke (PROTECTED)
    revocation = {"human_id": human_id, "contribution_id": contrib_id, "reason": "Privacy concerns"}
    res = client.post("/api/revocations", json=revocation, headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "COMPLETED"
