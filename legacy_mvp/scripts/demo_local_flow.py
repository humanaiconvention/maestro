import os
import json
import sys
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app

client = TestClient(app)

# Default dev API key — matches MAESTRO_API_KEY env default in app/auth.py
API_KEY = os.environ.get("MAESTRO_API_KEY", "dev-insecure-key-change-me")
AUTH = {"X-Maestro-API-Key": API_KEY}

def print_step(msg):
    print(f"\n{'='*60}\n{msg}\n{'='*60}")

def print_gate_results(evaluation):
    print(f"Overall Status: {evaluation['status']}")
    print(f"Decision Rationale: {evaluation.get('decision_rationale', 'N/A')}")
    print("Gate Details:")
    for gate in evaluation.get('gates_evaluated', evaluation.get('gates', [])):
        status_icon = "[PASS]" if gate['status'] == "PASS" else "[FAIL]" if gate['status'] == "FAIL" else "[REVIEW]"
        val = gate.get('value', 'N/A')
        thresh = gate.get('threshold', 'N/A')
        print(f"  {status_icon} {gate['gate']}: {gate['status']} (Value: {val}, Threshold: {thresh})")
        if gate.get('heuristic_extraction_risk'):
            print(f"     Heuristic Risk: {gate['heuristic_extraction_risk']}")
        if gate.get('reason'):
            print(f"     Reason: {gate['reason']}")

def run_demo():
    print_step("1. Registering Actors")
    res = client.post("/api/agents", json={"display_name": "Agent_Alpha"}, headers=AUTH)
    agent_id = res.json()["id"]
    print(f"Agent registered: {agent_id}")

    res = client.post("/api/humans", json={"display_name": "Human_Contributor_1"})
    human_id = res.json()["id"]
    print(f"Human registered: {human_id}")

    print_step("2. Human sets Bounded Consent")
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
    print("Consent scope configured.")

    print_step("3. Agent creates Grounding Request")
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
    res = client.post("/api/grounding-requests", json=req, headers=AUTH)
    req_id = res.json()["id"]
    print(f"Request created: {req_id}")

    print_step("4. Request Viability Check")
    res = client.post(f"/api/grounding-requests/{req_id}/evaluate", headers=AUTH)
    print_gate_results(res.json())

    print_step("5. Human submits Contribution with Provenance")
    contrib = {
        "human_id": human_id,
        "request_id": req_id,
        "content_payload": {"observed_rate": 0.35},
        "integrity_hash": "mock_sha256_hash",
        "creation_context": {"humanity_score": 0.98, "typing_cadence_std": 65}
    }
    res = client.post("/api/contributions", json=contrib, headers=AUTH)
    contrib_id = res.json()["id"]
    print(f"Contribution submitted: {contrib_id}")

    print_step("6. Contribution Viability Check")
    res = client.post(f"/api/contributions/{contrib_id}/evaluate", headers=AUTH)
    print_gate_results(res.json())

    print_step("7. Thermodynamic Settlement")
    settlement = {
        "prior_distribution": {"rate_0_20": 0.2},
        "posterior_distribution": {"rate_0_20": 0.05},
        "observation_count": 1,
        "claimed_entropy_reduction": 0.42,
        "total_payout": 10.0
    }
    res = client.post(f"/api/settlements/{req_id}/attempt", json=settlement, headers=AUTH)
    data = res.json()
    print(f"Settlement status: {data['status']}")
    print_gate_results(data['evaluation'])
    
    print_step("8. Audit Trail")
    res = client.get("/api/audit-events")
    events = res.json()
    for e in events[:5]: 
        print(f"[{e['created_at']}] {e['event_type']} - Entity: {e['entity_id']}")

if __name__ == "__main__":
    if os.path.exists("./maestro.db"):
        try:
            os.remove("./maestro.db")
        except PermissionError:
            pass
    run_demo()
