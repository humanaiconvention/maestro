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


def test_full_governed_run_flow():
    """
    End-to-end: create run → plan → approval → action → verify → settle.
    Exercises every new agentic-run endpoint through the FastAPI router.
    """
    # 1. Create a LearningRun
    res = client.post("/api/learning-runs", json={
        "title": "Observation Run Alpha",
        "objective": "Reduce uncertainty on retail patterns",
        "hypothesis_or_goal": "Prior entropy > posterior entropy",
        "model_or_agent_version": "agent-v2"
    }, headers=API_HEADERS)
    assert res.status_code == 200, res.text
    run = res.json()
    run_id = run["id"]
    assert run["status"] == "ACTIVE"

    # 2. Add a Plan
    res = client.post(f"/api/learning-runs/{run_id}/plans", json={
        "title": "Observe 50 customers",
        "objective": "Collect baseline behavioural data",
        "steps": ["station at entrance", "log help requests"],
        "risk_level": "low"
    }, headers=API_HEADERS)
    assert res.status_code == 200, res.text
    plan = res.json()
    plan_id = plan["id"]
    assert plan["status"] == "SUBMITTED"
    assert plan["run_id"] == run_id

    # 3. Add Approval for the plan
    res = client.post(f"/api/learning-runs/{run_id}/approvals", json={
        "target_ref": plan_id,
        "decision": "APPROVED",
        "approver": "operator-1",
        "rationale": "Low risk, consent verified"
    }, headers=API_HEADERS)
    assert res.status_code == 200, res.text
    approval = res.json()
    assert approval["decision"] == "APPROVED"

    # 4. Log an Action
    res = client.post(f"/api/learning-runs/{run_id}/actions", json={
        "plan_id": plan_id,
        "action_type": "observe",
        "actor": "agent-v2",
        "environment": "retail-store-sim"
    }, headers=API_HEADERS)
    assert res.status_code == 200, res.text
    action = res.json()
    action_id = action["id"]
    assert action["status"] == "PENDING"

    # 5. Verify a claim (empirical, clean evidence)
    res = client.post(f"/api/learning-runs/{run_id}/verify", json={
        "claim_text": "Observation rate was 0.35",
        "lane": "empirical",
        "evidence": {"variance_high": False},
        "target_ref": action_id
    }, headers=API_HEADERS)
    assert res.status_code == 200, res.text
    verification = res.json()
    assert verification["verdict"] == "verified"
    assert verification["review_status"] == "PENDING"
    assert verification["policy_state"] == "allowed"

    # 5.5. Resolve the verification
    res = client.patch(f"/api/learning-runs/{run_id}/verifications/{verification['id']}", json={
        "review_status": "RESOLVED",
        "rationale": "Operator confirms evidence is valid."
    }, headers=API_HEADERS)
    assert res.status_code == 200, res.text
    patched_verification = res.json()
    assert patched_verification["review_status"] == "RESOLVED"

    # 6. Settle the run
    res = client.post(f"/api/learning-runs/{run_id}/settle", json={
        "prior_distribution": {"rate_lt_0_3": 0.5, "rate_gte_0_3": 0.5},
        "posterior_distribution": {"rate_lt_0_3": 0.15, "rate_gte_0_3": 0.85},
        "observation_count": 50,
        "claimed_entropy_reduction": 0.42,
        "total_payout": 50.0
    }, headers=API_HEADERS)
    assert res.status_code == 200, res.text
    settlement = res.json()
    assert settlement["status"] == "COMPLETED"
    assert settlement["learning_run_id"] == run_id

    # 7. Retrieve the full run and validate all artifacts
    res = client.get(f"/api/learning-runs/{run_id}")
    assert res.status_code == 200, res.text
    detail = res.json()
    assert len(detail["plans"]) == 1
    assert len(detail["approvals"]) == 1
    assert len(detail["actions"]) == 1
    assert len(detail["verifications"]) == 1


def test_verification_blocks_endogenous_only():
    """Correlated-verifier policy blocks settlement when endogenous_only=True."""
    res = client.post("/api/learning-runs", json={
        "title": "Theory Run", "objective": "Test correlated policy"
    }, headers=API_HEADERS)
    run_id = res.json()["id"]

    res = client.post(f"/api/learning-runs/{run_id}/verify", json={
        "claim_text": "X implies Y",
        "lane": "theory",
        "evidence": {"formalizable": True, "proof_failed": False},
        "endogenous_only": True
    }, headers=API_HEADERS)
    assert res.status_code == 200
    v = res.json()
    assert v["policy_state"] == "blocked_pending_exogenous_check"
    assert v["review_status"] == "ESCALATED"


def test_verification_flags_metaphorical_claim():
    """Language safety hook rejects metaphorical entropy claims."""
    res = client.post("/api/learning-runs", json={
        "title": "Bad Claim Run", "objective": "Test metaphorical claim"
    }, headers=API_HEADERS)
    run_id = res.json()["id"]

    res = client.post(f"/api/learning-runs/{run_id}/verify", json={
        "claim_text": "We observed literal negative entropy in grounding.",
        "lane": "empirical",
        "evidence": {}
    }, headers=API_HEADERS)
    assert res.status_code == 200
    v = res.json()
    assert v["verdict"] == "metaphorical_or_nonliteral"


def test_rollback_on_run():
    """Rollback can be recorded against a run and action."""
    res = client.post("/api/learning-runs", json={
        "title": "Rollback Test", "objective": "Test rollback"
    }, headers=API_HEADERS)
    run_id = res.json()["id"]

    res = client.post(f"/api/learning-runs/{run_id}/actions", json={
        "action_type": "deploy_config",
        "actor": "agent-v1"
    }, headers=API_HEADERS)
    action_id = res.json()["id"]

    res = client.post(f"/api/learning-runs/{run_id}/rollbacks", json={
        "action_id": action_id,
        "prior_state_ref": "config_snapshot_abc",
        "rollback_trigger": "variance_exceeded",
        "executed_by": "operator-1"
    }, headers=API_HEADERS)
    assert res.status_code == 200
    rb = res.json()
    assert rb["action_id"] == action_id
    assert rb["prior_state_ref"] == "config_snapshot_abc"


def test_learning_run_not_found():
    """404 when accessing a non-existent run."""
    res = client.get("/api/learning-runs/nonexistent-id")
    assert res.status_code == 404

    res = client.post("/api/learning-runs/nonexistent-id/plans", json={
        "title": "X", "objective": "Y"
    }, headers=API_HEADERS)
    assert res.status_code == 404


# ── Part C: State Machine Tests ──────────────────────────────────────

def _make_run(title="Test Run"):
    """Helper: create a LearningRun and return the JSON."""
    res = client.post("/api/learning-runs", json={
        "title": title, "objective": "State machine testing"
    }, headers=API_HEADERS)
    assert res.status_code == 200
    return res.json()


def test_plan_auto_submitted():
    """Plan status must be SUBMITTED immediately after POST (not DRAFT)."""
    run = _make_run("Plan-State-Test")
    res = client.post(f"/api/learning-runs/{run['id']}/plans", json={
        "title": "Auto-submit plan", "objective": "Check status"
    }, headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "SUBMITTED"


def test_plan_status_cascades_on_approval():
    """Approving a plan's ID as target_ref must set plan.status to APPROVED."""
    run = _make_run("Cascade-Approval-Test")
    run_id = run["id"]

    plan = client.post(f"/api/learning-runs/{run_id}/plans", json={
        "title": "Plan A", "objective": "To be approved"
    }, headers=API_HEADERS).json()
    plan_id = plan["id"]
    assert plan["status"] == "SUBMITTED"

    # Approve the plan
    client.post(f"/api/learning-runs/{run_id}/approvals", json={
        "target_ref": plan_id, "decision": "APPROVED", "approver": "operator-1"
    }, headers=API_HEADERS)

    # Verify the plan status cascaded
    detail = client.get(f"/api/learning-runs/{run_id}").json()
    plan_in_run = next(p for p in detail["plans"] if p["id"] == plan_id)
    assert plan_in_run["status"] == "APPROVED"


def test_plan_status_cascades_on_rejection():
    """Rejecting a plan's ID as target_ref must set plan.status to REJECTED."""
    run = _make_run("Cascade-Rejection-Test")
    run_id = run["id"]

    plan = client.post(f"/api/learning-runs/{run_id}/plans", json={
        "title": "Plan B", "objective": "To be rejected"
    }, headers=API_HEADERS).json()
    plan_id = plan["id"]

    client.post(f"/api/learning-runs/{run_id}/approvals", json={
        "target_ref": plan_id, "decision": "REJECTED", "approver": "operator-1",
        "rationale": "Out of scope"
    }, headers=API_HEADERS)

    detail = client.get(f"/api/learning-runs/{run_id}").json()
    plan_in_run = next(p for p in detail["plans"] if p["id"] == plan_id)
    assert plan_in_run["status"] == "REJECTED"


def test_action_patch_updates_status():
    """PATCH /actions/{id} must update action status and output_refs."""
    run = _make_run("Action-Patch-Test")
    run_id = run["id"]

    action = client.post(f"/api/learning-runs/{run_id}/actions", json={
        "action_type": "compute", "actor": "agent-v2"
    }, headers=API_HEADERS).json()
    action_id = action["id"]
    assert action["status"] == "PENDING"

    res = client.patch(
        f"/api/learning-runs/{run_id}/actions/{action_id}",
        json={"status": "COMPLETED", "output_refs": {"result_key": "val-42"}},
        headers=API_HEADERS
    )
    assert res.status_code == 200
    patched = res.json()
    assert patched["status"] == "COMPLETED"


def test_settlement_blocked_by_escalation():
    """Settlement must return SETTLEMENT_PENDING when escalated verifications exist."""
    run = _make_run("Escalation-Block-Test")
    run_id = run["id"]

    # Create a verification that triggers endogenous-only block → ESCALATED + blocked
    verify_res = client.post(f"/api/learning-runs/{run_id}/verify", json={
        "claim_text": "X implies Y",
        "lane": "theory",
        "evidence": {"formalizable": True},
        "endogenous_only": True   # triggers correlated-verifier block
    }, headers=API_HEADERS)
    assert verify_res.status_code == 200
    v = verify_res.json()
    assert v["policy_state"] == "blocked_pending_exogenous_check"
    assert v["review_status"] == "ESCALATED"

    # Attempt settlement — should be blocked
    res = client.post(f"/api/learning-runs/{run_id}/settle", json={
        "prior_distribution": {"a": 0.5}, "posterior_distribution": {"a": 0.2},
        "observation_count": 10, "claimed_entropy_reduction": 0.3, "total_payout": 20.0
    }, headers=API_HEADERS)
    assert res.status_code == 200
    result = res.json()
    assert result["status"] == "SETTLEMENT_PENDING"
    assert "blocked_reason" in result

    # Verify run status also shows SETTLEMENT_PENDING
    run_detail = client.get(f"/api/learning-runs/{run_id}").json()
    assert run_detail["status"] == "SETTLEMENT_PENDING"


def test_rollback_sets_run_rolled_back():
    """Recording a rollback must transition run.status to ROLLED_BACK."""
    run = _make_run("Rollback-State-Test")
    run_id = run["id"]

    action = client.post(f"/api/learning-runs/{run_id}/actions", json={
        "action_type": "deploy", "actor": "agent-v1"
    }, headers=API_HEADERS).json()

    client.post(f"/api/learning-runs/{run_id}/rollbacks", json={
        "action_id": action["id"], "rollback_trigger": "gate_failure"
    }, headers=API_HEADERS)

    run_detail = client.get(f"/api/learning-runs/{run_id}").json()
    assert run_detail["status"] == "ROLLED_BACK"


def test_successful_settlement_sets_run_settled():
    """A passing settlement must transition run.status to SETTLED."""
    run = _make_run("Settlement-State-Test")
    run_id = run["id"]

    res = client.post(f"/api/learning-runs/{run_id}/settle", json={
        "prior_distribution": {"rate": 0.5}, "posterior_distribution": {"rate": 0.2},
        "observation_count": 30, "claimed_entropy_reduction": 0.42, "total_payout": 30.0
    }, headers=API_HEADERS)
    assert res.status_code == 200
    assert res.json()["status"] == "COMPLETED"

    run_detail = client.get(f"/api/learning-runs/{run_id}").json()
    assert run_detail["status"] == "SETTLED"


def test_action_blocked_if_plan_not_approved():
    """An action that links to a plan_id should be blocked if plan is not APPROVED."""
    run = _make_run("Action-Blocked-Test")
    run_id = run["id"]

    plan = client.post(f"/api/learning-runs/{run_id}/plans", json={
        "title": "Unapproved Plan", "objective": "Try acting on it"
    }, headers=API_HEADERS).json()

    res = client.post(f"/api/learning-runs/{run_id}/actions", json={
        "plan_id": plan["id"],
        "action_type": "deploy", 
        "actor": "agent-v1"
    }, headers=API_HEADERS)
    
    assert res.status_code == 400
    assert "Cannot execute action for plan in status" in res.json()["detail"]


def test_settlement_blocked_by_pending_review():
    """Settlement must return SETTLEMENT_PENDING even on standard PENDING verifications."""
    run = _make_run("Pending-Block-Test")
    run_id = run["id"]

    # Verify claim cleanly (but it still enters PENDING state)
    verify_res = client.post(f"/api/learning-runs/{run_id}/verify", json={
        "claim_text": "Good claim",
        "lane": "empirical",
        "evidence": {}
    }, headers=API_HEADERS)
    assert verify_res.status_code == 200
    assert verify_res.json()["review_status"] == "PENDING"

    # Attempt to settle — should block
    res = client.post(f"/api/learning-runs/{run_id}/settle", json={
        "prior_distribution": {"x": 0.5}, "posterior_distribution": {"x": 0.2},
        "observation_count": 10, "claimed_entropy_reduction": 0.3, "total_payout": 20.0
    }, headers=API_HEADERS)
    
    assert res.status_code == 200
    assert res.json()["status"] == "SETTLEMENT_PENDING"


