import os
import pytest
from datetime import datetime, timezone
from sqlalchemy.orm import Session

# Set TESTING environment variable BEFORE importing the app
os.environ["TESTING"] = "true"

from app.db import Base, engine, SessionLocal
from app.models.domain import (
    LearningRun, Plan, ActionReceipt, ApprovalDecision, 
    VerificationResult, RollbackRecord, Settlement, GroundingRequest, 
    Contribution, ProvenanceRecord, Actor, ConsentProfile
)
from app.models.enums import (
    LearningRunStatus, PlanStatus, ActionStatus, ApprovalDecisionEnum, 
    LaneType, Verdict, ReviewStatus, PolicyState, ActorType, PrivacyLevel, PayoutMethod
)
from app.services.verification import VerificationPipeline
from app.services.settlement import attempt_run_settlement
from app.schemas import SettlementAttempt

@pytest.fixture(autouse=True)
def reset_db():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)

@pytest.fixture
def db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_schema_creation_and_linkage(db_session: Session):
    # 1. Create a LearningRun
    run = LearningRun(
        title="Test Run",
        objective="Test the bridge",
        hypothesis_or_goal="Reduce uncertainty",
        model_or_agent_version="agent-v1",
        status=LearningRunStatus.active
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)
    
    # 2. Add Plan
    plan = Plan(
        run_id=run.id,
        title="Observation Plan",
        objective="Observe",
        status=PlanStatus.active
    )
    db_session.add(plan)
    db_session.commit()
    db_session.refresh(plan)
    
    # 3. Add Approval
    approval = ApprovalDecision(
        run_id=run.id,
        target_ref=plan.id,
        decision=ApprovalDecisionEnum.approved
    )
    db_session.add(approval)
    
    # 4. Add Action
    action = ActionReceipt(
        run_id=run.id,
        plan_id=plan.id,
        action_type="observe",
        status=ActionStatus.completed
    )
    db_session.add(action)
    db_session.commit()
    db_session.refresh(action)
    
    # 5. Add Verification
    verification = VerificationResult(
        run_id=run.id,
        target_ref=action.id,
        claim_text="Observation complete",
        lane=LaneType.empirical,
        verdict=Verdict.verified,
        review_status=ReviewStatus.resolved
    )
    db_session.add(verification)
    
    # 6. Add Rollback
    rollback = RollbackRecord(
        run_id=run.id,
        action_id=action.id,
        prior_state_ref="snapshot_x"
    )
    db_session.add(rollback)
    db_session.commit()

    # Validate relationships from run
    db_session.refresh(run)
    assert len(run.plans) == 1
    assert len(run.actions) == 1
    assert len(run.approvals) == 1
    assert len(run.verifications) == 1
    assert len(run.rollbacks) == 1
    assert run.plans[0].id == plan.id
    assert run.actions[0].id == action.id

def test_verifier_lane_routing():
    pipeline = VerificationPipeline()
    
    # Empirical lane - Needs review (high variance)
    evidence_emp = {"variance_high": True}
    res_emp = pipeline.run_pipeline("The measurements match", "empirical", evidence_emp)
    assert res_emp["lane"] == LaneType.empirical
    assert res_emp["verdict"] == Verdict.unresolved
    assert res_emp["review_status"] == ReviewStatus.escalated
    
    # Theory lane - PASS
    evidence_thy = {"formalizable": True, "proof_failed": False}
    res_thy = pipeline.run_pipeline("X implies Y", "theory", evidence_thy)
    assert res_thy["lane"] == LaneType.theory
    assert res_thy["verdict"] == Verdict.verified
    assert res_thy["review_status"] == ReviewStatus.pending

    # Literature lane - conflicting sources
    evidence_lit = {"sources_conflict": True, "weak_retrieval": False}
    res_lit = pipeline.run_pipeline("Studies show X", "literature", evidence_lit)
    assert res_lit["lane"] == LaneType.literature
    assert res_lit["verdict"] == Verdict.unresolved
    assert res_lit["review_status"] == ReviewStatus.escalated

def test_language_claim_safety_hook():
    pipeline = VerificationPipeline()
    
    # Should flag literal negative entropy without mapping
    res = pipeline.run_pipeline("We observed literal negative entropy in grounding.", "empirical", {})
    assert res["verdict"] == Verdict.metaphorical_or_nonliteral
    assert res["policy_state"] == PolicyState.allowed.value  # It flags verdict, not block policy here
    
def test_correlated_verifier_policy():
    pipeline = VerificationPipeline()
    
    # Endogenous only
    evidence = {"formalizable": True, "proof_failed": False}
    res = pipeline.run_pipeline("X implies Y", "theory", evidence, endogenous_only=True)
    assert res["policy_state"] == PolicyState.blocked_pending_exogenous_check.value
    assert res["review_status"] == ReviewStatus.escalated

def test_settlement_on_learning_run(db_session: Session):
    # Setup learning run
    run = LearningRun(
        title="Settlement Run",
        objective="Testing settlement",
        status=LearningRunStatus.active
    )
    db_session.add(run)
    db_session.commit()
    db_session.refresh(run)

    # Attempt Settlement
    settlement_data = SettlementAttempt(
        prior_distribution={"A": 0.5, "B": 0.5},
        posterior_distribution={"A": 0.9, "B": 0.1},
        observation_count=1,
        claimed_entropy_reduction=0.4,
        total_payout=10.0
    )
    
    res = attempt_run_settlement(db_session, run.id, settlement_data)
    
    assert res["status"] == "COMPLETED"
    assert res["learning_run_id"] == run.id
    
    # Check DB
    settlement = db_session.query(Settlement).filter_by(learning_run_id=run.id).first()
    assert settlement is not None
    assert settlement.status.value == "COMPLETED"
