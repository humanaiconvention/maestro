from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel
from app.db import get_db
from app.auth import require_api_key
from app.schemas import (
    HumanCreate, AgentCreate, ActorResponse, ConsentProfileUpdate, ConsentProfileResponse,
    GroundingRequestCreate, GroundingRequestResponse, ContributionCreate, ContributionResponse,
    ViabilityEvaluationResult, SettlementAttempt, RevocationCreate, RevocationResponse,
    ReviewQueueResponse, AuditEventResponse,
    LearningRunCreate, LearningRunResponse, LearningRunDetailResponse,
    PlanCreate, PlanResponse, ActionReceiptCreate, ActionReceiptResponse,
    ApprovalDecisionCreate, ApprovalDecisionResponse,
    VerificationRequest, VerificationResponse, VerificationStatusPatch,
    RollbackRecordCreate, RollbackRecordResponse,
    ApiKeyCreate, ApiKeyResponse,
)
from app.services import actors, requests as req_service, contributions, settlement, revocation
from app.models.domain import (
    Actor, ConsentProfile, GroundingRequest, Contribution, ReviewQueueItem, AuditEvent,
    LearningRun, Plan, ActionReceipt, ApprovalDecision, VerificationResult, RollbackRecord,
    ApiKey,
)
from app.models.enums import (
    ActorType, ContributionStatus, RequestStatus, LaneType,
    AuditEventType, PlanStatus, ActionStatus, LearningRunStatus, ApprovalDecisionEnum,
)
from app.services.verification import VerificationPipeline
from app.services.audit import emit_audit_event
import hashlib
import secrets
from datetime import datetime, timezone

# ── Patch Request Schemas (lightweight inline) ───────────────────────
class ActionStatusPatch(BaseModel):
    status: ActionStatus
    output_refs: Optional[dict] = None

class RunStatusPatch(BaseModel):
    status: LearningRunStatus
    reason: Optional[str] = None


router = APIRouter()

@router.post("/humans", response_model=ActorResponse)
def create_human(data: HumanCreate, db: Session = Depends(get_db)):
    return actors.create_human(db, data)

@router.post("/agents", response_model=ActorResponse)
def create_agent(data: AgentCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return actors.create_agent(db, data)

@router.post("/consent-profiles/{actor_id}", response_model=ConsentProfileResponse)
def update_consent(actor_id: str, data: ConsentProfileUpdate, db: Session = Depends(get_db)):
    return actors.upsert_consent_profile(db, actor_id, data)

@router.get("/consent-profiles/{actor_id}", response_model=ConsentProfileResponse)
def get_consent(actor_id: str, db: Session = Depends(get_db)):
    profile = db.query(ConsentProfile).filter_by(actor_id=actor_id).first()
    if not profile:
        raise HTTPException(status_code=404, detail="Not found")
    return profile

@router.post("/grounding-requests", response_model=GroundingRequestResponse)
def create_request(data: GroundingRequestCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return req_service.create_grounding_request(db, data)

@router.get("/grounding-requests/{request_id}", response_model=GroundingRequestResponse)
def get_request(request_id: str, db: Session = Depends(get_db)):
    req = db.query(GroundingRequest).filter_by(id=request_id).first()
    if not req:
        raise HTTPException(status_code=404, detail="Not found")
    return req

@router.post("/grounding-requests/{request_id}/evaluate", response_model=ViabilityEvaluationResult)       
def evaluate_request(request_id: str, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return req_service.evaluate_request_viability(db, request_id)

@router.post("/contributions", response_model=ContributionResponse)
def submit_contribution(data: ContributionCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    try:
        return contributions.submit_contribution(db, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/contributions/{contribution_id}/evaluate", response_model=ViabilityEvaluationResult)       
def evaluate_contribution(contribution_id: str, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return contributions.evaluate_contribution(db, contribution_id)

@router.post("/settlements/{request_id}/attempt")
def attempt_settlement(request_id: str, data: SettlementAttempt, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return settlement.attempt_settlement(db, request_id, data)

@router.post("/revocations", response_model=RevocationResponse)
def request_revocation(data: RevocationCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return revocation.request_revocation(db, data)

@router.get("/review-queue", response_model=List[ReviewQueueResponse])
def get_review_queue(db: Session = Depends(get_db)):
    return actors.get_review_queue(db)

@router.post("/review-queue/{item_id}/resolve")
def resolve_review_item(item_id: str, operator_id: str, action: str, rationale: str, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    return actors.resolve_review_item(db, item_id, operator_id, action, rationale)

@router.get("/audit-events", response_model=List[AuditEventResponse])
def get_audit_events(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    return (
        db.query(AuditEvent)
        .order_by(AuditEvent.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )


# ── Collection / List Endpoints ──────────────────────────────────────

@router.get("/actors", response_model=List[ActorResponse])
def list_actors(
    actor_type: Optional[ActorType] = Query(default=None, description="Filter by actor type"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Actor)
    if actor_type is not None:
        q = q.filter(Actor.actor_type == actor_type)
    return q.order_by(Actor.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/grounding-requests", response_model=List[GroundingRequestResponse])
def list_grounding_requests(
    agent_id: Optional[str] = Query(default=None, description="Filter by agent ID"),
    status: Optional[RequestStatus] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(GroundingRequest)
    if agent_id:
        q = q.filter(GroundingRequest.agent_id == agent_id)
    if status:
        q = q.filter(GroundingRequest.status == status)
    return q.order_by(GroundingRequest.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/contributions", response_model=List[ContributionResponse])
def list_contributions(
    human_id: Optional[str] = Query(default=None, description="Filter by human ID"),
    request_id: Optional[str] = Query(default=None, description="Filter by grounding request ID"),
    status: Optional[ContributionStatus] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(Contribution)
    if human_id:
        q = q.filter(Contribution.human_id == human_id)
    if request_id:
        q = q.filter(Contribution.request_id == request_id)
    if status:
        q = q.filter(Contribution.status == status)
    return q.order_by(Contribution.created_at.desc()).offset(offset).limit(limit).all()


@router.get("/learning-runs", response_model=List[LearningRunResponse])
def list_learning_runs(
    status: Optional[LearningRunStatus] = Query(default=None, description="Filter by status"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    q = db.query(LearningRun)
    if status:
        q = q.filter(LearningRun.status == status)
    return q.order_by(LearningRun.created_at.desc()).offset(offset).limit(limit).all()


# ── API Key Management ───────────────────────────────────────────────

@router.post("/api-keys", response_model=ApiKeyResponse, status_code=201)
def create_api_key(
    data: ApiKeyCreate,
    db: Session = Depends(get_db),
    _key: str = Depends(require_api_key),
):
    """Create a new per-agent API key.

    The raw key is returned **once** in ``raw_key`` and is never stored.
    Only the SHA-256 hash is persisted.  Callers must save the raw key
    immediately — it cannot be recovered after this response.
    """
    # Verify agent exists
    agent = db.query(Actor).filter_by(id=data.agent_id, actor_type=ActorType.agent).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    raw_key = f"mk_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(raw_key.encode()).hexdigest()

    api_key_record = ApiKey(
        agent_id=data.agent_id,
        key_hash=key_hash,
        label=data.label,
    )
    db.add(api_key_record)
    db.flush()
    emit_audit_event(
        db,
        AuditEventType.api_key_created,
        actor_id=data.agent_id,
        entity_id=api_key_record.id,
        metadata={"label": data.label},
    )
    db.commit()
    db.refresh(api_key_record)

    return ApiKeyResponse(
        id=api_key_record.id,
        agent_id=api_key_record.agent_id,
        label=api_key_record.label,
        created_at=api_key_record.created_at,
        revoked_at=api_key_record.revoked_at,
        raw_key=raw_key,  # returned once only
    )


@router.delete("/api-keys/{key_id}", status_code=204)
def revoke_api_key(
    key_id: str,
    db: Session = Depends(get_db),
    _key: str = Depends(require_api_key),
):
    """Revoke a per-agent API key by setting its ``revoked_at`` timestamp."""
    api_key_record = db.query(ApiKey).filter_by(id=key_id).first()
    if not api_key_record:
        raise HTTPException(status_code=404, detail="API key not found")
    if api_key_record.revoked_at is not None:
        raise HTTPException(status_code=409, detail="API key already revoked")

    api_key_record.revoked_at = datetime.now(timezone.utc)
    db.flush()
    emit_audit_event(
        db,
        AuditEventType.api_key_revoked,
        actor_id=api_key_record.agent_id,
        entity_id=api_key_record.id,
    )
    db.commit()


# ── Agentic Run Endpoints ────────────────────────────────────────────

def _get_run_or_404(db: Session, run_id: str) -> LearningRun:
    run = db.query(LearningRun).filter_by(id=run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="LearningRun not found")
    return run

@router.post("/learning-runs", response_model=LearningRunResponse)
def create_learning_run(data: LearningRunCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    run = LearningRun(**data.model_dump())
    db.add(run)
    db.flush()
    emit_audit_event(db, AuditEventType.actor_created, entity_id=run.id, metadata={"type": "learning_run"})
    db.commit()
    db.refresh(run)
    return run

@router.get("/learning-runs/{run_id}", response_model=LearningRunDetailResponse)
def get_learning_run(run_id: str, db: Session = Depends(get_db)):
    run = _get_run_or_404(db, run_id)
    return LearningRunDetailResponse(
        id=run.id, title=run.title, objective=run.objective,
        hypothesis_or_goal=run.hypothesis_or_goal, environment=run.environment,
        model_or_agent_version=run.model_or_agent_version,
        status=run.status, created_at=run.created_at,
        plans=[PlanResponse(id=p.id, run_id=p.run_id, title=p.title, objective=p.objective, status=p.status, created_at=p.created_at) for p in run.plans],
        actions=[ActionReceiptResponse(id=a.id, run_id=a.run_id, plan_id=a.plan_id, action_type=a.action_type, status=a.status, started_at=a.started_at) for a in run.actions],
        approvals=[ApprovalDecisionResponse(id=a.id, run_id=a.run_id, target_ref=a.target_ref, decision=a.decision, rationale=a.rationale, created_at=a.created_at) for a in run.approvals],
        verifications=[VerificationResponse(id=v.id, run_id=v.run_id, claim_text=v.claim_text, lane=v.lane, verdict=v.verdict, confidence=v.confidence, rationale=v.rationale, review_status=v.review_status, correlated_verifier_risk=v.correlated_verifier_risk, policy_state=v.policy_state.value if v.policy_state else None) for v in run.verifications],
        rollbacks=[RollbackRecordResponse(id=r.id, run_id=r.run_id, action_id=r.action_id, prior_state_ref=r.prior_state_ref, rollback_trigger=r.rollback_trigger, executed_at=r.executed_at) for r in run.rollbacks]
    )

@router.post("/learning-runs/{run_id}/plans", response_model=PlanResponse)
def add_plan(run_id: str, data: PlanCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    _get_run_or_404(db, run_id)
    plan = Plan(run_id=run_id, **data.model_dump())
    # Auto-advance to SUBMITTED when plan is lodged with the platform
    plan.status = PlanStatus.submitted
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan

@router.post("/learning-runs/{run_id}/actions", response_model=ActionReceiptResponse)
def add_action(run_id: str, data: ActionReceiptCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    run = _get_run_or_404(db, run_id)
    
    if run.status in [LearningRunStatus.settled, LearningRunStatus.failed, LearningRunStatus.rolled_back]:
        raise HTTPException(status_code=400, detail="Cannot add actions to a closed LearningRun")
        
    if data.plan_id:
        plan = db.query(Plan).filter_by(id=data.plan_id, run_id=run_id).first()
        if not plan:
            raise HTTPException(status_code=404, detail="Plan not found")
        if plan.status != PlanStatus.approved:
            raise HTTPException(status_code=400, detail=f"Cannot execute action for plan in status: {plan.status.value}")

    action = ActionReceipt(run_id=run_id, **data.model_dump())
    db.add(action)
    db.commit()
    db.refresh(action)
    return action

@router.post("/learning-runs/{run_id}/approvals", response_model=ApprovalDecisionResponse)
def add_approval(run_id: str, data: ApprovalDecisionCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    _get_run_or_404(db, run_id)
    approval = ApprovalDecision(run_id=run_id, **data.model_dump())
    db.add(approval)

    # ── Plan state cascade ───────────────────────────────────────────────
    # If the approval targets a Plan in this run, update the plan's status.
    if data.target_ref:
        plan = db.query(Plan).filter_by(id=data.target_ref, run_id=run_id).first()
        if plan:
            if data.decision == ApprovalDecisionEnum.approved:
                plan.status = PlanStatus.approved
            elif data.decision == ApprovalDecisionEnum.rejected:
                plan.status = PlanStatus.rejected
    # ────────────────────────────────────────────────────────────────────

    db.commit()
    db.refresh(approval)
    return approval

@router.post("/learning-runs/{run_id}/verify", response_model=VerificationResponse)
def verify_claim(run_id: str, data: VerificationRequest, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    _get_run_or_404(db, run_id)
    pipeline = VerificationPipeline()
    result = pipeline.run_pipeline(data.claim_text, data.lane, data.evidence, data.endogenous_only)
    
    vr = VerificationResult(
        run_id=run_id,
        target_ref=data.target_ref,
        claim_text=data.claim_text,
        lane=result["lane"],
        verdict=result["verdict"],
        confidence=result["confidence"],
        rationale=result["rationale"],
        review_status=result["review_status"],
        correlated_verifier_risk=result["correlated_verifier_risk"],
        policy_state=result["policy_state"] if isinstance(result["policy_state"], str) else result["policy_state"].value,
        provenance_bundle=result["provenance_bundle"]
    )
    db.add(vr)
    db.commit()
    db.refresh(vr)
    
    return VerificationResponse(
        id=vr.id, run_id=vr.run_id, claim_text=vr.claim_text,
        lane=vr.lane, verdict=vr.verdict, confidence=vr.confidence,
        rationale=vr.rationale, review_status=vr.review_status,
        correlated_verifier_risk=vr.correlated_verifier_risk,
        policy_state=vr.policy_state.value if hasattr(vr.policy_state, 'value') else vr.policy_state,
        provenance_bundle=vr.provenance_bundle
    )

@router.post("/learning-runs/{run_id}/rollbacks", response_model=RollbackRecordResponse)
def add_rollback(run_id: str, data: RollbackRecordCreate, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    run = _get_run_or_404(db, run_id)
    rollback = RollbackRecord(run_id=run_id, **data.model_dump())
    db.add(rollback)
    # Transition the run to ROLLED_BACK state
    run.status = LearningRunStatus.rolled_back
    db.commit()
    db.refresh(rollback)
    return rollback

@router.post("/learning-runs/{run_id}/settle")
def settle_learning_run(run_id: str, data: SettlementAttempt, db: Session = Depends(get_db), api_key: str = Depends(require_api_key)):
    _get_run_or_404(db, run_id)
    return settlement.attempt_run_settlement(db, run_id, data)


# ── Action PATCH ─────────────────────────────────────────────────────

@router.patch("/learning-runs/{run_id}/actions/{action_id}", response_model=ActionReceiptResponse)
def update_action(
    run_id: str,
    action_id: str,
    data: ActionStatusPatch,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key)
):
    _get_run_or_404(db, run_id)
    action = db.query(ActionReceipt).filter_by(id=action_id, run_id=run_id).first()
    if not action:
        raise HTTPException(status_code=404, detail="ActionReceipt not found")
    action.status = data.status
    if data.output_refs is not None:
        action.output_refs = data.output_refs
    db.commit()
    db.refresh(action)
    return action


# ── Verification PATCH ───────────────────────────────────────────────

@router.patch("/learning-runs/{run_id}/verifications/{verification_id}", response_model=VerificationResponse)
def update_verification(
    run_id: str,
    verification_id: str,
    data: VerificationStatusPatch,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key)
):
    run = _get_run_or_404(db, run_id)
    verification = db.query(VerificationResult).filter_by(id=verification_id, run_id=run_id).first()
    if not verification:
        raise HTTPException(status_code=404, detail="Verification not found")
        
    if data.review_status is not None:
        verification.review_status = data.review_status
    if data.policy_state is not None:
        verification.policy_state = data.policy_state
    if data.rationale is not None:
        verification.rationale = f"{verification.rationale} | {data.rationale}" if verification.rationale else data.rationale

    db.commit()
    db.refresh(verification)
    
    return VerificationResponse(
        id=verification.id, run_id=verification.run_id, claim_text=verification.claim_text,
        lane=verification.lane, verdict=verification.verdict, confidence=verification.confidence,
        rationale=verification.rationale, review_status=verification.review_status,
        correlated_verifier_risk=verification.correlated_verifier_risk,
        policy_state=verification.policy_state.value if hasattr(verification.policy_state, 'value') else verification.policy_state,
        provenance_bundle=verification.provenance_bundle
    )


# ── Run PATCH (operator override) ─────────────────────────────────────

@router.patch("/learning-runs/{run_id}", response_model=LearningRunResponse)
def update_run_status(
    run_id: str,
    data: RunStatusPatch,
    db: Session = Depends(get_db),
    api_key: str = Depends(require_api_key)
):
    run = _get_run_or_404(db, run_id)
    run.status = data.status
    db.flush()
    emit_audit_event(db, AuditEventType.actor_created, entity_id=run.id, metadata={
        "type": "run_status_override",
        "new_status": data.status.value,
        "reason": data.reason
    })
    db.commit()
    db.refresh(run)
    return run

