from sqlalchemy.orm import Session
from app.models.domain import GroundingRequest, ViabilityEvaluation, ReviewQueueItem, Actor
from app.models.enums import RequestStatus, AuditEventType, ViabilityDecision, ActorType
from app.schemas import GroundingRequestCreate
from app.services.audit import emit_audit_event
from app.services.policy import load_policy
from app.services.viability.engines import RequestViabilityEngine
import logging

logger = logging.getLogger(__name__)

def create_grounding_request(db: Session, data: GroundingRequestCreate) -> GroundingRequest:
    # GAP 1 Fix: Verify agent exists
    actor = db.query(Actor).filter_by(id=data.agent_id, actor_type=ActorType.agent).first()
    if not actor:
        raise ValueError(f"Agent actor '{data.agent_id}' not found")

    req = GroundingRequest(**data.model_dump())
    req.status = RequestStatus.draft
    db.add(req)
    db.flush()
    emit_audit_event(db, AuditEventType.request_created, actor_id=req.agent_id, entity_id=req.id)
    db.commit()
    db.refresh(req)
    return req

def evaluate_request_viability(db: Session, request_id: str) -> ViabilityEvaluation:
    req = db.query(GroundingRequest).filter_by(id=request_id).first()
    if not req:
        raise ValueError("Request not found")

    policy = load_policy(db)

    # Fetch previous evaluation for hysteresis
    previous_eval_obj = db.query(ViabilityEvaluation).filter_by(
        entity_type="request", entity_id=request_id
    ).order_by(ViabilityEvaluation.created_at.desc()).first()
    previous_eval = None
    if previous_eval_obj:
        previous_eval = {"status": previous_eval_obj.status.value, "gates": previous_eval_obj.gates_evaluated}

    engine = RequestViabilityEngine(policy)
    # Map ORM fields to the engine's expected keys explicitly — never pass __dict__
    req_dict = {
        "id": req.id,
        "request_class": req.lived_experience_axis.value if req.lived_experience_axis else None,
        "description": req.description,
        "effective_hourly_estimate": req.effective_hourly_estimate,
        "entropy_reduction_hypothesis": req.entropy_reduction_hypothesis,
        "phase_synchronization_hypothesis": req.phase_synchronization_hypothesis,
        "evidence_artifact_description": req.evidence_artifact_description,
    }
    result = engine.evaluate(req_dict, previous_eval=previous_eval)

    eval_record = ViabilityEvaluation(
        entity_type="request",
        entity_id=req.id,
        overall_viability=result["overall_viability"],
        status=ViabilityDecision(result["status"]),
        gates_evaluated=result["gates"],
        decision_rationale=result["decision_rationale"],
        evaluation_hash=result["evaluation_hash"]
    )
    db.add(eval_record)

    if result["status"] == "PASS":
        req.status = RequestStatus.open
    elif result["status"] == "FAIL":
        req.status = RequestStatus.cancelled
    else:
        req.status = RequestStatus.review
        rqi = ReviewQueueItem(
            entity_type="request",
            entity_id=req.id,
            reason=result["decision_rationale"],
            evidence_links={
                "evaluation_hash": result["evaluation_hash"],
                "gates_evaluated": result["gates"],
                "policy_snapshot": req.viability_policy_snapshot,
            },
        )
        db.add(rqi)

    db.flush()
    emit_audit_event(db, AuditEventType.request_evaluated, entity_id=req.id, metadata={"result": result["status"]})
    db.commit()
    db.refresh(eval_record)
    return eval_record
