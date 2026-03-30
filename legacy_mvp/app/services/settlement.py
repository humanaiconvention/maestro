from sqlalchemy.orm import Session
from app.models.domain import Settlement, GroundingRequest, ViabilityEvaluation, ReviewQueueItem, Contribution, ProvenanceRecord, LearningRun, VerificationResult
from app.models.enums import SettlementStatus, AuditEventType, ViabilityDecision, ReviewStatus, PolicyState, LearningRunStatus
from app.schemas import SettlementAttempt
from app.services.audit import emit_audit_event
from app.services.policy import load_policy
from app.services.viability.engines import SettlementViabilityEngine

def attempt_settlement(db: Session, request_id: str, data: SettlementAttempt) -> dict:
    req = db.query(GroundingRequest).filter_by(id=request_id).first()
    if not req:
        raise ValueError("Request not found")
        
    policy = load_policy(db)
    engine = SettlementViabilityEngine(policy)

    # Pull policy context from first contribution
    contrib = db.query(Contribution).filter_by(request_id=request_id).first()
    usage_rights = []
    allow_model_training = True
    if contrib:
        allow_model_training = contrib.consent_snapshot.get("allow_model_training", True)
        prov = db.query(ProvenanceRecord).filter_by(contribution_id=contrib.id).first()
        if prov:
            u_rights = prov.usage_rights
            usage_rights = u_rights if isinstance(u_rights, list) else list(u_rights.keys())

    s_dict = {
        "claimed_entropy_reduction": data.claimed_entropy_reduction,
        "payout_amount": data.total_payout,
        "effective_hourly_estimate": req.effective_hourly_estimate,
        "duration_minutes": req.estimated_duration_minutes,
        "usage_rights": usage_rights,
        "allow_model_training": allow_model_training
    }

    result = engine.evaluate(req.id, s_dict)

    eval_record = ViabilityEvaluation(
        entity_type="settlement",
        entity_id=req.id,
        overall_viability=result["overall_viability"],
        status=ViabilityDecision(result["status"]),
        gates_evaluated=result["gates"],
        decision_rationale=result["decision_rationale"],
        evaluation_hash=result["evaluation_hash"]
    )
    db.add(eval_record)

    settlement = Settlement(
        request_id=req.id,
        epistemic_value_proof=data.model_dump(),
        viability_proof_hash=result["evaluation_hash"],
        total_payout=data.total_payout,
        contributions_settled=data.observation_count,
        status=SettlementStatus.completed if result["status"] == "PASS" else SettlementStatus.review      
    )
    db.add(settlement)

    if result["status"] != "PASS":
        rqi = ReviewQueueItem(
            entity_type="settlement",
            entity_id=req.id,
            reason=result["decision_rationale"],
            evidence_links={
                "evaluation_hash": result["evaluation_hash"],
                "gates_evaluated": result["gates"],
                "claimed_entropy_reduction": data.claimed_entropy_reduction,
                "payout_amount": data.total_payout,
            },
        )
        db.add(rqi)

    db.flush()
    emit_audit_event(db, AuditEventType.settlement_attempted, entity_id=req.id, metadata={"status": settlement.status.value})
    db.commit()
    db.refresh(settlement)
    
    return {
        "id": settlement.id,
        "status": settlement.status.value,
        "total_payout": settlement.total_payout,
        "evaluation": result
    }

def attempt_run_settlement(db: Session, learning_run_id: str, data: SettlementAttempt) -> dict:
    run = db.query(LearningRun).filter_by(id=learning_run_id).first()
    if not run:
        raise ValueError("LearningRun not found")

    # ── Escalation Block ────────────────────────────────────────────────
    # Do not allow settlement if any verifications are pending or escalated.
    open_reviews = (
        db.query(VerificationResult)
        .filter_by(run_id=learning_run_id)
        .filter(VerificationResult.review_status.in_([ReviewStatus.pending, ReviewStatus.escalated]))
        .count()
    )
    if open_reviews > 0:
        run.status = LearningRunStatus.settlement_pending
        db.commit()
        return {
            "status": "SETTLEMENT_PENDING",
            "learning_run_id": run.id,
            "blocked_reason": f"{open_reviews} verification(s) require review or resolution before settlement",
            "evaluation": None
        }
    # ────────────────────────────────────────────────────────────────────

    policy = load_policy(db)
    engine = SettlementViabilityEngine(policy)

    s_dict = {
        "claimed_entropy_reduction": data.claimed_entropy_reduction,
        "payout_amount": data.total_payout,
        "effective_hourly_estimate": 60.0,
        "duration_minutes": 60,
        "usage_rights": [],
        "allow_model_training": True
    }

    result = engine.evaluate(run.id, s_dict)
    passed = result["status"] == "PASS"

    eval_record = ViabilityEvaluation(
        entity_type="settlement_run",
        entity_id=run.id,
        overall_viability=result["overall_viability"],
        status=ViabilityDecision(result["status"]),
        gates_evaluated=result["gates"],
        decision_rationale=result["decision_rationale"],
        evaluation_hash=result["evaluation_hash"]
    )
    db.add(eval_record)

    settlement_status = SettlementStatus.completed if passed else SettlementStatus.review
    settlement = Settlement(
        learning_run_id=run.id,
        epistemic_value_proof=data.model_dump(),
        viability_proof_hash=result["evaluation_hash"],
        total_payout=data.total_payout,
        contributions_settled=data.observation_count,
        status=settlement_status
    )
    db.add(settlement)

    # ── Transition run status ───────────────────────────────────────────
    run.status = LearningRunStatus.settled if passed else LearningRunStatus.failed
    # ────────────────────────────────────────────────────────────────────

    if not passed:
        rqi = ReviewQueueItem(
            entity_type="settlement_run",
            entity_id=run.id,
            reason=result["decision_rationale"],
            evidence_links={
                "evaluation_hash": result["evaluation_hash"],
                "gates_evaluated": result["gates"],
                "claimed_entropy_reduction": data.claimed_entropy_reduction,
                "payout_amount": data.total_payout,
                "learning_run_id": run.id,
            },
        )
        db.add(rqi)

    db.flush()
    emit_audit_event(db, AuditEventType.settlement_attempted, entity_id=run.id, metadata={
        "status": settlement.status.value,
        "run_status": run.status.value,
        "scope": "learning_run"
    })
    db.commit()
    db.refresh(settlement)

    return {
        "id": settlement.id,
        "status": settlement.status.value,
        "learning_run_id": settlement.learning_run_id,
        "total_payout": settlement.total_payout,
        "evaluation": result
    }

