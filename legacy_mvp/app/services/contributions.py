from sqlalchemy.orm import Session
from app.models.domain import Contribution, ProvenanceRecord, ConsentProfile, ViabilityEvaluation, ReviewQueueItem, GroundingRequest
from app.models.enums import ContributionStatus, AuditEventType, ViabilityDecision
from app.schemas import ContributionCreate
from app.services.audit import emit_audit_event
from app.services.policy import load_policy
from app.services.viability.engines import ContributionViabilityEngine
from app.services.attestation import AttestationPipeline

def submit_contribution(db: Session, data: ContributionCreate) -> Contribution:
    # Snapshot consent
    consent = db.query(ConsentProfile).filter_by(actor_id=data.human_id).first()
    if not consent:
        raise ValueError("Consent profile required")

    # Full point-in-time snapshot of consent profile
    consent_snapshot = {
        "allowed_axes": [a.value if hasattr(a, 'value') else str(a) for a in consent.allowed_axes],
        "max_task_duration_minutes": consent.max_task_duration_minutes,
        "privacy_level": consent.privacy_level.value if hasattr(consent.privacy_level, 'value') else str(consent.privacy_level),
        "allow_model_training": consent.allow_model_training,
        "allow_public_aggregate": consent.allow_public_aggregate,
        "min_hourly_equivalent_usd": consent.min_hourly_equivalent_usd,
        "payout_threshold_usd": consent.payout_threshold_usd,
        "payout_method": consent.payout_method,
        "daily_max_contributions": consent.daily_max_contributions,
        "weekly_max_hours": consent.weekly_max_hours,
        "require_full_provenance": consent.require_full_provenance,
        "revocation_window_days": consent.revocation_window_days
    }

    contrib = Contribution(
        human_id=data.human_id,
        request_id=data.request_id,
        content_payload=data.content_payload,
        consent_snapshot=consent_snapshot
    )
    db.add(contrib)
    db.flush()

    attestation_bundle_dict = (
        data.attestation_bundle.model_dump() if data.attestation_bundle else None
    )

    prov = ProvenanceRecord(
        contribution_id=contrib.id,
        human_id=data.human_id,
        request_id=data.request_id,
        integrity_hash=data.integrity_hash,
        creation_context=data.creation_context,
        attestation_bundle=attestation_bundle_dict,
        usage_rights={"retention": consent.revocation_window_days}
    )
    db.add(prov)
    
    emit_audit_event(db, AuditEventType.contribution_submitted, actor_id=data.human_id, entity_id=contrib.id)
    db.commit()
    db.refresh(contrib)
    return contrib

def evaluate_contribution(db: Session, contribution_id: str) -> ViabilityEvaluation:
    contrib = db.query(Contribution).filter_by(id=contribution_id).first()
    if not contrib:
        raise ValueError("Contribution not found")

    prov = db.query(ProvenanceRecord).filter_by(contribution_id=contribution_id).first()

    policy = load_policy(db)
    engine = ContributionViabilityEngine(policy)

    req = db.query(GroundingRequest).filter_by(id=contrib.request_id).first()

    # Phase D: run attestation pipeline to derive composite provenance score.
    # Pass the attestation sub-section of the policy so thresholds and the
    # require_cryptographic_attestation flag are read from the YAML at runtime.
    attestation_policy = policy.get("contribution_gates", {}).get("attestation", {})
    pipeline_result = AttestationPipeline(attestation_policy=attestation_policy).run(
        creation_context=prov.creation_context,
        attestation_bundle=prov.attestation_bundle,
    )

    c_dict = {
        "id": contrib.id,
        "axis": req.lived_experience_axis.value if req else "unknown",
        "content_payload": contrib.content_payload,
        "integrity_hash": prov.integrity_hash,
        "provenance_score": pipeline_result["composite_provenance_score"],
    }

    result = engine.evaluate(c_dict, contrib.consent_snapshot)

    # Escalate to REVIEW_REQUIRED if the attestation pipeline raised a flag,
    # even when the composite score passes the viability gate.  Two causes:
    #   (a) timing anomalies — behavioral signals inconsistent with human input
    #   (b) require_cryptographic_attestation is True but no valid proof supplied
    if pipeline_result["escalate"] and result["status"] == "PASS":
        result["status"] = "REVIEW_REQUIRED"
        reasons: list[str] = []
        if pipeline_result["correlation"]["anomalies"]:
            reasons.append(
                "Timing anomalies: "
                + "; ".join(pipeline_result["correlation"]["anomalies"])
            )
        if pipeline_result["crypto_requirement_unmet"]:
            reasons.append(
                "Cryptographic attestation required by policy but no valid proof supplied"
            )
        result["decision_rationale"] = " | ".join(reasons)

    eval_record = ViabilityEvaluation(
        entity_type="contribution",
        entity_id=contrib.id,
        overall_viability=result["overall_viability"],
        status=ViabilityDecision(result["status"]),
        gates_evaluated=result["gates"],
        decision_rationale=result["decision_rationale"],
        evaluation_hash=result["evaluation_hash"]
    )
    db.add(eval_record)

    if result["status"] == "PASS":
        contrib.status = ContributionStatus.verified
    elif result["status"] == "FAIL":
        contrib.status = ContributionStatus.rejected
    else:
        contrib.status = ContributionStatus.review
        rqi = ReviewQueueItem(
            entity_type="contribution",
            entity_id=contrib.id,
            reason=result["decision_rationale"],
            evidence_links={
                "evaluation_hash": result["evaluation_hash"],
                "gates_evaluated": result["gates"],
                "composite_provenance_score": pipeline_result["composite_provenance_score"],
                "base_humanity_score": pipeline_result["base_humanity_score"],
                "crypto_bonus": pipeline_result["crypto_bonus"],
                "timing_penalty": pipeline_result["timing_penalty"],
                "timing_anomalies": pipeline_result["correlation"]["anomalies"],
                "verifier_results": pipeline_result["verifier_results"],
                "crypto_requirement_unmet": pipeline_result["crypto_requirement_unmet"],
            },
        )
        db.add(rqi)

    db.flush()
    emit_audit_event(
        db, AuditEventType.contribution_evaluated, entity_id=contrib.id,
        metadata={
            "result": result["status"],
            "composite_provenance_score": pipeline_result["composite_provenance_score"],
            "timing_penalty": pipeline_result["timing_penalty"],
            "crypto_bonus": pipeline_result["crypto_bonus"],
            "escalated": pipeline_result["escalate"],
        }
    )
    db.commit()
    db.refresh(eval_record)
    return eval_record
