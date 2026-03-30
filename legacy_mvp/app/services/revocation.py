from sqlalchemy.orm import Session
from app.models.domain import RevocationRequest, Contribution, ProvenanceRecord
from app.models.enums import RevocationStatus, ContributionStatus, AuditEventType
from app.schemas import RevocationCreate
from app.services.audit import emit_audit_event
from datetime import datetime, timezone, timedelta
from fastapi import HTTPException
import yaml
import os
import logging

logger = logging.getLogger(__name__)

def request_revocation(db: Session, data: RevocationCreate) -> RevocationRequest:
    contrib = db.query(Contribution).filter_by(id=data.contribution_id).first()
    if not contrib:
        raise ValueError("Contribution not found")
        
    if contrib.human_id != data.human_id:
        raise ValueError("Unauthorized")

    # Load policy
    policy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "viability_policy.yaml")
    try:
        with open(policy_path, "r") as f:
            policy = yaml.safe_load(f)
    except FileNotFoundError:
        policy = {}
    
    # revocation.window_days default 90
    window_days = policy.get("revocation", {}).get("window_days", 90)
    
    # Check if within window
    now = datetime.now(timezone.utc)
    contrib_created_at = contrib.created_at
    if contrib_created_at.tzinfo is None:
        contrib_created_at = contrib_created_at.replace(tzinfo=timezone.utc)
        
    if now - contrib_created_at > timedelta(days=window_days):
        # Window expired
        rev = RevocationRequest(
            contribution_id=data.contribution_id,
            human_id=data.human_id,
            reason=data.reason,
            status=RevocationStatus.failed
        )
        db.add(rev)
        db.flush()
        
        # Use revocation_requested and add metadata for the reason
        emit_audit_event(
            db, 
            AuditEventType.revocation_requested, 
            actor_id=data.human_id, 
            entity_id=rev.id, 
            metadata={"reason": "revocation_window_expired"}
        )
        
        db.commit()
        raise HTTPException(status_code=400, detail="Revocation window has expired")

    prov = db.query(ProvenanceRecord).filter_by(contribution_id=contrib.id).first()

    rev = RevocationRequest(
        contribution_id=data.contribution_id,
        human_id=data.human_id,
        reason=data.reason,
        status=RevocationStatus.completed
    )
    db.add(rev)

    # Update downstream
    contrib.status = ContributionStatus.withdrawn
    if prov:
        prov.withdrawal_status = ContributionStatus.withdrawn

    db.flush()
    emit_audit_event(db, AuditEventType.revocation_completed, actor_id=data.human_id, entity_id=rev.id)   
    db.commit()
    db.refresh(rev)
    return rev
