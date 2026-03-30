from sqlalchemy.orm import Session
from app.models.domain import Actor, ConsentProfile, ReviewQueueItem
from app.models.enums import ActorType, AuditEventType
from app.services.audit import emit_audit_event
from app.schemas import HumanCreate, AgentCreate, ConsentProfileUpdate
import json

def create_human(db: Session, data: HumanCreate) -> Actor:
    actor = Actor(actor_type=ActorType.human, display_name=data.display_name)
    db.add(actor)
    db.flush()
    emit_audit_event(db, AuditEventType.actor_created, actor_id=actor.id, metadata={"type": "human"})
    db.commit()
    db.refresh(actor)
    return actor

def create_agent(db: Session, data: AgentCreate) -> Actor:
    actor = Actor(actor_type=ActorType.agent, display_name=data.display_name)
    db.add(actor)
    db.flush()
    emit_audit_event(db, AuditEventType.actor_created, actor_id=actor.id, metadata={"type": "agent"})
    db.commit()
    db.refresh(actor)
    return actor

def upsert_consent_profile(db: Session, actor_id: str, data: ConsentProfileUpdate) -> ConsentProfile:
    profile = db.query(ConsentProfile).filter_by(actor_id=actor_id).first()
    if not profile:
        profile = ConsentProfile(actor_id=actor_id)
        db.add(profile)
    
    for k, v in data.model_dump().items():
        if v is not None:
            setattr(profile, k, v)

    db.flush()
    emit_audit_event(db, AuditEventType.consent_updated, actor_id=actor_id, metadata={"profile_id": profile.id})
    db.commit()
    db.refresh(profile)
    return profile

def get_review_queue(db: Session) -> list[ReviewQueueItem]:
    return db.query(ReviewQueueItem).filter_by(status="pending").all()

def resolve_review_item(db: Session, item_id: str, operator_id: str, action: str, rationale: str) -> ReviewQueueItem:
    item = db.query(ReviewQueueItem).filter_by(id=item_id).first()
    if not item:
        raise ValueError("Item not found")

    item.status = "resolved"
    item.operator_id = operator_id
    item.action_taken = action
    item.rationale = rationale
    
    db.flush()
    emit_audit_event(db, AuditEventType.operator_decision, actor_id=operator_id, entity_id=item.id, metadata={"action": action})
    db.commit()
    db.refresh(item)
    return item
