from sqlalchemy.orm import Session
from app.models.domain import AuditEvent
from app.models.enums import AuditEventType

def emit_audit_event(db: Session, event_type: AuditEventType, actor_id: str = None, entity_id: str = None, metadata: dict = None):
    event = AuditEvent(
        event_type=event_type,
        actor_id=actor_id,
        entity_id=entity_id,
        metadata_json=metadata or {}
    )
    db.add(event)
    db.flush()   # Stage the row — caller's commit will persist it
    return event
