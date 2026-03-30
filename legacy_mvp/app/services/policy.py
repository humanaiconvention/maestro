"""Policy loader with audit trail.

Every call to :func:`load_policy` reads ``viability_policy.yaml``, computes a
SHA-256 fingerprint of the raw YAML, and emits an ``AuditEvent`` with:

  * ``hash``    — current SHA-256 of the file
  * ``changed`` — True when the hash differs from the previous load in this
                  process (module-level cache).  In production with multiple
                  workers each worker tracks its own last-seen hash; a shared
                  Redis key could be used for process-wide change detection but
                  is out of scope for the MVP.

All three service modules (requests, contributions, settlement) call this
function instead of opening the YAML directly so that every policy read is
recorded in the append-only audit log.
"""
from __future__ import annotations

import hashlib
import logging
import os
from typing import Any

import yaml
from sqlalchemy.orm import Session

from app.models.enums import AuditEventType
from app.services.audit import emit_audit_event

logger = logging.getLogger(__name__)

# Module-level cache: absolute path → last-seen SHA-256 hex
_last_policy_hash: dict[str, str] = {}

_DEFAULT_POLICY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "config",
    "viability_policy.yaml",
)


def load_policy(db: Session, path: str | None = None) -> dict[str, Any]:
    """Load ``viability_policy.yaml`` and emit an audit event.

    Args:
        db:   SQLAlchemy session — used only to emit the audit event.
        path: Override the default config path (useful in tests).

    Returns:
        Parsed policy dict, or ``{}`` if the file is not found.
    """
    resolved = path or _DEFAULT_POLICY_PATH

    try:
        with open(resolved, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except FileNotFoundError:
        logger.warning("viability_policy.yaml not found at %s — using empty policy", resolved)
        emit_audit_event(
            db,
            AuditEventType.policy_loaded,
            metadata={
                "policy_path": os.path.basename(resolved),
                "hash": None,
                "changed": None,
                "error": "file_not_found",
            },
        )
        return {}

    current_hash = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    previous_hash = _last_policy_hash.get(resolved)
    changed = previous_hash is not None and previous_hash != current_hash
    _last_policy_hash[resolved] = current_hash

    emit_audit_event(
        db,
        AuditEventType.policy_loaded,
        metadata={
            "policy_path": os.path.basename(resolved),
            "hash": current_hash,
            "changed": changed,
            "previous_hash": previous_hash,
        },
    )

    return yaml.safe_load(raw) or {}
