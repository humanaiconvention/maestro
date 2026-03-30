"""Authentication helpers.

Two-tier key validation:
  1. Per-agent keys — SHA-256 of the raw key is looked up in the ``api_keys``
     table.  A key is valid if it exists and has not been revoked.
  2. Master key fallback — the ``MAESTRO_API_KEY`` env-var is accepted when no
     matching per-agent key is found.  This preserves backward compatibility
     for existing deployments and automated tests.

Key generation is handled by ``POST /api/api-keys``.  The raw key is returned
once at creation; only the hash is stored.
"""
from __future__ import annotations

import hashlib
import os

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.orm import Session

from app.db import get_db

API_KEY_HEADER = APIKeyHeader(name="X-Maestro-API-Key", auto_error=False)
_MASTER_KEY = os.getenv("MAESTRO_API_KEY", "dev-insecure-key-change-me")


def _hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def require_api_key(
    api_key: str = Security(API_KEY_HEADER),
    db: Session = Depends(get_db),
) -> str:
    """FastAPI dependency — returns the validated key or raises 401."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key",
        )

    # Tier 1: per-agent DB lookup (import here to avoid circular import at module load)
    from app.models.domain import ApiKey  # noqa: PLC0415

    key_hash = _hash_key(api_key)
    db_key = (
        db.query(ApiKey)
        .filter_by(key_hash=key_hash)
        .filter(ApiKey.revoked_at.is_(None))
        .first()
    )
    if db_key:
        return api_key

    # Tier 2: master env-var fallback
    if api_key == _MASTER_KEY:
        return api_key

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
