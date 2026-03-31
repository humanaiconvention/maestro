"""
consent.py — One-way consent gate for Maestro gateway.

Consent model (2026 revision):
──────────────────────────────
1. **Binary gate**: participants consent or they don't.  No granular tiers at
   the gate level.  (The 5-layer participation receipt model in
   participation.py is retained for historical records.)
2. **Kernel type**: the participant chooses *transcript-based* or
   *concept-based* grounding at consent time.  This choice is fixed for the
   session.
3. **Attribution token**: a deterministic HMAC-SHA256 token derived from
   (session_id, kernel_type, server secret).  Binds the session to the
   consent choice so it can be re-verified later without storing extra state.

Consent is **one-way**: once the attribution token is issued the participant
can request data removal via the formal withdrawal flow (see
docs/revocation-and-withdrawal.md), but the attribution token itself remains
valid as proof that consent was given at this point in time.

Flow
────
1. Participant solves Altcha PoW → ``POST /v1/session/verify``
   Returns a base JWT with ``human_verified=true``.

2. Participant chooses kernel type → ``POST /v1/session/consent``
   Body: ``{"kernel_type": "transcript" | "concept"}``
   Returns an **upgraded JWT** with consent claims + attribution_token.

3. Chat completions check for consent when ``CONSENT_GATE_ENABLED=true``.
   The ``require_consent`` FastAPI dependency raises HTTP 403 if the JWT
   lacks ``consent=true`` / ``kernel_type``.

   When ``CONSENT_GATE_ENABLED`` is absent or ``"false"`` (the default)
   the gate is a no-op, preserving backward compatibility for deployments
   that do not yet expose the consent step to participants.

Archived model reference
────────────────────────
Pre-2026 the consent model had five independent axes (transcript, felt_state,
gfs_activations, training_signal, retention) with per-axis choices.  That
model is preserved in ``participation.py`` for receipt generation but is no
longer used as a gate condition.
"""

import hashlib
import hmac as _hmac
import logging
import os
import time
from typing import Literal, Optional

import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"

KERNEL_TRANSCRIPT = "transcript"
KERNEL_CONCEPT    = "concept"
KernelType = Literal["transcript", "concept"]

# ─────────────────────────────────────────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────────────────────────────────────────

class ConsentRequest(BaseModel):
    kernel_type: KernelType


class ConsentResponse(BaseModel):
    token:             str   # upgraded JWT carrying consent claims
    attribution_token: str   # opaque HMAC proof of consent (keep for receipt)
    kernel_type:       str
    expires_in:        int   # seconds until token expiry


# ─────────────────────────────────────────────────────────────────────────────
# Attribution token
# ─────────────────────────────────────────────────────────────────────────────

def _make_attribution_token(session_id: str, kernel_type: str, secret: str) -> str:
    """
    Deterministic HMAC-SHA256 attribution token.

    Binds ``(session_id, kernel_type)`` to the server secret so it can be
    re-derived at any future point for verification.  Not forgeable without
    the secret.  The first 16 hex chars are safe to log (128-bit security
    margin is preserved by the full 64-char digest).
    """
    msg = f"{session_id}:{kernel_type}".encode()
    return _hmac.new(secret.encode(), msg, hashlib.sha256).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Route registration
# ─────────────────────────────────────────────────────────────────────────────

def register_consent_routes(app, jwt_secret: str, session_expiry_seconds: int) -> None:
    """
    Register the ``POST /v1/session/consent`` route on *app*.

    Called from ``main.py`` after session routes are registered.
    """
    _security = HTTPBearer()

    @app.post(
        "/v1/session/consent",
        response_model=ConsentResponse,
        tags=["session"],
    )
    def session_consent(
        body: ConsentRequest,
        credentials: HTTPAuthorizationCredentials = Security(_security),
    ):
        """
        Upgrade a verified session JWT with consent and kernel-type claims.

        **Prerequisites:** the caller must hold a JWT issued by
        ``POST /v1/session/verify`` (``human_verified=true``).

        On success a new JWT is returned that additionally carries:
        ``consent=true``, ``kernel_type``, and ``attribution_token``.

        **Idempotent**: submitting the same kernel_type again returns a new
        JWT with an identical attribution_token.  Changing the kernel_type
        after initial consent is not supported and returns HTTP 409.
        """
        token = credentials.credentials
        try:
            payload = jwt.decode(token, jwt_secret, algorithms=[_ALGORITHM])
        except jwt.PyJWTError:
            raise HTTPException(
                status_code=401,
                detail="Invalid or expired session token",
            )

        if not payload.get("human_verified"):
            raise HTTPException(
                status_code=403,
                detail="Session must be human-verified before consent can be recorded",
            )

        session_id = payload.get("session_id") or payload.get("sub")
        if not session_id:
            raise HTTPException(
                status_code=400,
                detail="Session ID missing from token",
            )

        # Idempotency: if consent already recorded with a *different* kernel_type, reject.
        existing_kernel = payload.get("kernel_type")
        if existing_kernel and existing_kernel != body.kernel_type:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Consent already recorded with kernel_type={existing_kernel!r}. "
                    "Kernel type cannot be changed after initial consent."
                ),
            )

        attribution_token = _make_attribution_token(session_id, body.kernel_type, jwt_secret)

        now = int(time.time())
        original_exp = payload.get("exp", now + session_expiry_seconds)
        remaining = max(60, original_exp - now)

        new_claims = {
            **payload,
            "consent":           True,
            "kernel_type":       body.kernel_type,
            "attribution_token": attribution_token,
            "iat":               now,
            "exp":               now + remaining,
        }
        new_token = jwt.encode(new_claims, jwt_secret, algorithm=_ALGORITHM)

        logger.info(
            "consent_recorded",
            extra={
                "session_id":        session_id,
                "kernel_type":       body.kernel_type,
                "attribution_token": attribution_token[:16] + "...",
            },
        )

        return ConsentResponse(
            token=new_token,
            attribution_token=attribution_token,
            kernel_type=body.kernel_type,
            expires_in=remaining,
        )


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency: require_consent
# ─────────────────────────────────────────────────────────────────────────────

_consent_security = HTTPBearer()


async def require_consent(
    credentials: HTTPAuthorizationCredentials = Security(_consent_security),
) -> dict:
    """
    FastAPI dependency that enforces the consent gate.

    Raises:
        HTTP 401 — token invalid / missing
        HTTP 403 — token valid but consent claims absent

    Returns the consent context dict on success:
        {
            "consent":           True,
            "kernel_type":       "transcript" | "concept",
            "attribution_token": "<hex>",
            "session_id":        "<uuid>",
        }

    Usage:
        @app.post("/v1/chat/completions")
        async def chat(consent: dict = Depends(require_consent), ...):
            ...

    This dependency re-decodes the JWT independently of ``verify_auth``.
    When both are used, include both as separate Depends parameters.
    """
    jwt_secret = os.environ.get("MAESTRO_JWT_SECRET")
    if not jwt_secret:
        raise HTTPException(status_code=503, detail="Session signing not configured")

    token = credentials.credentials
    try:
        payload = jwt.decode(token, jwt_secret, algorithms=[_ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")

    if not payload.get("consent"):
        raise HTTPException(
            status_code=403,
            detail=(
                "Consent required before starting an interview session. "
                "Call POST /v1/session/consent first."
            ),
        )
    kernel_type = payload.get("kernel_type")
    if not kernel_type:
        raise HTTPException(
            status_code=403,
            detail="Kernel type not specified in consent record",
        )

    return {
        "consent":           payload["consent"],
        "kernel_type":       kernel_type,
        "attribution_token": payload.get("attribution_token"),
        "session_id":        payload.get("session_id") or payload.get("sub"),
    }
