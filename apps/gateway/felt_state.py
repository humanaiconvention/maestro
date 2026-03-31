"""
felt_state.py -- Per-turn affective state collection for Maestro sessions.

Implements the felt_state channel: participants can optionally label their
emotional/affective state after each turn in a session.  These labels flow
into the lattice as felt_state nodes, enabling the signal scorer's affective
granularity measurement and the collapse function's precision-weighting.

Theoretical grounding (Adolphs 2017):
    "An emotion is a functional state of the mind that puts your brain in a
    particular mode of operation that adjusts your goals, directs your
    attention, and modifies the weights you assign to various factors as
    you do mental calculations."

Felt_state labels make the participant's precision-weighting mechanism
observable.  High emotional granularity (Barrett 2004) -- the ability to
make fine-grained distinctions between emotional states -- predicts better
emotion regulation, decision-making, and more informative corrective signals
for the training pipeline.

Consent gate:
    felt_state collection is gated by the consent record's `felt_state` field.
    If consent is "denied", the endpoint accepts but discards the data and
    returns a 200 with `stored: false`.  The participant is never penalised
    for declining.

Endpoints:
    POST /v1/session/felt-state      Record a felt_state for a specific turn
    GET  /v1/session/felt-state      Retrieve all felt_states for the session

Schema:
    FeltStateRequest:
        turn_index: int          Required. Which turn this label refers to.
        label: str               Required. The affect word/phrase (1-100 chars).
        valence: float           Optional. -1.0 (negative) to 1.0 (positive).
        arousal: float           Optional. 0.0 (calm) to 1.0 (activated).
        note: str                Optional. Free-text elaboration (max 500 chars).

    Labels are deliberately free-form, not constrained to a fixed taxonomy.
    Barrett's research shows that richer, more differentiated vocabularies
    correlate with higher emotional granularity -- imposing a fixed list
    would ceiling the measurement.  The signal scorer handles vocabulary
    diversity natively via embedding-space clustering.
"""

import logging
import os
from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import jwt as pyjwt

from apps.gateway.session_store import session_store

logger = logging.getLogger(__name__)

_ALGORITHM = "HS256"
_security = HTTPBearer()


# -----------------------------------------------------------------------
# Request / Response models
# -----------------------------------------------------------------------

class FeltStateRequest(BaseModel):
    turn_index: int = Field(
        ...,
        ge=0,
        description="Zero-based index of the turn this label refers to",
    )
    label: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Affect word or short phrase (e.g., 'curious', 'uneasy', 'surprised')",
    )
    valence: Optional[float] = Field(
        None,
        ge=-1.0,
        le=1.0,
        description="Emotional valence: -1.0 (negative) to 1.0 (positive)",
    )
    arousal: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Activation level: 0.0 (calm) to 1.0 (highly activated)",
    )
    note: Optional[str] = Field(
        None,
        max_length=500,
        description="Optional free-text elaboration",
    )

    @field_validator("label")
    @classmethod
    def clean_label(cls, v: str) -> str:
        return v.strip()


class FeltStateResponse(BaseModel):
    stored: bool
    turn_index: int
    label: str
    message: str


class FeltStateListResponse(BaseModel):
    session_id: str
    felt_states: list[dict]
    count: int


# -----------------------------------------------------------------------
# Helper: extract session_id from JWT
# -----------------------------------------------------------------------

def _extract_session_id(credentials: HTTPAuthorizationCredentials) -> str:
    jwt_secret = os.environ.get("MAESTRO_JWT_SECRET")
    if not jwt_secret:
        raise HTTPException(status_code=503, detail="Session signing not configured")

    try:
        payload = pyjwt.decode(
            credentials.credentials, jwt_secret, algorithms=[_ALGORITHM]
        )
    except pyjwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired session token")

    session_id = payload.get("session_id") or payload.get("sub")
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID missing from token")

    return session_id


def _check_felt_state_consent(credentials: HTTPAuthorizationCredentials) -> bool:
    """
    Check if the session's consent allows felt_state collection.

    Returns True if consent allows it, False if denied.
    Does NOT raise -- felt_state denial is not an error, just a gate.
    """
    jwt_secret = os.environ.get("MAESTRO_JWT_SECRET")
    if not jwt_secret:
        return False

    try:
        payload = pyjwt.decode(
            credentials.credentials, jwt_secret, algorithms=[_ALGORITHM]
        )
    except pyjwt.PyJWTError:
        return False

    # The consent model stores felt_state preference in the JWT
    # or in the session store.  For the binary gate model (consent.py),
    # felt_state is opt-in: the participant must have consented to the
    # session AND the felt_state_consent claim must be present.
    #
    # Default: if consent is True and no explicit felt_state denial,
    # allow collection.  The lattice consent node (5-layer model)
    # will be the authoritative gate at receipt time.
    if not payload.get("consent"):
        return False

    # Check for explicit felt_state denial in JWT claims
    felt_pref = payload.get("felt_state_consent")
    if felt_pref == "denied":
        return False

    return True


# -----------------------------------------------------------------------
# Route registration
# -----------------------------------------------------------------------

def register_felt_state_routes(app) -> None:
    """Register felt_state collection endpoints on the FastAPI app."""

    @app.post(
        "/v1/session/felt-state",
        response_model=FeltStateResponse,
        tags=["session", "felt-state"],
    )
    def record_felt_state(
        body: FeltStateRequest,
        credentials: HTTPAuthorizationCredentials = Security(_security),
    ):
        """
        Record an affective state label for a specific turn in the session.

        The participant calls this after each assistant response to label
        how they felt during or after reading the response.  Labels are
        free-form -- the richer and more differentiated the vocabulary,
        the higher the affective granularity score.

        Examples of high-granularity labels:
            "curious but slightly unsettled"
            "recognized something familiar"
            "frustrated -- felt redirected"
            "peaceful, grounded"

        Examples of low-granularity labels:
            "fine"
            "ok"
            "bad"

        Both are accepted.  The system never judges or rejects labels.
        """
        session_id = _extract_session_id(credentials)
        consent_ok = _check_felt_state_consent(credentials)

        if not consent_ok:
            logger.debug(
                "felt_state_declined",
                extra={"session_id": session_id, "turn_index": body.turn_index},
            )
            return FeltStateResponse(
                stored=False,
                turn_index=body.turn_index,
                label=body.label,
                message="Felt state acknowledged but not stored (consent not granted)",
            )

        # Validate turn_index against session turn count
        meta = session_store.get(session_id)
        if meta and body.turn_index > meta.turn_count + 1:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"turn_index {body.turn_index} is beyond the current "
                    f"session turn count ({meta.turn_count})"
                ),
            )

        result = session_store.record_felt_state(
            session_id=session_id,
            turn_index=body.turn_index,
            label=body.label,
            valence=body.valence,
            arousal=body.arousal,
            note=body.note,
        )

        if result is None:
            raise HTTPException(
                status_code=404,
                detail="Session not found. Start a chat session first.",
            )

        logger.info(
            "felt_state_recorded",
            extra={
                "session_id": session_id,
                "turn_index": body.turn_index,
                "label": body.label,
            },
        )

        return FeltStateResponse(
            stored=True,
            turn_index=body.turn_index,
            label=body.label,
            message="Felt state recorded",
        )

    @app.get(
        "/v1/session/felt-state",
        response_model=FeltStateListResponse,
        tags=["session", "felt-state"],
    )
    def list_felt_states(
        credentials: HTTPAuthorizationCredentials = Security(_security),
    ):
        """
        Retrieve all felt_state labels recorded for this session.

        Returns them ordered by turn_index.  The client can use this to
        display the affect arc alongside the conversation transcript.
        """
        session_id = _extract_session_id(credentials)
        states = session_store.get_felt_states(session_id)

        return FeltStateListResponse(
            session_id=session_id,
            felt_states=states,
            count=len(states),
        )
