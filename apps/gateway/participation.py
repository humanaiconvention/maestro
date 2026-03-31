"""
participation.py — Participation receipt generation for session lattices.

A participation receipt is the participant's verifiable record of their session.
It contains:

  - merkle_root    : SHA-256 Merkle root of their session lattice (base nodes)
  - session_id     : anonymous session identifier (from the JWT)
  - consent_summary: plain-English version of the consent choices
  - node_count     : number of committed nodes
  - created_at     : timestamp
  - qr_payload     : compact string for QR code encoding

The receipt is NOT a secret.  The Merkle root allows anyone to verify that a
specific session lattice corresponds to this receipt, but reveals nothing about
the content.

Two routes are added to the gateway:
  POST /v1/session/receipt   — submit lattice, receive receipt + QR data URL
  GET  /v1/session/receipt/{merkle_root}  — retrieve receipt by Merkle root

Receipts are persisted to Postgres when DATABASE_URL is set, with an
in-memory dict as fallback for local development.
"""

import asyncio
import base64
import functools
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from typing import Any, Optional

from apps.gateway import db

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_USE_DB = bool(os.environ.get("DATABASE_URL"))


# ──────────────────────────────────────────────────────────────────────────────
# Receipt dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ParticipationReceipt:
    merkle_root:      str
    session_id:       str
    turn_count:       int
    node_count:       int
    consent_summary:  dict[str, str]
    created_at:       str
    qr_payload:       str      # compact JSON string for QR encoding
    qr_data_url:      Optional[str]  # data:image/png;base64,... (None if qrcode unavailable)

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Consent label helpers
# ──────────────────────────────────────────────────────────────────────────────

_TRANSCRIPT_LABELS: dict[str, str] = {
    "train":        "Used verbatim for AI training",
    "quote":        "May be quoted in research",
    "retain_only":  "Retained but not used",
    "denied":       "Not retained",
}

_FELT_STATE_LABELS: dict[str, str] = {
    "train_abstracted": "Abstracted emotional signal allowed for training",
    "denied":           "Not recorded",
}

_GFS_LABELS: dict[str, str] = {
    "improve_model": "Grounding scores used for model improvement",
    "denied":        "Grounding scores not computed",
}

_SIGNAL_LABELS: dict[str, str] = {
    "sft_dpo": "Session used for training signal (SFT/DPO)",
    "denied":  "Not used for training signal",
}

_RETENTION_LABELS: dict[str, str] = {
    "6mo":      "Deleted after 6 months",
    "2yr":      "Retained for up to 2 years",
    "extended": "Retained in research archive (deletion on request)",
}


def _consent_to_summary(consent: Optional[dict]) -> dict[str, str]:
    if not consent:
        return {"note": "Consent record not available"}
    return {
        "transcript":       _TRANSCRIPT_LABELS.get(consent.get("transcript", ""), consent.get("transcript", "")),
        "felt_state":       _FELT_STATE_LABELS.get(consent.get("felt_state", ""), consent.get("felt_state", "")),
        "gfs_activations":  _GFS_LABELS.get(consent.get("gfs_activations", ""), consent.get("gfs_activations", "")),
        "training_signal":  _SIGNAL_LABELS.get(consent.get("training_signal", ""), consent.get("training_signal", "")),
        "retention":        _RETENTION_LABELS.get(consent.get("retention", ""), consent.get("retention", "")),
        "agreed_at":        consent.get("agreed_at", ""),
    }


# ──────────────────────────────────────────────────────────────────────────────
# QR code generation
# ──────────────────────────────────────────────────────────────────────────────

def _build_qr_payload(merkle_root: str, session_id: str, created_at: str) -> str:
    """
    Compact payload string for QR code.

    Format: "HAIC:v1:<merkle_root>:<session_id>:<created_at_epoch>"

    Short enough to fit in a QR code at error-correction level M.
    The participant can use this string to verify their lattice later:
      1. Load lattice JSON
      2. Recompute Merkle root
      3. Compare with the QR code value
    """
    try:
        import datetime
        ts = int(datetime.datetime.fromisoformat(created_at).timestamp())
    except Exception:
        ts = int(time.time())
    return f"HAIC:v1:{merkle_root}:{session_id}:{ts}"


def _generate_qr_data_url(payload: str) -> Optional[str]:
    """
    Generate a QR code as a base64-encoded PNG data URL.

    Returns None if the `qrcode` package is not installed.
    The deployment should install: pip install qrcode[pil]
    """
    try:
        import io
        import qrcode  # type: ignore
        qr = qrcode.QRCode(
            version=None,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=6,
            border=4,
        )
        qr.add_data(payload)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except ImportError:
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Receipt builder
# ──────────────────────────────────────────────────────────────────────────────

def build_receipt(receipt_input: dict) -> ParticipationReceipt:
    """
    Build a ParticipationReceipt from a to_receipt_dict() dict.

    receipt_input keys:
        merkle_root, session_id, turn_count, consent, node_count, created_at
    """
    merkle_root = receipt_input["merkle_root"]
    session_id  = receipt_input.get("session_id", "")
    created_at  = receipt_input.get("created_at", "")
    consent     = receipt_input.get("consent")

    qr_payload  = _build_qr_payload(merkle_root, session_id, created_at)
    qr_data_url = _generate_qr_data_url(qr_payload)

    return ParticipationReceipt(
        merkle_root     = merkle_root,
        session_id      = session_id,
        turn_count      = receipt_input.get("turn_count") or 0,
        node_count      = receipt_input.get("node_count") or 0,
        consent_summary = _consent_to_summary(consent),
        created_at      = created_at,
        qr_payload      = qr_payload,
        qr_data_url     = qr_data_url,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Receipt store — Postgres-backed when DATABASE_URL is set, in-memory fallback
# ──────────────────────────────────────────────────────────────────────────────

_receipt_store: dict[str, ParticipationReceipt] = {}


async def store_receipt(receipt: ParticipationReceipt) -> None:
    if _USE_DB:
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, functools.partial(
                db.store_receipt,
                receipt_id=receipt.merkle_root,
                session_id=receipt.session_id,
                merkle_root=receipt.merkle_root,
                payload_dict=receipt.to_dict(),
            ))
            return
        except Exception as exc:
            logger.warning("db.store_receipt failed, falling back to memory: %s", exc)
    _receipt_store[receipt.merkle_root] = receipt


async def get_receipt(merkle_root: str) -> Optional[ParticipationReceipt]:
    if _USE_DB:
        try:
            loop = asyncio.get_event_loop()
            payload = await loop.run_in_executor(None, functools.partial(db.get_receipt, merkle_root))
            if payload is not None:
                return ParticipationReceipt(**payload)
            return None
        except Exception as exc:
            logger.warning("db.get_receipt failed, falling back to memory: %s", exc)
    return _receipt_store.get(merkle_root)


async def list_receipts(since: Optional[str] = None) -> list[ParticipationReceipt]:
    """Return all stored receipts, optionally filtered to those after `since` (ISO 8601)."""
    if _USE_DB:
        try:
            loop = asyncio.get_event_loop()
            rows = await loop.run_in_executor(None, functools.partial(db.list_receipts, since=since))
            return [ParticipationReceipt(**row["payload"]) for row in rows]
        except Exception as exc:
            logger.warning("db.list_receipts failed, falling back to memory: %s", exc)

    receipts = list(_receipt_store.values())
    if since:
        receipts = [r for r in receipts if r.created_at >= since]
    return receipts


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic request/response models (module-level for FastAPI annotation resolution)
# ──────────────────────────────────────────────────────────────────────────────

class LatticeSubmitRequest(BaseModel):
    lattice: dict   # Either SessionLattice.to_dict() OR lightweight skeleton


class ReceiptResponse(BaseModel):
    merkle_root:     str
    session_id:      str
    turn_count:      int
    node_count:      int
    consent_summary: dict
    created_at:      str
    qr_payload:      str
    qr_data_url:     Optional[str]


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI route registration helper
# ──────────────────────────────────────────────────────────────────────────────

def register_receipt_routes(app: Any) -> None:
    """
    Register participation receipt endpoints on a FastAPI app.

    Called from main.py:  register_receipt_routes(app)
    """
    from fastapi import HTTPException

    @app.post("/v1/session/receipt", response_model=ReceiptResponse, tags=["session"])
    async def create_receipt(body: LatticeSubmitRequest):
        """
        Submit a session lattice and receive a verifiable participation receipt.

        Accepts two formats:
          1. Full SessionLattice.to_dict() — verified as-is.
          2. Lightweight skeleton {session_id, messages, consent} — lattice is built
             server-side before issuing the receipt.

        The Merkle root is always derived server-side, never trusted from the client.
        """
        from libs.lattice import (
            SessionLattice, SessionInput, ConsentRecord,
            build_session_lattice, verify_lattice, to_receipt_dict,
        )

        d = body.lattice

        # Detect lightweight skeleton (has 'messages' key, not 'base_nodes')
        if "messages" in d and "base_nodes" not in d:
            try:
                consent_d = d.get("consent", {})
                consent = ConsentRecord(
                    transcript      = consent_d.get("transcript",      "retain_only"),
                    felt_state      = consent_d.get("felt_state",       "denied"),
                    gfs_activations = consent_d.get("gfs_activations",  "denied"),
                    training_signal = consent_d.get("training_signal",  "denied"),
                    retention       = consent_d.get("retention",        "6mo"),
                    agreed_at       = consent_d.get("agreed_at",        ""),
                )
                inp = SessionInput(
                    session_id   = d.get("session_id", "unknown"),
                    messages     = d.get("messages", []),
                    felt_states  = [None] * len(d.get("messages", [])),
                    context_text = d.get("context_text"),
                    consent      = consent,
                )
                lattice = build_session_lattice(inp)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Failed to build lattice from skeleton: {e}")
        else:
            try:
                lattice = SessionLattice.from_dict(d)
            except Exception as e:
                raise HTTPException(status_code=422, detail=f"Invalid lattice format: {e}")

            ok, errors = verify_lattice(lattice)
            if not ok:
                raise HTTPException(status_code=422, detail=f"Lattice integrity check failed: {errors}")

        receipt_input = to_receipt_dict(lattice)
        receipt = build_receipt(receipt_input)
        await store_receipt(receipt)

        return ReceiptResponse(**receipt.to_dict())

    @app.get("/v1/session/receipt/{merkle_root_param}", response_model=ReceiptResponse, tags=["session"])
    async def fetch_receipt(merkle_root_param: str):
        """Retrieve a previously issued participation receipt by Merkle root."""
        receipt = await get_receipt(merkle_root_param)
        if not receipt:
            raise HTTPException(status_code=404, detail="Receipt not found")
        return ReceiptResponse(**receipt.to_dict())
