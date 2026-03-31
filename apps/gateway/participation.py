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

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Persistence: Postgres if DATABASE_URL is set, SQLite otherwise, in-memory as last resort.
_USE_POSTGRES = bool(os.environ.get("DATABASE_URL"))
_USE_SQLITE = not _USE_POSTGRES  # SQLite is the default local persistence
_USE_DB = _USE_POSTGRES or _USE_SQLITE

if _USE_POSTGRES:
    from apps.gateway import db
    logger.info("Receipt persistence: Postgres (DATABASE_URL set)")
elif _USE_SQLITE:
    from apps.gateway import db_sqlite as db  # type: ignore[assignment]
    try:
        db.init_db()
        logger.info("Receipt persistence: SQLite at %s", db.SQLITE_DB_PATH)
    except Exception as exc:
        logger.warning("SQLite init failed, falling back to in-memory: %s", exc)
        _USE_DB = False


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

def _run_cognitive_pipeline(lattice_path: str, lattice_dir: str) -> None:
    """
    Run the Orch-OS-aligned cognitive pipeline on a single lattice file.

    This is a best-effort background operation triggered at receipt time.
    It scores the session through all cognitive cores, updates the memory
    field, and appends the training signal to the signals file.

    Non-fatal: any failure here does NOT affect the receipt.
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        from libs.lattice.signal_scorer import SignalScorer
        from libs.lattice.memory_field import MemoryField
        from libs.lattice.collapse import CollapseFunction
    except ImportError:
        # Running from a context where libs isn't on path — try relative
        import sys
        _lib_root = str(_Path(__file__).resolve().parents[2])
        if _lib_root not in sys.path:
            sys.path.insert(0, _lib_root)
        from libs.lattice.signal_scorer import SignalScorer
        from libs.lattice.memory_field import MemoryField
        from libs.lattice.collapse import CollapseFunction

    data_dir = _Path(lattice_dir).parent  # data/lattices -> data/
    field_path = str(data_dir / "memory_field.pkl")
    signals_path = str(data_dir / "training_signals.json")

    with open(lattice_path, "r", encoding="utf-8") as f:
        lattice_data = _json.load(f)

    scorer = SignalScorer()
    mf = MemoryField(field_path)
    collapse_fn = CollapseFunction(scorer, mf)

    signal, cognitive = collapse_fn.collapse_with_routing(lattice_data)

    # Store centroid in memory field
    from libs.lattice.signal_scorer import _get_embedder
    all_texts, _, _ = scorer._extract_turns(lattice_data)
    if all_texts:
        embedder = _get_embedder()
        import numpy as _np
        embs = embedder.encode(all_texts, convert_to_numpy=True)
        centroid = embs.mean(axis=0)
        score = scorer.score_from_dict(lattice_data)
        mf.store_from_score(score, centroid)
        mf.save()

    # Append signal to training_signals.json
    existing_signals = []
    if _Path(signals_path).exists():
        with open(signals_path, "r", encoding="utf-8") as f:
            existing_signals = _json.load(f)

    # Deduplicate by session_id
    existing_ids = {s.get("session_id") for s in existing_signals}
    if signal.session_id not in existing_ids:
        existing_signals.append(signal.to_dict())
        with open(signals_path, "w", encoding="utf-8") as f:
            _json.dump(existing_signals, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Cognitive pipeline complete: {signal.session_id} "
        f"type={signal.signal_type} weight={signal.weight:.4f} "
        f"cores={collapse_fn.router._dominant_cores(cognitive.signals)}"
    )


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

    # CS5 (Storage Exhaustion): max session size and turn count limits
    _MAX_LATTICE_SIZE_BYTES = 65_536   # 64 KB max lattice payload
    _MAX_SESSION_TURNS      = 20       # hard cap on conversation turns
    _MAX_MESSAGES           = 40       # hard cap on messages (user + assistant)

    @app.post("/v1/session/receipt", response_model=ReceiptResponse, tags=["session"])
    async def create_receipt(body: LatticeSubmitRequest):
        """
        Submit a session lattice and receive a verifiable participation receipt.

        Accepts two formats:
          1. Full SessionLattice.to_dict() — verified as-is.
          2. Lightweight skeleton {session_id, messages, consent} — lattice is built
             server-side before issuing the receipt.

        The Merkle root is always derived server-side, never trusted from the client.

        CS5 hardening: rejects sessions exceeding size/turn limits.
        """
        import json as _json
        from libs.lattice import (
            SessionLattice, SessionInput, ConsentRecord,
            build_session_lattice, verify_lattice, to_receipt_dict,
        )

        d = body.lattice

        # CS5: Size guard — reject oversized payloads before processing
        raw_size = len(_json.dumps(d, ensure_ascii=False).encode("utf-8"))
        if raw_size > _MAX_LATTICE_SIZE_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Lattice payload ({raw_size:,} bytes) exceeds max ({_MAX_LATTICE_SIZE_BYTES:,} bytes)"
            )

        # CS5: Turn count guard
        messages = d.get("messages", [])
        if len(messages) > _MAX_MESSAGES:
            raise HTTPException(
                status_code=422,
                detail=f"Session has {len(messages)} messages — max is {_MAX_MESSAGES}"
            )

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

                # Assemble felt_states from three possible sources:
                # 1. Client-submitted felt_states in the skeleton body
                # 2. Session store accumulation (from POST /v1/session/felt-state)
                # 3. Fall back to [None] * len(messages)
                messages = d.get("messages", [])
                raw_felt = d.get("felt_states")

                if raw_felt and isinstance(raw_felt, list):
                    # Client provided felt_states inline with the skeleton
                    fs_list = raw_felt
                else:
                    # Pull from session store (accumulated via per-turn endpoint)
                    _sid = d.get("session_id", "unknown")
                    try:
                        from apps.gateway.session_store import session_store as _ss
                        stored_fs = _ss.get_felt_states(_sid)
                    except Exception:
                        stored_fs = []

                    if stored_fs:
                        # Convert session store format (list of dicts with
                        # turn_index) to the parallel list format expected
                        # by SessionInput.  Build a sparse list indexed by
                        # turn_index, with None for turns without labels.
                        fs_list = [None] * len(messages)
                        for fs_entry in stored_fs:
                            idx = fs_entry.get("turn_index", -1)
                            if 0 <= idx < len(messages):
                                fs_list[idx] = fs_entry
                    else:
                        fs_list = [None] * len(messages)

                # Respect consent: if felt_state consent is denied, zero out
                if consent.felt_state == "denied":
                    fs_list = [None] * len(messages)

                inp = SessionInput(
                    session_id   = d.get("session_id", "unknown"),
                    messages     = messages,
                    felt_states  = fs_list,
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

        # Persist lattice to pool for improvement pipeline
        # This bridges the gap: receipt → SQLite AND lattice → data/lattices/
        try:
            from libs.lattice import save_lattice as _save_lattice
            from pathlib import Path as _Path
            _pool_dir = _Path(__file__).resolve().parents[3] / "data" / "lattices"
            # Worktree detection: if .git is a file (not dir), walk up to main repo
            _repo_root = _Path(__file__).resolve().parents[3]
            if (_repo_root / ".git").is_file():
                _pool_dir = _repo_root.parent.parent.parent / "data" / "lattices"
            _pool_dir.mkdir(parents=True, exist_ok=True)
            _lattice_path = _pool_dir / f"{lattice.session_id}.json"
            _save_lattice(lattice, str(_lattice_path))
            logger.info(f"Lattice saved to pool: {_lattice_path.name} ({len(lattice.base_nodes)} nodes)")

            # Run cognitive routing + collapse on the saved lattice (best-effort)
            # This triggers the full Orch-OS-aligned pipeline: signal generation,
            # core activation, memory field update, and collapse scoring.
            try:
                _run_cognitive_pipeline(str(_lattice_path), str(_pool_dir))
            except Exception as _cog_exc:
                logger.warning(f"Cognitive pipeline (non-fatal): {_cog_exc}")

        except Exception as exc:
            logger.warning(f"Failed to save lattice to pool: {exc}")
            # Non-fatal: receipt is already stored, lattice save is best-effort

        return ReceiptResponse(**receipt.to_dict())

    @app.get("/v1/session/receipt/{merkle_root_param}", response_model=ReceiptResponse, tags=["session"])
    async def fetch_receipt(merkle_root_param: str):
        """Retrieve a previously issued participation receipt by Merkle root."""
        receipt = await get_receipt(merkle_root_param)
        if not receipt:
            raise HTTPException(status_code=404, detail="Receipt not found")
        return ReceiptResponse(**receipt.to_dict())
