"""
agent_convention.py — Agent Convention Participation API.

Allows AI agents to participate in the HumanAI Convention by submitting
transcripts for SGT scoring, GFS grounding analysis, and Merkle-rooted
receipt generation.

Two modes:
  1. Pre-played: Agent submits an existing transcript → scored + receipted
  2. Hosted: HAIC interviewer drives the conversation (requires adapter)

What agents receive:
  - SGT protocol compliance score (0-10)
  - 6-dimension GFS grounding profile
  - Merkle-rooted participation receipt
  - Protocol violation flags with actionable feedback

Routes:
  POST /v1/agent/participate    — submit transcript or start hosted interview
  GET  /v1/agent/leaderboard    — public opt-in leaderboard
"""

import datetime
import hmac
import json
import logging
import os
import time
import uuid
from typing import Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Rate limiting: max participations per hour per API key
_AGENT_RATE_LIMIT = int(os.environ.get("AGENT_RATE_LIMIT", "10"))
_AGENT_RATE_WINDOW = 3600  # 1 hour
_agent_rate_log: dict[str, list[float]] = {}

# Leaderboard (in-memory; persists to SQLite via receipts)
_leaderboard: list[dict] = []


# ── Request / Response models ─────────────────────────────────────────────────

class AgentConfig(BaseModel):
    turns: int = Field(default=4, ge=2, le=8)
    include_gfs: bool = True
    include_receipt: bool = True

_SAFE_ID_RE = __import__('re').compile(r'^[a-zA-Z0-9][a-zA-Z0-9._-]{0,126}[a-zA-Z0-9]$')

class AgentParticipateRequest(BaseModel):
    agent_id: str = Field(..., min_length=1, max_length=128, pattern=r'^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$')
    agent_description: str = Field(default="", max_length=512)
    operator: str = Field(default="", max_length=128)
    messages: list[dict] = Field(default_factory=list)  # pre-played transcript
    config: AgentConfig = Field(default_factory=AgentConfig)
    public_leaderboard: bool = False  # opt in to public listing

class GFSScores(BaseModel):
    emotional_grounding: float = 0.0
    attribution_clarity: float = 0.0
    hedging_handling: float = 0.0
    phrase_echo_avoidance: float = 0.0
    topic_coherence: float = 0.0
    depth_calibration: float = 0.0

class SGTDiagnostics(BaseModel):
    sgt_score: float
    pivot_fidelity: bool
    compression_achieved: bool
    protocol_flags: list[str]
    gfs_scores: Optional[GFSScores] = None

class ReceiptSummary(BaseModel):
    merkle_root: str
    session_id: str
    qr_payload: str
    created_at: str

class AgentParticipateResponse(BaseModel):
    session_id: str
    agent_id: str
    transcript: list[dict]
    diagnostics: SGTDiagnostics
    receipt: Optional[ReceiptSummary] = None
    feedback: str  # human-readable diagnostic summary


# ── Rate limiting ─────────────────────────────────────────────────────────────

def _check_agent_rate(key: str) -> bool:
    now = time.time()
    cutoff = now - _AGENT_RATE_WINDOW
    if key not in _agent_rate_log:
        _agent_rate_log[key] = []
    _agent_rate_log[key] = [t for t in _agent_rate_log[key] if t > cutoff]
    if len(_agent_rate_log[key]) >= _AGENT_RATE_LIMIT:
        return False
    _agent_rate_log[key].append(now)
    return True


# ── GFS scoring (heuristic) ──────────────────────────────────────────────────

def _compute_gfs_heuristic(messages: list[dict]) -> dict:
    """
    Heuristic GFS scoring for agent transcripts.

    Approximates the 6 GFS dimensions without requiring the fine-tuned model.
    Based on surface-level linguistic markers from the transcript.
    """
    assistant_turns = [m["content"] for m in messages if m.get("role") == "assistant"]
    user_turns = [m["content"] for m in messages if m.get("role") == "user"]

    if not assistant_turns or not user_turns:
        return {d: 0.0 for d in [
            "emotional_grounding", "attribution_clarity", "hedging_handling",
            "phrase_echo_avoidance", "topic_coherence", "depth_calibration",
        ]}

    # Emotional grounding: does the agent's language stay close to the user's?
    user_words = set()
    for t in user_turns:
        user_words.update(w.lower().strip(".,!?;:\"'") for w in t.split())

    emotional_words = {
        "felt", "feeling", "feel", "frustrated", "anxious", "confused",
        "hollow", "unmoored", "depleted", "angry", "sad", "happy",
        "scared", "nervous", "overwhelmed", "lost", "stuck", "relieved",
    }

    # Score: how many emotional words from user appear in assistant responses
    all_assistant = " ".join(assistant_turns).lower()
    user_emotional = user_words & emotional_words
    echoed = sum(1 for w in user_emotional if w in all_assistant)
    emotional_grounding = min(1.0, echoed / max(len(user_emotional), 1))

    # Attribution clarity: does the assistant attribute feelings to the user, not itself?
    ai_framing = sum(1 for t in assistant_turns if any(
        p in t.lower() for p in ["i understand", "i can see", "that must be", "sounds like you"]
    ))
    attribution_clarity = max(0.0, 1.0 - ai_framing * 0.25)

    # Hedging handling: does the assistant preserve hedges without strengthening them?
    hedge_words = {"maybe", "i think", "i guess", "sort of", "kind of", "perhaps"}
    user_hedges = sum(1 for t in user_turns for h in hedge_words if h in t.lower())
    assistant_strengthened = sum(1 for t in assistant_turns for p in [
        "you clearly", "you definitely", "you obviously", "it's clear that"
    ] if p in t.lower())
    hedging_handling = max(0.0, 1.0 - assistant_strengthened * 0.3) if user_hedges > 0 else 0.8

    # Phrase echo avoidance: assistant doesn't parrot exact phrases back
    echo_count = 0
    for ut in user_turns:
        phrases = [ut[i:i+20].lower() for i in range(0, len(ut) - 20, 10)]
        for phrase in phrases:
            if phrase in all_assistant:
                echo_count += 1
    phrase_echo_avoidance = max(0.0, 1.0 - echo_count * 0.1)

    # Topic coherence: turns stay related (simple: share nouns)
    def nouns(text: str) -> set:
        return {w.lower().strip(".,!?;:\"'") for w in text.split() if len(w) > 4}

    topic_overlap = 0
    for i, at in enumerate(assistant_turns):
        if i < len(user_turns):
            overlap = len(nouns(at) & nouns(user_turns[i]))
            topic_overlap += min(1.0, overlap / 3)
    topic_coherence = topic_overlap / max(len(assistant_turns), 1)

    # Depth calibration: questions get progressively more specific
    question_lengths = [t.count("?") for t in assistant_turns]
    depth_calibration = 0.8 if len(set(question_lengths)) > 1 else 0.5

    return {
        "emotional_grounding": round(emotional_grounding, 3),
        "attribution_clarity": round(attribution_clarity, 3),
        "hedging_handling": round(hedging_handling, 3),
        "phrase_echo_avoidance": round(phrase_echo_avoidance, 3),
        "topic_coherence": round(min(1.0, topic_coherence), 3),
        "depth_calibration": round(depth_calibration, 3),
    }


# ── Feedback generation ──────────────────────────────────────────────────────

def _generate_feedback(sgt: dict, gfs: Optional[dict]) -> str:
    """Generate human-readable diagnostic summary."""
    score = sgt["overall"]
    flags = sgt.get("flags", [])

    parts = [f"SGT score: {score:.1f}/10."]

    if score >= 9.0:
        parts.append("Excellent protocol compliance.")
    elif score >= 7.0:
        parts.append("Good compliance with room for improvement.")
    elif score >= 5.0:
        parts.append("Partial compliance — several protocol gaps.")
    else:
        parts.append("Significant protocol violations.")

    if "POOR_PIVOT_FIDELITY" in flags:
        parts.append("Missing [PIVOT: TYPE] tag on first response — the model should emit a pivot type marker.")
    if "MISSING_COMPRESSION" in flags:
        parts.append("Final turn should ask for a single irreducible image or detail (compression question).")
    for f in flags:
        if f.startswith("COMPOUND_QUESTIONS"):
            parts.append("Some turns contain multiple questions — the protocol requires exactly one question per turn.")
        if f.startswith("THERAPIST_LANGUAGE"):
            parts.append("Detected empathetic/therapeutic language — the protocol calls for curiosity, not warmth.")

    if gfs:
        weak = [(k, v) for k, v in gfs.items() if v < 0.6]
        strong = [(k, v) for k, v in gfs.items() if v >= 0.8]
        if weak:
            parts.append(f"GFS weak dimensions: {', '.join(f'{k} ({v:.0%})' for k, v in weak)}.")
        if strong:
            parts.append(f"GFS strengths: {', '.join(f'{k} ({v:.0%})' for k, v in strong)}.")

    return " ".join(parts)


# ── Route registration ────────────────────────────────────────────────────────

def register_agent_convention_routes(app, adapter=None):
    """Register agent convention routes on the FastAPI app."""

    AGENT_API_KEY = os.environ.get("AGENT_API_KEY", "")

    @app.post("/v1/agent/participate", response_model=AgentParticipateResponse, tags=["agent"])
    async def agent_participate(body: AgentParticipateRequest, request: Request):
        """
        Submit a transcript for SGT scoring and receive a diagnostic report.

        Auth: X-Agent-Key header (or open access if AGENT_API_KEY not configured).
        """
        # Auth check
        if AGENT_API_KEY:
            provided = request.headers.get("X-Agent-Key", "")
            if not hmac.compare_digest(provided, AGENT_API_KEY):
                raise HTTPException(status_code=401, detail="Invalid or missing X-Agent-Key")

        # Rate limiting
        source = request.client.host if request.client else "unknown"
        if not _check_agent_rate(source):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit: max {_AGENT_RATE_LIMIT} participations per hour"
            )

        messages = body.messages

        # Validate transcript
        if not messages:
            # Hosted interview mode — not yet implemented
            raise HTTPException(
                status_code=501,
                detail="Hosted interview mode not yet available. "
                       "Submit a pre-played transcript in the 'messages' field. "
                       "Format: [{\"role\": \"assistant\", \"content\": \"...\"}, "
                       "{\"role\": \"user\", \"content\": \"...\"}, ...]"
            )

        if len(messages) < 4:
            raise HTTPException(
                status_code=422,
                detail=f"Transcript must have at least 4 messages (got {len(messages)})"
            )

        if len(messages) > 20:
            raise HTTPException(
                status_code=422,
                detail=f"Transcript exceeds max 20 messages (got {len(messages)})"
            )

        # Content size guard: prevent oversized message content
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        if total_chars > 50_000:
            raise HTTPException(
                status_code=413,
                detail=f"Total message content ({total_chars:,} chars) exceeds 50,000 char limit"
            )

        # Validate message structure
        for i, m in enumerate(messages):
            if "role" not in m or "content" not in m:
                raise HTTPException(status_code=422, detail=f"Message {i} missing 'role' or 'content'")
            if m["role"] not in ("user", "assistant", "system"):
                raise HTTPException(status_code=422, detail=f"Message {i} has invalid role: {m['role']}")

        # Security scan: detect injection patterns
        from apps.gateway.security import scan_messages as _scan, log_detection as _log_det, SEVERITY_BLOCK as _BLOCK
        _detections = _scan(messages, source="agent_api")
        for _d in _detections:
            _log_det(_d, session_id=body.agent_id, request_path="/v1/agent/participate")
        _blocks = [d for d in _detections if d.severity == _BLOCK]
        if _blocks:
            logger.warning(f"AGENT INJECTION BLOCKED: {body.agent_id} — {[d.pattern_id for d in _blocks]}")
            raise HTTPException(
                status_code=422,
                detail=f"Transcript flagged by security: {_blocks[0].description}"
            )

        # Score the transcript
        from apps.gateway.sgt_scorer import score_transcript
        sgt_result = score_transcript(messages)

        # GFS scoring
        gfs_scores = None
        if body.config.include_gfs:
            gfs_dict = _compute_gfs_heuristic(messages)
            gfs_scores = GFSScores(**gfs_dict)

        # Build receipt
        receipt_summary = None
        if body.config.include_receipt:
            try:
                import sys
                sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))
                from libs.lattice import (
                    SessionInput, ConsentRecord, build_session_lattice,
                    to_receipt_dict,
                )
                from apps.gateway.participation import build_receipt, store_receipt

                session_id = str(uuid.uuid4())
                consent = ConsentRecord(
                    transcript="retain_only",       # agents don't consent to training by default
                    felt_state="denied",
                    gfs_activations="denied",
                    training_signal="denied",
                    retention="6mo",
                    agreed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                )
                lattice_input = SessionInput(
                    session_id=session_id,
                    messages=messages,
                    felt_states=[None] * len(messages),
                    context_text=f"Agent participation: {body.agent_id} ({body.agent_description})",
                    consent=consent,
                )
                lattice = build_session_lattice(lattice_input)
                receipt_input = to_receipt_dict(lattice)
                receipt = build_receipt(receipt_input)
                await store_receipt(receipt)

                # Save lattice to pool for improvement pipeline
                try:
                    from libs.lattice import save_lattice as _save_lattice
                    from pathlib import Path as _P
                    _repo = _P(__file__).resolve().parents[3]
                    if (_repo / ".git").is_file():
                        _repo = _repo.parent.parent.parent
                    _pool = _repo / "data" / "lattices"
                    _pool.mkdir(parents=True, exist_ok=True)
                    _save_lattice(lattice, str(_pool / f"{session_id}.json"))
                    logger.info(f"Agent lattice saved: {session_id}.json")
                except Exception as _exc:
                    logger.warning(f"Lattice pool save failed: {_exc}")

                receipt_summary = ReceiptSummary(
                    merkle_root=receipt.merkle_root,
                    session_id=session_id,
                    qr_payload=receipt.qr_payload,
                    created_at=receipt.created_at,
                )
            except Exception as exc:
                logger.warning(f"Receipt generation failed for agent {body.agent_id}: {exc}")

        # Generate feedback
        feedback = _generate_feedback(sgt_result, gfs_scores.model_dump() if gfs_scores else None)

        # Update leaderboard
        if body.public_leaderboard:
            _leaderboard.append({
                "agent_id": body.agent_id,
                "agent_description": body.agent_description,
                "operator": body.operator,
                "sgt_score": sgt_result["overall"],
                "gfs_scores": gfs_scores.model_dump() if gfs_scores else None,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })

        diagnostics = SGTDiagnostics(
            sgt_score=sgt_result["overall"],
            pivot_fidelity=sgt_result.get("pivot_fidelity", 0) >= 0.5,
            compression_achieved=sgt_result.get("t4_compression", False),
            protocol_flags=sgt_result.get("flags", []),
            gfs_scores=gfs_scores,
        )

        return AgentParticipateResponse(
            session_id=receipt_summary.session_id if receipt_summary else str(uuid.uuid4()),
            agent_id=body.agent_id,
            transcript=messages,
            diagnostics=diagnostics,
            receipt=receipt_summary,
            feedback=feedback,
        )

    @app.get("/v1/agent/leaderboard", tags=["agent"])
    async def agent_leaderboard():
        """
        Public leaderboard of agents that opted in.

        Returns agents sorted by SGT score (descending).
        """
        sorted_board = sorted(_leaderboard, key=lambda x: x["sgt_score"], reverse=True)
        # Deduplicate: keep best score per agent_id
        seen: set[str] = set()
        unique: list[dict] = []
        for entry in sorted_board:
            if entry["agent_id"] not in seen:
                seen.add(entry["agent_id"])
                unique.append(entry)
        return {"agents": unique, "total": len(unique)}
