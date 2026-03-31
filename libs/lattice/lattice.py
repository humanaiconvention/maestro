"""
lattice.py — LatticeNode schema and Merkle DAG for participant knowledge-packets.

Each participant session produces a Merkle DAG of typed nodes:

  session          Root node — session metadata and consent pointer
  transcript       One conversation turn (role + content)
  context          Background context provided before the interview
  evidence         Specific example or piece of evidence cited by the participant
  felt_state       Affective/emotional state captured from the turn
  gfs_dimension    GFS activation weight for one semantic dimension (DERIVED — versioned)
  phenomenological Structured lived-experience data distilled from the transcript
  training_signal  Export-ready training example (SFT pair, DPO pair, or classifier label)
  consent          Consent record — layer flags committed upfront

Hashing protocol (content-addressed, like git objects):
  content_hash = SHA-256(canonical_json(content))
  node_hash    = SHA-256(node_kind + "\\0" + content_hash + "\\0" + ":".join(sorted(parent_hashes)))

The node_hash is stable across rebuilds given the same content and parents.
The Merkle root of a session is the hash of the session node (which descends from all others).

GFS layer nodes are marked as derived=True. They are NOT included in the base participant
commitment but are stored separately and recomputed when the semantic grounding model improves.

Usage:
    from tools.lattice import LatticeNode, build_session_lattice, merkle_root, to_receipt_dict
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Literal, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Types
# ──────────────────────────────────────────────────────────────────────────────

NodeKind = Literal[
    "session",
    "transcript",
    "context",
    "evidence",
    "felt_state",
    "gfs_dimension",
    "phenomenological",
    "training_signal",
    "consent",
]

TrainingUse = Literal["verbatim", "abstracted", "denied"]


# ──────────────────────────────────────────────────────────────────────────────
# Core node
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LatticeNode:
    """
    A single node in the participant knowledge-packet lattice.

    Fields
    ------
    node_id       : Stable UUID (for database FK; NOT used in hashing)
    node_kind     : Discriminated type tag
    content       : Kind-specific payload dict (see KIND_SCHEMAS below)
    parent_hashes : Sorted list of parent node_hash values
    content_hash  : SHA-256 of canonical JSON of `content`
    node_hash     : SHA-256(kind \\0 content_hash \\0 sorted_parents_joined)
    training_use  : Consent gate for this node's content
    derived       : True for GFS layer nodes (excluded from base commitment)
    gfs_version   : Semantic grounding model version that produced this node (derived only)
    created_at    : ISO 8601 UTC timestamp
    """

    node_id:       str
    node_kind:     NodeKind
    content:       dict[str, Any]
    parent_hashes: list[str]
    content_hash:  str
    node_hash:     str
    training_use:  TrainingUse
    derived:       bool
    gfs_version:   Optional[str]
    created_at:    str

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "LatticeNode":
        return LatticeNode(**d)


# ──────────────────────────────────────────────────────────────────────────────
# Hashing helpers
# ──────────────────────────────────────────────────────────────────────────────

def _canonical_json(obj: Any) -> str:
    """Stable, sorted-key JSON for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def compute_content_hash(content: dict) -> str:
    return _sha256(_canonical_json(content))


def compute_node_hash(node_kind: str, content_hash: str, parent_hashes: list[str]) -> str:
    """
    Deterministic node hash.  Includes kind, content, and sorted parent set.
    Changing any field changes the hash — content-addressed like a git blob.
    """
    parents_part = ":".join(sorted(parent_hashes))
    preimage = f"{node_kind}\x00{content_hash}\x00{parents_part}"
    return _sha256(preimage)


# ──────────────────────────────────────────────────────────────────────────────
# Node factory
# ──────────────────────────────────────────────────────────────────────────────

def make_node(
    node_kind:     NodeKind,
    content:       dict[str, Any],
    parent_hashes: list[str],
    training_use:  TrainingUse = "verbatim",
    derived:       bool = False,
    gfs_version:   Optional[str] = None,
) -> LatticeNode:
    """Create a fully hashed LatticeNode from raw inputs."""
    content_hash = compute_content_hash(content)
    node_hash    = compute_node_hash(node_kind, content_hash, parent_hashes)
    return LatticeNode(
        node_id       = str(uuid.uuid4()),
        node_kind     = node_kind,
        content       = content,
        parent_hashes = sorted(parent_hashes),
        content_hash  = content_hash,
        node_hash     = node_hash,
        training_use  = training_use,
        derived       = derived,
        gfs_version   = gfs_version,
        created_at    = datetime.now(timezone.utc).isoformat(),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session lattice builder
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConsentRecord:
    """
    Upfront consent choices committed as a consent node in the lattice.

    Each field is the participant's choice for that layer.
    This record is Merkle-committed and cannot be retroactively altered.
    """
    transcript:      Literal["train", "quote", "retain_only", "denied"]
    felt_state:      Literal["train_abstracted", "denied"]
    gfs_activations: Literal["improve_model", "denied"]
    training_signal: Literal["sft_dpo", "denied"]
    retention:       Literal["6mo", "2yr", "extended"]
    agreed_at:       str  # ISO timestamp


@dataclass
class SessionInput:
    """
    All inputs needed to build a full session lattice.

    `messages` is the interview transcript in OpenAI-style format.
    `felt_states` is a list parallel to messages — one dict per turn, or None.
    `context_text` is optional free-text context provided by the participant.
    `consent` is the upfront consent record.
    `session_id` is the JWT session_id from the anonymous session.
    """
    session_id:   str
    messages:     list[dict[str, str]]           # [{role, content}, ...]
    felt_states:  list[Optional[dict[str, Any]]] # parallel to messages
    context_text: Optional[str]
    consent:      ConsentRecord


@dataclass
class SessionLattice:
    """
    The complete set of nodes for one participant session.

    `base_nodes`    — immutable participant commitment (transcript, consent, etc.)
    `derived_nodes` — GFS layer; versioned, recomputable, separate commitment
    `merkle_root`   — SHA-256 root of the base lattice (session node hash)
    `session_id`    — links back to the anonymous JWT session
    """
    base_nodes:    list[LatticeNode]
    derived_nodes: list[LatticeNode]
    merkle_root:   str
    session_id:    str

    def all_nodes(self) -> list[LatticeNode]:
        return self.base_nodes + self.derived_nodes

    def base_by_kind(self, kind: NodeKind) -> list[LatticeNode]:
        return [n for n in self.base_nodes if n.node_kind == kind]

    def to_dict(self) -> dict:
        return {
            "merkle_root":    self.merkle_root,
            "session_id":     self.session_id,
            "base_nodes":     [n.to_dict() for n in self.base_nodes],
            "derived_nodes":  [n.to_dict() for n in self.derived_nodes],
        }

    @staticmethod
    def from_dict(d: dict) -> "SessionLattice":
        return SessionLattice(
            base_nodes    = [LatticeNode.from_dict(n) for n in d["base_nodes"]],
            derived_nodes = [LatticeNode.from_dict(n) for n in d["derived_nodes"]],
            merkle_root   = d["merkle_root"],
            session_id    = d["session_id"],
        )


def build_session_lattice(inp: SessionInput) -> SessionLattice:
    """
    Construct the full Merkle DAG for a participant session.

    Build order (parents must precede children):
      1. consent node  (no parents)
      2. context node  (no parents; optional)
      3. transcript nodes (chain: each turn's parent = previous turn's hash)
      4. felt_state nodes (parent = corresponding transcript node)
      5. phenomenological node (parents = all transcript hashes)
      6. session node (root; parents = all other base node hashes)

    Returns a SessionLattice with base_nodes (immutable) and derived_nodes=[].
    Call gfs_layer.annotate_lattice() to populate derived_nodes.
    """
    base: list[LatticeNode] = []
    _tu = _consent_to_training_use(inp.consent)

    # ── 1. Consent node ──────────────────────────────────────────────────────
    consent_content = {
        "transcript":      inp.consent.transcript,
        "felt_state":      inp.consent.felt_state,
        "gfs_activations": inp.consent.gfs_activations,
        "training_signal": inp.consent.training_signal,
        "retention":       inp.consent.retention,
        "agreed_at":       inp.consent.agreed_at,
    }
    consent_node = make_node("consent", consent_content, [], training_use="denied")
    base.append(consent_node)

    # ── 2. Context node (optional) ───────────────────────────────────────────
    context_hash: Optional[str] = None
    if inp.context_text:
        ctx_node = make_node(
            "context",
            {"text": inp.context_text},
            [consent_node.node_hash],
            training_use=_tu["transcript"],
        )
        base.append(ctx_node)
        context_hash = ctx_node.node_hash

    # ── 3. Transcript nodes (chain) ──────────────────────────────────────────
    transcript_hashes: list[str] = []
    prev_hash = consent_node.node_hash

    for i, msg in enumerate(inp.messages):
        parents = [prev_hash]
        if context_hash:
            parents.append(context_hash)

        t_node = make_node(
            "transcript",
            {
                "turn_index": i,
                "role":       msg["role"],
                "content":    msg["content"],
            },
            parents,
            training_use=_tu["transcript"],
        )
        base.append(t_node)
        transcript_hashes.append(t_node.node_hash)
        prev_hash = t_node.node_hash

        # ── 4. Felt-state node (paired with this turn) ────────────────────
        felt = inp.felt_states[i] if i < len(inp.felt_states) else None
        if felt:
            fs_node = make_node(
                "felt_state",
                {"turn_index": i, **felt},
                [t_node.node_hash],
                training_use=_tu["felt_state"],
            )
            base.append(fs_node)

    # ── 5. Phenomenological summary node ────────────────────────────────────
    # Content is a structured distillation; populated later by analysis tools.
    phenom_node = make_node(
        "phenomenological",
        {
            "session_id":    inp.session_id,
            "turn_count":    len(inp.messages),
            "summary":       None,   # filled by analysis pipeline
            "themes":        [],
            "affect_arc":    [],
        },
        transcript_hashes,
        training_use=_tu["transcript"],
    )
    base.append(phenom_node)

    # ── 6. Session root node ─────────────────────────────────────────────────
    all_base_hashes = [n.node_hash for n in base]
    session_node = make_node(
        "session",
        {
            "session_id":    inp.session_id,
            "turn_count":    len(inp.messages),
            "consent_hash":  consent_node.node_hash,
            "has_context":   context_hash is not None,
        },
        all_base_hashes,
        training_use="denied",   # session metadata not used in training
    )
    base.append(session_node)

    return SessionLattice(
        base_nodes    = base,
        derived_nodes = [],
        merkle_root   = session_node.node_hash,
        session_id    = inp.session_id,
    )


def _consent_to_training_use(c: ConsentRecord) -> dict[str, TrainingUse]:
    """Map consent choices to per-field TrainingUse values."""
    return {
        "transcript": (
            "verbatim"   if c.transcript in ("train", "quote")
            else "abstracted" if c.transcript == "retain_only"
            else "denied"
        ),
        "felt_state": (
            "abstracted" if c.felt_state == "train_abstracted"
            else "denied"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Merkle proof helpers
# ──────────────────────────────────────────────────────────────────────────────

def verify_node(node: LatticeNode) -> bool:
    """Re-derive and check both hashes for a single node."""
    ch = compute_content_hash(node.content)
    if ch != node.content_hash:
        return False
    nh = compute_node_hash(node.node_kind, node.content_hash, node.parent_hashes)
    return nh == node.node_hash


def verify_lattice(lattice: SessionLattice) -> tuple[bool, list[str]]:
    """
    Verify all base nodes and confirm the session root matches merkle_root.
    Returns (all_ok, list_of_error_messages).
    """
    errors: list[str] = []
    for node in lattice.base_nodes:
        if not verify_node(node):
            errors.append(f"Hash mismatch on node {node.node_id} ({node.node_kind})")

    session_nodes = [n for n in lattice.base_nodes if n.node_kind == "session"]
    if not session_nodes:
        errors.append("No session root node found")
    elif session_nodes[-1].node_hash != lattice.merkle_root:
        errors.append("Merkle root mismatch")

    return len(errors) == 0, errors


def merkle_root(lattice: SessionLattice) -> str:
    """Return the session's Merkle root (session node hash)."""
    return lattice.merkle_root


# ──────────────────────────────────────────────────────────────────────────────
# Receipt summary helper
# ──────────────────────────────────────────────────────────────────────────────

def to_receipt_dict(lattice: SessionLattice) -> dict:
    """
    Return a compact dict suitable for participant receipt display / QR encoding.

    Contains:
      - merkle_root: the participant's cryptographic commitment
      - session_id:  anonymous session identifier
      - turn_count:  number of conversation turns
      - consent:     the consent choices made
      - node_count:  total base nodes committed
      - created_at:  timestamp of the session node
    """
    session_node = next(
        (n for n in reversed(lattice.base_nodes) if n.node_kind == "session"),
        None,
    )
    consent_node = next(
        (n for n in lattice.base_nodes if n.node_kind == "consent"),
        None,
    )
    return {
        "merkle_root": lattice.merkle_root,
        "session_id":  lattice.session_id,
        "turn_count":  session_node.content.get("turn_count") if session_node else None,
        "consent":     consent_node.content if consent_node else None,
        "node_count":  len(lattice.base_nodes),
        "created_at":  session_node.created_at if session_node else None,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_lattice(lattice: SessionLattice, path: str) -> None:
    """Save lattice to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lattice.to_dict(), f, indent=2, ensure_ascii=False)


def load_lattice(path: str) -> SessionLattice:
    """Load lattice from a JSON file and verify integrity."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    lattice = SessionLattice.from_dict(d)
    ok, errors = verify_lattice(lattice)
    if not ok:
        raise ValueError(f"Lattice integrity check failed: {errors}")
    return lattice
