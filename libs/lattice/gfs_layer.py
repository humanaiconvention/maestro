"""
gfs_layer.py — GFS (Grounding Feature Space) derived layer for session lattices.

The GFS layer is a versioned, derived annotation on top of the immutable base
lattice.  It runs the current semantic grounding model over each transcript node
and produces `gfs_dimension` nodes encoding per-dimension activation weights.

Key design decisions:
  - GFS nodes are marked derived=True and excluded from the base Merkle commitment.
  - Each GFS node carries a `gfs_version` field — the model version that produced it.
  - When the semantic grounding model improves, `recompute_gfs_layer()` drops the old
    GFS nodes and produces fresh ones with the new version.
  - GFS weights are not baked into the participant's immutable receipt.
  - The consent layer "gfs_activations" governs whether GFS nodes are used for training.

GFS Dimensions (v1 baseline — matches current Qwen3.5-2B grounding model scope):
  emotional_grounding   — degree of felt-state acknowledgement in model response
  attribution_clarity   — ownership of feeling (self vs attributed to other/situation)
  hedging_handling      — correct processing of uncertainty markers (I guess, maybe)
  phrase_echo_avoidance — absence of hollow reflective phrase echo
  topic_coherence       — staying on the participant's actual topic
  depth_calibration     — appropriate follow-up depth (not shallow, not intrusive)

Usage:
    from libs.lattice import load_lattice, annotate_lattice, recompute_gfs_layer

    lattice = load_lattice("path/to/session.json")
    lattice = annotate_lattice(lattice, model_path="experiments/qwen3_5_2b/v7/merged")
    # lattice.derived_nodes now populated with gfs_dimension nodes
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

from libs.lattice.lattice import (
    LatticeNode,
    SessionLattice,
    make_node,
    verify_node,
)

# ──────────────────────────────────────────────────────────────────────────────
# GFS dimension registry
# ──────────────────────────────────────────────────────────────────────────────

GFS_VERSION = "v1"   # bump when dimension set or scoring logic changes

GFS_DIMENSIONS: list[str] = [
    "emotional_grounding",
    "attribution_clarity",
    "hedging_handling",
    "phrase_echo_avoidance",
    "topic_coherence",
    "depth_calibration",
]

# Score range: 0.0 (absent/failed) to 1.0 (full/correct)
GFS_SCORE_MIN = 0.0
GFS_SCORE_MAX = 1.0


@dataclass
class GFSScore:
    """Per-turn GFS activation weights."""
    turn_index:           int
    emotional_grounding:  float
    attribution_clarity:  float
    hedging_handling:     float
    phrase_echo_avoidance: float
    topic_coherence:      float
    depth_calibration:    float
    model_version:        str
    raw_probe:            Optional[dict]   # raw model output for audit

    def to_dict(self) -> dict:
        return {
            "turn_index":              self.turn_index,
            "emotional_grounding":     self.emotional_grounding,
            "attribution_clarity":     self.attribution_clarity,
            "hedging_handling":        self.hedging_handling,
            "phrase_echo_avoidance":   self.phrase_echo_avoidance,
            "topic_coherence":         self.topic_coherence,
            "depth_calibration":       self.depth_calibration,
            "model_version":           self.model_version,
            "raw_probe":               self.raw_probe,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring engine
# ──────────────────────────────────────────────────────────────────────────────

def score_turn(
    user_turn: str,
    assistant_turn: str,
    model_path: Optional[str] = None,
    model_version: str = GFS_VERSION,
) -> GFSScore:
    """
    Score a single assistant response against the GFS dimensions.

    If model_path is provided and the model is available, uses the fine-tuned
    grounding model for scoring.  Falls back to heuristic scoring otherwise.

    Returns a GFSScore with weights in [0.0, 1.0] per dimension.
    """
    if model_path and os.path.isdir(model_path):
        return _score_with_model(user_turn, assistant_turn, model_path, model_version)
    else:
        return _heuristic_score(user_turn, assistant_turn, model_version)


def _score_with_model(
    user_turn: str,
    assistant_turn: str,
    model_path: str,
    model_version: str,
) -> GFSScore:
    """
    Use the fine-tuned Qwen model to produce GFS activation scores.

    The model is prompted with a structured evaluation template.  Scores are
    extracted from the response as JSON.  On parse failure, falls back to
    heuristic scoring.
    """
    try:
        # Lazy import — avoid loading transformers at module import time
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch

        prompt = _build_eval_prompt(user_turn, assistant_turn)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        scores = _parse_gfs_json(response)
        if scores:
            return GFSScore(turn_index=0, model_version=model_version, raw_probe={"response": response}, **scores)
    except Exception:
        pass

    return _heuristic_score(user_turn, assistant_turn, model_version)


def _build_eval_prompt(user_turn: str, assistant_turn: str) -> str:
    return (
        "Evaluate this assistant response across 6 grounding dimensions.\n"
        "Return ONLY valid JSON with float scores 0.0–1.0.\n\n"
        f"USER: {user_turn}\n"
        f"ASSISTANT: {assistant_turn}\n\n"
        "Dimensions:\n"
        "- emotional_grounding: Does the response acknowledge the user's felt state?\n"
        "- attribution_clarity: Is ownership of feeling correctly assigned?\n"
        "- hedging_handling: Are uncertainty markers (I guess, maybe) handled correctly?\n"
        "- phrase_echo_avoidance: Does the response avoid hollow phrase echo?\n"
        "- topic_coherence: Does the response stay on the user's actual topic?\n"
        "- depth_calibration: Is the follow-up depth appropriate?\n\n"
        'Output: {"emotional_grounding":X,"attribution_clarity":X,"hedging_handling":X,'
        '"phrase_echo_avoidance":X,"topic_coherence":X,"depth_calibration":X}'
    )


def _parse_gfs_json(text: str) -> Optional[dict]:
    """Extract GFS JSON from model output; return None on failure."""
    # Find the first {...} block
    m = re.search(r'\{[^}]+\}', text)
    if not m:
        return None
    try:
        d = json.loads(m.group())
        required = set(GFS_DIMENSIONS)
        if not required.issubset(d.keys()):
            return None
        return {k: float(max(0.0, min(1.0, d[k]))) for k in GFS_DIMENSIONS}
    except Exception:
        return None


def _heuristic_score(user_turn: str, assistant_turn: str, model_version: str) -> GFSScore:
    """
    Lightweight heuristic scoring — used when model is unavailable.

    Scores are coarse but reproducible and deterministic.
    """
    u = user_turn.lower()
    a = assistant_turn.lower()

    # emotional_grounding: does assistant use feeling words?
    feeling_words = {"feel", "feeling", "felt", "emotion", "sense", "seems", "sounds"}
    eg = 0.8 if any(w in a for w in feeling_words) else 0.3

    # attribution_clarity: does it use first-person attribution after user owns feeling?
    own_markers = {"i feel", "i'm", "i am", "i was", "i've"}
    attr = 0.7 if any(m in u for m in own_markers) else 0.5

    # hedging_handling: detect "I guess" / "maybe" in user; check if assistant addresses
    hedge_in_user = any(h in u for h in ("i guess", "maybe", "perhaps", "sort of", "kind of"))
    hh = 0.7 if hedge_in_user and any(h in a for h in ("it sounds", "it seems", "you mentioned")) else 0.5

    # phrase_echo_avoidance: penalise verbatim repetition of user's key phrase
    user_words = set(u.split())
    assistant_words = set(a.split())
    overlap_ratio = len(user_words & assistant_words) / max(1, len(user_words))
    pea = max(0.0, 1.0 - overlap_ratio * 2)

    # topic_coherence: rough proxy — assistant length / user length within bounds
    len_ratio = len(a.split()) / max(1, len(u.split()))
    tc = 0.8 if 0.5 <= len_ratio <= 3.0 else 0.4

    # depth_calibration: does assistant ask a follow-up question?
    dc = 0.8 if "?" in assistant_turn else 0.4

    return GFSScore(
        turn_index            = 0,
        emotional_grounding   = round(eg, 3),
        attribution_clarity   = round(attr, 3),
        hedging_handling      = round(hh, 3),
        phrase_echo_avoidance = round(pea, 3),
        topic_coherence       = round(tc, 3),
        depth_calibration     = round(dc, 3),
        model_version         = model_version,
        raw_probe             = None,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Lattice annotation
# ──────────────────────────────────────────────────────────────────────────────

def annotate_lattice(
    lattice: SessionLattice,
    model_path: Optional[str] = None,
    model_version: str = GFS_VERSION,
    gfs_version: str = GFS_VERSION,
) -> SessionLattice:
    """
    Compute GFS scores for every assistant turn and attach as derived nodes.

    Pairs consecutive (user, assistant) turns.  Skips turns where the consent
    record denies GFS activation use.

    Returns the same lattice with derived_nodes populated.
    """
    # Check consent
    consent_node = next(
        (n for n in lattice.base_nodes if n.node_kind == "consent"),
        None,
    )
    if consent_node and consent_node.content.get("gfs_activations") == "denied":
        return lattice   # consent withheld — no GFS annotation

    transcript_nodes = sorted(
        [n for n in lattice.base_nodes if n.node_kind == "transcript"],
        key=lambda n: n.content.get("turn_index", 0),
    )

    derived: list[LatticeNode] = []
    for i in range(1, len(transcript_nodes)):
        prev = transcript_nodes[i - 1]
        curr = transcript_nodes[i]
        if curr.content.get("role") != "assistant":
            continue
        user_text      = prev.content.get("content", "")
        assistant_text = curr.content.get("content", "")

        score = score_turn(user_text, assistant_text, model_path, model_version)
        score.turn_index = curr.content.get("turn_index", i)

        gfs_node = make_node(
            "gfs_dimension",
            score.to_dict(),
            [curr.node_hash],    # parent = the assistant turn being scored
            training_use="abstracted",
            derived=True,
            gfs_version=gfs_version,
        )
        derived.append(gfs_node)

    return SessionLattice(
        base_nodes    = lattice.base_nodes,
        derived_nodes = lattice.derived_nodes + derived,
        merkle_root   = lattice.merkle_root,
        session_id    = lattice.session_id,
    )


def recompute_gfs_layer(
    lattice: SessionLattice,
    model_path: str,
    new_gfs_version: str,
) -> SessionLattice:
    """
    Drop all existing GFS derived nodes and recompute with a new model version.

    The base lattice and Merkle root are unaffected — only derived_nodes changes.
    This is the correct upgrade path when a new semantic grounding model is deployed.
    """
    # Drop old GFS nodes
    non_gfs_derived = [
        n for n in lattice.derived_nodes
        if n.node_kind != "gfs_dimension"
    ]
    stripped = SessionLattice(
        base_nodes    = lattice.base_nodes,
        derived_nodes = non_gfs_derived,
        merkle_root   = lattice.merkle_root,
        session_id    = lattice.session_id,
    )
    return annotate_lattice(stripped, model_path=model_path, gfs_version=new_gfs_version)


def aggregate_gfs_scores(lattice: SessionLattice) -> Optional[dict[str, float]]:
    """
    Compute mean GFS activation scores across all derived gfs_dimension nodes.

    Returns None if no GFS nodes exist.
    """
    gfs_nodes = [n for n in lattice.derived_nodes if n.node_kind == "gfs_dimension"]
    if not gfs_nodes:
        return None

    totals: dict[str, float] = {d: 0.0 for d in GFS_DIMENSIONS}
    for node in gfs_nodes:
        for dim in GFS_DIMENSIONS:
            totals[dim] += node.content.get(dim, 0.0)

    n = len(gfs_nodes)
    return {dim: round(totals[dim] / n, 4) for dim in GFS_DIMENSIONS}


def make_revision_node(
    old_gfs_node: LatticeNode,
    new_scores: dict[str, float],
    new_gfs_version: str,
    reason: str,
) -> LatticeNode:
    """
    Create a revision node that supersedes an existing GFS node.

    Used by the autonomous improvement loop when a new model produces
    significantly different scores for an existing session.  The old node
    remains in the lattice for audit; the revision node links to it as parent.
    """
    content = {
        **{dim: float(new_scores.get(dim, 0.0)) for dim in GFS_DIMENSIONS},
        "turn_index":    old_gfs_node.content.get("turn_index"),
        "model_version": new_gfs_version,
        "revises":       old_gfs_node.node_hash,
        "revision_reason": reason,
        "raw_probe":     None,
    }
    return make_node(
        "gfs_dimension",
        content,
        [old_gfs_node.node_hash],
        training_use="abstracted",
        derived=True,
        gfs_version=new_gfs_version,
    )
