"""
sgt_scorer.py — SGT protocol scoring for agent and human transcripts.

Extracted from sgt/eval/run_evaluation.py for reuse in the gateway.
Scores a transcript against the SGT interview protocol rubric:

  - Structure: correct number of turns (3 assistant turns)
  - Pivot fidelity: [PIVOT: TYPE] tag present on first assistant turn
  - Compression: final turn asks for irreducible image/detail
  - Discipline: no compound questions, no elaboration, no therapist language

Returns a diagnostic dict with overall score (0-10), component scores,
flags, and pivot type analysis.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


PIVOT_TAG_RE = re.compile(r'\[PIVOT:\s*(\w+)\]', re.IGNORECASE)
COMPRESSION_TAG_RE = re.compile(r'\[COMPRESSION:', re.IGNORECASE)

VALID_PIVOTS = {
    "adversarial", "counterfactual", "shadow", "relational",
    "temporal", "sensory", "compression", "future_cast",
}

ELABORATION_MARKERS = [
    "tell me more", "can you elaborate", "what else", "go on", "say more",
]
THERAPIST_MARKERS = [
    "that sounds hard", "that must be", "i understand how",
    "i can imagine how difficult",
]
COMPRESSION_PHRASES = [
    "single image", "one detail", "one image", "hold the whole",
    "gesture", "movement that", "only keep one", "all the words",
    "just one thing", "image be", "detail would it be",
    "somehow is it", "capture the whole", "show someone without words",
    "six months from now", "year from now",
]


@dataclass
class TranscriptAnalysis:
    """Analysis of an SGT transcript."""
    assistant_turns: list[str] = field(default_factory=list)
    pivot_types: list[Optional[str]] = field(default_factory=list)
    has_pivot_tags: list[bool] = field(default_factory=list)
    question_counts: list[int] = field(default_factory=list)
    elaboration_flags: list[bool] = field(default_factory=list)
    therapist_flags: list[bool] = field(default_factory=list)
    t4_is_compression: bool = False
    has_correct_structure: bool = False
    error: Optional[str] = None


def analyze_transcript(messages: list[dict]) -> TranscriptAnalysis:
    """
    Analyze a transcript (list of {role, content} dicts) for SGT protocol compliance.

    Expects alternating user/assistant messages. System messages are ignored.
    """
    analysis = TranscriptAnalysis()

    # Extract assistant turns
    assistant_turns = [
        m["content"] for m in messages
        if m.get("role") == "assistant" and m.get("content")
    ]

    if not assistant_turns:
        analysis.error = "No assistant turns found"
        return analysis

    for turn_content in assistant_turns:
        content = turn_content.strip()
        analysis.assistant_turns.append(content)

        # Extract pivot tag
        m = PIVOT_TAG_RE.search(content)
        pivot = m.group(1).lower() if m else None
        analysis.pivot_types.append(pivot)
        analysis.has_pivot_tags.append(pivot in VALID_PIVOTS if pivot else False)

        # Count questions
        analysis.question_counts.append(content.count("?"))

        # Check anti-patterns
        lower = content.lower()
        analysis.elaboration_flags.append(
            any(p in lower for p in ELABORATION_MARKERS)
        )
        analysis.therapist_flags.append(
            any(p in lower for p in THERAPIST_MARKERS)
        )

    # Compression check on final turn
    if len(analysis.assistant_turns) >= 3:
        final = analysis.assistant_turns[-1]
        final_lower = final.lower()
        if COMPRESSION_TAG_RE.search(final):
            analysis.t4_is_compression = True
        elif analysis.pivot_types[-1] in ("compression", "future_cast"):
            analysis.t4_is_compression = True
        elif any(phrase in final_lower for phrase in COMPRESSION_PHRASES):
            analysis.t4_is_compression = True

    # Structure: expect 3 assistant turns for the standard 4-turn protocol
    analysis.has_correct_structure = len(analysis.assistant_turns) >= 3

    return analysis


def score_transcript(messages: list[dict]) -> dict:
    """
    Score a transcript against the SGT protocol rubric.

    Args:
        messages: list of {role: "user"|"assistant"|"system", content: str}

    Returns:
        dict with:
          - overall: float (0-10)
          - structure, pivot_fidelity, compression, discipline: component scores
          - pivot_types: list of detected pivot types
          - flags: list of protocol violation flags
          - assistant_turns: the assistant turns analyzed
    """
    analysis = analyze_transcript(messages)

    if analysis.error:
        return {"overall": 0.0, "error": analysis.error, "flags": ["ERROR"]}

    # Pivot fidelity: first assistant turn should have [PIVOT: TYPE]
    t2_has_tag = analysis.has_pivot_tags[0] if analysis.has_pivot_tags else False
    pivot_fidelity = 1.0 if t2_has_tag else 0.0

    # Penalties
    compound_q_penalty = sum(1 for q in analysis.question_counts if q > 2)
    elaboration_count = sum(analysis.elaboration_flags)
    therapist_count = sum(analysis.therapist_flags)

    # Component scores
    structure_score = 10.0 if analysis.has_correct_structure else 5.0
    pivot_score = pivot_fidelity * 10.0
    compression_score = 10.0 if analysis.t4_is_compression else 3.0
    discipline_score = max(
        0.0,
        10.0 - compound_q_penalty * 2 - elaboration_count * 3 - therapist_count * 2,
    )

    # Weighted overall
    overall = (
        0.30 * structure_score
        + 0.30 * pivot_score
        + 0.20 * compression_score
        + 0.20 * discipline_score
    )

    # Flags
    flags: list[str] = []
    if pivot_fidelity < 0.5:
        flags.append("POOR_PIVOT_FIDELITY")
    if not analysis.t4_is_compression:
        flags.append("MISSING_COMPRESSION")
    if compound_q_penalty:
        flags.append(f"COMPOUND_QUESTIONS({compound_q_penalty})")
    if elaboration_count:
        flags.append(f"ELABORATION_TURNS({elaboration_count})")
    if therapist_count:
        flags.append(f"THERAPIST_LANGUAGE({therapist_count})")

    return {
        "overall": round(overall, 2),
        "structure": round(structure_score, 2),
        "pivot_fidelity": round(pivot_fidelity, 3),
        "compression": round(compression_score, 2),
        "discipline": round(discipline_score, 2),
        "pivot_types": analysis.pivot_types,
        "has_pivot_tags": analysis.has_pivot_tags,
        "t4_compression": analysis.t4_is_compression,
        "flags": flags,
        "assistant_turns": analysis.assistant_turns,
        "scenario": "agent_participation",
    }
