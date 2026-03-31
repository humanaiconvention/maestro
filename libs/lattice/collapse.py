"""
collapse.py -- Weighted collapse of lattice signals into training_signal nodes.

This is the Maestro equivalent of Brescia's AICollapseStrategyService:
multiple orthogonal signals held in superposition across a session lattice
are collapsed into a single training_signal node with a precision-weighted
training value.

Unlike Brescia's weighted sum (alpha*valence + beta*coherence + gamma*contradiction),
the collapse function here is grounded in the E <= C viability condition:

    training_weight = integration_depth * oracle_capacity * (1 + contradiction)

Where:
  - integration_depth: how many orthogonal meaning-dimensions were traversed
    (Tononi's phi proxy -- sessions that engage diverse signal types are
    more informative for landscape shaping)
  - oracle_capacity: whether the human participant was in a flourishing state
    (Friston's precision-weighting -- high-capacity oracle corrections carry
    more weight because they're more likely to be genuine gradient signal)
  - contradiction: the Shadow Core signal -- divergence between participant
    and model framings amplifies training weight because these are the
    highest-information moments (Brescia's key insight: contradiction is
    fuel, not noise)

The memory field provides cross-session context:
  - Attractor density: how many prior sessions explored this region
  - Regional weight: mean training weight in this region
  - These inform a novelty bonus (unexplored regions get upweighted)
    and a drift penalty (dense regions with declining oracle capacity
    suggest solipsistic drift)

Output: training_signal nodes appended to lattice.derived_nodes, each
containing the collapse weights, component scores, and the session
transcript formatted as an SFT training pair.

Usage:
    from maestro.libs.lattice.collapse import CollapseFunction
    from maestro.libs.lattice.signal_scorer import SignalScorer
    from maestro.libs.lattice.memory_field import MemoryField

    scorer = SignalScorer()
    field = MemoryField("data/memory_field.pkl")
    collapse = CollapseFunction(scorer, field)

    lattice_dict = json.load(open("session.json"))
    signals = collapse.collapse(lattice_dict)
    # signals is a list of TrainingSignal dataclasses
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from libs.lattice.signal_scorer import SignalScorer, SessionSignalScore, _get_embedder
    from libs.lattice.memory_field import MemoryField
    from libs.lattice.cognitive_cores import CognitiveRouter, CognitiveProcessingResult
except ImportError:
    from signal_scorer import SignalScorer, SessionSignalScore, _get_embedder  # type: ignore[no-redef]
    from memory_field import MemoryField  # type: ignore[no-redef]
    from cognitive_cores import CognitiveRouter, CognitiveProcessingResult  # type: ignore[no-redef]


# -----------------------------------------------------------------------
# Output schema
# -----------------------------------------------------------------------

@dataclass
class TrainingSignal:
    """A collapsed training signal ready for SGT pipeline consumption."""

    session_id:       str
    signal_type:      str          # "sft_pair" | "dpo_pair" | "weight_only"
    weight:           float        # Final precision-weighted training weight
    raw_weight:       float        # From SignalScorer (before memory adjustment)

    # Component scores (for auditability)
    integration_depth:   float
    semantic_entropy:    float
    temporal_coherence:  float
    oracle_capacity:     float
    oracle_effective:    float      # oracle_capacity * (1 + affective_granularity)
    affective_granularity: float    # Barrett emotional granularity
    affect_arc_coherence:  float    # Temporal coherence of affect trajectory
    novelty:             float
    contradiction_score: float

    # Memory field context
    attractor_density:   int        # How many prior sessions in this region
    regional_weight:     float      # Mean weight in this region
    novelty_bonus:       float      # Upweight for unexplored regions
    drift_penalty:       float      # Downweight for potential solipsistic drift

    # The actual training content
    turns:            list[dict]    # [{"role": "...", "content": "..."}]
    turn_count:       int

    # Consent gate
    consent_allows_training: bool

    def to_dict(self) -> dict:
        return asdict(self)


# -----------------------------------------------------------------------
# Drift detection
# -----------------------------------------------------------------------

def _detect_drift(
    attractor_density: int,
    regional_weight: float,
    current_oracle: float,
    density_threshold: int = 3,
    oracle_decay_ratio: float = 0.7,
) -> float:
    """
    Detect potential solipsistic drift in an attractor region.

    Drift signature: dense region (many visits) where oracle capacity
    is declining relative to regional average.  This means the model
    is attracting sessions to this region but participants are becoming
    less engaged / more compliant -- exactly the Shadow-wins-4-of-5
    pattern Brescia observed.

    Returns a penalty in [0, 1] where 0 = no drift, 1 = severe drift.
    """
    if attractor_density < density_threshold:
        return 0.0  # Too few visits to diagnose drift

    if regional_weight < 1e-6:
        return 0.0  # No prior signal to compare against

    # Expected oracle capacity from regional average:
    # regional_weight = integration * oracle * (1 + contradiction)
    # We can't decompose perfectly, but if current oracle is well below
    # what the region historically produced, that's a drift signal.
    if current_oracle < 1e-6:
        return 0.8  # Near-zero oracle in a populated region is concerning

    # Simple ratio: if current oracle is < 70% of what it used to be
    # in this region, apply drift penalty proportional to the decay
    # This is a proxy; with more data we'd track oracle_capacity trends
    # per attractor region in the memory field.
    # For now, use a heuristic: regions with high density and low
    # current oracle relative to regional weight are drifting.
    expected_floor = oracle_decay_ratio
    if current_oracle < expected_floor:
        penalty = (expected_floor - current_oracle) / expected_floor
        return min(penalty, 1.0)

    return 0.0


def _novelty_bonus(
    attractor_density: int,
    novelty_score: float,
    max_bonus: float = 0.5,
) -> float:
    """
    Bonus multiplier for sessions exploring new attractor regions.

    Novel sessions (high cosine distance from corpus centroid, low
    attractor density) get upweighted because they're adding new
    structure to the landscape rather than reinforcing existing patterns.

    Returns a bonus in [0, max_bonus].
    """
    if attractor_density == 0:
        # Completely new region
        return max_bonus * min(novelty_score * 2, 1.0)

    # Diminishing bonus as density increases
    density_factor = 1.0 / (1.0 + attractor_density)
    return max_bonus * novelty_score * density_factor


# -----------------------------------------------------------------------
# Core collapse function
# -----------------------------------------------------------------------

class CollapseFunction:
    """
    Collapses lattice signals into precision-weighted training signals.

    The collapse integrates:
    1. Five landscape observables (from SignalScorer)
    2. Cross-session memory context (from MemoryField)
    3. Consent gates (from the lattice consent node)
    4. Drift detection (solipsistic drift penalty)
    5. Novelty bonus (unexplored attractor regions)
    """

    def __init__(
        self,
        scorer: SignalScorer,
        memory_field: MemoryField,
        attractor_radius: float = 0.3,
        min_weight_threshold: float = 0.1,
    ):
        self.scorer = scorer
        self.memory_field = memory_field
        self.attractor_radius = attractor_radius
        self.min_weight_threshold = min_weight_threshold
        self.router = CognitiveRouter(scorer, memory_field, attractor_radius)

    def _check_consent(self, lattice_data: dict) -> bool:
        """Check if the session's consent allows training signal generation."""
        for node in lattice_data.get("base_nodes", []):
            if node["node_kind"] == "consent":
                ts = node["content"].get("training_signal", "denied")
                return ts != "denied"
        return False

    def _extract_conversation(self, lattice_data: dict) -> list[dict]:
        """Extract ordered conversation turns for SFT formatting."""
        transcripts = []
        for node in lattice_data.get("base_nodes", []):
            if node["node_kind"] == "transcript":
                transcripts.append(node)

        transcripts.sort(key=lambda n: n["content"].get("turn_index", 0))

        turns = []
        for t in transcripts:
            role = t["content"].get("role", "")
            content = t["content"].get("content", "")
            if role in ("user", "assistant"):
                turns.append({"role": role, "content": content})

        return turns

    def collapse(self, lattice_data: dict) -> TrainingSignal:
        """
        Collapse a session lattice into a training signal.

        Steps:
        1. Score the session (five observables + contradiction)
        2. Compute session centroid for memory field queries
        3. Query memory field for attractor context
        4. Apply drift detection and novelty bonus
        5. Compute final precision-weighted training weight
        6. Package as TrainingSignal with full auditability
        """
        # 1. Score
        score = self.scorer.score_from_dict(lattice_data)

        # 2. Compute session centroid
        all_texts, _, _ = self.scorer._extract_turns(lattice_data)
        centroid = None
        if all_texts:
            embedder = _get_embedder()
            all_embs = embedder.encode(all_texts, convert_to_numpy=True)
            centroid = all_embs.mean(axis=0)

        # 3. Memory field context
        density = 0
        reg_weight = 0.0
        if centroid is not None:
            density = self.memory_field.attractor_density(
                centroid, radius=self.attractor_radius
            )
            reg_weight = self.memory_field.mean_weight_in_region(
                centroid, radius=self.attractor_radius
            )

        # 4. Drift and novelty adjustments
        # Use oracle_effective (which includes affective precision)
        # for drift detection — a participant with high emotional
        # granularity is less likely to be in a compliance/drift state
        drift = _detect_drift(
            density, reg_weight, score.oracle_effective
        )
        nov_bonus = _novelty_bonus(density, score.novelty)

        # 5. Final weight
        #    raw_weight * (1 + novelty_bonus) * (1 - drift_penalty)
        adjusted = score.training_weight * (1.0 + nov_bonus) * (1.0 - drift)

        # 6. Consent gate
        consent_ok = self._check_consent(lattice_data)

        # If consent denies training, weight goes to zero but we still
        # record the scores for monitoring (E <= C tracking)
        if not consent_ok:
            adjusted = 0.0

        # 7. Package
        turns = self._extract_conversation(lattice_data)

        signal_type = "weight_only"
        if consent_ok and adjusted >= self.min_weight_threshold and len(turns) >= 2:
            signal_type = "sft_pair"

        return TrainingSignal(
            session_id=score.session_id,
            signal_type=signal_type,
            weight=round(adjusted, 4),
            raw_weight=round(score.training_weight, 4),
            integration_depth=score.integration_depth,
            semantic_entropy=score.semantic_entropy,
            temporal_coherence=score.temporal_coherence,
            oracle_capacity=score.oracle_capacity,
            oracle_effective=score.oracle_effective,
            affective_granularity=score.affective_granularity,
            affect_arc_coherence=score.affect_arc_coherence,
            novelty=score.novelty,
            contradiction_score=score.contradiction_score,
            attractor_density=density,
            regional_weight=round(reg_weight, 4),
            novelty_bonus=round(nov_bonus, 4),
            drift_penalty=round(drift, 4),
            turns=turns,
            turn_count=len(turns),
            consent_allows_training=consent_ok,
        )

    def collapse_with_routing(self, lattice_data: dict) -> tuple[TrainingSignal, CognitiveProcessingResult]:
        """
        Full Orch-OS-aligned pipeline: cognitive routing -> collapse.

        Phase I:   NeuralSignal generation (via CognitiveRouter)
        Phase II:  Parallel core activation (via CognitiveRouter)
        Phase III: Symbolic collapse (via this CollapseFunction)

        Returns both the collapsed TrainingSignal and the full
        CognitiveProcessingResult with timeline for auditability.
        """
        # Phases I + II: route through cognitive cores
        cognitive_result = self.router.process(lattice_data)

        # Phase III: collapse using the scored superposition
        signal = self.collapse(lattice_data)

        # Log the collapse event into the cognitive timeline
        from cognitive_cores import TimelineEvent
        import time as _time
        cognitive_result.timeline.append(TimelineEvent(
            timestamp=_time.time(),
            event_type="symbolic_collapse",
            core="system",
            data={
                "signal_type": signal.signal_type,
                "raw_weight": signal.raw_weight,
                "novelty_bonus": signal.novelty_bonus,
                "drift_penalty": signal.drift_penalty,
                "final_weight": signal.weight,
                "consent": signal.consent_allows_training,
                "dominant_cores": self.router._dominant_cores(cognitive_result.signals),
            },
        ))

        return signal, cognitive_result

    def collapse_and_store(self, lattice_data: dict) -> TrainingSignal:
        """
        Collapse + update the memory field with this session.

        Call this during the normal pipeline flow so the memory field
        accumulates session centroids for future cross-session queries.
        """
        signal = self.collapse(lattice_data)

        # Store in memory field (even if consent denies training,
        # we store the centroid for landscape monitoring)
        all_texts, _, _ = self.scorer._extract_turns(lattice_data)
        if all_texts:
            embedder = _get_embedder()
            all_embs = embedder.encode(all_texts, convert_to_numpy=True)
            centroid = all_embs.mean(axis=0)

            score = self.scorer.score_from_dict(lattice_data)
            created = lattice_data.get("base_nodes", [{}])[-1].get(
                "created_at", ""
            )
            self.memory_field.store_from_score(
                score, centroid, timestamp=created
            )

        return signal


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    lattice_dir = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..",
            "data", "lattices",
        )
    )
    lattice_dir = os.path.abspath(lattice_dir)

    field_path = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(
            os.path.dirname(__file__),
            "..", "..", "..", "..",
            "data", "memory_field.pkl",
        )
    )
    field_path = os.path.abspath(field_path)

    print(f"Lattice dir:  {lattice_dir}")
    print(f"Field path:   {field_path}")
    print()

    scorer = SignalScorer()
    mf = MemoryField(field_path)
    collapse_fn = CollapseFunction(scorer, mf)

    files = sorted(Path(lattice_dir).glob("*.json"))
    signals = []

    use_routing = "--route" in sys.argv

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        if use_routing:
            signal, cognitive = collapse_fn.collapse_with_routing(data)
            # Store in memory field
            all_texts, _, _ = collapse_fn.scorer._extract_turns(data)
            if all_texts:
                embedder = _get_embedder()
                all_embs = embedder.encode(all_texts, convert_to_numpy=True)
                centroid = all_embs.mean(axis=0)
                score = collapse_fn.scorer.score_from_dict(data)
                created = data.get("base_nodes", [{}])[-1].get("created_at", "")
                mf.store_from_score(score, centroid, timestamp=created)
            # Print cognitive timeline
            print(collapse_fn.router.format_timeline(cognitive))
            print()
        else:
            signal = collapse_fn.collapse_and_store(data)

        signals.append(signal)

        consent_str = "YES" if signal.consent_allows_training else "NO"
        print(f"-- {signal.session_id} --")
        print(f"  Signal type:         {signal.signal_type}")
        print(f"  Consent:             {consent_str}")
        print(f"  Raw weight:          {signal.raw_weight:.4f}")
        print(f"  Novelty bonus:       +{signal.novelty_bonus:.4f}")
        print(f"  Drift penalty:       -{signal.drift_penalty:.4f}")
        print(f"  Final weight:        {signal.weight:.4f}")
        print(f"  Attractor density:   {signal.attractor_density}")
        print(f"  Regional weight:     {signal.regional_weight:.4f}")
        print(f"  Turns:               {signal.turn_count}")
        print()

    # Save updated memory field
    mf.save()

    # Summary
    trainable = [s for s in signals if s.signal_type == "sft_pair"]
    denied = [s for s in signals if not s.consent_allows_training]
    zero_weight = [s for s in signals if s.weight == 0 and s.consent_allows_training]

    print(f"=== Collapse summary ({len(signals)} sessions) ===")
    print(f"  Trainable (sft_pair):     {len(trainable)}")
    print(f"  Consent denied:           {len(denied)}")
    print(f"  Zero weight (consented):  {len(zero_weight)}")
    if trainable:
        weights = [s.weight for s in trainable]
        print(f"  Mean trainable weight:    {np.mean(weights):.4f}")
        print(f"  Max trainable weight:     {max(weights):.4f}")

    # Write signals to JSON for SGT pipeline
    output_path = os.path.join(
        os.path.dirname(lattice_dir), "training_signals.json"
    )
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(
            [s.to_dict() for s in signals],
            fh, indent=2, ensure_ascii=False,
        )
    print(f"\n  Signals written to: {output_path}")
