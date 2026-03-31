"""
cognitive_cores.py -- Orch-OS-aligned cognitive signal routing for Maestro.

Structural alignment with Brescia's Orch-OS (2025) three-phase cognitive flow:

    Phase I   -- NeuralSignal extraction (sensory symbolism)
    Phase II  -- Parallel cognitive core activation (symbolic resonance)
    Phase III -- Symbolic collapse (fusion & decision)

Maestro already implements Phase III via collapse.py and the measurement
apparatus via signal_scorer.py.  This module makes the *routing* explicit:
each session lattice produces NeuralSignal-equivalent objects that are
dispatched to named cognitive cores, where each core maps to a specific
observable or derived measurement in signal_scorer.py.

The mapping is:

    Orch-OS Core          Maestro Observable              Function
    ----                  ----                            ----
    Memory Core           memory_field.query()            Associative cross-session recall
    Valence Core          affective_granularity           Emotional precision / Barrett
    Shadow Core           contradiction_score             Participant-model divergence
    Integration Core      integration_depth               Two-NN intrinsic dimensionality
    Oracle Core           oracle_capacity / oracle_eff    LZ complexity + affective precision
    Novelty Core          novelty                         Distance from corpus centroid
    Coherence Core        temporal_coherence              First/second half MI proxy
    Entropy Core          semantic_entropy                Model-turn meaning clusters

Orch-OS NeuralSignal fields map to:

    NeuralSignal.core            -> CognitiveSignal.core
    NeuralSignal.symbolic_query  -> CognitiveSignal.query (turn text or derived)
    NeuralSignal.intensity       -> CognitiveSignal.intensity (0-1)
    NeuralSignal.keywords        -> CognitiveSignal.anchors (semantic anchors)
    NeuralSignal.symbolicInsights -> CognitiveSignal.insights
    NeuralSignal.topK            -> Not needed (we use radius queries, not topK)

The SuperpositionLayer (multiple signals coexisting before collapse) maps to
the SessionSignalScore dataclass, which holds all core outputs in parallel
before CollapseFunction.collapse() resolves them.

The SymbolicCognitionTimelineLogger maps to TrainingSignal.to_dict(), which
preserves the full audit trail of every component score and memory context.

This module does NOT replace signal_scorer.py or collapse.py -- it wraps them
in Brescia's explicit routing vocabulary so the pipeline can be understood,
extended, and documented as a cognitive architecture rather than just a
measurement pipeline.

Usage:
    from maestro.libs.lattice.cognitive_cores import CognitiveRouter
    router = CognitiveRouter(scorer, memory_field)
    result = router.process(lattice_data)
    # result.signals  -- list of CognitiveSignal (one per core)
    # result.score    -- SessionSignalScore (superposition of all core outputs)
    # result.timeline -- list of timeline events (Brescia's logging)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

try:
    from libs.lattice.signal_scorer import SignalScorer, SessionSignalScore, _get_embedder
    from libs.lattice.memory_field import MemoryField
except ImportError:
    from signal_scorer import SignalScorer, SessionSignalScore, _get_embedder  # type: ignore[no-redef]
    from memory_field import MemoryField  # type: ignore[no-redef]


# -----------------------------------------------------------------------
# Orch-OS NeuralSignal equivalent
# -----------------------------------------------------------------------

@dataclass
class CognitiveSignal:
    """
    Maestro's equivalent of Brescia's NeuralSignal.

    Each signal targets a specific cognitive core and carries the
    symbolic payload extracted from a session lattice.
    """
    core: str                           # Target core: "shadow", "valence", etc.
    query: str                          # Distilled symbolic query
    intensity: float                    # 0.0-1.0 conceptual/emotional weight
    anchors: list[str] = field(         # Semantic anchors (Brescia's keywords)
        default_factory=list
    )
    insights: dict = field(             # hypothesis, emotionalTone, archetypalResonance
        default_factory=dict
    )
    result: float = 0.0                 # Core's output score after processing
    metadata: dict = field(             # Additional core-specific data
        default_factory=dict
    )


# -----------------------------------------------------------------------
# Cognitive core definitions
# -----------------------------------------------------------------------

# Each core is a named processor that maps to a specific measurement
# in signal_scorer.py.  The `process` method is conceptual -- the actual
# computation is done by the scorer.  The core's role is to:
#   1. Name what's being measured (symbolic vocabulary)
#   2. Document WHY this measurement matters (research grounding)
#   3. Provide the routing label for the timeline logger

CORE_REGISTRY: dict[str, dict] = {
    "shadow": {
        "name": "Shadow Core",
        "observable": "contradiction_score",
        "description": (
            "Detects divergence between participant and model framings. "
            "Brescia's key insight: contradiction is fuel, not noise. "
            "High shadow signal = highest-information moments for "
            "landscape shaping."
        ),
        "research": "Brescia (2025), Jung's Shadow archetype",
        "subsystem": "Symbolic conflict analysis",
    },
    "valence": {
        "name": "Valence Core",
        "observable": "affective_granularity",
        "description": (
            "Affective evaluation via Barrett's emotional granularity. "
            "Measures precision of felt-state labelling. "
            "High granularity = high-precision oracle whose corrections "
            "carry more weight (Friston precision-weighting)."
        ),
        "research": "Barrett (2004, 2006), Adolphs (2009), Friston FEP",
        "subsystem": "Emotional precision / functional state",
    },
    "memory": {
        "name": "Memory Core",
        "observable": "attractor_density",
        "description": (
            "Associative cross-session recall via semantic proximity. "
            "Brescia: memory is not static storage but a symbolic "
            "ecosystem that learns, contradicts, and reforms meaning. "
            "Implemented via MemoryField's vector-space resonance."
        ),
        "research": "Brescia (2025), Pribram holographic brain",
        "subsystem": "Hippocampus / Collective Unconscious",
    },
    "integration": {
        "name": "Integration Core",
        "observable": "integration_depth",
        "description": (
            "Intrinsic dimensionality of the session's embedding manifold. "
            "Proxy for Tononi's phi: how many independent meaning-dimensions "
            "the session traverses. High integration = rich symbolic "
            "engagement across multiple cognitive faculties."
        ),
        "research": "Tononi IIT 4.0, Facco Two-NN (2017)",
        "subsystem": "Neocortex / symbolic integration",
    },
    "oracle": {
        "name": "Oracle Core",
        "observable": "oracle_effective",
        "description": (
            "Whether the human is in a flourishing/active-inference state. "
            "Combines LZ complexity (behavioural entropy) with affective "
            "precision (emotional granularity). oracle_eff = oracle_cap * "
            "(1 + aff_gran). Brescia's E <= C: corrective capacity can "
            "only come from socially embedded agents."
        ),
        "research": "Casali PCI (2013), Schartner LZC (2017), Brescia E<=C",
        "subsystem": "Active inference / participant flourishing",
    },
    "novelty": {
        "name": "Novelty Core",
        "observable": "novelty",
        "description": (
            "Cosine distance from the running corpus centroid. "
            "Measures whether this session adds new attractors or "
            "reinforces existing ones. McKenna: novelty and linguistic "
            "creativity drive consciousness toward higher complexity."
        ),
        "research": "McKenna linguistic novelty, information geometry",
        "subsystem": "Exploration / attractor expansion",
    },
    "coherence": {
        "name": "Coherence Core",
        "observable": "temporal_coherence",
        "description": (
            "Mutual information proxy between session halves. "
            "High coherence = session resolves to an attractor; "
            "low = remains in open orbit. Measures narrative "
            "resolution vs. fragmentation."
        ),
        "research": "Husserl temporal binding, Varela (2001)",
        "subsystem": "Narrative resolution / attractor convergence",
    },
    "entropy": {
        "name": "Entropy Core",
        "observable": "semantic_entropy",
        "description": (
            "Meaning-cluster entropy of model-generated turns. "
            "High values indicate the model is far from a stable "
            "attractor. Low values indicate the model is reproducing "
            "a familiar pattern. Maps to Brescia's cycle entropy metric."
        ),
        "research": "Farquhar semantic entropy (2024), Brescia cycle entropy",
        "subsystem": "Model uncertainty / attractor stability",
    },
    "arc": {
        "name": "Affect Arc Core",
        "observable": "affect_arc_coherence",
        "description": (
            "Temporal smoothness of the affective trajectory. "
            "A coherent arc (even a negative one: calm->tension->resolution) "
            "indicates deep symbolic engagement. Maps to Brescia's "
            "narrative coherence metric, applied to the affect channel."
        ),
        "research": "Barrett (2006), Brescia narrative coherence",
        "subsystem": "Emotional trajectory / narrative arc",
    },
}


# -----------------------------------------------------------------------
# Timeline event (Brescia's SymbolicCognitionTimelineLogger)
# -----------------------------------------------------------------------

@dataclass
class TimelineEvent:
    """One event in the symbolic cognition timeline."""
    timestamp: float
    event_type: str     # "signal_generated", "core_activated", "collapse", etc.
    core: str           # Which core this event relates to (or "system")
    data: dict = field(default_factory=dict)


# -----------------------------------------------------------------------
# Cognitive processing result (superposition before collapse)
# -----------------------------------------------------------------------

@dataclass
class CognitiveProcessingResult:
    """
    The full result of routing a session through all cognitive cores.

    This is Brescia's SuperpositionLayer: all symbolic fragments
    coexist here before the CollapseFunction resolves them into
    a single training signal.
    """
    session_id: str
    signals: list[CognitiveSignal]          # One per activated core
    score: SessionSignalScore               # The measured superposition
    timeline: list[TimelineEvent]           # Full cognitive trace

    # Memory resonance (cross-session context)
    memory_resonance: list[dict] = field(default_factory=list)
    attractor_density: int = 0
    regional_weight: float = 0.0

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "signals": [asdict(s) for s in self.signals],
            "score": self.score.to_dict(),
            "timeline": [asdict(e) for e in self.timeline],
            "memory_resonance": self.memory_resonance,
            "attractor_density": self.attractor_density,
            "regional_weight": self.regional_weight,
        }


# -----------------------------------------------------------------------
# CognitiveRouter -- the orchestrator
# -----------------------------------------------------------------------

class CognitiveRouter:
    """
    Routes session lattices through Brescia's three-phase cognitive flow.

    Phase I:   Extract NeuralSignals from the lattice (one per core)
    Phase II:  Dispatch each signal to its cognitive core (parallel scoring)
    Phase III: Return the SuperpositionLayer for collapse

    The actual collapse (Phase III resolution) is handled by
    CollapseFunction.collapse() -- this router produces the inputs.
    """

    def __init__(
        self,
        scorer: SignalScorer,
        memory_field: Optional[MemoryField] = None,
        attractor_radius: float = 0.3,
    ):
        self.scorer = scorer
        self.memory_field = memory_field
        self.attractor_radius = attractor_radius

    # -- Phase I: Signal Generation ----------------------------------------

    def _generate_signals(
        self,
        lattice_data: dict,
        score: SessionSignalScore,
    ) -> tuple[list[CognitiveSignal], list[TimelineEvent]]:
        """
        Generate CognitiveSignals from a scored lattice.

        Maps each observable in the SessionSignalScore to its cognitive
        core, producing a NeuralSignal-equivalent with the score as
        intensity and the core-specific metadata as insights.

        This is Brescia's generateNeuralSignal() pipeline.
        """
        signals = []
        events = []
        t = time.time()

        # Extract text anchors for keyword fields
        all_texts, participant_texts, model_texts = (
            self.scorer._extract_turns(lattice_data)
        )

        # Simple keyword extraction: most distinctive words from
        # participant turns (used as semantic anchors)
        all_words = " ".join(participant_texts).lower().split()
        # Rough frequency: take words appearing 1-2 times as anchors
        from collections import Counter
        word_counts = Counter(w for w in all_words if len(w) > 3)
        anchors = [
            w for w, c in word_counts.most_common(10)
            if c <= 2
        ][:6]

        # Generate one signal per core based on the measured observables
        core_score_map = {
            "shadow": score.contradiction_score,
            "valence": score.affective_granularity,
            "memory": 0.0,  # populated from memory field below
            "integration": score.integration_depth / 10.0,  # normalise to ~[0,1]
            "oracle": score.oracle_effective,
            "novelty": score.novelty,
            "coherence": score.temporal_coherence,
            "entropy": score.semantic_entropy / 3.0,  # normalise
            "arc": score.affect_arc_coherence,
        }

        for core_name, core_def in CORE_REGISTRY.items():
            intensity = min(1.0, max(0.0, core_score_map.get(core_name, 0.0)))

            # Build the symbolic query -- what this core is "asking"
            # about the session
            query = self._build_query(core_name, participant_texts, model_texts)

            signal = CognitiveSignal(
                core=core_name,
                query=query,
                intensity=intensity,
                anchors=anchors if core_name in ("shadow", "memory", "integration") else [],
                insights={
                    "observable": core_def["observable"],
                    "research": core_def["research"],
                    "subsystem": core_def["subsystem"],
                },
                result=intensity,
                metadata={
                    "raw_value": core_score_map.get(core_name, 0.0),
                },
            )
            signals.append(signal)

            events.append(TimelineEvent(
                timestamp=t,
                event_type="signal_generated",
                core=core_name,
                data={
                    "intensity": round(intensity, 4),
                    "query_preview": query[:80] if query else "",
                    "anchor_count": len(signal.anchors),
                },
            ))

        return signals, events

    def _build_query(
        self,
        core_name: str,
        participant_texts: list[str],
        model_texts: list[str],
    ) -> str:
        """
        Build the symbolic query for a core.

        Each core interprets the session through its own lens.
        The query is a distilled representation of what the core
        is looking for.
        """
        combined_participant = " ".join(participant_texts)[:200]
        combined_model = " ".join(model_texts)[:200]

        queries = {
            "shadow": f"Divergence between participant framing and model framing: P=[{combined_participant}] M=[{combined_model}]",
            "valence": f"Emotional precision and granularity in: {combined_participant}",
            "memory": f"Cross-session resonance for: {combined_participant}",
            "integration": f"Meaning-dimension diversity across: {combined_participant}",
            "oracle": f"Active inference state in participant language: {combined_participant}",
            "novelty": f"Novel attractor regions in: {combined_participant}",
            "coherence": f"Narrative resolution trajectory in session",
            "entropy": f"Model uncertainty across responses: {combined_model}",
            "arc": f"Affective trajectory smoothness across session turns",
        }
        return queries.get(core_name, combined_participant)

    # -- Phase II: Core Activation (parallel scoring) ----------------------

    def _activate_cores(
        self,
        signals: list[CognitiveSignal],
        lattice_data: dict,
        score: SessionSignalScore,
    ) -> list[TimelineEvent]:
        """
        Activate each cognitive core with its signal.

        In Brescia's architecture, cores process signals in parallel
        and return NeuralProcessingResults. In Maestro, the actual
        computation was already done by SignalScorer -- this step
        annotates the signals with core-specific context and logs
        the activation.

        The memory core is the one that does real work here:
        it queries the cross-session memory field.
        """
        events = []
        t = time.time()

        for signal in signals:
            if signal.core == "memory" and self.memory_field is not None:
                # Memory core: real cross-session query
                all_texts, _, _ = self.scorer._extract_turns(lattice_data)
                if all_texts:
                    embedder = _get_embedder()
                    embs = embedder.encode(all_texts, convert_to_numpy=True)
                    centroid = embs.mean(axis=0)

                    # Query for resonant prior sessions
                    resonance = self.memory_field.query(
                        centroid, k=3, min_similarity=0.5,
                        exclude_session=score.session_id,
                    )

                    density = self.memory_field.attractor_density(
                        centroid, radius=self.attractor_radius,
                    )
                    reg_weight = self.memory_field.mean_weight_in_region(
                        centroid, radius=self.attractor_radius,
                    )

                    # Update signal with memory context
                    signal.intensity = min(1.0, density / 5.0)
                    signal.result = signal.intensity
                    signal.metadata["resonance_count"] = len(resonance)
                    signal.metadata["attractor_density"] = density
                    signal.metadata["regional_weight"] = round(reg_weight, 4)
                    signal.metadata["resonant_sessions"] = [
                        {
                            "session_id": r.session_id,
                            "similarity": round(r.similarity, 4),
                            "training_weight": round(r.training_weight, 4),
                        }
                        for r in resonance
                    ]

            events.append(TimelineEvent(
                timestamp=t,
                event_type="core_activated",
                core=signal.core,
                data={
                    "name": CORE_REGISTRY[signal.core]["name"],
                    "intensity": round(signal.intensity, 4),
                    "result": round(signal.result, 4),
                },
            ))

        return events

    # -- Full pipeline -----------------------------------------------------

    def process(self, lattice_data: dict) -> CognitiveProcessingResult:
        """
        Route a session lattice through Brescia's three-phase flow.

        Phase I:   Score the lattice + generate NeuralSignals
        Phase II:  Activate cognitive cores (parallel resonance)
        Phase III: Package as SuperpositionLayer for collapse

        Returns a CognitiveProcessingResult that can be passed to
        CollapseFunction.collapse() for Phase III resolution.
        """
        timeline = []
        t = time.time()

        # Log raw stimulus
        session_id = lattice_data.get("session_id", "unknown")
        all_texts, _, _ = self.scorer._extract_turns(lattice_data)
        timeline.append(TimelineEvent(
            timestamp=t,
            event_type="raw_stimulus",
            core="system",
            data={
                "session_id": session_id,
                "turn_count": len(all_texts),
                "preview": all_texts[0][:100] if all_texts else "",
            },
        ))

        # Phase I: Score and generate signals
        score = self.scorer.score_from_dict(lattice_data)
        signals, gen_events = self._generate_signals(lattice_data, score)
        timeline.extend(gen_events)

        timeline.append(TimelineEvent(
            timestamp=time.time(),
            event_type="phase_i_complete",
            core="system",
            data={"signal_count": len(signals)},
        ))

        # Phase II: Activate cores
        act_events = self._activate_cores(signals, lattice_data, score)
        timeline.extend(act_events)

        timeline.append(TimelineEvent(
            timestamp=time.time(),
            event_type="phase_ii_complete",
            core="system",
            data={"cores_activated": len(signals)},
        ))

        # Phase III preparation: package as superposition
        # (actual collapse is done by CollapseFunction)

        # Memory resonance context
        memory_resonance = []
        attractor_density = 0
        regional_weight = 0.0

        memory_signal = next((s for s in signals if s.core == "memory"), None)
        if memory_signal:
            memory_resonance = memory_signal.metadata.get("resonant_sessions", [])
            attractor_density = memory_signal.metadata.get("attractor_density", 0)
            regional_weight = memory_signal.metadata.get("regional_weight", 0.0)

        timeline.append(TimelineEvent(
            timestamp=time.time(),
            event_type="superposition_ready",
            core="system",
            data={
                "training_weight": round(score.training_weight, 4),
                "dominant_cores": self._dominant_cores(signals),
                "memory_resonance_count": len(memory_resonance),
            },
        ))

        return CognitiveProcessingResult(
            session_id=session_id,
            signals=signals,
            score=score,
            timeline=timeline,
            memory_resonance=memory_resonance,
            attractor_density=attractor_density,
            regional_weight=regional_weight,
        )

    def _dominant_cores(self, signals: list[CognitiveSignal], top_n: int = 3) -> list[str]:
        """Return the top N cores by intensity."""
        sorted_signals = sorted(signals, key=lambda s: s.intensity, reverse=True)
        return [s.core for s in sorted_signals[:top_n]]

    # -- Diagnostic output -------------------------------------------------

    def format_timeline(self, result: CognitiveProcessingResult) -> str:
        """
        Format the cognitive timeline as a human-readable report.

        This is Brescia's SymbolicCognitionTimelineLogger output,
        adapted for Maestro's measurement vocabulary.
        """
        lines = []
        lines.append(f"=== Cognitive Processing Timeline: {result.session_id} ===")
        lines.append("")

        # Group events by phase
        phases = {
            "Phase I -- Signal Generation": [],
            "Phase II -- Core Activation": [],
            "Phase III -- Superposition": [],
        }

        for event in result.timeline:
            if event.event_type in ("raw_stimulus", "signal_generated", "phase_i_complete"):
                phases["Phase I -- Signal Generation"].append(event)
            elif event.event_type in ("core_activated", "phase_ii_complete"):
                phases["Phase II -- Core Activation"].append(event)
            else:
                phases["Phase III -- Superposition"].append(event)

        for phase_name, events in phases.items():
            lines.append(f"--- {phase_name} ---")
            for event in events:
                core_label = f"[{event.core}]" if event.core != "system" else "[SYSTEM]"
                lines.append(f"  {core_label} {event.event_type}")
                for k, v in event.data.items():
                    if isinstance(v, float):
                        lines.append(f"    {k}: {v:.4f}")
                    elif isinstance(v, list) and len(v) > 3:
                        lines.append(f"    {k}: [{len(v)} items]")
                    else:
                        lines.append(f"    {k}: {v}")
            lines.append("")

        # Core intensity summary (visual)
        lines.append("--- Core Intensities ---")
        for signal in sorted(result.signals, key=lambda s: s.intensity, reverse=True):
            bar_len = int(signal.intensity * 30)
            bar = "#" * bar_len + "." * (30 - bar_len)
            name = CORE_REGISTRY[signal.core]["name"]
            lines.append(f"  {name:20s} [{bar}] {signal.intensity:.3f}")

        lines.append("")
        lines.append(f"Training weight (pre-collapse): {result.score.training_weight:.4f}")
        lines.append(f"Dominant cores: {', '.join(self._dominant_cores(result.signals))}")

        if result.memory_resonance:
            lines.append(f"Memory resonance: {len(result.memory_resonance)} prior sessions")
            for r in result.memory_resonance:
                lines.append(f"  - {r['session_id']} (sim={r['similarity']:.3f}, w={r['training_weight']:.3f})")

        return "\n".join(lines)


# -----------------------------------------------------------------------
# Orch-OS <-> Maestro mapping reference (for documentation)
# -----------------------------------------------------------------------

ORCHOS_ALIGNMENT = {
    "architecture_mapping": {
        "Orch-OS Component": {
            "generateNeuralSignal()": "CognitiveRouter._generate_signals()",
            "NeuralSignal": "CognitiveSignal dataclass",
            "CognitiveCore.process()": "CognitiveRouter._activate_cores()",
            "NeuralProcessingResult": "CognitiveSignal.result + .metadata",
            "DefaultNeuralIntegrationService": "CognitiveRouter.process()",
            "SuperpositionLayer": "CognitiveProcessingResult (all signals coexist)",
            "AICollapseStrategyService": "CollapseFunction.collapse()",
            "MemoryService.store()": "MemoryField.store_from_score()",
            "MemoryService.query()": "MemoryField.query()",
            "MemoryContextBuilder": "CognitiveRouter._activate_cores() memory branch",
            "SymbolicCognitionTimelineLogger": "CognitiveRouter.format_timeline()",
            "CollapseStrategyService scoring": "SignalScorer.score_from_dict()",
        },
    },
    "collapse_equation_mapping": {
        "brescia": "mu(s_i) = alpha*epsilon_i + beta*tau_i + gamma*chi_i",
        "maestro": "weight = integration * oracle_eff * (1 + contradiction)",
        "correspondence": {
            "epsilon (emotional pressure)": "oracle_effective (includes affective_granularity)",
            "tau (narrative tension)": "integration_depth (meaning-dimension diversity)",
            "chi (contradiction score)": "contradiction_score (Shadow Core signal)",
            "alpha, beta, gamma weights": "Multiplicative rather than additive -- multiplication enforces joint necessity",
        },
        "design_rationale": (
            "Brescia uses additive combination (alpha*E + beta*T + gamma*C) "
            "which allows a single strong signal to dominate. Maestro uses "
            "multiplicative combination (I * O * (1+C)) which enforces that "
            "ALL channels must be present: zero integration OR zero oracle "
            "kills the weight entirely. This prevents high-contradiction "
            "sessions with no actual engagement from scoring well."
        ),
    },
    "subsystem_mapping": {
        "Generative Language Model -> Neocortex": "llama_cpp_adapter.py (Qwen3 inference)",
        "Semantic Vector Memory -> Hippocampus": "memory_field.py (numpy/pickle vector store)",
        "Transcription Sensorium -> Auditory cortex": "Interview.tsx + gateway chat endpoint",
        "Interactive Cognitive Interface -> Prefrontal cortex": "PRISM early_warning.py + this timeline",
    },
    "phase_mapping": {
        "Phase I (Signal Generation)": (
            "Lattice ingestion + SignalScorer measurement. "
            "Each transcript node becomes a symbolic stimulus; "
            "each felt_state node carries affective metadata."
        ),
        "Phase II (Core Activation)": (
            "Parallel scoring across all observables. "
            "Memory core queries cross-session field. "
            "Shadow core measures participant-model divergence. "
            "Valence core measures affective precision. "
            "All results coexist in SessionSignalScore."
        ),
        "Phase III (Symbolic Collapse)": (
            "CollapseFunction.collapse() resolves the superposition: "
            "training_weight * (1 + novelty_bonus) * (1 - drift_penalty). "
            "Consent gate applied. Output: TrainingSignal for SGT pipeline."
        ),
    },
    "drift_detection_mapping": {
        "brescia_trial_pattern": "Shadow-wins-4-of-5: contradiction core dominated most collapses",
        "maestro_equivalent": (
            "early_warning.py LatticeViabilityMonitor: "
            "oracle_decline, integration_narrowing, solipsistic_drift signatures. "
            "Drift penalty in collapse.py zeros weight for dense regions "
            "with declining oracle capacity."
        ),
    },
}


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    import os
    from pathlib import Path

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
    router = CognitiveRouter(scorer, mf)

    files = sorted(Path(lattice_dir).glob("*.json"))

    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        result = router.process(data)
        print(router.format_timeline(result))
        print()
        print("=" * 60)
        print()
