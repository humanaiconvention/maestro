"""
signal_scorer.py — Landscape-observable measurement for session lattices.

Computes five observables that characterise a session's contribution to the
shaped space in which correct answers emerge as stable attractors.

The five observables derive from convergent research traditions:

  1. integration_depth   — Intrinsic dimensionality (Two-NN estimator) of turn
                           embeddings.  Proxy for Tononi's phi: how many
                           independent meaning-dimensions the session traverses.
                           (Facco et al. 2017; Ansuini et al. 2019)

  2. semantic_entropy    — Meaning-cluster entropy of model-generated turns.
                           High values indicate the model is far from a stable
                           attractor in that region of the space.
                           (Farquhar et al. 2024)

  3. temporal_coherence  — Mutual information between the first and second
                           halves of the session embedding trajectory.
                           Measures whether the session resolves to an attractor
                           or remains in open orbit.
                           (Husserl 1905/Varela et al. 2001 — temporal binding)

  4. oracle_capacity     — Lempel-Ziv complexity of participant language,
                           conditioned on model turns.  Measures whether the
                           human is in a flourishing / active-inference state
                           (high LZC) or a habit / compliance state (low LZC).
                           (Casali et al. 2013; Schartner et al. 2017;
                            Bréscia E ≤ C viability condition)

  5. novelty             — Cosine distance of the session centroid from the
                           running lattice centroid.  Measures whether this
                           session adds new attractors to the space or
                           reinforces existing ones.

Orch-OS Alignment (Brescia 2025):
  This scorer implements Phase I of Brescia's three-phase cognitive flow
  (NeuralSignal extraction).  Each observable maps to a named cognitive core:
    integration_depth   -> Integration Core (Tononi phi proxy)
    semantic_entropy    -> Entropy Core (model uncertainty)
    temporal_coherence  -> Coherence Core (narrative resolution)
    oracle_capacity     -> Oracle Core (participant flourishing)
    novelty             -> Novelty Core (attractor expansion)
    contradiction_score -> Shadow Core (divergence as fuel)
    affective_granularity -> Valence Core (Barrett precision)
  See cognitive_cores.py for the full routing vocabulary.

Usage:
    from maestro.libs.lattice.signal_scorer import SignalScorer, SessionSignalScore
    scorer = SignalScorer()                           # loads embedder once
    score  = scorer.score(lattice)                    # on a single SessionLattice
    scores = scorer.score_corpus(lattice_dir)         # on a directory of lattice JSON
"""

from __future__ import annotations

import math
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

# Lazy-loaded on first use to avoid import-time GPU initialisation.
_EMBEDDER = None
_EMBEDDER_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, good for cosine sim


def _get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(_EMBEDDER_NAME)
    return _EMBEDDER


# ──────────────────────────────────────────────────────────────────────────────
# Output schema
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SessionSignalScore:
    """Five landscape observables + affective precision for one session lattice."""

    session_id:          str
    integration_depth:   float   # Two-NN intrinsic dim estimate
    semantic_entropy:    float   # Meaning-cluster entropy of model turns
    temporal_coherence:  float   # I(first_half; second_half) proxy
    oracle_capacity:     float   # Normalised LZ complexity of participant text
    novelty:             float   # Cosine distance from corpus centroid

    # Derived composite — precision-weighted training value
    training_weight:     float   # integration * oracle_effective * (1 + contradiction)

    # Raw contradiction signal (Brescia's Shadow Core equivalent)
    contradiction_score: float   # Mean cosine distance between participant
                                 # turn embeddings and model turn embeddings
                                 # for adjacent pairs

    # Affective precision (Adolphs/Barrett)
    # Emotion as functional state: "adjusts goals, directs attention,
    # and modifies the weights you assign to various factors."
    # High granularity = high-precision oracle whose corrections carry
    # more weight (Friston precision-weighting via felt_state channel).
    affective_granularity: float  # Diversity + specificity of felt_state labels
    affect_arc_coherence:  float  # Temporal coherence of affect trajectory
    oracle_effective:      float  # oracle_capacity * (1 + affective_granularity)

    turn_count:          int
    participant_turns:   int
    model_turns:         int

    def to_dict(self) -> dict:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# Observable 1:  Integration depth  (Two-NN intrinsic dimension)
# ──────────────────────────────────────────────────────────────────────────────

def _two_nn_id(embeddings: np.ndarray) -> float:
    """
    Two-NN intrinsic dimension estimator (Facco et al. 2017).

    For each point, compute the ratio mu = d2/d1 of the distances to its
    second and first nearest neighbours.  The maximum-likelihood estimate
    of intrinsic dimension is:

        ID = n / sum(log(mu_i))

    Requires at least 6 points for a stable estimate.  Returns 0.0 for
    degenerate inputs, and caps output at the ambient dimension as a
    sanity bound.
    """
    n = len(embeddings)
    if n < 6:
        return 0.0

    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)

    # distances[:, 0] is self (≈0), [:, 1] is d₁, [:, 2] is d₂
    d1 = distances[:, 1]
    d2 = distances[:, 2]

    # Guard against zero distances (identical embeddings)
    valid = d1 > 1e-12
    if valid.sum() < 2:
        return 0.0

    mu = d2[valid] / d1[valid]
    log_mu = np.log(mu)
    log_mu = log_mu[log_mu > 1e-12]  # drop degenerate ratios

    if len(log_mu) == 0:
        return 0.0

    id_est = float(len(log_mu) / np.sum(log_mu))
    # Cap at ambient dimension (384 for all-MiniLM-L6-v2)
    return min(id_est, float(embeddings.shape[1]))


# ──────────────────────────────────────────────────────────────────────────────
# Observable 2:  Semantic entropy
# ──────────────────────────────────────────────────────────────────────────────

def _semantic_entropy(embeddings: np.ndarray, threshold: float = 0.30) -> float:
    """
    Approximate semantic entropy via meaning-cluster counting.

    Clusters model-turn embeddings by cosine similarity.  Two turns are in the
    same meaning cluster if their cosine distance is below `threshold`.
    Entropy is computed over the cluster size distribution.

    Returns 0.0 if there are fewer than 2 model turns.
    """
    n = len(embeddings)
    if n < 2:
        return 0.0

    # Simple greedy agglomerative clustering
    from sklearn.metrics.pairwise import cosine_distances
    dists = cosine_distances(embeddings)
    assigned = [-1] * n
    cluster_id = 0

    for i in range(n):
        if assigned[i] >= 0:
            continue
        assigned[i] = cluster_id
        for j in range(i + 1, n):
            if assigned[j] < 0 and dists[i, j] < threshold:
                assigned[j] = cluster_id
        cluster_id += 1

    # Entropy over cluster distribution
    counts = np.bincount(assigned)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


# ──────────────────────────────────────────────────────────────────────────────
# Observable 3:  Temporal coherence  (MI proxy)
# ──────────────────────────────────────────────────────────────────────────────

def _temporal_coherence(embeddings: np.ndarray) -> float:
    """
    Proxy for mutual information between the first and second halves of the
    session embedding trajectory.

    Computed as the cosine similarity between the mean embedding of the first
    half and the mean embedding of the second half.  High similarity means
    the session resolves coherently; low means it fragments.

    Returns 0.0 for sessions with fewer than 4 turns.
    """
    n = len(embeddings)
    if n < 4:
        return 0.0

    mid = n // 2
    first_centroid = embeddings[:mid].mean(axis=0)
    second_centroid = embeddings[mid:].mean(axis=0)

    dot = np.dot(first_centroid, second_centroid)
    norm = np.linalg.norm(first_centroid) * np.linalg.norm(second_centroid)
    if norm < 1e-12:
        return 0.0

    return float(dot / norm)


# ──────────────────────────────────────────────────────────────────────────────
# Observable 4:  Oracle capacity  (normalised Lempel-Ziv complexity)
# ──────────────────────────────────────────────────────────────────────────────

def _lz_complexity(text: str) -> int:
    """
    Lempel-Ziv 76 complexity: count of distinct substrings encountered
    during a left-to-right scan.  Higher = less compressible = higher
    behavioural entropy.
    """
    if not text:
        return 0

    n = len(text)
    i = 0
    c = 1  # complexity counter
    l = 1  # current prefix length

    while i + l <= n:
        # Check if the substring text[i:i+l] has appeared in text[0:i+l-1]
        if text[i:i + l] in text[0:i + l - 1]:
            l += 1
        else:
            c += 1
            i += l
            l = 1

    return c


def _normalised_lz(text: str, min_chars: int = 40) -> float:
    """
    LZ complexity normalised by the theoretical maximum for the text length.
    Maximum LZ complexity for length n ~ n / log2(n).
    Returns a value in [0, 1] where 1 = maximally complex (incompressible).

    Texts shorter than `min_chars` are penalised proportionally because LZ
    complexity is unreliable on very short strings (it saturates at 1.0
    regardless of actual complexity).
    """
    if len(text) < 2:
        return 0.0

    raw = _lz_complexity(text)
    n = len(text)
    max_c = n / max(math.log2(n), 1.0)
    score = float(min(raw / max_c, 1.0))

    # Penalise short texts — we can't reliably measure complexity
    # in a handful of words.  Linear ramp from 0 at len=2 to full
    # weight at len=min_chars.
    if n < min_chars:
        score *= n / min_chars

    return score


def _oracle_capacity(participant_texts: list[str]) -> float:
    """
    Mean normalised LZ complexity across all participant turns.
    """
    if not participant_texts:
        return 0.0

    scores = [_normalised_lz(t) for t in participant_texts]
    return float(np.mean(scores))


# ──────────────────────────────────────────────────────────────────────────────
# Observable 5:  Novelty (cosine distance from corpus centroid)
# ──────────────────────────────────────────────────────────────────────────────

def _novelty(session_centroid: np.ndarray,
             corpus_centroid: Optional[np.ndarray]) -> float:
    """
    Cosine distance between this session's mean embedding and the running
    corpus centroid.  Returns 0.0 if no corpus centroid is available (first
    session).
    """
    if corpus_centroid is None:
        return 0.0

    dot = np.dot(session_centroid, corpus_centroid)
    norm = np.linalg.norm(session_centroid) * np.linalg.norm(corpus_centroid)
    if norm < 1e-12:
        return 0.0

    cos_sim = dot / norm
    return float(1.0 - cos_sim)  # distance, not similarity


# ──────────────────────────────────────────────────────────────────────────────
# Contradiction score  (Bréscia Shadow Core)
# ──────────────────────────────────────────────────────────────────────────────

def _contradiction_score(participant_embs: np.ndarray,
                         model_embs: np.ndarray,
                         random_baseline: float = 0.55) -> float:
    """
    Mean cosine distance between each participant turn embedding and the
    preceding model turn embedding, corrected for the high-dimensional
    baseline.

    In 384-dim space, random unit vectors have expected cosine similarity
    ~0, so the expected cosine distance is ~1.0.  Sentence embeddings are
    NOT random — they cluster on a manifold — so empirical baseline for
    unrelated sentence pairs is ~0.55.  We subtract this baseline so that
    contradiction_score = 0 means "typical unrelatedness" and positive
    values mean "genuine divergence from the model's offered framing."

    This is the Shadow Core signal: contradiction as fuel, not noise.
    """
    n = min(len(participant_embs), len(model_embs))
    if n == 0:
        return 0.0

    distances = []
    for i in range(n):
        dot = np.dot(participant_embs[i], model_embs[i])
        norm = (np.linalg.norm(participant_embs[i])
                * np.linalg.norm(model_embs[i]))
        if norm > 1e-12:
            distances.append(1.0 - dot / norm)

    if not distances:
        return 0.0

    raw = float(np.mean(distances))
    # Subtract baseline and clamp to [0, 1]
    corrected = max(0.0, (raw - random_baseline) / (1.0 - random_baseline))
    return corrected


# ──────────────────────────────────────────────────────────────────────────────
# Observable 6:  Affective precision  (Adolphs / Barrett)
# ──────────────────────────────────────────────────────────────────────────────

def _affective_granularity(felt_states: list[dict]) -> float:
    """
    Measure the emotional granularity of a participant's felt_state labels.

    Emotional granularity (Barrett 2004, 2006) is the ability to make
    fine-grained distinctions between emotional states.  High granularity
    predicts better emotion regulation, decision-making, and -- for our
    purposes -- higher-quality corrective signals.

    Per Adolphs: emotion is a "functional state that adjusts goals, directs
    attention, and modifies the weights you assign."  A participant who can
    label these functional states with specificity is providing direct
    observational access to their precision-weighting mechanism.

    Measured as:
      granularity = (unique_labels / total_labels) * entropy(label_distribution)

    The ratio captures vocabulary breadth; the entropy captures how evenly
    that vocabulary is used (not just one label repeated).

    Returns 0.0 if no felt_state data is available.
    """
    if not felt_states:
        return 0.0

    # Extract emotion labels from felt_state dicts
    labels = []
    for fs in felt_states:
        if not fs:
            continue
        # felt_state nodes may carry various fields; look for the
        # most common patterns: "label", "emotion", "state", or
        # any string value that looks like an affect label
        for key in ("label", "emotion", "state", "felt", "affect"):
            if key in fs and isinstance(fs[key], str) and fs[key]:
                labels.append(fs[key].lower().strip())
                break
        else:
            # Fall back to any string value in the dict
            for v in fs.values():
                if isinstance(v, str) and v and v not in ("user", "assistant"):
                    labels.append(v.lower().strip())
                    break

    if not labels:
        return 0.0

    n = len(labels)
    unique = len(set(labels))

    # Breadth: what fraction of labels are unique
    breadth = unique / n

    # Entropy over label distribution
    from collections import Counter
    counts = Counter(labels)
    probs = np.array(list(counts.values()), dtype=float) / n
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))

    # Normalise entropy by log2(n) to get [0, 1]
    max_entropy = math.log2(n) if n > 1 else 1.0
    norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Granularity = breadth * normalised entropy
    return float(breadth * norm_entropy)


def _affect_arc_coherence(felt_states: list[dict]) -> float:
    """
    Measure the temporal coherence of the affect trajectory.

    A coherent affect arc (even a negative one: calm -> tension -> resolution)
    indicates deep engagement.  An incoherent arc (random jumps between
    unrelated states) indicates confusion or disengagement.

    Measured via embedding similarity of consecutive felt_state labels.
    High coherence = smooth trajectory through affect space.

    Falls back to 0.0 if insufficient data or no embedder available.
    """
    if len(felt_states) < 3:
        return 0.0

    # Extract label strings
    labels = []
    for fs in felt_states:
        if not fs:
            labels.append("")
            continue
        for key in ("label", "emotion", "state", "felt", "affect"):
            if key in fs and isinstance(fs[key], str) and fs[key]:
                labels.append(fs[key])
                break
        else:
            for v in fs.values():
                if isinstance(v, str) and v and v not in ("user", "assistant"):
                    labels.append(v)
                    break
            else:
                labels.append("")

    # Filter to non-empty
    valid = [l for l in labels if l]
    if len(valid) < 3:
        return 0.0

    try:
        embedder = _get_embedder()
        embs = embedder.encode(valid, convert_to_numpy=True)

        # Consecutive cosine similarities
        sims = []
        for i in range(len(embs) - 1):
            dot = np.dot(embs[i], embs[i + 1])
            norm = np.linalg.norm(embs[i]) * np.linalg.norm(embs[i + 1])
            if norm > 1e-12:
                sims.append(dot / norm)

        if not sims:
            return 0.0

        # Mean consecutive similarity = arc coherence
        return float(np.mean(sims))

    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Main scorer
# ──────────────────────────────────────────────────────────────────────────────

class SignalScorer:
    """
    Computes SessionSignalScore for lattice sessions.

    Maintains a running corpus centroid for novelty computation.
    """

    def __init__(self, embedder_name: str = _EMBEDDER_NAME):
        global _EMBEDDER_NAME
        _EMBEDDER_NAME = embedder_name
        self._corpus_embeddings: list[np.ndarray] = []
        self._corpus_centroid: Optional[np.ndarray] = None

    def _update_centroid(self, session_centroid: np.ndarray) -> None:
        """Update the running corpus centroid with a new session."""
        self._corpus_embeddings.append(session_centroid)
        self._corpus_centroid = np.mean(self._corpus_embeddings, axis=0)

    def _extract_turns(self, lattice_data: dict) -> tuple[
        list[str], list[str], list[str]
    ]:
        """
        Extract turn texts from a lattice, split by role.

        Returns:
            (all_texts, participant_texts, model_texts)
        """
        transcripts = []
        for node in lattice_data.get("base_nodes", []):
            if node["node_kind"] == "transcript":
                transcripts.append(node)

        # Sort by turn_index
        transcripts.sort(key=lambda n: n["content"].get("turn_index", 0))

        all_texts = []
        participant_texts = []
        model_texts = []

        for t in transcripts:
            text = t["content"].get("content", "")
            role = t["content"].get("role", "")
            all_texts.append(text)

            if role == "user":
                participant_texts.append(text)
            elif role == "assistant":
                model_texts.append(text)

        return all_texts, participant_texts, model_texts

    def _extract_felt_states(self, lattice_data: dict) -> list[dict]:
        """
        Extract felt_state node contents from the lattice.

        Returns a list of felt_state content dicts, ordered by turn_index.
        Empty list if no felt_state nodes exist or consent denies them.
        """
        felt_nodes = []
        for node in lattice_data.get("base_nodes", []):
            if node["node_kind"] == "felt_state":
                felt_nodes.append(node)

        # Sort by turn_index if present
        felt_nodes.sort(
            key=lambda n: n["content"].get("turn_index", 0)
        )

        return [n["content"] for n in felt_nodes]

    def score_from_dict(self, lattice_data: dict) -> SessionSignalScore:
        """
        Score a session lattice from its dict representation.
        """
        session_id = lattice_data.get("session_id", "unknown")
        all_texts, participant_texts, model_texts = self._extract_turns(
            lattice_data
        )
        felt_states = self._extract_felt_states(lattice_data)

        if not all_texts:
            return SessionSignalScore(
                session_id=session_id,
                integration_depth=0.0,
                semantic_entropy=0.0,
                temporal_coherence=0.0,
                oracle_capacity=0.0,
                novelty=0.0,
                training_weight=0.0,
                contradiction_score=0.0,
                affective_granularity=0.0,
                affect_arc_coherence=0.0,
                oracle_effective=0.0,
                turn_count=0,
                participant_turns=0,
                model_turns=0,
            )

        embedder = _get_embedder()

        # Embed all turns
        all_embs = embedder.encode(all_texts, convert_to_numpy=True)

        # Embed by role
        participant_embs = (
            embedder.encode(participant_texts, convert_to_numpy=True)
            if participant_texts else np.zeros((0, all_embs.shape[1]))
        )
        model_embs = (
            embedder.encode(model_texts, convert_to_numpy=True)
            if model_texts else np.zeros((0, all_embs.shape[1]))
        )

        # 1. Integration depth
        integration = _two_nn_id(all_embs)

        # 2. Semantic entropy (over model turns only)
        sem_entropy = _semantic_entropy(model_embs)

        # 3. Temporal coherence
        coherence = _temporal_coherence(all_embs)

        # 4. Oracle capacity (behavioural channel — LZ complexity)
        oracle_cap = _oracle_capacity(participant_texts)

        # 5. Novelty
        session_centroid = all_embs.mean(axis=0)
        nov = _novelty(session_centroid, self._corpus_centroid)
        self._update_centroid(session_centroid)

        # Contradiction (Shadow Core)
        contradiction = _contradiction_score(participant_embs, model_embs)

        # 6. Affective precision (Adolphs/Barrett)
        # "An emotion is a functional state that puts your brain in a
        # particular mode of operation that adjusts your goals, directs
        # your attention, and modifies the weights you assign to various
        # factors as you do mental calculations." — Ralph Adolphs
        #
        # felt_state is NOT a sixth independent observable — it is a
        # precision modifier on oracle_capacity.  A participant who can
        # label their emotional functional states with granularity is
        # demonstrating that their precision-weighting mechanism is
        # active and legible.  Their corrections carry more weight.
        aff_gran = _affective_granularity(felt_states)
        aff_arc = _affect_arc_coherence(felt_states)

        # Oracle effective = oracle_capacity * (1 + affective_granularity)
        # When felt_state is absent (granularity=0), oracle_effective
        # equals oracle_capacity — the behavioural channel alone.
        # When felt_state is rich, the affective channel amplifies it.
        oracle_eff = oracle_cap * (1.0 + aff_gran)

        # Composite training weight:
        #   integration × oracle_effective × (1 + contradiction)
        # This embodies the research convergence:
        #   - High integration = high phi = many orthogonal signals engaged
        #   - High oracle_effective = flourishing participant with legible
        #     precision-weighting = reliable gradient (Friston + Adolphs)
        #   - High contradiction = shadow signal = most informative for
        #     landscape shaping (prevents premature attractor convergence)
        training_weight = integration * oracle_eff * (1.0 + contradiction)

        return SessionSignalScore(
            session_id=session_id,
            integration_depth=round(integration, 4),
            semantic_entropy=round(sem_entropy, 4),
            temporal_coherence=round(coherence, 4),
            oracle_capacity=round(oracle_cap, 4),
            novelty=round(nov, 4),
            training_weight=round(training_weight, 4),
            contradiction_score=round(contradiction, 4),
            affective_granularity=round(aff_gran, 4),
            affect_arc_coherence=round(aff_arc, 4),
            oracle_effective=round(oracle_eff, 4),
            turn_count=len(all_texts),
            participant_turns=len(participant_texts),
            model_turns=len(model_texts),
        )

    def score(self, lattice) -> SessionSignalScore:
        """Score a SessionLattice object."""
        return self.score_from_dict(lattice.to_dict())

    def score_corpus(self, lattice_dir: str) -> list[SessionSignalScore]:
        """
        Score all lattice JSON files in a directory.

        Files are processed in filename order.  The corpus centroid is
        updated incrementally, so novelty is relative to all previously
        scored sessions.
        """
        results = []
        lattice_path = Path(lattice_dir)

        files = sorted(lattice_path.glob("*.json"))
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            score = self.score_from_dict(data)
            results.append(score)

        return results


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point — run against the lattice directory
# ──────────────────────────────────────────────────────────────────────────────

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
    print(f"Scoring lattices in: {lattice_dir}\n")

    scorer = SignalScorer()
    scores = scorer.score_corpus(lattice_dir)

    for s in scores:
        print(f"-- {s.session_id} ({s.turn_count} turns, "
              f"{s.participant_turns}p/{s.model_turns}m) --")
        print(f"  Integration depth:   {s.integration_depth:.4f}")
        print(f"  Semantic entropy:    {s.semantic_entropy:.4f}")
        print(f"  Temporal coherence:  {s.temporal_coherence:.4f}")
        print(f"  Oracle capacity:     {s.oracle_capacity:.4f}")
        print(f"  Affect granularity:  {s.affective_granularity:.4f}")
        print(f"  Affect arc coher.:   {s.affect_arc_coherence:.4f}")
        print(f"  Oracle effective:    {s.oracle_effective:.4f}")
        print(f"  Novelty:             {s.novelty:.4f}")
        print(f"  Contradiction:       {s.contradiction_score:.4f}")
        print(f"  Training weight:     {s.training_weight:.4f}")
        print()

    # Summary
    if scores:
        weights = [s.training_weight for s in scores]
        print(f"=== Corpus summary ({len(scores)} sessions) ===")
        print(f"  Mean training weight: {np.mean(weights):.4f}")
        print(f"  Std training weight:  {np.std(weights):.4f}")
        print(f"  Max:                  {max(weights):.4f}")
        print(f"  Min:                  {min(weights):.4f}")
