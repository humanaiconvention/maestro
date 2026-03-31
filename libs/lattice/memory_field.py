"""
memory_field.py -- Cross-session embedding memory for the Maestro lattice.

Implements a persistent vector memory field that stores session-level and
turn-level embeddings, enabling cross-session resonance retrieval.  This is
the structural equivalent of Brescia's Pinecone vector memory, but local-
first and backed by numpy + pickle (swappable to FAISS/ChromaDB for scale).

The memory field serves the Collapse Function by answering:
  "Which prior sessions resonate with this one?"
  "Has this attractor region been visited before?"
  "What was the training outcome last time this region was active?"

Theoretical grounding:
  - Hopfield networks: memory as energy minima in a landscape
  - Friston FEP: the generative model IS the shaped landscape
  - Brescia Orch-OS: Pinecone memory enables temporal binding across sessions
  - Consilience E <= C: memory field tracks where C has been invested,
    so new sessions can inherit prior oracle corrections

Usage:
    from maestro.libs.lattice.memory_field import MemoryField
    field = MemoryField("path/to/field.pkl")
    field.store(session_id, centroid, score, metadata)
    neighbours = field.query(centroid, k=5)
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MemoryRecord:
    """One record in the memory field."""

    session_id:    str
    centroid:      np.ndarray           # Session embedding centroid (384-dim)
    training_weight: float              # From SignalScorer
    oracle_capacity: float              # Oracle capacity at time of storage
    contradiction_score: float          # Shadow signal at time of storage
    integration_depth: float            # Integration depth at time of storage
    timestamp:     str                  # ISO 8601
    metadata:      dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid"] = self.centroid.tolist()
        return d


@dataclass
class ResonanceResult:
    """A neighbour returned by a memory field query."""

    session_id:      str
    similarity:      float              # Cosine similarity to query
    training_weight: float
    oracle_capacity: float
    contradiction_score: float
    integration_depth: float
    timestamp:       str
    metadata:        dict


class MemoryField:
    """
    Persistent vector memory field backed by numpy arrays.

    Stores session centroids and metadata, supports cosine-similarity
    nearest-neighbour queries.  Persists to a pickle file.

    For corpora up to ~10K sessions this is efficient.  Beyond that,
    swap the _query implementation for FAISS IndexFlatIP.
    """

    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._records: list[MemoryRecord] = []
        self._centroids: Optional[np.ndarray] = None   # (N, D) matrix
        self._dirty = False

        if self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        with open(self._path, "rb") as f:
            data = pickle.load(f)
        self._records = data["records"]
        self._rebuild_index()

    def save(self) -> None:
        """Persist the memory field to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "wb") as f:
            pickle.dump({"records": self._records}, f)
        self._dirty = False

    def _rebuild_index(self) -> None:
        """Rebuild the (N, D) centroid matrix from records."""
        if not self._records:
            self._centroids = None
            return
        self._centroids = np.stack(
            [r.centroid for r in self._records], axis=0
        )
        # L2-normalise for cosine similarity via dot product
        norms = np.linalg.norm(self._centroids, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        self._centroids = self._centroids / norms

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        session_id: str,
        centroid: np.ndarray,
        training_weight: float = 0.0,
        oracle_capacity: float = 0.0,
        contradiction_score: float = 0.0,
        integration_depth: float = 0.0,
        timestamp: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Add a session embedding to the memory field.

        Deduplicates by session_id: if the session already exists, its
        record is updated in place.
        """
        # Remove existing record for this session (update semantics)
        self._records = [
            r for r in self._records if r.session_id != session_id
        ]

        record = MemoryRecord(
            session_id=session_id,
            centroid=centroid.copy(),
            training_weight=training_weight,
            oracle_capacity=oracle_capacity,
            contradiction_score=contradiction_score,
            integration_depth=integration_depth,
            timestamp=timestamp,
            metadata=metadata or {},
        )
        self._records.append(record)
        self._rebuild_index()
        self._dirty = True

    def store_from_score(
        self,
        score,
        centroid: np.ndarray,
        timestamp: str = "",
        metadata: Optional[dict] = None,
    ) -> None:
        """
        Convenience: store from a SessionSignalScore dataclass.
        """
        self.store(
            session_id=score.session_id,
            centroid=centroid,
            training_weight=score.training_weight,
            oracle_capacity=score.oracle_capacity,
            contradiction_score=score.contradiction_score,
            integration_depth=score.integration_depth,
            timestamp=timestamp,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        centroid: np.ndarray,
        k: int = 5,
        min_similarity: float = 0.0,
        exclude_session: Optional[str] = None,
    ) -> list[ResonanceResult]:
        """
        Find the k most similar sessions to the given centroid.

        Args:
            centroid: Query embedding (D-dim).
            k: Maximum neighbours to return.
            min_similarity: Minimum cosine similarity threshold.
            exclude_session: Session ID to exclude (e.g., the query session).

        Returns:
            List of ResonanceResult sorted by descending similarity.
        """
        if self._centroids is None or len(self._records) == 0:
            return []

        # Normalise query
        q = centroid.copy()
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-12:
            return []
        q = q / q_norm

        # Cosine similarity via dot product (centroids already normalised)
        similarities = self._centroids @ q

        # Build (similarity, index) pairs, excluding self
        candidates = []
        for i, sim in enumerate(similarities):
            rec = self._records[i]
            if exclude_session and rec.session_id == exclude_session:
                continue
            if sim >= min_similarity:
                candidates.append((float(sim), i))

        # Sort descending by similarity
        candidates.sort(key=lambda x: -x[0])
        candidates = candidates[:k]

        results = []
        for sim, idx in candidates:
            rec = self._records[idx]
            results.append(ResonanceResult(
                session_id=rec.session_id,
                similarity=round(sim, 4),
                training_weight=rec.training_weight,
                oracle_capacity=rec.oracle_capacity,
                contradiction_score=rec.contradiction_score,
                integration_depth=rec.integration_depth,
                timestamp=rec.timestamp,
                metadata=rec.metadata,
            ))

        return results

    # ------------------------------------------------------------------
    # Landscape queries
    # ------------------------------------------------------------------

    def attractor_density(self, centroid: np.ndarray, radius: float = 0.3) -> int:
        """
        Count how many prior sessions fall within `radius` cosine distance
        of the given centroid.  High density = well-explored attractor region.
        """
        results = self.query(
            centroid, k=len(self._records) if self._records else 1,
            min_similarity=(1.0 - radius),
        )
        return len(results)

    def mean_weight_in_region(
        self, centroid: np.ndarray, radius: float = 0.3
    ) -> float:
        """
        Mean training weight of sessions within `radius` cosine distance.
        Low mean weight in a dense region = potential solipsistic drift zone.
        """
        results = self.query(
            centroid, k=len(self._records) if self._records else 1,
            min_similarity=(1.0 - radius),
        )
        if not results:
            return 0.0
        return float(np.mean([r.training_weight for r in results]))

    def corpus_centroid(self) -> Optional[np.ndarray]:
        """Return the mean centroid of all stored sessions, or None."""
        if self._centroids is None:
            return None
        return self._centroids.mean(axis=0)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return len(self._records)

    def session_ids(self) -> list[str]:
        return [r.session_id for r in self._records]

    def __repr__(self) -> str:
        return f"MemoryField({self._path}, {self.size} records)"


# -----------------------------------------------------------------------
# CLI entry point -- populate the memory field from scored lattices
# -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json
    from signal_scorer import SignalScorer

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

    from signal_scorer import _get_embedder

    files = sorted(Path(lattice_dir).glob("*.json"))
    for f in files:
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        score = scorer.score_from_dict(data)

        # Compute session centroid from all turn embeddings
        all_texts, _, _ = scorer._extract_turns(data)
        if not all_texts:
            print(f"  SKIP {score.session_id} (no turns)")
            continue

        embedder = _get_embedder()
        all_embs = embedder.encode(all_texts, convert_to_numpy=True)
        centroid = all_embs.mean(axis=0)

        created = data.get("base_nodes", [{}])[-1].get("created_at", "")

        mf.store_from_score(score, centroid, timestamp=created)
        print(f"  Stored {score.session_id}  weight={score.training_weight:.4f}")

    mf.save()
    print(f"\nMemory field saved: {mf.size} records -> {field_path}")

    # Demo: query each session against the field
    print("\n=== Cross-session resonance ===")
    for f in sorted(Path(lattice_dir).glob("*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        sid = data.get("session_id", "?")
        all_texts, _, _ = scorer._extract_turns(data)
        if not all_texts:
            continue
        embs = embedder.encode(all_texts, convert_to_numpy=True)
        c = embs.mean(axis=0)
        neighbours = mf.query(c, k=3, exclude_session=sid)
        print(f"\n  {sid}:")
        for nb in neighbours:
            print(f"    -> {nb.session_id}  sim={nb.similarity:.3f}  "
                  f"weight={nb.training_weight:.4f}")
