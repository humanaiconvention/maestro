"""
Prism API routes for the Maestro gateway.

Serves PRISM run data files from the data/records/ directory so the
frontend Prism module can display run history without requiring the
legacy_mvp service (port 8001) to be running.

Endpoints:
    GET  /v1/prism/runs          — list available run records
    GET  /v1/prism/runs/{run_id} — get a specific run record by ID
    GET  /v1/prism/status        — PRISM system status (tool versions, model track)
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ── Data directories ──────────────────────────────────────────────────────────

# Project root: maestro/apps/gateway/ → up 3 levels → project root
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Records directory (existing PRISM run JSON files)
_RECORDS_DIR = Path(os.environ.get(
    "PRISM_RECORDS_DIR",
    _PROJECT_ROOT / "data" / "records",
))

# Experiments directory for real training runs.
# When running from a git worktree (.git is a file, not a dir), large training
# artifacts live only in the main repo checkout — walk up out of the worktree.
def _resolve_experiment_dir() -> Path:
    if env_val := os.environ.get("EXPERIMENT_DIR"):
        return Path(env_val)
    primary = _PROJECT_ROOT / "experiments" / "qwen3_5_2b"
    # Detect worktree: .git is a regular file (contains "gitdir: ...")
    if (_PROJECT_ROOT / ".git").is_file():
        # Layout: main_repo/.claude/worktrees/<name>/
        # parents[0]=<name>, parents[1]=worktrees, parents[2]=.claude, parents[3]=main_repo
        candidate = _PROJECT_ROOT.parent.parent.parent / "experiments" / "qwen3_5_2b"
        if candidate.exists():
            return candidate
    return primary

_EXPERIMENT_DIR = _resolve_experiment_dir()

# Map version → experiment file paths
_TRAINING_VERSIONS: dict[str, dict] = {
    "v5": {
        "label": "v5 — rubric baseline",
        "trainer_state": _EXPERIMENT_DIR / "v5" / "adapter" / "checkpoint-384" / "trainer_state.json",
        "benchmark": _EXPERIMENT_DIR / "t3_benchmark" / "results.json",
        "steps": 384,
        "examples": 2154,
        "dataset": "grounding_mix_v5.jsonl",
    },
    "v6": {
        "label": "v6 — first grounding run",
        "trainer_state": _EXPERIMENT_DIR / "v6" / "adapter" / "checkpoint-396" / "trainer_state.json",
        "benchmark": _EXPERIMENT_DIR / "t3_benchmark_v6v7" / "results.json",
        "steps": 396,
        "examples": 2215,
        "dataset": "grounding_mix_v6.jsonl",
    },
    "v7": {
        "label": "v7 — expanded dataset",
        "trainer_state": _EXPERIMENT_DIR / "v7" / "adapter" / "checkpoint-408" / "trainer_state.json",
        "benchmark": None,
        "steps": 408,
        "examples": 2275,
        "dataset": "grounding_mix_v7.jsonl",
    },
    "v8": {
        "label": "v8 — active model",
        "trainer_state": _EXPERIMENT_DIR / "v8" / "adapter" / "checkpoint-414" / "trainer_state.json",
        "benchmark": _EXPERIMENT_DIR / "t3_benchmark_v8" / "results.json",
        "steps": 414,
        "examples": 2325,
        "dataset": "grounding_mix_v8.jsonl",
    },
}

def _load_json_file(path: Path) -> dict:
    """Load a JSON file, raising HTTPException on missing/corrupt."""
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse {path.name}: {exc}") from exc

# Active model track — overridable to avoid hardcoding the version string.
_MODEL_TRACK = os.environ.get("PRISM_MODEL_TRACK", "qwen3.5-2b-haic-v8")

# run_id must be alphanumeric with hyphens/underscores/dots, max 128 chars.
# Validated before any filesystem lookup to prevent path traversal.
_SAFE_RUN_ID_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9\-_\.]{0,127}$')

# Warn at import time if the directory doesn't exist so operators see it in
# the startup log rather than silently getting empty /runs responses.
if not _RECORDS_DIR.exists():
    logger.warning(
        f"prism: records directory not found: {_RECORDS_DIR} — "
        "all /v1/prism/runs requests will return []. "
        "Set PRISM_RECORDS_DIR env var to override."
    )


# ── Models ─────────────────────────────────────────────────────────────────────

class PrismRunSummary(BaseModel):
    """Lightweight summary returned in the list endpoint."""
    id: str
    source: str          # filename stem
    model_name: Optional[str] = None
    run_kind: Optional[str] = None
    status: Optional[str] = None
    updated_at: Optional[str] = None
    has_entropy_proof: bool = False
    has_layer_profiles: bool = False
    # TurboQuant-inspired outlier geometry aggregates (from EntropySnapshot)
    has_outlier_geometry: bool = False
    mean_outlier_ratio: Optional[float] = None
    mean_cardinal_proximity: Optional[float] = None
    mean_quantization_hostility: Optional[float] = None
    worst_layer_idx: Optional[int] = None


class PrismStatus(BaseModel):
    """PRISM system status snapshot."""
    records_dir: str
    record_count: int
    model_track: str
    tools: list[dict[str, str]]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_record(path: Path) -> dict[str, Any]:
    """Load a JSON record file, returning {} on parse error."""
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"prism: failed to load record {path.name}: {exc}")
        return {}


def _records() -> list[Path]:
    """Return all .json files in the records directory, sorted by mtime desc."""
    if not _RECORDS_DIR.exists():
        logger.warning(f"prism: records directory does not exist: {_RECORDS_DIR}")
        return []
    paths = sorted(
        _RECORDS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return paths


def _make_summary(path: Path, data: dict[str, Any]) -> PrismRunSummary:
    """Build a PrismRunSummary from a record dict."""
    # Records may be raw PRISM probes or Maestro learning-run exports.
    # Try several known shapes.
    run_id = (
        data.get("id")
        or data.get("run_id")
        or data.get("session_id")
        or path.stem
    )
    model_name = (
        data.get("model_name")
        or data.get("base_model_ref")
        or data.get("model")
        or None
    )
    run_kind = data.get("run_kind") or data.get("kind") or "probe"
    status   = data.get("status") or "unknown"
    updated_at = (
        data.get("updated_at")
        or data.get("timestamp")
        or data.get("created_at")
        or None
    )

    entropy_proof = data.get("entropy_delta_proof") or data.get("delta_proof") or {}
    before = data.get("entropy_snapshot_before") or data.get("snapshot_before") or {}
    after  = data.get("entropy_snapshot_after")  or data.get("snapshot_after")  or {}

    has_proof   = bool(entropy_proof)
    has_layers  = bool(
        (before.get("layer_profiles") if isinstance(before, dict) else None)
        or (after.get("layer_profiles") if isinstance(after, dict) else None)
    )

    # Extract TurboQuant aggregates — prefer the "after" snapshot (post-training),
    # fall back to "before", then to top-level fields on the record itself.
    tq_source = (after if isinstance(after, dict) and after else
                 before if isinstance(before, dict) and before else
                 data)
    mean_outlier_ratio        = tq_source.get("mean_outlier_ratio")
    mean_cardinal_proximity   = tq_source.get("mean_cardinal_proximity")
    mean_quantization_hostility = tq_source.get("mean_quantization_hostility")
    worst_layer_idx           = tq_source.get("worst_layer_idx")

    has_outlier_geometry = any(
        v is not None for v in [
            mean_outlier_ratio, mean_cardinal_proximity,
            mean_quantization_hostility, worst_layer_idx,
        ]
    )

    return PrismRunSummary(
        id=str(run_id),
        source=path.stem,
        model_name=model_name,
        run_kind=run_kind,
        status=status,
        updated_at=updated_at,
        has_entropy_proof=has_proof,
        has_layer_profiles=has_layers,
        has_outlier_geometry=has_outlier_geometry,
        mean_outlier_ratio=float(mean_outlier_ratio) if mean_outlier_ratio is not None else None,
        mean_cardinal_proximity=float(mean_cardinal_proximity) if mean_cardinal_proximity is not None else None,
        mean_quantization_hostility=float(mean_quantization_hostility) if mean_quantization_hostility is not None else None,
        worst_layer_idx=int(worst_layer_idx) if worst_layer_idx is not None else None,
    )


def _normalize_run(path: Path, data: dict[str, Any]) -> dict[str, Any]:
    """
    Return a normalized record dict that matches the frontend LearningRun shape.

    Records on disk may be raw PRISM probe results, SGT output dicts, or Maestro
    learning-run exports.  This function ensures all required LearningRun fields
    are present with sensible fallbacks so the frontend can render without errors.

    The original dict is not mutated; a shallow-merged copy is returned.
    """
    run_id = str(
        data.get("id")
        or data.get("run_id")
        or data.get("session_id")
        or path.stem
    )
    base_model_ref = str(
        data.get("base_model_ref")
        or data.get("model_name")
        or data.get("model")
        or "unknown"
    )
    run_kind = str(data.get("run_kind") or data.get("kind") or "probe")
    status   = str(data.get("status") or "unknown")
    source_axis = str(data.get("source_axis") or data.get("axis") or "unknown")
    updated_at = str(
        data.get("updated_at")
        or data.get("timestamp")
        or data.get("created_at")
        or ""
    )

    # Carry entropy fields through using both naming conventions.
    entropy_before = (
        data.get("entropy_snapshot_before")
        or data.get("snapshot_before")
        or None
    )
    entropy_after = (
        data.get("entropy_snapshot_after")
        or data.get("snapshot_after")
        or None
    )
    entropy_proof = (
        data.get("entropy_delta_proof")
        or data.get("delta_proof")
        or None
    )

    # Hoist TurboQuant aggregates to the top level so consumers don't have to
    # dig into nested snapshot dicts.  Prefer the "after" snapshot (post-training)
    # then fall back to "before", then to any top-level fields already present.
    tq_source = (entropy_after if isinstance(entropy_after, dict) and entropy_after else
                 entropy_before if isinstance(entropy_before, dict) and entropy_before else
                 data)
    tq_fields: dict[str, Any] = {}
    for _tq_key in (
        "mean_outlier_ratio",
        "mean_cardinal_proximity",
        "mean_quantization_hostility",
        "worst_layer_idx",
    ):
        _val = tq_source.get(_tq_key)
        if _val is not None:
            tq_fields[_tq_key] = _val

    normalized: dict[str, Any] = {
        **data,
        "id": run_id,
        "base_model_ref": base_model_ref,
        "run_kind": run_kind,
        "status": status,
        "source_axis": source_axis,
        "updated_at": updated_at,
        "entropy_snapshot_before": entropy_before,
        "entropy_snapshot_after": entropy_after,
        "entropy_delta_proof": entropy_proof,
        "_source_file": path.stem,
        **tq_fields,
    }
    return normalized


# ── Router ─────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1/prism", tags=["prism"])


@router.get("/status", response_model=PrismStatus)
def prism_status():
    """
    Return PRISM system status: record count, model track, and tool registry.
    No auth required — public read-only metadata.
    """
    records = _records()
    return PrismStatus(
        records_dir=_RECORDS_DIR.name,
        record_count=len(records),
        model_track=_MODEL_TRACK,
        tools=[
            {"name": "TransformerLens", "role": "activation-inspection", "url": "https://github.com/TransformerMechInterp/TransformerLens"},
            {"name": "CircuitsVis",     "role": "notebook-visualization", "url": "https://github.com/TransformerMechInterp/CircuitsVis"},
            {"name": "NNsight",         "role": "causal-intervention",    "url": "https://nnsight.net/"},
            {"name": "Neuronpedia",     "role": "sae-feature-dashboard",  "url": "https://docs.neuronpedia.org/features"},
            {"name": "AttributionGraphs","role": "circuit-tracing",       "url": "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"},
            {"name": "GemmaScope2",     "role": "sae-weights-reference",  "url": "https://deepmind.google/models/gemma/gemma-scope/"},
            {"name": "BertViz",         "role": "attention-visualization","url": "https://github.com/jessevig/bertviz"},
        ],
    )


@router.get("/runs/summary", response_model=list[PrismRunSummary])
def list_prism_runs_summary():
    """
    List lightweight summaries of all PRISM run records.
    Used by dashboards that only need metadata (count, flags) without full records.
    """
    summaries = []
    for path in _records():
        data = _load_record(path)
        if not data:
            continue
        try:
            summaries.append(_make_summary(path, data))
        except Exception as exc:
            logger.warning(f"prism: skipping {path.name}: {exc}")
    return summaries


@router.get("/runs", response_model=list[dict])
def list_prism_runs():
    """
    List all available PRISM run records from data/records/ as full normalized dicts.

    Returns each record in a shape compatible with the frontend LearningRun type:
    required fields (id, base_model_ref, run_kind, status, source_axis, updated_at)
    are always present with fallbacks; optional entropy snapshot and proof fields
    are passed through as-is from the underlying record file.

    Sorted by file modification time (newest first).
    """
    runs = []
    for path in _records():
        data = _load_record(path)
        if not data:
            continue
        try:
            runs.append(_normalize_run(path, data))
        except Exception as exc:
            logger.warning(f"prism: skipping {path.name}: {exc}")
    return runs


@router.get("/runs/{run_id}", response_model=dict)
def get_prism_run(run_id: str):
    """
    Return the full JSON record for a given run ID or filename stem.
    Searches by id/run_id field first, then by filename stem.
    """
    if not _SAFE_RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=404, detail=f"PRISM run not found: {run_id}")

    for path in _records():
        data = _load_record(path)
        if not data:
            continue
        # Match by logical ID field (falls back to filename stem when no id fields present)
        if str(data.get("id") or data.get("run_id") or data.get("session_id") or path.stem) == run_id:
            return _normalize_run(path, data)
        # Match by filename stem (handles case where record has a different logical id)
        if path.stem == run_id:
            return _normalize_run(path, data)

    raise HTTPException(status_code=404, detail=f"PRISM run not found: {run_id}")


@router.get("/versions")
def list_versions():
    """
    List all fine-tuning versions with availability flags.
    No auth required.
    """
    out = []
    for version, meta in _TRAINING_VERSIONS.items():
        out.append({
            "version": version,
            "label": meta["label"],
            "steps": meta["steps"],
            "examples": meta["examples"],
            "dataset": meta["dataset"],
            "has_training": meta["trainer_state"] is not None and meta["trainer_state"].exists(),
            "has_benchmark": meta["benchmark"] is not None and meta["benchmark"].exists(),
        })
    return out


@router.get("/training/{version}")
def get_training(version: str):
    """
    Return full training log for a version: per-step loss, grad_norm,
    learning_rate, entropy, token_accuracy + eval checkpoints.
    Source: experiments/qwen3_5_2b/{version}/adapter/checkpoint-*/trainer_state.json
    """
    if version not in _TRAINING_VERSIONS:
        raise HTTPException(status_code=404, detail=f"Unknown version: {version}")
    meta = _TRAINING_VERSIONS[version]
    if not meta["trainer_state"]:
        raise HTTPException(status_code=404, detail=f"No training data for {version}")

    state = _load_json_file(meta["trainer_state"])
    log_history = state.get("log_history", [])

    train_log = [
        {
            "step": e["step"],
            "loss": e["loss"],
            "grad_norm": e.get("grad_norm"),
            "learning_rate": e.get("learning_rate"),
            "entropy": e.get("entropy"),
            "token_accuracy": e.get("mean_token_accuracy"),
            "epoch": e.get("epoch"),
        }
        for e in log_history if "loss" in e
    ]
    eval_log = [
        {
            "step": e["step"],
            "eval_loss": e["eval_loss"],
            "eval_entropy": e.get("eval_entropy"),
            "eval_token_accuracy": e.get("eval_mean_token_accuracy"),
            "epoch": e.get("epoch"),
        }
        for e in log_history if "eval_loss" in e
    ]

    return {
        "version": version,
        "label": meta["label"],
        "total_steps": state.get("global_step"),
        "best_step": state.get("best_global_step"),
        "best_eval_loss": state.get("best_metric"),
        "epoch": state.get("epoch"),
        "train_log": train_log,
        "eval_log": eval_log,
    }


@router.get("/benchmark/{version}")
def get_benchmark(version: str):
    """
    Return per-scenario T3 benchmark breakdown for a version.
    Source: experiments/qwen3_5_2b/t3_benchmark*/results.json
    Includes per-scenario pass rates, t1/t2/t3 rates, and first 3 run examples.
    """
    if version not in _TRAINING_VERSIONS:
        raise HTTPException(status_code=404, detail=f"Unknown version: {version}")
    meta = _TRAINING_VERSIONS[version]
    if not meta["benchmark"]:
        raise HTTPException(status_code=404, detail=f"No benchmark data for {version}")

    results = _load_json_file(meta["benchmark"])
    models = results.get("models", {})
    model_key = next(iter(models), None)
    if not model_key:
        raise HTTPException(status_code=404, detail="No model key in benchmark results")

    model_data = models[model_key]

    # Per-scenario summary — strip conversation runs to a small sample
    scenarios_out = {}
    for scenario_id, sd in model_data.get("scenarios", {}).items():
        parts = scenario_id.split("--", 1)
        ownership = parts[0] if len(parts) == 2 else scenario_id
        emotion   = parts[1] if len(parts) == 2 else "unknown"
        scenarios_out[scenario_id] = {
            "emotion":       emotion,
            "ownership":     ownership,
            "n_runs":        sd.get("n_runs"),
            "t3_correct":    sd.get("t3_correct"),
            "t3_pass_rate":  sd.get("t3_pass_rate"),
            "t1_pass_rate":  sd.get("t1_all_pass_rate"),
            "t2_pass_rate":  sd.get("t2_pass_rate"),
            # First 3 example runs for the conversation viewer
            "examples": [
                {
                    "outcome": r.get("t3_outcome"),
                    "turns":   [t.get("reply", "") for t in r.get("turns", [])],
                }
                for r in sd.get("runs", [])[:3]
            ],
        }

    return {
        "version":            version,
        "label":              meta["label"],
        "model_key":          model_key,
        "overall_pass_rate":  model_data.get("overall_t3_pass_rate"),
        "generated_at":       results.get("generated_at"),
        "scenarios":          scenarios_out,
    }


# ── Cognitive / Viability endpoints ──────────────────────────────────────────
# Orch-OS-aligned cognitive core monitoring and E <= C viability checks.
# These expose the lattice pipeline's cognitive routing and drift detection
# to the PRISM dashboard for real-time landscape health monitoring.

_LATTICE_DIR = _PROJECT_ROOT / "data" / "lattices"
_SIGNALS_PATH = _PROJECT_ROOT / "data" / "training_signals.json"
_FIELD_PATH = _PROJECT_ROOT / "data" / "memory_field.pkl"


@router.get("/cognitive/status")
def cognitive_status():
    """
    Return cognitive pipeline status: signal counts, viability, core activity.
    Lightweight endpoint for dashboard polling.
    """
    signals = _load_training_signals()
    viability = _run_viability_check(signals)

    trainable = [s for s in signals if s.get("signal_type") == "sft_pair"]
    denied = [s for s in signals if not s.get("consent_allows_training")]
    weights = [s["weight"] for s in trainable if s.get("weight", 0) > 0]

    return {
        "total_sessions": len(signals),
        "trainable": len(trainable),
        "consent_denied": len(denied),
        "mean_weight": round(sum(weights) / len(weights), 4) if weights else 0,
        "max_weight": round(max(weights), 4) if weights else 0,
        "viability": viability,
        "lattice_dir_exists": _LATTICE_DIR.exists(),
        "signals_file_exists": _SIGNALS_PATH.exists(),
    }


@router.get("/cognitive/signals")
def cognitive_signals():
    """
    Return all training signals with component scores.
    Used by the cognitive core visualization in the PRISM dashboard.
    """
    signals = _load_training_signals()

    # Add core intensity data for visualization
    for s in signals:
        s["core_intensities"] = _compute_core_intensities(s)

    return signals


@router.get("/cognitive/viability")
def cognitive_viability():
    """
    Run the E <= C viability check and return the full report.
    """
    signals = _load_training_signals()
    return _run_viability_check(signals)


@router.get("/cognitive/session/{session_id}")
def cognitive_session(session_id: str):
    """
    Run the full Orch-OS cognitive routing pipeline on a single session
    and return the timeline + core intensities.
    """
    if not _SAFE_RUN_ID_RE.match(session_id):
        raise HTTPException(status_code=400, detail="Invalid session ID format")

    lattice_path = _LATTICE_DIR / f"{session_id}.json"
    if not lattice_path.exists():
        raise HTTPException(status_code=404, detail=f"Lattice not found: {session_id}")

    try:
        import sys
        _lib_root = str(Path(__file__).resolve().parents[2])
        if _lib_root not in sys.path:
            sys.path.insert(0, _lib_root)

        from libs.lattice.signal_scorer import SignalScorer
        from libs.lattice.memory_field import MemoryField
        from libs.lattice.cognitive_cores import CognitiveRouter

        lattice_data = _load_json_file(lattice_path)
        scorer = SignalScorer()
        mf = MemoryField(str(_FIELD_PATH)) if _FIELD_PATH.exists() else None
        router_cog = CognitiveRouter(scorer, mf)
        result = router_cog.process(lattice_data)

        return {
            "session_id": result.session_id,
            "score": result.score.to_dict(),
            "signals": [
                {
                    "core": s.core,
                    "intensity": round(s.intensity, 4),
                    "query": s.query[:200],
                    "insights": s.insights,
                }
                for s in result.signals
            ],
            "dominant_cores": router_cog._dominant_cores(result.signals),
            "memory_resonance": result.memory_resonance,
            "timeline_summary": {
                "events": len(result.timeline),
                "phases": {
                    "signal_generation": sum(1 for e in result.timeline if e.event_type == "signal_generated"),
                    "core_activation": sum(1 for e in result.timeline if e.event_type == "core_activated"),
                },
            },
        }
    except ImportError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cognitive pipeline not available: {exc}. Install sentence-transformers."
        )
    except Exception as exc:
        logger.warning(f"Cognitive routing failed for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


def _load_training_signals() -> list[dict]:
    """Load training_signals.json if it exists."""
    if not _SIGNALS_PATH.exists():
        return []
    try:
        with _SIGNALS_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        logger.warning(f"Failed to load training signals: {exc}")
        return []


def _run_viability_check(signals: list[dict]) -> dict:
    """Run the E <= C viability check."""
    try:
        import sys
        _prism_root = str(Path(__file__).resolve().parents[3] / "prism" / "src")
        if _prism_root not in sys.path:
            sys.path.insert(0, _prism_root)
        from prism.eval.early_warning import LatticeViabilityMonitor
        monitor = LatticeViabilityMonitor()
        return monitor.check(signals)
    except ImportError:
        return {"status": "UNAVAILABLE", "message": "PRISM not installed"}
    except Exception as exc:
        return {"status": "ERROR", "message": str(exc)}


def _compute_core_intensities(signal: dict) -> dict:
    """
    Compute Orch-OS cognitive core intensities from a training signal's
    component scores. Used for the core radar/bar chart visualization.
    """
    return {
        "shadow": min(1.0, signal.get("contradiction_score", 0.0)),
        "valence": min(1.0, signal.get("affective_granularity", 0.0)),
        "integration": min(1.0, signal.get("integration_depth", 0.0) / 10.0),
        "oracle": min(1.0, signal.get("oracle_effective", signal.get("oracle_capacity", 0.0))),
        "novelty": min(1.0, signal.get("novelty", 0.0) if "novelty" in signal else 0.0),
        "coherence": min(1.0, signal.get("temporal_coherence", 0.0)),
        "entropy": min(1.0, signal.get("semantic_entropy", 0.0) / 3.0),
        "arc": min(1.0, signal.get("affect_arc_coherence", 0.0)),
    }


def register_prism_routes(app: Any) -> None:
    """Register the PRISM router on the FastAPI app."""
    app.include_router(router)
