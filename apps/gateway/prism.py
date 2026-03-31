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

# ── Data directory ────────────────────────────────────────────────────────────

# Records directory relative to the project root. Can be overridden via env.
_RECORDS_DIR = Path(os.environ.get(
    "PRISM_RECORDS_DIR",
    Path(__file__).resolve().parents[3] / "data" / "records",
))

# Active model track — overridable to avoid hardcoding the version string.
_MODEL_TRACK = os.environ.get("PRISM_MODEL_TRACK", "qwen3.5-2b-haic-v7")

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


def register_prism_routes(app: Any) -> None:
    """Register the PRISM router on the FastAPI app."""
    app.include_router(router)
