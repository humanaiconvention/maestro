"""
Prism API routes for the Maestro gateway.

Serves PRISM run data files from the data/records/ directory so the
frontend Prism module can display run history without requiring the
legacy_mvp service (port 8001) to be running.

Endpoints:
    GET  /v1/prism/runs             — list available run records (full)
    GET  /v1/prism/runs/summary     — lightweight summaries (metadata only)
    GET  /v1/prism/runs/{run_id}    — get a specific run record by ID
    GET  /v1/prism/status           — PRISM system status (tool versions, model track)

Outlier geometry fields (TurboQuant-inspired, v8 STG):
    PrismRunSummary surfaces mean_quantization_hostility, mean_outlier_ratio,
    mean_cardinal_proximity, and worst_layer_idx from EntropySnapshot records
    so dashboards can flag quantization-hostile runs without loading full blobs.
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

_RECORDS_DIR = Path(os.environ.get(
    "PRISM_RECORDS_DIR",
    Path(__file__).resolve().parents[3] / "data" / "records",
))

_MODEL_TRACK = os.environ.get("PRISM_MODEL_TRACK", "qwen3.5-2b-haic-v7")

# run_id validated before any filesystem lookup to prevent path traversal.
_SAFE_RUN_ID_RE = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9\-_\.]{0,127}$')

if not _RECORDS_DIR.exists():
    logger.warning(
        f"prism: records directory not found: {_RECORDS_DIR} — "
        "all /v1/prism/runs requests will return []. "
        "Set PRISM_RECORDS_DIR env var to override."
    )


# ── Models ─────────────────────────────────────────────────────────────────────

class PrismRunSummary(BaseModel):
    """Lightweight summary returned in the list-summary endpoint."""
    id: str
    source: str
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
    try:
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(f"prism: failed to load record {path.name}: {exc}")
        return {}


def _records() -> list[Path]:
    if not _RECORDS_DIR.exists():
        return []
    return sorted(
        _RECORDS_DIR.glob("*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def _make_summary(path: Path, data: dict[str, Any]) -> PrismRunSummary:
    run_id = (
        data.get("id") or data.get("run_id") or data.get("session_id") or path.stem
    )
    model_name = (
        data.get("model_name") or data.get("base_model_ref") or data.get("model") or None
    )
    run_kind   = data.get("run_kind") or data.get("kind") or "probe"
    status     = data.get("status") or "unknown"
    updated_at = (
        data.get("updated_at") or data.get("timestamp") or data.get("created_at") or None
    )

    entropy_proof = data.get("entropy_delta_proof") or data.get("delta_proof") or {}
    before = data.get("entropy_snapshot_before") or data.get("snapshot_before") or {}
    after  = data.get("entropy_snapshot_after")  or data.get("snapshot_after")  or {}

    has_proof  = bool(entropy_proof)
    has_layers = bool(
        (before.get("layer_profiles") if isinstance(before, dict) else None)
        or (after.get("layer_profiles") if isinstance(after, dict) else None)
    )

    # TurboQuant aggregates — prefer "after" snapshot (post-training state),
    # fall back to "before", then to top-level fields on the record itself.
    tq_source = (
        after  if isinstance(after, dict) and after else
        before if isinstance(before, dict) and before else
        data
    )
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
    """Return a normalized record dict matching the frontend LearningRun shape."""
    run_id = str(
        data.get("id") or data.get("run_id") or data.get("session_id") or path.stem
    )
    base_model_ref = str(
        data.get("base_model_ref") or data.get("model_name") or data.get("model") or "unknown"
    )
    run_kind    = str(data.get("run_kind") or data.get("kind") or "probe")
    status      = str(data.get("status") or "unknown")
    source_axis = str(data.get("source_axis") or data.get("axis") or "unknown")
    updated_at  = str(
        data.get("updated_at") or data.get("timestamp") or data.get("created_at") or ""
    )

    entropy_before = data.get("entropy_snapshot_before") or data.get("snapshot_before") or None
    entropy_after  = data.get("entropy_snapshot_after")  or data.get("snapshot_after")  or None
    entropy_proof  = data.get("entropy_delta_proof")     or data.get("delta_proof")      or None

    return {
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
    }


# ── Router ─────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/v1/prism", tags=["prism"])


@router.get("/status", response_model=PrismStatus)
def prism_status():
    """PRISM system status: record count, model track, tool registry. Public read-only."""
    records = _records()
    return PrismStatus(
        records_dir=str(_RECORDS_DIR),
        record_count=len(records),
        model_track=_MODEL_TRACK,
        tools=[
            {"name": "TransformerLens",    "role": "activation-inspection",    "url": "https://github.com/TransformerMechInterp/TransformerLens"},
            {"name": "CircuitsVis",         "role": "notebook-visualization",   "url": "https://github.com/TransformerMechInterp/CircuitsVis"},
            {"name": "NNsight",             "role": "causal-intervention",      "url": "https://nnsight.net/"},
            {"name": "Neuronpedia",         "role": "sae-feature-dashboard",    "url": "https://docs.neuronpedia.org/features"},
            {"name": "AttributionGraphs",   "role": "circuit-tracing",          "url": "https://transformer-circuits.pub/2025/attribution-graphs/methods.html"},
            {"name": "GemmaScope2",         "role": "sae-weights-reference",    "url": "https://deepmind.google/models/gemma/gemma-scope/"},
            {"name": "BertViz",             "role": "attention-visualization",  "url": "https://github.com/jessevig/bertviz"},
        ],
    )


@router.get("/runs/summary", response_model=list[PrismRunSummary])
def list_prism_runs_summary():
    """Lightweight summaries of all PRISM run records (metadata + outlier geometry flags)."""
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
    """List all PRISM run records as full normalized dicts. Sorted newest-first."""
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
    """Return the full JSON record for a given run ID or filename stem."""
    if not _SAFE_RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=404, detail=f"PRISM run not found: {run_id}")

    for path in _records():
        data = _load_record(path)
        if not data:
            continue
        if str(data.get("id") or data.get("run_id") or data.get("session_id") or path.stem) == run_id:
            return _normalize_run(path, data)
        if path.stem == run_id:
            return _normalize_run(path, data)

    raise HTTPException(status_code=404, detail=f"PRISM run not found: {run_id}")


def register_prism_routes(app: Any) -> None:
    """Register the PRISM router on the FastAPI app."""
    app.include_router(router)
