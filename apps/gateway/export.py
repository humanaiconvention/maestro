"""
export.py — Admin data-export endpoint for consent-verified participation receipts.

GET /v1/export/kernels
    Returns a provenance-only export of all participation receipts (no raw content).
    Columns: receipt_id, session_id, merkle_root, consent_level, timestamp, payload_hash

    Auth:    Bearer <EXPORT_API_KEY>  OR  ?key=<EXPORT_API_KEY>
    Params:
        format=csv   (default) — Content-Type: text/csv
        format=json            — Content-Type: application/json
        since=<ISO 8601 date>  — only receipts created at or after this timestamp

The payload_hash is SHA-256 of the canonical receipt payload JSON (sans qr_data_url),
proving data provenance without exposing any conversation content.
"""

import hashlib
import io
import json
import logging
import os
from csv import writer as csv_writer
from typing import Any, Optional

from fastapi import HTTPException, Query, Request
from fastapi.responses import Response

from apps.gateway.participation import list_receipts

logger = logging.getLogger(__name__)

_EXPORT_API_KEY: Optional[str] = None


def _get_export_key() -> str:
    """Lazy-load EXPORT_API_KEY so tests can set it after module import."""
    global _EXPORT_API_KEY
    if _EXPORT_API_KEY is None:
        _EXPORT_API_KEY = os.environ.get("EXPORT_API_KEY", "")
    return _EXPORT_API_KEY


def _verify_export_auth(request: Request) -> None:
    """
    Validate the export API key from Authorization: Bearer header only.
    Query-param key auth is intentionally not supported — secrets must not
    appear in URLs, which are logged by proxies, CDNs, and browser history.
    Raises HTTP 401/403 on failure.
    """
    export_key = _get_export_key()
    if not export_key:
        raise HTTPException(status_code=503, detail="Export not configured (EXPORT_API_KEY not set)")

    provided: Optional[str] = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        provided = auth_header[len("Bearer "):]

    if not provided:
        raise HTTPException(status_code=401, detail="Export API key required (Authorization: Bearer <key>)")

    # Constant-time comparison to prevent timing attacks
    import hmac
    if not hmac.compare_digest(provided, export_key):
        raise HTTPException(status_code=403, detail="Invalid export API key")


def _payload_hash(receipt) -> str:
    """SHA-256 of canonical receipt payload (qr_data_url excluded — it's derived)."""
    d = receipt.to_dict()
    d.pop("qr_data_url", None)
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _consent_level(receipt) -> str:
    """Compact single-field summary of consent choices (JSON)."""
    return json.dumps(receipt.consent_summary, ensure_ascii=True)


def _to_csv(receipts: list) -> str:
    buf = io.StringIO()
    w = csv_writer(buf)
    w.writerow(["receipt_id", "session_id", "merkle_root", "consent_level", "timestamp", "payload_hash"])
    for r in receipts:
        w.writerow([
            r.merkle_root,          # receipt_id == merkle_root (used as PK)
            r.session_id,
            r.merkle_root,
            _consent_level(r),
            r.created_at,
            _payload_hash(r),
        ])
    return buf.getvalue()


def _to_json(receipts: list) -> str:
    rows = []
    for r in receipts:
        rows.append({
            "receipt_id":    r.merkle_root,
            "session_id":    r.session_id,
            "merkle_root":   r.merkle_root,
            "consent_level": r.consent_summary,
            "timestamp":     r.created_at,
            "payload_hash":  _payload_hash(r),
        })
    return json.dumps({"count": len(rows), "kernels": rows}, indent=2, ensure_ascii=True)


def register_export_routes(app: Any) -> None:
    """Register GET /v1/export/kernels on the FastAPI app. Called from main.py."""

    @app.get("/v1/export/kernels", tags=["export"])
    async def export_kernels(
        request: Request,
        format: str = Query(default="csv", pattern="^(csv|json)$"),
        since: Optional[str] = Query(default=None),
    ):
        """
        Export consent-verified data kernel metadata for training/research pipelines.

        Returns provenance hashes only — no raw conversation content is included.
        Requires EXPORT_API_KEY via `Authorization: Bearer <key>` header only.
        Query-param key auth is not supported to prevent secret logging in proxies/CDNs.
        """
        _verify_export_auth(request)

        receipts = await list_receipts(since=since)

        if format == "json":
            return Response(
                content=_to_json(receipts),
                media_type="application/json",
            )

        return Response(
            content=_to_csv(receipts),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=\"kernels.csv\""},
        )
