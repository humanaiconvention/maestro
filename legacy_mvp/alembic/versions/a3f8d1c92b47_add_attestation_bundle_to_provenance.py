"""add attestation_bundle to provenance_records

Revision ID: a3f8d1c92b47
Revises: 599f0a63e444
Create Date: 2026-03-17

Phase D (Cryptographic Provenance): adds the attestation_bundle JSON column to
provenance_records so that TLSNotary proofs, Secure Enclave attestation quotes,
and future ZK proof payloads can be stored alongside the existing
creation_context telemetry.

The column is nullable so that contributions submitted before Phase D (which
carry no cryptographic attestation) are unaffected — the AttestationPipeline
handles a None bundle gracefully and falls back to the legacy humanity_score.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import sqlite


# revision identifiers, used by Alembic.
revision: str = "a3f8d1c92b47"
down_revision: Union[str, None] = "599f0a63e444"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "provenance_records",
        sa.Column("attestation_bundle", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("provenance_records", "attestation_bundle")
