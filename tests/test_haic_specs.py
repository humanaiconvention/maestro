from __future__ import annotations

import json
from pathlib import Path

from libs.schemas.haic_schemas import (
    ClaimPacket,
    EvidencePacket,
    FederatedExchangeStandard,
    KnowledgePacketDocument,
    ObjectionPacket,
    ParticipationCovenant,
    ParticipationIntegrityStandard,
    RevisionPacket,
    SoftLaunchStandard,
    validate_federated_exchange_standard,
    validate_knowledge_packet,
    validate_participation_covenant,
    validate_participation_integrity_standard,
    validate_soft_launch_standard,
)


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "specs" / "haic" / "examples"
SCHEMAS = ROOT / "specs" / "haic"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_claim_example_validates() -> None:
    packet = validate_knowledge_packet(_load_json(EXAMPLES / "claim.packet.json"))
    assert isinstance(packet, ClaimPacket)
    assert packet.kind == "claim"
    assert packet.provenance.created_by_mode == "hybrid"
    assert packet.alignment_axis is not None
    assert "honesty" in packet.alignment_axis.commitments


def test_evidence_example_validates() -> None:
    packet = validate_knowledge_packet(_load_json(EXAMPLES / "evidence.packet.json"))
    assert isinstance(packet, EvidencePacket)
    assert packet.ledger_refs[0].immutable is True


def test_objection_example_validates() -> None:
    packet = validate_knowledge_packet(_load_json(EXAMPLES / "objection.packet.json"))
    assert isinstance(packet, ObjectionPacket)
    assert packet.target_packet_id == "haic.claim.semantic-viability-condition.v0"


def test_revision_example_validates() -> None:
    packet = validate_knowledge_packet(_load_json(EXAMPLES / "revision.packet.json"))
    assert isinstance(packet, RevisionPacket)
    assert packet.supersedes_target is True
    assert packet.changes[0].field_path == "scope.domain"


def test_covenant_example_validates() -> None:
    covenant = validate_participation_covenant(
        _load_json(EXAMPLES / "grounded-correction.covenant.json")
    )
    assert isinstance(covenant, ParticipationCovenant)
    assert covenant.memory_policy.retention_mode == "bounded"
    assert covenant.approvals.writes == "explicit"
    assert covenant.alignment_axis is not None
    assert covenant.alignment_axis.mode == "geometric_orientation"
    assert covenant.standards[0].standard_id == "haic.standard.participation-integrity.v0"
    assert covenant.soft_launch is not None
    assert covenant.soft_launch.participation_mode == "calibration_only"


def test_participation_integrity_standard_example_validates() -> None:
    standard = validate_participation_integrity_standard(
        _load_json(EXAMPLES / "participation-integrity.standard.json")
    )
    assert isinstance(standard, ParticipationIntegrityStandard)
    assert standard.status == "active"
    assert "bounded_consent" in standard.commitments
    assert standard.architecture_stack[0].layer == "public_website"


def test_federated_exchange_standard_example_validates() -> None:
    standard = validate_federated_exchange_standard(
        _load_json(EXAMPLES / "federated-exchange.standard.json")
    )
    assert isinstance(standard, FederatedExchangeStandard)
    assert standard.status == "active"
    assert "federated_exchange" in standard.commitments
    assert standard.default_memory_policy.central_storage_mode == "verification_only"
    assert standard.default_memory_policy.edge_exchange_required is True


def test_soft_launch_standard_example_validates() -> None:
    standard = validate_soft_launch_standard(
        _load_json(EXAMPLES / "soft-launch.standard.json")
    )
    assert isinstance(standard, SoftLaunchStandard)
    assert standard.status == "active"
    assert standard.default_policy.phase == "seed"
    assert standard.default_policy.participation_mode == "calibration_only"


def test_federated_covenant_example_validates() -> None:
    covenant = validate_participation_covenant(
        _load_json(EXAMPLES / "grounded-correction-federated.covenant.json")
    )
    assert isinstance(covenant, ParticipationCovenant)
    assert covenant.memory_policy.central_storage_mode == "verification_only"
    assert covenant.memory_policy.edge_exchange_required is True
    standard_ids = [standard.standard_id for standard in covenant.standards]
    assert "haic.standard.federated-exchange.v0" in standard_ids
    assert covenant.soft_launch is not None
    assert covenant.soft_launch.phase == "seed"
    assert "haic.standard.soft-launch.v0" in standard_ids


def test_bounded_memory_policy_requires_retention_days() -> None:
    payload = _load_json(EXAMPLES / "grounded-correction.covenant.json")
    payload["memory_policy"]["max_retention_days"] = None

    try:
        validate_participation_covenant(payload)
    except Exception as exc:
        assert "max_retention_days" in str(exc)
    else:
        raise AssertionError("Expected validation error for bounded retention without days")


def test_verification_only_memory_policy_rejects_raw_artifacts() -> None:
    payload = _load_json(EXAMPLES / "grounded-correction-federated.covenant.json")
    payload["memory_policy"]["allowed_central_artifacts"].append("raw_content")

    try:
        validate_participation_covenant(payload)
    except Exception as exc:
        assert "verification_only" in str(exc)
    else:
        raise AssertionError("Expected validation error for raw artifacts in verification-only mode")


def test_knowledge_packet_schema_is_discriminated_union() -> None:
    schema = _load_json(SCHEMAS / "knowledge-packet.v0.schema.json")
    assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert "oneOf" in schema
    assert schema["discriminator"]["propertyName"] == "kind"


def test_participation_covenant_schema_has_required_sections() -> None:
    schema = _load_json(SCHEMAS / "participation-covenant.v0.schema.json")
    required = set(schema["required"])
    assert {"participants", "preconditions", "hard_invariants", "hard_goals", "recovery"} <= required


def test_participation_integrity_standard_schema_has_required_sections() -> None:
    schema = _load_json(SCHEMAS / "participation-integrity-standard.v0.schema.json")
    required = set(schema["required"])
    assert {
        "commitments",
        "prohibited_patterns",
        "required_controls",
        "architecture_stack",
    } <= required


def test_federated_exchange_standard_schema_has_required_sections() -> None:
    schema = _load_json(SCHEMAS / "federated-exchange-standard.v0.schema.json")
    required = set(schema["required"])
    assert {
        "commitments",
        "prohibited_patterns",
        "required_controls",
        "allowed_system_artifacts",
        "disallowed_system_artifacts",
        "default_memory_policy",
        "architecture_stack",
    } <= required


def test_soft_launch_standard_schema_has_required_sections() -> None:
    schema = _load_json(SCHEMAS / "soft-launch-standard.v0.schema.json")
    required = set(schema["required"])
    assert {
        "commitments",
        "prohibited_patterns",
        "required_controls",
        "default_policy",
        "architecture_stack",
    } <= required


def test_root_model_round_trip() -> None:
    raw = _load_json(EXAMPLES / "claim.packet.json")
    model = KnowledgePacketDocument.model_validate(raw)
    dumped = model.model_dump(mode="json")
    assert dumped["packet_id"] == raw["packet_id"]
