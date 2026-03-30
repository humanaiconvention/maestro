"""
Unit tests for the test_case module.
"""

import pytest
from semantic_grounding.test_case import SemanticGroundingTestCase, GroundingType


class TestSemanticGroundingTestCase:
    """Tests for SemanticGroundingTestCase class."""

    def test_create_valid_test_case(self):
        """Test creating a valid test case."""
        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="The red ball is on the table.",
            expected_grounding=["red ball", "table"],
        )

        assert test_case.test_id == "test_001"
        assert test_case.grounding_type == GroundingType.OBJECT_REFERENCE
        assert test_case.prompt == "The red ball is on the table."
        assert test_case.expected_grounding == ["red ball", "table"]
        assert test_case.context is None
        assert test_case.metadata is None

    def test_create_test_case_with_context(self):
        """Test creating a test case with context."""
        test_case = SemanticGroundingTestCase(
            test_id="test_002",
            grounding_type=GroundingType.SPATIAL_REASONING,
            prompt="The library is between the coffee shop and the park.",
            expected_grounding=["library", "coffee shop", "park"],
            context="Spatial relationships",
        )

        assert test_case.context == "Spatial relationships"

    def test_create_test_case_with_metadata(self):
        """Test creating a test case with metadata."""
        metadata = {"difficulty": "easy", "category": "indoor"}
        test_case = SemanticGroundingTestCase(
            test_id="test_003",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
            metadata=metadata,
        )

        assert test_case.metadata == metadata

    def test_empty_test_id_raises_error(self):
        """Test that empty test_id raises ValueError."""
        with pytest.raises(ValueError, match="test_id cannot be empty"):
            SemanticGroundingTestCase(
                test_id="",
                grounding_type=GroundingType.OBJECT_REFERENCE,
                prompt="Test prompt",
                expected_grounding=["test"],
            )

    def test_empty_prompt_raises_error(self):
        """Test that empty prompt raises ValueError."""
        with pytest.raises(ValueError, match="prompt cannot be empty"):
            SemanticGroundingTestCase(
                test_id="test_001",
                grounding_type=GroundingType.OBJECT_REFERENCE,
                prompt="",
                expected_grounding=["test"],
            )

    def test_empty_expected_grounding_raises_error(self):
        """Test that empty expected_grounding raises ValueError."""
        with pytest.raises(ValueError, match="expected_grounding cannot be empty"):
            SemanticGroundingTestCase(
                test_id="test_001",
                grounding_type=GroundingType.OBJECT_REFERENCE,
                prompt="Test prompt",
                expected_grounding=[],
            )

    def test_to_dict(self):
        """Test converting test case to dictionary."""
        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
            context="Test context",
            metadata={"key": "value"},
        )

        result = test_case.to_dict()

        assert result["test_id"] == "test_001"
        assert result["grounding_type"] == "object_reference"
        assert result["prompt"] == "Test prompt"
        assert result["expected_grounding"] == ["test"]
        assert result["context"] == "Test context"
        assert result["metadata"] == {"key": "value"}

    def test_from_dict(self):
        """Test creating test case from dictionary."""
        data = {
            "test_id": "test_001",
            "grounding_type": "spatial_reasoning",
            "prompt": "Test prompt",
            "expected_grounding": ["test1", "test2"],
            "context": "Test context",
            "metadata": {"key": "value"},
        }

        test_case = SemanticGroundingTestCase.from_dict(data)

        assert test_case.test_id == "test_001"
        assert test_case.grounding_type == GroundingType.SPATIAL_REASONING
        assert test_case.prompt == "Test prompt"
        assert test_case.expected_grounding == ["test1", "test2"]
        assert test_case.context == "Test context"
        assert test_case.metadata == {"key": "value"}

    def test_from_dict_without_optional_fields(self):
        """Test creating test case from dictionary without optional fields."""
        data = {
            "test_id": "test_001",
            "grounding_type": "object_reference",
            "prompt": "Test prompt",
            "expected_grounding": ["test"],
        }

        test_case = SemanticGroundingTestCase.from_dict(data)

        assert test_case.context is None
        assert test_case.metadata is None


class TestGroundingType:
    """Tests for GroundingType enum."""

    def test_grounding_types_exist(self):
        """Test that all expected grounding types exist."""
        assert GroundingType.OBJECT_REFERENCE.value == "object_reference"
        assert GroundingType.SPATIAL_REASONING.value == "spatial_reasoning"
        assert GroundingType.TEMPORAL_REASONING.value == "temporal_reasoning"
        assert GroundingType.CAUSAL_REASONING.value == "causal_reasoning"
        assert GroundingType.CONCEPTUAL_MAPPING.value == "conceptual_mapping"

    def test_grounding_type_from_value(self):
        """Test creating GroundingType from value."""
        grounding_type = GroundingType("object_reference")
        assert grounding_type == GroundingType.OBJECT_REFERENCE
