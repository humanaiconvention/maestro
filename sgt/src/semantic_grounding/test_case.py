"""
Test case framework for semantic grounding evaluation.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum


class GroundingType(Enum):
    """Types of semantic grounding to test."""

    OBJECT_REFERENCE = "object_reference"  # Grounding object references to real entities
    SPATIAL_REASONING = "spatial_reasoning"  # Understanding spatial relationships
    TEMPORAL_REASONING = "temporal_reasoning"  # Understanding temporal relationships
    CAUSAL_REASONING = "causal_reasoning"  # Understanding cause and effect
    CONCEPTUAL_MAPPING = "conceptual_mapping"  # Mapping abstract concepts to concrete examples
    RECURSIVE_DRIFT = "recursive_drift"  # Testing semantic drift over multiple generations
    INFORMATION_PRESERVATION = "information_preservation"  # Testing retention of ground truth
    EPISTEMIC_GROUNDING = "epistemic_grounding"  # Testing exogenous corrective grounding


@dataclass
class SemanticGroundingTestCase:
    """
    A test case for evaluating semantic grounding.

    Attributes:
        test_id: Unique identifier for the test case
        grounding_type: The type of semantic grounding being tested
        prompt: The input prompt to test grounding capabilities
        expected_grounding: Expected grounding elements or concepts
        context: Optional context information for the test
        metadata: Additional metadata for the test case
    """

    test_id: str
    grounding_type: GroundingType
    prompt: str
    expected_grounding: List[str]
    context: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate test case after initialization."""
        if not self.test_id:
            raise ValueError("test_id cannot be empty")
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if not self.expected_grounding:
            raise ValueError("expected_grounding cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        """Convert test case to dictionary representation."""
        return {
            "test_id": self.test_id,
            "grounding_type": self.grounding_type.value,
            "prompt": self.prompt,
            "expected_grounding": self.expected_grounding,
            "context": self.context,
            "metadata": self.metadata or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SemanticGroundingTestCase":
        """Create test case from dictionary representation."""
        return cls(
            test_id=data["test_id"],
            grounding_type=GroundingType(data["grounding_type"]),
            prompt=data["prompt"],
            expected_grounding=data["expected_grounding"],
            context=data.get("context"),
            metadata=data.get("metadata"),
        )
