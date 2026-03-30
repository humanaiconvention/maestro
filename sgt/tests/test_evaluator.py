"""
Unit tests for the evaluator module.
"""

import pytest
from unittest.mock import Mock, patch
from semantic_grounding.evaluator import SemanticGroundingEvaluator, EvaluationResult
from semantic_grounding.test_case import SemanticGroundingTestCase, GroundingType
from semantic_grounding.lm_adapters import BaseLMAdapter


class MockAdapter(BaseLMAdapter):
    """Mock adapter for testing."""

    def __init__(self, response_text="The red ball is a spherical object. The table is furniture."):
        self.response_text = response_text
        self.generate_called = False
        self.last_prompt = None

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        self.generate_called = True
        self.last_prompt = prompt
        return self.response_text


class TestSemanticGroundingEvaluator:
    """Tests for SemanticGroundingEvaluator class."""

    def test_init_with_adapter(self):
        """Test initializing evaluator with a custom adapter."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        assert evaluator.model_name == "qwen"
        assert evaluator.adapter == mock_adapter
        assert evaluator.success_threshold == 0.7
        assert evaluator.metrics_mode == "exact"
        assert evaluator.embedding_model_name == "all-MiniLM-L6-v2"
        assert evaluator.embedding_threshold == 0.7
        assert evaluator.structured_output is False

    @patch("semantic_grounding.evaluator_core.QwenHFAdapter")
    def test_init_creates_qwen_adapter(self, mock_qwen_class):
        """Test that initializing with 'qwen' creates QwenHFAdapter."""
        mock_adapter_instance = Mock()
        mock_qwen_class.return_value = mock_adapter_instance

        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            model_path="/fake/path",
        )

        mock_qwen_class.assert_called_once_with(model_path="/fake/path")
        assert evaluator.adapter == mock_adapter_instance

    @patch("semantic_grounding.evaluator_core.LlamaHFAdapter")
    def test_init_creates_llama_adapter(self, mock_llama_class):
        """Test that initializing with 'llama' creates LlamaHFAdapter."""
        mock_adapter_instance = Mock()
        mock_llama_class.return_value = mock_adapter_instance

        evaluator = SemanticGroundingEvaluator(
            model_name="llama",
            model_path="/fake/path",
        )

        mock_llama_class.assert_called_once_with(model_path="/fake/path")
        assert evaluator.adapter == mock_adapter_instance

    def test_init_with_unsupported_model_raises_error(self):
        """Test that initializing with unsupported model raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model name"):
            SemanticGroundingEvaluator(model_name="unsupported-model")

    def test_evaluate(self):
        """Test evaluating a test case."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="The red ball is on the table.",
            expected_grounding=["red ball", "table"],
        )

        result = evaluator.evaluate(test_case)

        assert isinstance(result, EvaluationResult)
        assert result.test_case == test_case
        assert result.model_response == mock_adapter.response_text
        assert isinstance(result.extracted_groundings, list)
        assert result.metrics is not None
        assert isinstance(result.success, bool)
        assert mock_adapter.generate_called

    def test_evaluate_batch(self):
        """Test evaluating multiple test cases."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_cases = [
            SemanticGroundingTestCase(
                test_id=f"test_{i:03d}",
                grounding_type=GroundingType.OBJECT_REFERENCE,
                prompt=f"Test prompt {i}",
                expected_grounding=["test"],
            )
            for i in range(3)
        ]

        results = evaluator.evaluate_batch(test_cases)

        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)

    def test_build_prompt(self):
        """Test building prompt from test case."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
            context="Test context",
        )

        prompt = evaluator._build_prompt(test_case)

        assert "Test context" in prompt
        assert "Test prompt" in prompt
        assert "semantic grounding" in prompt.lower()

    def test_build_prompt_without_context(self):
        """Test building prompt without context."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
        )

        prompt = evaluator._build_prompt(test_case)

        assert "Test prompt" in prompt
        assert "Context:" not in prompt

    def test_get_summary_statistics(self):
        """Test calculating summary statistics."""
        mock_adapter = MockAdapter(response_text="Test response with test")
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
            success_threshold=0.5,
        )

        test_cases = [
            SemanticGroundingTestCase(
                test_id=f"test_{i:03d}",
                grounding_type=GroundingType.OBJECT_REFERENCE,
                prompt="Test prompt",
                expected_grounding=["test"],
            )
            for i in range(3)
        ]

        results = evaluator.evaluate_batch(test_cases)
        summary = evaluator.get_summary_statistics(results)

        assert summary["total_tests"] == 3
        assert "successful_tests" in summary
        assert "success_rate" in summary
        assert "average_precision" in summary
        assert "average_recall" in summary
        assert "average_f1_score" in summary
        assert "average_accuracy" in summary

    def test_get_summary_statistics_empty(self):
        """Test calculating summary statistics with empty results."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        summary = evaluator.get_summary_statistics([])

        assert summary == {}

    def test_embedding_mode_initialization(self):
        """Test initializing evaluator with embedding mode."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
            metrics_mode="embedding",
            embedding_model_name="all-mpnet-base-v2",
            embedding_threshold=0.8,
        )

        assert evaluator.metrics_mode == "embedding"
        assert evaluator.embedding_model_name == "all-mpnet-base-v2"
        assert evaluator.embedding_threshold == 0.8

    @patch("semantic_grounding.metrics.GroundingMetrics.calculate")
    def test_evaluate_with_embedding_mode(self, mock_calculate):
        """Test evaluating with embedding mode."""
        from semantic_grounding.metrics import GroundingMetrics

        # Mock the metrics calculation
        mock_metrics = GroundingMetrics(
            precision=0.8,
            recall=0.9,
            f1_score=0.85,
            accuracy=0.85,
            grounded_concepts=["concept1", "concept2"],
            missing_concepts=[],
            extra_concepts=[],
        )
        mock_calculate.return_value = mock_metrics

        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
            metrics_mode="embedding",
            embedding_model_name="all-MiniLM-L6-v2",
            embedding_threshold=0.75,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["concept1", "concept2"],
        )

        _ = evaluator.evaluate(test_case)

        # Verify that calculate was called with embedding mode
        mock_calculate.assert_called_once()
        call_args = mock_calculate.call_args
        assert call_args.kwargs["mode"] == "embedding"
        assert call_args.kwargs["embedding_model_name"] == "all-MiniLM-L6-v2"
        assert call_args.kwargs["embedding_threshold"] == 0.75

    def test_structured_output_initialization(self):
        """Test initializing evaluator with structured output."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
            structured_output=True,
        )

        assert evaluator.structured_output is True

    def test_extract_groundings_with_json(self):
        """Test extracting groundings from structured JSON output."""
        json_response = '{"groundings": ["red ball", "table", "blue cup"]}'
        mock_adapter = MockAdapter(response_text=json_response)
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
            structured_output=True,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["red ball", "table", "blue cup"],
        )

        result = evaluator.evaluate(test_case)

        assert result.extracted_groundings == ["red ball", "table", "blue cup"]

    def test_extract_groundings_fallback_on_invalid_json(self):
        """Test that invalid JSON falls back to text extraction."""
        invalid_json = "This is not valid JSON"
        mock_adapter = MockAdapter(response_text=invalid_json)
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
            structured_output=True,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
        )

        result = evaluator.evaluate(test_case)

        # Should fall back to text extraction
        assert isinstance(result.extracted_groundings, list)
        assert len(result.extracted_groundings) > 0


class TestEvaluationResult:
    """Tests for EvaluationResult class."""

    def test_to_dict(self):
        """Test converting evaluation result to dictionary."""
        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
        )

        from semantic_grounding.metrics import GroundingMetrics

        metrics = GroundingMetrics.calculate(["test"], ["test"])

        result = EvaluationResult(
            test_case=test_case,
            model_response="Test response",
            extracted_groundings=["test"],
            metrics=metrics,
            success=True,
            metadata={"key": "value"},
        )

        result_dict = result.to_dict()

        assert result_dict["model_response"] == "Test response"
        assert result_dict["extracted_groundings"] == ["test"]
        assert result_dict["success"] is True
        assert result_dict["metadata"] == {"key": "value"}
        assert "test_case" in result_dict
        assert "metrics" in result_dict

    def test_to_dict_with_landscape_metrics(self):
        """Test converting evaluation result with landscape metrics to dictionary."""
        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
        )

        from semantic_grounding.metrics import GroundingMetrics
        from semantic_grounding.hessian_diagnostics import LandscapeMetrics

        metrics = GroundingMetrics.calculate(["test"], ["test"])
        landscape_metrics = LandscapeMetrics(
            condition_number=100.0,
            spectral_sharpness=0.5,
            min_eigenvalue=0.01,
            eigenvalue_spread=0.49,
            power_law_compliance=0.2,
            has_negative_eigenvalues=False,
            stability_score=0.9,
        )

        result = EvaluationResult(
            test_case=test_case,
            model_response="Test response",
            extracted_groundings=["test"],
            metrics=metrics,
            success=True,
            landscape_metrics=landscape_metrics,
            failure_mode_analysis={"overall_risk": "SAFE"},
        )

        result_dict = result.to_dict()

        assert "landscape_metrics" in result_dict
        assert "failure_mode_analysis" in result_dict
        assert result_dict["landscape_metrics"]["condition_number"] == 100.0
        assert result_dict["failure_mode_analysis"]["overall_risk"] == "SAFE"


class TestBackwardCompatibility:
    """Tests for backward compatibility with existing code."""

    def test_evaluate_without_landscape_metrics(self):
        """Test that evaluate works without landscape metrics parameter (backward compatibility)."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="The red ball is on the table.",
            expected_grounding=["red ball", "table"],
        )

        # Call without the new parameter (should work as before)
        result = evaluator.evaluate(test_case)

        assert isinstance(result, EvaluationResult)
        assert result.landscape_metrics is None
        assert result.failure_mode_analysis is None

    def test_evaluate_with_landscape_metrics_false(self):
        """Test that evaluate with compute_landscape_metrics=False works."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="The red ball is on the table.",
            expected_grounding=["red ball", "table"],
        )

        result = evaluator.evaluate(test_case, compute_landscape_metrics=False)

        assert isinstance(result, EvaluationResult)
        assert result.landscape_metrics is None

    def test_evaluate_with_landscape_metrics_true(self):
        """Test that evaluate with compute_landscape_metrics=True computes metrics."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="The red ball is on the table.",
            expected_grounding=["red ball", "table"],
        )

        result = evaluator.evaluate(test_case, compute_landscape_metrics=True)

        assert isinstance(result, EvaluationResult)
        assert result.landscape_metrics is not None
        assert result.failure_mode_analysis is not None
        assert "overall_risk" in result.failure_mode_analysis
        assert result.landscape_metrics.condition_number > 0

    def test_evaluate_batch_without_landscape_metrics(self):
        """Test that evaluate_batch works without landscape metrics parameter."""
        mock_adapter = MockAdapter()
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_cases = [
            SemanticGroundingTestCase(
                test_id=f"test_{i:03d}",
                grounding_type=GroundingType.OBJECT_REFERENCE,
                prompt=f"Test prompt {i}",
                expected_grounding=["test"],
            )
            for i in range(3)
        ]

        # Call without the new parameter (should work as before)
        results = evaluator.evaluate_batch(test_cases)

        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert all(r.landscape_metrics is None for r in results)

    def test_evaluation_result_fields_optional(self):
        """Test that new fields in EvaluationResult are optional."""
        test_case = SemanticGroundingTestCase(
            test_id="test_001",
            grounding_type=GroundingType.OBJECT_REFERENCE,
            prompt="Test prompt",
            expected_grounding=["test"],
        )

        from semantic_grounding.metrics import GroundingMetrics

        metrics = GroundingMetrics.calculate(["test"], ["test"])

        # Create result without new fields
        result = EvaluationResult(
            test_case=test_case,
            model_response="Test response",
            extracted_groundings=["test"],
            metrics=metrics,
            success=True,
        )

        assert result.landscape_metrics is None
        assert result.failure_mode_analysis is None

    def test_evaluate_recursive(self):
        """Test evaluating semantic grounding recursively."""
        mock_adapter = MockAdapter(response_text="grounded concept")
        evaluator = SemanticGroundingEvaluator(
            model_name="qwen",
            adapter=mock_adapter,
        )

        test_case = SemanticGroundingTestCase(
            test_id="test_recursive_001",
            grounding_type=GroundingType.RECURSIVE_DRIFT,
            prompt="Initial prompt",
            expected_grounding=["grounded concept"],
            context="Context for recursive test",
        )

        # Run for 3 generations
        results = evaluator.evaluate_recursive(test_case, num_generations=3)

        assert len(results) == 3
        for i, res in enumerate(results):
            assert res.generation == i
            assert res.response == "grounded concept"
            assert res.grounding_score == 1.0
            assert res.metrics.information_preservation_rate == 1.0
            assert res.metrics.status.value == "VIABLE"
            assert "Previous observation: grounded concept" in mock_adapter.last_prompt if i > 0 else True
