"""
Evaluator for semantic grounding using local language models (QWEN/LLAMA).
"""

import json
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from .test_case import SemanticGroundingTestCase
from .metrics import GroundingMetrics
from .lm_adapters import BaseLMAdapter, QwenHFAdapter, LlamaHFAdapter
from .viability import (
    RecursiveEvaluator,
    RecursiveStepResult,
    ViabilityMetrics,
    ViabilityStatus,
)


@dataclass
class EvaluationResult:
    """
    Result of a semantic grounding evaluation.

    Attributes:
        test_case: The test case that was evaluated
        model_response: Raw response from the model
        extracted_groundings: Grounding concepts extracted from the response
        metrics: Calculated metrics for the evaluation
        success: Whether the evaluation passed a threshold
        metadata: Additional metadata about the evaluation
        landscape_metrics: Optional Hessian-based landscape diagnostics
        failure_mode_analysis: Optional failure mode classification results
    """

    test_case: SemanticGroundingTestCase
    model_response: str
    extracted_groundings: List[str]
    metrics: GroundingMetrics
    success: bool
    metadata: Optional[Dict[str, Any]] = None
    landscape_metrics: Optional[Any] = None  # LandscapeMetrics from hessian_diagnostics
    failure_mode_analysis: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        result = {
            "test_case": self.test_case.to_dict(),
            "model_response": self.model_response,
            "extracted_groundings": self.extracted_groundings,
            "metrics": self.metrics.to_dict(),
            "success": self.success,
            "metadata": self.metadata or {},
        }

        if self.landscape_metrics is not None:
            result["landscape_metrics"] = self.landscape_metrics.to_dict()

        if self.failure_mode_analysis is not None:
            result["failure_mode_analysis"] = self.failure_mode_analysis

        return result


class SemanticGroundingEvaluator:
    """
    Evaluator for testing semantic grounding using local language models (QWEN/LLAMA).
    """

    def __init__(
        self,
        model_name: str = "qwen",
        model_path: Optional[str] = None,
        success_threshold: float = 0.7,
        adapter: Optional[BaseLMAdapter] = None,
        metrics_mode: str = "exact",
        embedding_model_name: str = "all-MiniLM-L6-v2",
        embedding_threshold: float = 0.7,
        structured_output: bool = False,
    ):
        """
        Initialize the semantic grounding evaluator.

        Args:
            model_name: Name or type of model to use ('qwen' or 'llama')
            model_path: Path to local model weights
                (or use env vars QWEN_MODEL_PATH/LLAMA_MODEL_PATH)
            success_threshold: F1 score threshold for considering a test successful
            adapter: Optional pre-initialized adapter (for testing/advanced use)
            metrics_mode: Matching mode - "exact" for n-gram or "embedding"
                for semantic similarity
            embedding_model_name: Name of sentence-transformers model
                (used when metrics_mode="embedding")
            embedding_threshold: Cosine similarity threshold for embedding
                matching (default: 0.7)
            structured_output: If True, expect model to return JSON with
                "groundings" key
        """
        self.model_name = model_name.lower()
        self.success_threshold = success_threshold
        self.metrics_mode = metrics_mode
        self.embedding_model_name = embedding_model_name
        self.embedding_threshold = embedding_threshold
        self.structured_output = structured_output

        # Use provided adapter or create one based on model_name
        if adapter is not None:
            self.adapter = adapter
        else:
            self.adapter = self._create_adapter(model_name, model_path)

    def _create_adapter(self, model_name: str, model_path: Optional[str]) -> BaseLMAdapter:
        """
        Create appropriate adapter based on model name.

        Args:
            model_name: Name or type of model ('qwen', 'llama', etc.)
            model_path: Path to local model weights

        Returns:
            Initialized adapter instance
        """
        model_name_lower = model_name.lower()

        if "qwen" in model_name_lower:
            return QwenHFAdapter(model_path=model_path)
        elif "llama" in model_name_lower:
            return LlamaHFAdapter(model_path=model_path)
        else:
            raise ValueError(
                f"Unsupported model name: {model_name}. "
                "Supported models: 'qwen', 'llama' (or variants containing these names)"
            )

    def evaluate(
        self,
        test_case: SemanticGroundingTestCase,
        extract_groundings: bool = True,
        compute_landscape_metrics: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate a semantic grounding test case.

        Args:
            test_case: The test case to evaluate
            extract_groundings: Whether to automatically extract groundings from response
            compute_landscape_metrics: Whether to compute Hessian-based landscape diagnostics
                Note: Requires model adapter to support Hessian computation

        Returns:
            EvaluationResult with metrics and details
        """
        # Build the prompt with context if available
        full_prompt = self._build_prompt(test_case)

        # Generate response using adapter
        model_response = self.adapter.generate(full_prompt, max_tokens=256, temperature=0.0)

        # Extract groundings from the response
        extracted_groundings = self._extract_groundings(model_response, extract_groundings)

        # Calculate metrics using the configured mode
        metrics = GroundingMetrics.calculate(
            expected_groundings=test_case.expected_grounding,
            actual_groundings=extracted_groundings,
            case_sensitive=False,
            mode=self.metrics_mode,
            embedding_model_name=self.embedding_model_name,
            embedding_threshold=self.embedding_threshold,
        )

        # Determine success based on F1 score
        success = metrics.f1_score >= self.success_threshold

        # Prepare metadata
        result_metadata = {
            "model_name": self.model_name,
            "success_threshold": self.success_threshold,
        }

        # Compute landscape metrics if requested
        landscape_metrics = None
        failure_mode_analysis = None

        if compute_landscape_metrics:
            # Import here to avoid circular dependency and allow graceful degradation
            try:
                from .hessian_diagnostics import (
                    HessianCompute,
                    FailureModeClassifier,
                    compute_landscape_metrics as compute_lm,
                )

                # Attempt real spectral computation via PRISM; fall back to mock if unavailable.
                # PRISM uses compute_spectral_metrics() + compute_top_eigenvalues() on the
                # model's mid-layer hidden states — a Hessian-free approximation that captures
                # the activation covariance spectrum without requiring backpropagation.
                prism_used = False
                eigenvalues = None
                if getattr(self, "model", None) is not None:
                    try:
                        import sys, os
                        _prism_src = os.path.normpath(
                            os.path.join(os.path.dirname(__file__),
                                         "../../../../prism/src")
                        )
                        if os.path.isdir(_prism_src) and _prism_src not in sys.path:
                            sys.path.insert(0, _prism_src)
                        from prism.analysis import compute_top_eigenvalues
                        import torch

                        # Encode the original prompt (test_case.prompt) to get hidden states
                        if hasattr(self, "tokenizer") and self.tokenizer is not None:
                            enc = self.tokenizer(
                                test_case.prompt,
                                return_tensors="pt",
                                truncation=True,
                                max_length=256,
                            )
                            enc = {k: v.to(next(self.model.parameters()).device) for k, v in enc.items()}
                            with torch.no_grad():
                                out = self.model(**enc, output_hidden_states=True)
                            mid = len(out.hidden_states) // 2
                            hidden = out.hidden_states[mid][0].float()  # (seq, dim)
                            k = min(10, hidden.shape[0], hidden.shape[-1])
                            eigenvalues = compute_top_eigenvalues(hidden, k=k)
                            prism_used = True
                            result_metadata["landscape_metrics_note"] = (
                                "Eigenvalue spectrum computed via PRISM spectral analysis "
                                "(activation covariance at mid-layer)."
                            )
                    except Exception as _prism_err:
                        pass  # Fall through to mock below

                if eigenvalues is None:
                    # Fallback mock: deterministic spectrum based on response length
                    eigenvalues = [float(i + 1) for i in range(min(len(model_response), 10))]
                    if not prism_used:
                        result_metadata["landscape_metrics_note"] = (
                            "Using mock eigenvalue spectrum (PRISM not available or model/tokenizer "
                            "not set on evaluator). For real metrics install spectral-microscope and "
                            "set evaluator.model + evaluator.tokenizer."
                        )

                landscape_metrics = compute_lm(eigenvalues)
                failure_mode_analysis = FailureModeClassifier().generate_diagnostic_report(landscape_metrics)
            except ImportError:
                result_metadata["landscape_metrics_error"] = (
                    "Hessian diagnostics module not available"
                )

        return EvaluationResult(
            test_case=test_case,
            model_response=model_response,
            extracted_groundings=extracted_groundings,
            metrics=metrics,
            success=success,
            metadata=result_metadata,
            landscape_metrics=landscape_metrics,
            failure_mode_analysis=failure_mode_analysis,
        )

    def evaluate_batch(
        self,
        test_cases: List[SemanticGroundingTestCase],
        compute_landscape_metrics: bool = False,
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple test cases.

        Args:
            test_cases: List of test cases to evaluate
            compute_landscape_metrics: Whether to compute Hessian-based landscape diagnostics

        Returns:
            List of evaluation results
        """
        results = []
        for test_case in test_cases:
            result = self.evaluate(test_case, compute_landscape_metrics=compute_landscape_metrics)
            results.append(result)
        return results

    def _extract_groundings(self, model_response: str, extract_groundings: bool) -> List[str]:
        """
        Extract groundings from model response.

        If structured_output is enabled, tries to parse JSON with "groundings" key.
        Otherwise, uses simple text extraction.

        Args:
            model_response: Raw model response
            extract_groundings: Whether to extract groundings

        Returns:
            List of extracted grounding concepts
        """
        if not extract_groundings:
            return [model_response]

        # Try structured JSON parsing if enabled
        if self.structured_output:
            try:
                # Try to find and extract complete JSON object
                # Look for JSON that contains "groundings" key
                # First try to parse entire response as JSON
                parsed = json.loads(model_response)
                if "groundings" in parsed and isinstance(parsed["groundings"], list):
                    return parsed["groundings"]
            except (json.JSONDecodeError, ValueError):
                # If full response isn't JSON, try to find JSON substring
                # Use a simple heuristic: find { followed by "groundings"
                try:
                    start_idx = model_response.find("{")
                    if start_idx != -1:
                        # Find the matching closing brace by counting braces
                        brace_count = 0
                        end_idx = start_idx
                        for i in range(start_idx, len(model_response)):
                            if model_response[i] == "{":
                                brace_count += 1
                            elif model_response[i] == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break

                        json_str = model_response[start_idx:end_idx]
                        parsed = json.loads(json_str)
                        if "groundings" in parsed and isinstance(parsed["groundings"], list):
                            return parsed["groundings"]
                except (json.JSONDecodeError, ValueError):
                    # Fall back to text extraction if JSON parsing fails
                    pass

        # Default: extract concepts from text
        return GroundingMetrics.extract_concepts_from_text(model_response)

    def evaluate_recursive(
        self,
        test_case: SemanticGroundingTestCase,
        num_generations: int = 5,
        compute_landscape_metrics: bool = False,
    ) -> List[RecursiveStepResult]:
        """
        Evaluate semantic grounding over multiple recursive generations.
        Each generation's output becomes the prompt for the next.

        Args:
            test_case: The initial test case
            num_generations: Number of recursive steps to perform
            compute_landscape_metrics: Whether to compute Hessian metrics at each step

        Returns:
            List of RecursiveStepResult for each generation
        """
        results = []
        current_prompt = self._build_prompt(test_case)
        initial_f1 = 0.0

        for gen in range(num_generations):
            # Generate response
            model_response = self.adapter.generate(current_prompt, max_tokens=256, temperature=0.7)

            # Extract and calculate base grounding metrics
            extracted = self._extract_groundings(model_response, True)
            base_metrics = GroundingMetrics.calculate(
                expected_groundings=test_case.expected_grounding,
                actual_groundings=extracted,
                mode=self.metrics_mode,
                embedding_threshold=self.embedding_threshold,
            )

            if gen == 0:
                initial_f1 = base_metrics.f1_score

            # Calculate Viability Metrics (Et, Ct, etc.)
            et = RecursiveEvaluator.calculate_epistemic_drift(initial_f1, base_metrics.f1_score, gen)
            ct = RecursiveEvaluator.calculate_corrective_capacity(
                base_metrics.f1_score, base_metrics.precision
            )
            status = RecursiveEvaluator.determine_viability(et, ct)
            
            # Information Preservation Rate (normalized F1)
            ipr = base_metrics.f1_score / (initial_f1 if initial_f1 > 0 else 1.0)
            
            # Shannon Entropy of the response
            entropy = RecursiveEvaluator.calculate_shannon_entropy(model_response)

            # Basic KL Divergence calculation based on concept preservation
            # p_reality is uniform over expected concepts
            # p_t is 1 if concept present, 0 if missing (normalized)
            n_expected = len(test_case.expected_grounding)
            p_reality = [1.0 / n_expected] * n_expected
            p_t = [1.0 if c in base_metrics.grounded_concepts else 0.0 for c in test_case.expected_grounding]
            
            kl_div = RecursiveEvaluator.calculate_kl_divergence(p_reality, p_t)

            viability_metrics = ViabilityMetrics(
                epistemic_drift=et,
                corrective_capacity=ct,
                viability_ratio=ct / (et if et > 0 else 1e-9),
                information_preservation_rate=ipr,
                shannon_entropy=entropy,
                kl_divergence=kl_div,
                status=status,
            )

            step_result = RecursiveStepResult(
                generation=gen,
                response=model_response,
                metrics=viability_metrics,
                grounding_score=base_metrics.f1_score,
            )
            results.append(step_result)

            # Prepare next prompt: feed back the model's own grounding interpretation
            current_prompt = (
                f"Context: {test_case.context}\n"
                f"Previous observation: {model_response}\n"
                "Task: Re-evaluate and refine the semantic grounding for this observation. "
                "Ensure all real-world concepts are correctly preserved."
            )

        return results

    def _build_prompt(self, test_case: SemanticGroundingTestCase) -> str:
        """
        Build a prompt for the model based on the test case.

        Args:
            test_case: The test case to build a prompt for

        Returns:
            Formatted prompt string
        """
        prompt_parts = []

        if test_case.context:
            prompt_parts.append(f"Context: {test_case.context}")

        prompt_parts.append(
            "Task: Demonstrate semantic grounding for the following prompt. "
            "Identify and explain the real-world concepts, entities, and relationships."
        )

        if self.structured_output:
            prompt_parts.append(
                '\nPlease return your response as JSON with a "groundings" key containing '
                "a list of the key concepts. For example: "
                '{"groundings": ["concept1", "concept2", "concept3"]}'
            )

        prompt_parts.append(f"\nPrompt: {test_case.prompt}")

        return "\n".join(prompt_parts)

    def get_summary_statistics(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """
        Calculate summary statistics across multiple evaluation results.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with summary statistics
        """
        if not results:
            return {}

        total = len(results)
        successful = sum(1 for r in results if r.success)

        avg_precision = sum(r.metrics.precision for r in results) / total
        avg_recall = sum(r.metrics.recall for r in results) / total
        avg_f1 = sum(r.metrics.f1_score for r in results) / total
        avg_accuracy = sum(r.metrics.accuracy for r in results) / total

        return {
            "total_tests": total,
            "successful_tests": successful,
            "success_rate": successful / total,
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1_score": avg_f1,
            "average_accuracy": avg_accuracy,
        }
