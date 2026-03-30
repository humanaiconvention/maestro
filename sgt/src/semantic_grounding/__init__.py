"""
Semantic Grounding Test Module

A module to test semantic grounding capabilities using local language models (QWEN/LLAMA).
Semantic grounding refers to the ability to connect abstract concepts, symbols,
or language to their real-world meanings and contexts.
"""

__version__ = "0.1.0"
__author__ = "HumanAI Convention"

from .test_case import SemanticGroundingTestCase, GroundingType
from .evaluator_core import SemanticGroundingEvaluator, EvaluationResult
from .metrics import GroundingMetrics
from .lm_adapters import BaseLMAdapter, QwenHFAdapter, LlamaHFAdapter
from .hessian_diagnostics import (
    LandscapeMetrics,
    HessianCompute,
    FailureModeClassifier,
    RiskLevel,
    compute_landscape_metrics,
)

__all__ = [
    "SemanticGroundingTestCase",
    "GroundingType",
    "SemanticGroundingEvaluator",
    "EvaluationResult",
    "GroundingMetrics",
    "BaseLMAdapter",
    "QwenHFAdapter",
    "LlamaHFAdapter",
    "LandscapeMetrics",
    "HessianCompute",
    "FailureModeClassifier",
    "RiskLevel",
    "compute_landscape_metrics",
]
