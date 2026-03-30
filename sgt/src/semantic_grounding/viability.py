"""
Semantic Viability and Information Preservation in Recursive Systems.

This module implements the theoretical framework for testing how information 
decays in recursive generative systems and whether exogenous grounding 
can satisfy the Semantic Viability Condition (Et <= Ct).
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from enum import Enum


class ViabilityStatus(Enum):
    """Classification of semantic viability."""
    VIABLE = "VIABLE"      # Ct >= Et: Corrective capacity keeps pace with drift
    UNSTABLE = "UNSTABLE"  # Ct < Et: System is slowly drifting towards collapse
    COLLAPSED = "COLLAPSED" # Ct << Et: Information preservation has failed


@dataclass
class ViabilityMetrics:
    """
    Metrics for the Semantic Viability Condition (Et <= Ct).
    
    Attributes:
        epistemic_drift (Et): Rate of internal semantic decay or hallucination amplification.
        corrective_capacity (Ct): Rate of external grounding or corrective information.
        viability_ratio: Ct / Et (Values >= 1.0 indicate viability).
        information_preservation_rate (IPR): Percentage of initial ground truth bits retained.
        shannon_entropy: Semantic entropy of the current generation.
        kl_divergence: Divergence from the original ground truth distribution.
    """
    epistemic_drift: float
    corrective_capacity: float
    viability_ratio: float
    information_preservation_rate: float
    shannon_entropy: float
    kl_divergence: float
    status: ViabilityStatus

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epistemic_drift": self.epistemic_drift,
            "corrective_capacity": self.corrective_capacity,
            "viability_ratio": self.viability_ratio,
            "information_preservation_rate": self.information_preservation_rate,
            "shannon_entropy": self.shannon_entropy,
            "kl_divergence": self.kl_divergence,
            "status": self.status.value,
        }


@dataclass
class RecursiveStepResult:
    """Result of a single step in a recursive generation chain."""
    generation: int
    response: str
    metrics: ViabilityMetrics
    grounding_score: float # F1 score of grounding in this generation


class RecursiveEvaluator:
    """
    Evaluator for monitoring information preservation across recursive generations.
    """
    
    @staticmethod
    def calculate_epistemic_drift(initial_f1: float, current_f1: float, generation: int) -> float:
        """
        Calculate Et (Epistemic Drift) as the average decay per generation.
        """
        if generation == 0:
            return 0.0
        return (initial_f1 - current_f1) / generation

    @staticmethod
    def calculate_corrective_capacity(grounding_score: float, precision: float) -> float:
        """
        Calculate Ct (Corrective Capacity) based on current grounding performance.
        Ct is boosted by high precision (accuracy of grounded elements).
        """
        return grounding_score * precision

    @staticmethod
    def determine_viability(et: float, ct: float) -> ViabilityStatus:
        """Apply the Semantic Viability Condition."""
        if et <= 0:
            return ViabilityStatus.VIABLE
        
        ratio = ct / et
        if ratio >= 1.0:
            return ViabilityStatus.VIABLE
        elif ratio >= 0.5:
            return ViabilityStatus.UNSTABLE
        else:
            return ViabilityStatus.COLLAPSED

    @staticmethod
    def calculate_shannon_entropy(text: str) -> float:
        """Calculate basic Shannon entropy of the response text."""
        if not text:
            return 0.0
        
        # Tokenize simply for now
        tokens = text.lower().split()
        if not tokens:
            return 0.0
            
        _, counts = np.unique(tokens, return_counts=True)
        probs = counts / len(tokens)
        return -np.sum(probs * np.log2(probs))

    @staticmethod
    def calculate_kl_divergence(p_reality: List[float], p_t: List[float]) -> float:
        """
        Calculate KL-Divergence between ground truth and current generation.
        Note: requires probability distributions of concepts.
        """
        p = np.array(p_reality)
        q = np.array(p_t)
        
        # Add epsilon to avoid log(0)
        p = p + 1e-12
        q = q + 1e-12
        
        # Re-normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return np.sum(p * np.log2(p / q))
