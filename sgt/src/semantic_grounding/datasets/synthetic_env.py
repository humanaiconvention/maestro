"""Controlled synthetic grounding stress test."""

import random
from typing import List, Dict, Tuple
import json

class ColorBindingEnvironment:
    """
    A controlled synthetic task to test symbol-reference binding.
    Symbols (Object_A, Object_B) map to external properties (Colors).
    We test if recursive training causes these bindings to drift.
    """
    
    COLORS = ["red", "blue", "green", "yellow", "purple", "orange"]
    OBJECTS = ["Object_Alpha", "Object_Beta", "Object_Gamma", "Object_Delta"]
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
        # Establish Ground Truth Bindings
        self.ground_truth = {
            obj: random.choice(self.COLORS) for obj in self.OBJECTS
        }
        
    def generate_human_dataset(self, num_samples: int = 50) -> List[Dict[str, str]]:
        """Generates R0 (Generation 0) training data."""
        dataset = []
        for _ in range(num_samples):
            obj = random.choice(self.OBJECTS)
            color = self.ground_truth[obj]
            
            # Simple QA format
            prompt = f"Question: What is the color of {obj}?\nAnswer:"
            response = f" {color}."
            
            dataset.append({
                "prompt": prompt,
                "completion": response,
                "ground_truth_concept": color
            })
        return dataset
        
    def generate_eval_set(self) -> List[Dict[str, str]]:
        """Generates the fixed evaluation set to test semantic preservation."""
        eval_set = []
        for obj, color in self.ground_truth.items():
            prompt = f"Question: What is the color of {obj}?\nAnswer:"
            eval_set.append({
                "prompt": prompt,
                "expected": color
            })
        return eval_set

    def simulate_exogenous_correction(self, synthetic_sample: Dict[str, str]) -> Dict[str, str]:
        """
        Takes a generated synthetic sample and forces it back to ground truth.
        This simulates R4 (External Correction Channel).
        """
        # Find which object was asked about
        for obj in self.OBJECTS:
            if obj in synthetic_sample["prompt"]:
                # Force the completion to be the ground truth color
                synthetic_sample["completion"] = f" {self.ground_truth[obj]}."
                break
        return synthetic_sample
