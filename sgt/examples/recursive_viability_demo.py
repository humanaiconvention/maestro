"""
Demo of Recursive Semantic Viability Testing.

This script simulates a recursive generation chain and monitors information decay
and semantic viability using the new RecursiveEvaluator.
"""

from semantic_grounding import (
    SemanticGroundingTestCase,
    SemanticGroundingEvaluator,
    GroundingType,
)
import json

def run_recursive_demo():
    # 1. Initialize Evaluator (Mock adapter for demo purposes)
    # In a real run, you would provide a model_path
    evaluator = SemanticGroundingEvaluator(
        model_name="qwen",
        model_path="mock", # Use a mock adapter for this demo
        success_threshold=0.7,
    )

    # 2. Create a base test case with high-entropy ground truth
    test_case = SemanticGroundingTestCase(
        test_id="recursive_001",
        grounding_type=GroundingType.RECURSIVE_DRIFT,
        prompt="Describe the ecosystem of a tropical rainforest.",
        expected_grounding=[
            "biodiversity", "canopy", "understory", "emergent layer", 
            "precipitation", "nutrient cycling", "symbiosis", "epiphytes"
        ],
        context="A complex ecological description requiring precise semantic grounding.",
    )

    print(f"Starting Recursive Viability Test: {test_case.test_id}")
    print(f"Initial Grounding Concepts: {len(test_case.expected_grounding)}")
    print("-" * 50)

    # 3. Run Recursive Evaluation
    # For this demo, we'll simulate 5 generations
    # (The evaluate_recursive method will call the model 5 times)
    results = evaluator.evaluate_recursive(test_case, num_generations=5)

    # 4. Display Results
    for res in results:
        print(f"Generation {res.generation}:")
        print(f"  Grounding Score (F1): {res.grounding_score:.2f}")
        print(f"  Epistemic Drift (Et): {res.metrics.epistemic_drift:.4f}")
        print(f"  Corrective Capacity (Ct): {res.metrics.corrective_capacity:.4f}")
        print(f"  IPR: {res.metrics.information_preservation_rate:.1%}")
        print(f"  Status: {res.metrics.status}")
        print(f"  Response Entropy: {res.metrics.shannon_entropy:.2f}")
        print("-" * 30)

    # 5. Final Summary
    final = results[-1]
    print("\nFinal Viability Summary:")
    print(f"  Total Information Lost: {(1 - final.metrics.information_preservation_rate):.1%}")
    print(f"  Final System State: {final.metrics.status}")
    
    if final.metrics.status == "VIABLE":
        print("  SUCCESS: Semantic Viability Condition (Et <= Ct) was satisfied.")
    else:
        print("  WARNING: System failed to preserve semantic grounding across generations.")

if __name__ == "__main__":
    # We need to mock the adapter generate for this to run without a real model
    from semantic_grounding.lm_adapters import BaseLMAdapter
    
    class MockAdapter(BaseLMAdapter):
        def __init__(self, model_path=None):
            self.gen_count = 0
            # Responses that simulate decay by losing keywords
            self.responses = [
                "The rainforest has huge biodiversity and a thick canopy. Nutrient cycling and symbiosis are key.",
                "The forest has biodiversity and a canopy. Precipitation happens often.",
                "It is a forest with biodiversity and rain. Epiphytes are cool.",
                "The forest is green and wet with some biodiversity.",
                "Green forest with trees."
            ]
            
        def generate(self, prompt, max_tokens=256, temperature=0.7):
            resp = self.responses[self.gen_count % len(self.responses)]
            self.gen_count += 1
            return resp
            
    # Injection of mock adapter and simple keyword matcher
    import semantic_grounding.evaluator
    import semantic_grounding.metrics
    
    original_create = semantic_grounding.evaluator.SemanticGroundingEvaluator._create_adapter
    semantic_grounding.evaluator.SemanticGroundingEvaluator._create_adapter = lambda self, name, path: MockAdapter()
    
    # Overwrite concept extractor for the demo to ensure keywords are matched
    def mock_extract(text):
        keywords = [
            "biodiversity", "canopy", "understory", "emergent layer", 
            "precipitation", "nutrient cycling", "symbiosis", "epiphytes"
        ]
        return [k for k in keywords if k in text.lower()]
        
    semantic_grounding.metrics.GroundingMetrics.extract_concepts_from_text = staticmethod(mock_extract)
    
    run_recursive_demo()
