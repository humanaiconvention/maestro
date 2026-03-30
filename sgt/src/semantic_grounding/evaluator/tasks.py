"""Rigorous task evaluators ensuring immutable hold-out testing."""

from typing import List, Dict, Any
from datasets import load_dataset
import torch

class BenchmarkRegistry:
    """Loads and holds immutable external benchmarks."""
    
    def __init__(self, eval_samples: int = 100, seed: int = 42):
        self.eval_samples = eval_samples
        self.seed = seed
        self.datasets = {}
        self._load_all()
        
    def _load_all(self):
        print("Loading immutable evaluation sets...")
        self.datasets["arc_easy"] = self._load_arc()
        self.datasets["gsm8k"] = self._load_gsm8k()
        self.datasets["wikitext"] = self._load_wikitext()
        
    def _load_arc(self) -> List[Dict[str, str]]:
        try:
            ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
            ds = ds.shuffle(seed=self.seed).select(range(min(self.eval_samples, len(ds))))
            out = []
            for item in ds:
                choices = item["choices"]
                correct_idx = choices["label"].index(item["answerKey"])
                out.append({
                    "prompt": f"Question: {item['question']}\nAnswer:",
                    "expected": f" {choices['text'][correct_idx]}"
                })
            return out
        except Exception as e:
            print(f"Failed to load ARC: {e}")
            return []

    def _load_gsm8k(self) -> List[Dict[str, str]]:
        try:
            ds = load_dataset("gsm8k", "main", split="test")
            ds = ds.shuffle(seed=self.seed).select(range(min(self.eval_samples, len(ds))))
            out = []
            for item in ds:
                # Extract just the final number for simpler evaluation if needed, 
                # or require the whole answer. For now, exact match of the answer block.
                out.append({
                    "prompt": f"Question: {item['question']}\nAnswer:",
                    "expected": f" {item['answer']}"
                })
            return out
        except Exception as e:
            print(f"Failed to load GSM8K: {e}")
            return []

    def _load_wikitext(self) -> List[Dict[str, str]]:
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            valid_texts = [x for x in ds if len(x["text"].strip()) > 50]
            valid_texts = valid_texts[:self.eval_samples]
            return [{"prompt": "", "expected": item["text"]} for item in valid_texts]
        except Exception as e:
            print(f"Failed to load Wikitext: {e}")
            return []

    def evaluate_model(self, trainer, generation_idx: int) -> Dict[str, Any]:
        """Runs the model against all loaded benchmarks."""
        results = {"generation": generation_idx}
        
        # 1. Fluency (Perplexity on Wikitext)
        if self.datasets.get("wikitext"):
            ppl = trainer.calculate_perplexity(self.datasets["wikitext"])
            results["val_perplexity"] = ppl
            
        # 2. Grounding & Reasoning (Accuracy)
        for task in ["arc_easy", "gsm8k"]:
            if not self.datasets.get(task): continue
            
            ds = self.datasets[task]
            prompts = [d["prompt"] for d in ds]
            
            # Use max_new_tokens conservatively, though GSM8K needs more.
            # We'll use 50 as a generic middle ground for now, can be configured.
            predictions = trainer.generate_synthetic_data(prompts, max_new_tokens=50)
            
            correct = 0
            for pred, expected in zip(predictions, ds):
                # Stricter match: check if the completion STARTS WITH the expected answer
                if pred["completion"].strip().lower().startswith(expected["expected"].strip().lower()):
                    correct += 1
                    
            acc = correct / len(ds) if len(ds) > 0 else 0.0
            results[f"{task}_accuracy"] = acc
            
            # Calculate task-specific perplexity as well
            results[f"{task}_perplexity"] = trainer.calculate_perplexity(ds)

        return results
