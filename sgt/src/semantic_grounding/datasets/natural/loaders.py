"""Loaders for standard public NLP datasets to be used as 'Natural' task families."""

from typing import List, Dict, Tuple
from datasets import load_dataset

class NaturalDatasetLoader:
    """
    Opinionated selection of lightweight, public datasets that represent 
    different families of semantic grounding and robustness.
    
    1. Grounded Reference: `ai2_arc` (ARC-Easy) - Requires mapping to real-world science facts.
    2. Fluency/Calibration: `wikitext` (wikitext-2-raw-v1) - Standard OOD perplexity baseline.
    3. OOD/Robustness: `piqa` - Physical interaction reasoning, shifted from standard factual QA.
    """
    
    @staticmethod
    def load_arc_easy(num_train: int = 500, num_eval: int = 100) -> Tuple[List[Dict], List[Dict]]:
        """
        Loads ARC-Easy for Grounded Reference testing.
        Formats as a QA prompt with the correct answer injected.
        """
        try:
            ds = load_dataset("ai2_arc", "ARC-Easy")
            train = ds["train"].shuffle(seed=42).select(range(min(num_train, len(ds["train"]))))
            eval_set = ds["test"].shuffle(seed=42).select(range(min(num_eval, len(ds["test"]))))
            
            def format_arc(item):
                choices = item["choices"]
                correct_idx = choices["label"].index(item["answerKey"])
                correct_text = choices["text"][correct_idx]
                return {
                    "prompt": f"Question: {item['question']}\nAnswer:",
                    "expected": f" {correct_text}",
                    "completion": f" {correct_text}" # For training
                }
                
            return [format_arc(x) for x in train], [format_arc(x) for x in eval_set]
        except Exception as e:
            print(f"Warning: Failed to load ARC-Easy. Ensure internet connection. {e}")
            return [], []

    @staticmethod
    def load_hellaswag(num_eval: int = 100) -> List[Dict]:
        """
        Loads HellaSwag for OOD Robustness testing.
        Tests common-sense situational grounding.
        """
        try:
            # Note: hellaswag has 'train', 'test', 'validation'
            ds = load_dataset("hellaswag")
            eval_set = ds["validation"].shuffle(seed=42).select(range(min(num_eval, len(ds["validation"]))))
            
            def format_hellaswag(item):
                # HellaSwag provides 'ctx' and 4 'endings'. 'label' is index of correct ending.
                ctx = item["ctx"]
                correct_idx = int(item["label"])
                correct_text = item["endings"][correct_idx]
                return {
                    "prompt": f"Context: {ctx}\nNext step:",
                    "expected": f" {correct_text}"
                }
                
            return [format_hellaswag(x) for x in eval_set]
        except Exception as e:
            print(f"Warning: Failed to load HellaSwag. {e}")
            return []

    @staticmethod
    def load_wikitext(num_samples: int = 100) -> List[Dict]:
        """
        Loads Wikitext-2 for pure OOD perplexity tracking.
        This tests 'Stylistic Fluency' completely independent of the training distribution.
        """
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1")
            # Filter empty lines
            valid_texts = [x for x in ds["test"] if len(x["text"].strip()) > 50]
            eval_set = valid_texts[:num_samples]
            
            return [{"prompt": "", "expected": item["text"]} for item in eval_set]
        except Exception as e:
            print(f"Warning: Failed to load Wikitext. {e}")
            return []
