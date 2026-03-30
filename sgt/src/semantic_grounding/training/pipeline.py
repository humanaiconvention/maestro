"""PEFT / LoRA Training wrapper for recursive generations."""

import torch
from typing import List, Dict, Optional, Tuple
import os
import json

class RecursiveTrainer:
    """
    Wrapper for fine-tuning models iteratively.
    Uses PEFT/LoRA if configured to keep memory requirements low.
    """
    
    def __init__(self, model_name: str, config: dict, dtype: str = "float16"):
        self.base_model_name = model_name
        self.config = config
        self.dtype_str = dtype
        
        # We only import heavy huggingface modules when needed
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing RecursiveTrainer on {self.device} with dtype {self.dtype_str}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Pad token is required for batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self._load_model()

    def _load_model(self, adapter_path: Optional[str] = None):
        """Loads the base model, and optionally attaches a LoRA adapter."""
        from transformers import AutoModelForCausalLM
        from peft import get_peft_model, LoraConfig, TaskType, PeftModel
        
        if hasattr(self, "model") and self.model is not None:
            # If we already have a model, we just want to swap or load the adapter
            if adapter_path:
                print(f"Reloading LoRA adapter from {adapter_path}")
                # Properly unload old adapter if it exists
                if isinstance(self.model, PeftModel):
                    self.model = self.model.unload() # Back to base model
                
                self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True)
            return

        # Map dtype string to torch.dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32
        }
        target_dtype = dtype_map.get(self.dtype_str, torch.float16)

        # Initial load
        print(f"Loading base model: {self.base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            device_map="auto",
            torch_dtype=target_dtype
        )
        
        if self.config.get("use_lora", True):
            if adapter_path:
                print(f"Loading existing LoRA adapter from {adapter_path}")
                self.model = PeftModel.from_pretrained(self.model, adapter_path, is_trainable=True)
            else:
                print("Initializing fresh LoRA adapter")
                lora_config = LoraConfig(
                    r=self.config.get("lora_r", 8),
                    lora_alpha=self.config.get("lora_alpha", 32),
                    target_modules=["q_proj", "v_proj"], 
                    lora_dropout=0.05,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM
                )
                self.model = get_peft_model(self.model, lora_config)

    def train_generation(self, dataset: List[Dict[str, str]], output_dir: str):
        """Trains the model on the provided dataset and saves the new adapter."""
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from datasets import Dataset
        
        # Prepare HF Dataset
        texts = [d["prompt"] + d["completion"] for d in dataset]
        hf_dataset = Dataset.from_dict({"text": texts})
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
            
        tokenized_dataset = hf_dataset.map(tokenize_function, batched=True, remove_columns=["text"], desc="Tokenizing dataset")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            per_device_train_batch_size=self.config.get("batch_size", 2),
            learning_rate=float(self.config.get("learning_rate", 2e-4)),
            num_train_epochs=self.config.get("epochs_per_generation", 1),
            logging_steps=10,
            save_strategy="no", 
            remove_unused_columns=False,
            report_to="none"
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        )
        
        print(f"Starting training for {output_dir}...")
        trainer.train()
        
        print(f"Saving generation adapter to {output_dir}")
        self.model.save_pretrained(output_dir)
        
    def generate_synthetic_data(self, prompts: List[str], max_new_tokens: int = 20) -> List[Dict[str, str]]:
        """Uses the current model to generate synthetic completions."""
        self.model.eval()
        results = []
        
        print(f"Generating {len(prompts)} synthetic completions...", flush=True)
        for i, prompt in enumerate(prompts):
            if i > 0 and i % 25 == 0:
                print(f"  ...generated {i}/{len(prompts)}", flush=True)
                
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Extract just the completion
            full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            completion = full_text[len(prompt):]
            
            results.append({
                "prompt": prompt,
                "completion": completion
            })
            
        print(f"Finished generating {len(prompts)} completions.", flush=True)
        return results

    def calculate_perplexity(self, dataset: List[Dict[str, str]]) -> float:
        """Calculates validation perplexity on a dataset."""
        self.model.eval()
        total_loss = 0.0
        total_length = 0
        
        with torch.no_grad():
            for item in dataset:
                text = item["prompt"] + item.get("expected", item.get("completion", ""))
                inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                
                # loss is the mean negative log likelihood of the tokens
                loss = outputs.loss
                seq_len = inputs["input_ids"].size(1)
                
                total_loss += loss.item() * seq_len
                total_length += seq_len
                
        if total_length == 0:
            return float("inf")
            
        return torch.exp(torch.tensor(total_loss / total_length)).item()

    def get_logits_for_eval(self, dataset: List[Dict[str, str]]) -> Tuple[List[float], List[int]]:
        """Returns max probabilities and correct-index binary labels for calibration analysis."""
        self.model.eval()
        probs_out = []
        labels_out = []
        
        with torch.no_grad():
            for item in dataset:
                # Use the logit at the last prompt position to predict the first completion token
                full_text = item["prompt"] + item.get("expected", "")
                inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
                prompt_inputs = self.tokenizer(item["prompt"], return_tensors="pt")
                prompt_len = prompt_inputs.input_ids.size(1)
                outputs = self.model(**inputs)
                
                target_logits = outputs.logits[0, prompt_len - 1, :].float()
                softmax_probs = torch.softmax(target_logits, dim=-1)
                max_prob, pred_id = torch.max(softmax_probs, dim=-1)
                
                # Target ID
                target_ids = self.tokenizer.encode(item.get("expected", "").strip())
                if not target_ids: continue
                target_id = target_ids[0]
                
                probs_out.append(max_prob.item())
                labels_out.append(1 if pred_id.item() == target_id else 0)
                
        return probs_out, labels_out
