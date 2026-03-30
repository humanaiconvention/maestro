"""
Language model adapters for local inference with QWEN and LLAMA models.

This module provides a unified interface for different language models,
supporting both local weights via Hugging Face Transformers and optional API access.
"""

import json as _json
import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseLMAdapter(ABC):
    """
    Abstract base class for language model adapters.

    All adapters must implement the generate method to provide
    a consistent interface for text generation.
    """

    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """
        Generate text based on the given prompt.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Generated text response
        """
        pass


class QwenHFAdapter(BaseLMAdapter):
    """
    Adapter for QWEN models using Hugging Face Transformers.

    Supports local weight inference and optional API access.
    Uses lazy imports to avoid heavy dependencies if not needed.
    """

    def __init__(self, model_path: Optional[str] = None, api_key: Optional[str] = None):
        """
        Initialize QWEN adapter.

        Args:
            model_path: Path to local QWEN model weights (or env var QWEN_MODEL_PATH)
            api_key: Optional API key for QWEN API (or env var QWEN_API_KEY)
        """
        self.model_path = model_path or os.getenv("QWEN_MODEL_PATH")
        self.api_key = api_key or os.getenv("QWEN_API_KEY")
        self.model = None
        self.tokenizer = None

        if self.model_path:
            self._load_local_model()
        elif self.api_key:
            raise NotImplementedError(
                "QWEN API integration is not yet implemented. "
                "Please provide a local model path via model_path or QWEN_MODEL_PATH env var."
            )
        else:
            raise ValueError(
                "Either model_path (or QWEN_MODEL_PATH env var) or api_key "
                "(or QWEN_API_KEY env var) must be provided for QWEN adapter."
            )

    def _load_local_model(self) -> None:
        """Load QWEN model from local weights using Hugging Face Transformers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
        except ImportError:
            raise ImportError(
                "transformers package is required for local model inference. "
                "Install it with: pip install transformers>=4.30.0"
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch package is required for local model inference. "
                "Install it with: pip install torch>=2.0.0"
            )

        if not self.model_path:
            raise ValueError("model_path must be provided for local inference")

        print(f"Loading QWEN model from {self.model_path}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path, device_map="auto", trust_remote_code=True
            )
            print("QWEN model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load QWEN model from {self.model_path}. "
                f"Error: {str(e)}\n"
                "Please ensure the model path is correct and the model files are available."
            )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """
        Generate text using QWEN model.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Generated text response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Cannot generate text.")

        try:
            import torch
        except ImportError:
            raise ImportError("torch package is required for generation")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode and return only the generated part
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        return generated_text


class LlamaServerAdapter(BaseLMAdapter):
    """Adapter for llama.cpp llama-server via HTTP API (GGUF models).

    Connects to a running llama-server instance and calls the /completion
    endpoint. Supports token logprob extraction (n_probs) for SGT protocol
    turn auditing without requiring access to model internals.

    Args:
        base_url: Server base URL, e.g. "http://localhost:8080"
        n_probs:  Default number of top token probabilities per position.
                  0 disables logprob collection. Per-call override via
                  generate_with_logprobs(n_probs=...).
        timeout:  HTTP request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        n_probs: int = 0,
        timeout: int = 60,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.n_probs = n_probs
        self.timeout = timeout

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """Generate text, discarding logprob metadata."""
        return self.generate_with_logprobs(
            prompt, max_tokens=max_tokens, temperature=temperature
        )["text"]

    def generate_with_logprobs(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        n_probs: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate text and return token logprobs from llama-server.

        Uses the raw /completion endpoint which exposes completion_probabilities
        when n_probs > 0.

        Returns:
            dict with:
                "text"     (str)  — generated text
                "logprobs" (list) — per-token top-probability dicts, may be []
        """
        n_probs_use = self.n_probs if n_probs is None else n_probs

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "n_probs": n_probs_use,
            "stop": [],
        }

        data = _json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/completion",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"llama-server unreachable at {self.base_url}: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"llama-server request failed: {exc}") from exc

        return {
            "text": body.get("content", ""),
            "logprobs": body.get("completion_probabilities", []),
        }

    def audit_sgt_turn(
        self,
        prompt: str,
        control_tokens: List[str],
        max_tokens: int = 64,
        n_probs: int = 10,
    ) -> Dict[str, Any]:
        """Audit the model's confidence at an SGT protocol turn boundary.

        Captures the logprob margin for the specified control tokens
        (pivot markers, compression signals) at the first generated
        token position.

        Args:
            prompt:         The prompt at a protocol turn boundary.
            control_tokens: Token strings to look for, e.g. ["<PIVOT>", "<COMPRESS>"].
            max_tokens:     Maximum tokens to generate.
            n_probs:        Number of top probabilities to capture per position.

        Returns:
            dict with:
                "text"                  — generated text
                "top_logprobs"          — first-position top-k entries
                "control_token_logprobs"— {token: logprob} for matched control tokens
        """
        result = self.generate_with_logprobs(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            n_probs=n_probs,
        )

        logprobs = result.get("logprobs", [])
        control_found: Dict[str, Optional[float]] = {}

        if logprobs:
            first_position = logprobs[0] if isinstance(logprobs, list) else {}
            top_entries = first_position.get("top_logprobs", [])
            for entry in top_entries:
                tok = entry.get("tok_str", "")
                for ct in control_tokens:
                    if ct in tok or tok in ct:
                        control_found[ct] = entry.get("logprob", None)

        return {
            "text": result["text"],
            "top_logprobs": logprobs[:3] if logprobs else [],
            "control_token_logprobs": control_found,
        }


class LlamaHFAdapter(BaseLMAdapter):
    """
    Adapter for LLAMA models using Hugging Face Transformers.

    Supports local weight inference via transformers library.
    Uses lazy imports to avoid heavy dependencies if not needed.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize LLAMA adapter.

        Args:
            model_path: Path to local LLAMA model weights (or env var LLAMA_MODEL_PATH)
        """
        self.model_path = model_path or os.getenv("LLAMA_MODEL_PATH")
        self.model = None
        self.tokenizer = None

        if not self.model_path:
            raise ValueError(
                "model_path (or LLAMA_MODEL_PATH env var) must be provided for LLAMA adapter."
            )

        self._load_local_model()

    def _load_local_model(self) -> None:
        """Load LLAMA model from local weights using Hugging Face Transformers."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: F401
        except ImportError:
            raise ImportError(
                "transformers package is required for local model inference. "
                "Install it with: pip install transformers>=4.30.0"
            )

        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError(
                "torch package is required for local model inference. "
                "Install it with: pip install torch>=2.0.0"
            )

        print(f"Loading LLAMA model from {self.model_path}...")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map="auto",
            )
            print("LLAMA model loaded successfully.")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load LLAMA model from {self.model_path}. "
                f"Error: {str(e)}\n"
                "Please ensure the model path is correct and the model files are available."
            )

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        """
        Generate text using LLAMA model.

        Args:
            prompt: Input text prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (0.0 = deterministic)

        Returns:
            Generated text response
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Cannot generate text.")

        try:
            import torch
        except ImportError:
            raise ImportError("torch package is required for generation")

        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")

        # Move to same device as model
        if hasattr(self.model, "device"):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode and return only the generated part
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        return generated_text
