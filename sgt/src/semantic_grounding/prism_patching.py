"""Activation patching for SGT protocol adherence analysis.

Implements causal tracing: patch hidden-state activations from a "source"
(compliant) run into a "target" (noncompliant) run at each transformer layer,
measuring the effect on the probability of the desired protocol token.

This localises *which layers* carry causal information about SGT protocol
compliance — the primary diagnostic for understanding where pivot/compression
signals are computed in the network.

Usage:
    patcher = ActivationPatcher(hf_model, tokenizer)
    results = patcher.patch_sweep(
        source_prompt="[prompt that correctly triggers <PIVOT>]",
        target_prompt="[prompt that fails to trigger <PIVOT>]",
        token_of_interest="<PIVOT>",
    )
    top = patcher.top_layers_by_effect(results, top_k=5)

Requirements: torch, transformers (already needed for HF model loading).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ActivationPatcher:
    """Causal activation patching across transformer layers.

    For each layer, replaces the target run's hidden states at that layer
    with the source run's hidden states, measuring the change in the
    log-probability of a token of interest at the final position.

    A large positive patch_effect (patched_logprob - baseline_logprob) at a
    layer means that layer carries causal information about the token.

    Args:
        model:     HF model with output_hidden_states and register_forward_hook support.
        tokenizer: HF tokenizer.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def patch_sweep(
        self,
        source_prompt: str,
        target_prompt: str,
        token_of_interest: str,
        layers: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Sweep activation patches from source into target across layers.

        Args:
            source_prompt:      Prompt from the compliant (correct) run.
            target_prompt:      Prompt from the noncompliant (incorrect) run.
            token_of_interest:  Token whose next-position log-probability to monitor.
            layers:             Layer indices to patch. None = all transformer layers.

        Returns:
            dict with:
                token_of_interest  — the monitored token string
                baseline_logprob   — log-prob of token on target run (unpatched)
                source_logprob     — log-prob of token on source run
                n_layers           — total number of transformer layers found
                patch_effects      — list of per-layer dicts:
                    layer_idx, baseline_logprob, patched_logprob,
                    patch_effect (patched - baseline),
                    normalized_effect (effect / |source - baseline|)
        """
        logger.info(
            "patch_sweep: collecting hidden states for source and target prompts"
        )
        _, source_hidden = self._collect_hidden_states(source_prompt)
        n_layers = len(source_hidden)

        if layers is None:
            layers = list(range(n_layers))
        else:
            layers = [l for l in layers if 0 <= l < n_layers]

        baseline_logprob = self._next_token_logprob(target_prompt, token_of_interest)
        source_logprob = self._next_token_logprob(source_prompt, token_of_interest)

        denom = abs(source_logprob - baseline_logprob) + 1e-9
        patch_effects = []

        for layer_idx in layers:
            patched_logprob = self._patch_layer_and_measure(
                target_prompt=target_prompt,
                source_hidden_at_layer=source_hidden[layer_idx],
                patch_layer_idx=layer_idx,
                token_of_interest=token_of_interest,
            )
            effect = patched_logprob - baseline_logprob
            patch_effects.append({
                "layer_idx": layer_idx,
                "baseline_logprob": baseline_logprob,
                "patched_logprob": patched_logprob,
                "patch_effect": effect,
                "normalized_effect": effect / denom,
            })

        return {
            "token_of_interest": token_of_interest,
            "baseline_logprob": baseline_logprob,
            "source_logprob": source_logprob,
            "n_layers": n_layers,
            "patch_effects": patch_effects,
        }

    def top_layers_by_effect(
        self,
        patch_results: Dict[str, Any],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Return the top-k layers ranked by absolute patch effect magnitude."""
        effects = patch_results.get("patch_effects", [])
        return sorted(effects, key=lambda x: abs(x["patch_effect"]), reverse=True)[:top_k]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_hidden_states(
        self, prompt: str
    ) -> Tuple[Any, List[Any]]:
        """Run a forward pass and return (logits, list_of_hidden_state_tensors)."""
        import torch

        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)

        hidden = [h.detach().clone() for h in out.hidden_states]
        return out.logits, hidden

    def _next_token_logprob(self, prompt: str, target_token: str) -> float:
        """Return the log-probability of target_token as the next token after prompt."""
        import torch

        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        target_ids = self.tokenizer.encode(target_token, add_special_tokens=False)
        if not target_ids:
            return float("-inf")
        target_id = target_ids[0]

        with torch.no_grad():
            out = self.model(**enc)

        log_probs = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
        return float(log_probs[target_id].item())

    def _patch_layer_and_measure(
        self,
        target_prompt: str,
        source_hidden_at_layer: Any,
        patch_layer_idx: int,
        token_of_interest: str,
    ) -> float:
        """Patch a single layer's output with source activations and measure effect."""
        import torch

        device = next(self.model.parameters()).device
        enc = self.tokenizer(
            target_prompt, return_tensors="pt", truncation=True, max_length=512
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        target_ids = self.tokenizer.encode(token_of_interest, add_special_tokens=False)
        if not target_ids:
            return float("-inf")
        target_id = target_ids[0]

        patch = source_hidden_at_layer.to(device)  # (1, src_seq, dim)
        result_logprob = [float("-inf")]

        def _hook(module: Any, input: Any, output: Any) -> Any:
            """Replace hidden states with (possibly length-adjusted) source patch."""
            if isinstance(output, tuple):
                hs = output[0]  # (1, tgt_seq, dim)
                tgt_seq = hs.shape[1]
                src_seq = patch.shape[1]
                if src_seq >= tgt_seq:
                    p = patch[:, :tgt_seq, :]
                else:
                    # Pad by repeating last token representation
                    pad = patch[:, -1:, :].expand(1, tgt_seq - src_seq, -1)
                    p = torch.cat([patch, pad], dim=1)
                return (p,) + output[1:]
            else:
                tgt_seq = output.shape[1]
                src_seq = patch.shape[1]
                if src_seq >= tgt_seq:
                    return patch[:, :tgt_seq, :]
                pad = patch[:, -1:, :].expand(1, tgt_seq - src_seq, -1)
                return torch.cat([patch, pad], dim=1)

        try:
            transformer_layers = self._get_transformer_layers()
        except RuntimeError as exc:
            logger.warning("ActivationPatcher: %s", exc)
            return float("-inf")

        if patch_layer_idx >= len(transformer_layers):
            return float("-inf")

        handle = transformer_layers[patch_layer_idx].register_forward_hook(_hook)
        try:
            with torch.no_grad():
                out = self.model(**enc)
            log_probs = torch.nn.functional.log_softmax(out.logits[0, -1, :], dim=-1)
            result_logprob[0] = float(log_probs[target_id].item())
        except Exception as exc:
            logger.warning(
                "ActivationPatcher: patch at layer %d failed — %s", patch_layer_idx, exc
            )
        finally:
            handle.remove()

        return result_logprob[0]

    def _get_transformer_layers(self) -> List[Any]:
        """Locate the list of transformer decoder layers in the model.

        Tries common attribute paths across model families (Qwen, Llama,
        GPT-2, T5 decoder, Falcon, Mistral, ...).
        """
        candidate_paths = [
            "model.layers",           # Llama, Qwen, Mistral
            "transformer.h",          # GPT-2, GPT-J, Falcon
            "model.decoder.layers",   # T5, BART decoder
            "decoder.layers",
            "layers",
        ]
        for path in candidate_paths:
            obj = self.model
            try:
                for attr in path.split("."):
                    obj = getattr(obj, attr)
                return list(obj)
            except AttributeError:
                continue
        raise RuntimeError(
            "Could not locate transformer layers on this model. "
            "Check the architecture and update _get_transformer_layers() if needed."
        )
