"""SGT-specific adapter for PRISM snapshot capture.

Wraps prism_adapter.take_snapshot to accept SGT's data formats
(RecursiveTrainer instance + List[Dict] eval dataset).
"""

from __future__ import annotations

import os
import sys
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve adapter path from the monorepo layout
_ADAPTER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../libs/adapters")
)
_SCHEMAS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../libs/schemas")
)
for _p in [_ADAPTER_DIR, _SCHEMAS_DIR]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from prism_adapter import take_snapshot, compute_entropy_delta, PrismUnavailableError  # noqa: E402
from prism_schemas import EntropySnapshot, EntropyDeltaProof  # noqa: E402


def take_sgt_snapshot(
    trainer: Any,
    eval_data: List[Dict[str, Any]],
    generation_idx: int,
    regime: str,
    n_samples: int = 10,
    layers_to_profile: Optional[List[int]] = None,
) -> EntropySnapshot:
    """Capture an EntropySnapshot from a RecursiveTrainer + SGT eval dataset.

    This is the primary entry point for SGT-side PRISM measurement.
    It adapts the SGT data format (list of dicts with a "prompt" key) to
    the plain list-of-strings format that prism_adapter.take_snapshot expects.

    Args:
        trainer: RecursiveTrainer instance. Must have .model, .tokenizer, .device attributes.
        eval_data: SGT evaluation dataset — list of dicts each containing at least "prompt".
                   Other keys ("completion", "label", etc.) are ignored.
        generation_idx: The current SGT generation (0-indexed).
        regime: SGT regime label, e.g. "R1", "R2", "R3", "R4", "R1_accum".
        n_samples: How many eval prompts to use (more = more accurate, slower).
        layers_to_profile: Override the default layer selection (every 3rd, up to 8).

    Returns:
        EntropySnapshot with generation_idx and regime set.

    Raises:
        PrismUnavailableError: If spectral-microscope is not installed.
        ValueError: If trainer has no .model attribute or eval_data is empty.
    """
    if not hasattr(trainer, "model") or trainer.model is None:
        raise ValueError("trainer.model is None — cannot take PRISM snapshot before model is loaded.")
    if not eval_data:
        raise ValueError("eval_data is empty — cannot take PRISM snapshot.")

    # Extract plain-text prompts from SGT eval dicts
    prompts: List[str] = []
    for item in eval_data[:n_samples]:
        prompt = item.get("prompt") or item.get("input") or item.get("question", "")
        if prompt:
            prompts.append(str(prompt))

    if not prompts:
        raise ValueError(
            "Could not extract any prompts from eval_data. "
            "Expected dicts with a 'prompt', 'input', or 'question' key."
        )

    device = getattr(trainer, "device", None)
    if device is not None:
        device = str(device)

    logger.info(
        "prism_integration: taking snapshot at gen=%d regime=%s using %d prompts",
        generation_idx, regime, len(prompts),
    )

    return take_snapshot(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        eval_prompts=prompts,
        generation_idx=generation_idx,
        regime=regime,
        layers_to_profile=layers_to_profile,
        n_samples=n_samples,
        device=device,
    )


__all__ = ["take_sgt_snapshot", "compute_entropy_delta", "PrismUnavailableError",
           "EntropySnapshot", "EntropyDeltaProof"]
