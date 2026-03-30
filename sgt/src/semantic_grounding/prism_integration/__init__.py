"""PRISM integration for the Semantic Grounding Test module.

Provides non-invasive measurement hooks that attach to RecursiveTrainer
and record per-generation spectral/geometric metrics via PRISM.

Exports:
    take_sgt_snapshot       — capture EntropySnapshot from a trainer + eval dataset
    RecursiveTrainerPrismHook — observer that records snapshots across generations
    compute_entropy_delta   — compute before/after delta proof
    EntropySnapshot         — shared data contract
    EntropyDeltaProof       — settlement proof artifact
    GeometricHealthScore    — demand-side health score
    PrismUnavailableError   — raised when PRISM is not installed
"""

from .snapshot import take_sgt_snapshot
from .hooks import RecursiveTrainerPrismHook

# Re-export shared contracts for convenience
import os, sys

_ADAPTER_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../libs/adapters")
)
_SCHEMAS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "../../../../libs/schemas")
)
for _p in [_ADAPTER_DIR, _SCHEMAS_DIR]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from prism_adapter import compute_entropy_delta, PrismUnavailableError
    from prism_schemas import EntropySnapshot, EntropyDeltaProof, GeometricHealthScore
except ImportError:
    # If running with fully installed packages these will be available elsewhere
    pass

__all__ = [
    "take_sgt_snapshot",
    "RecursiveTrainerPrismHook",
    "compute_entropy_delta",
    "EntropySnapshot",
    "EntropyDeltaProof",
    "GeometricHealthScore",
    "PrismUnavailableError",
]
