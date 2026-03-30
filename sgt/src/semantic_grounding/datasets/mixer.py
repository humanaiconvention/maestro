"""Dataset mixing logic for rigorous recursive regimes."""

import random
from typing import List, Dict

class RegimeMixer:
    """Mixes synthetic and real datasets according to the policy."""
    
    def __init__(self, config: dict):
        self.config = config
        self.data_policy = config.get("data_policy", "replace") # 'replace' or 'accumulate'
        self.accumulated_history = []
        
    def mix_dataset(self, 
                    synthetic_data: List[Dict[str, str]], 
                    correction_pool: List[Dict[str, str]], 
                    target_size: int,
                    correction_fraction: float) -> List[Dict[str, str]]:
        """
        Creates a new training dataset enforcing rigorous correction bandwidths.
        """
        n_corr_target = int(target_size * correction_fraction)
        
        mixed_dataset = []
        
        # 1. Determine base synthetic pool
        if self.data_policy == "replace":
            pool = synthetic_data
        elif self.data_policy == "accumulate":
            self.accumulated_history.extend(synthetic_data)
            pool = self.accumulated_history
        else:
            raise ValueError(f"Unknown data_policy: {self.data_policy}")

        # 2. Exogenous Correction
        n_corr_actual = 0
        if n_corr_target > 0 and correction_pool:
            n_corr_actual = min(n_corr_target, len(correction_pool))
            mixed_dataset.extend(random.choices(correction_pool, k=n_corr_actual))
            
        # 3. Base Synthetic / Fallback
        # We need to fill the remainder of target_size from the synthetic pool
        n_syn_required = target_size - n_corr_actual
        if n_syn_required > 0 and pool:
            mixed_dataset.extend(random.choices(pool, k=n_syn_required))
            
        random.shuffle(mixed_dataset)
        return mixed_dataset
