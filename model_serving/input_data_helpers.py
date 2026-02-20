from __future__ import annotations

from typing import List, Tuple
import numpy as np

def get_padded_vector_and_mask(
    history: List[np.ndarray], 
    max_history_len: int, 
    embed_dim: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    """
    hist_len = len(history)

    # validate input data 
    if hist_len > 0:
        for h in history:
            if len(h) != embed_dim:
                raise ValueError(f"History length ({len(h)}) and embedding dim ({embed_dim} do not match)")
            
    seq_len = min(hist_len, max_history_len)
    
    # Initialize padded array
    padded = np.zeros((max_history_len, embed_dim), dtype=np.float32)
    mask = np.zeros(max_history_len, dtype=bool)

    if seq_len > 0:
        # Truncate to max_history_len if needed, load from memmap
        padded[:seq_len] = history[: max_history_len]
        mask[:seq_len] = True

    return padded, mask
