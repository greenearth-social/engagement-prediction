from __future__ import annotations

from typing import Any, Tuple, Dict, List, Optional, Union
import numpy as np
import base64
import struct
import zlib

# ----------------------------------------
# Embeddings helpers
# ----------------------------------------

# Known embedding model dimensions
EMBEDDING_MODEL_DIMS: Dict[str, int] = {
    "all_MiniLM_L6_v2": 384,
    "all_MiniLM_L12_v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
}


def get_embedding_dim_for_known_model(embedding_model: str) -> int:
    """
    Get the embedding dimension for a known model name.
    
    Args:
        embedding_model: Name of the embedding model
        
    Returns:
        Embedding dimension (e.g., 384 for MiniLM models)
        
    Raises:
        ValueError: If model name is not in EMBEDDING_MODEL_DIMS
    """
    if embedding_model not in EMBEDDING_MODEL_DIMS:
        known_models = ", ".join(sorted(EMBEDDING_MODEL_DIMS.keys()))
        raise ValueError(
            f"Unknown embedding model '{embedding_model}'. "
            f"Known models: {known_models}. "
            f"Add new models to EMBEDDING_MODEL_DIMS in input_data_helpers.py."
        )
    return EMBEDDING_MODEL_DIMS[embedding_model]


def _extract_compressed_embedding_vector_from_struct(embeddings: Any, embedding_model: str) -> Optional[str]:
    """
    Extract the base85-encoded embedding string for a given model from a single row's
    `embeddings` value.

    This is intentionally pure-Python (non-Polars) so it can be used inside
    `map_elements()` without relying on Polars struct/list expressions.
    """
    if embeddings is None:
        return None

    for item in embeddings:
        if item is None:
            continue

        if isinstance(item, dict):
            if item.get("key") == embedding_model:
                return item.get("value")
            continue

        if isinstance(item, (tuple, list)) and len(item) >= 2:
            if item[0] == embedding_model:
                return item[1]
            continue

        key = getattr(item, "key", None)
        if key == embedding_model:
            return getattr(item, "value", None)

    return None


def _decompress_and_unpack_embedding(s: str, decompress: Optional[bool] = None) -> list[float]:
    """
    Convert an embedding from a base85-encoded string to a list of floats.

    If `decompress` is `True`, decompress with zlib and throw an error if decompression fails.

    If `decompress` is `False`, do not decompress before unpacking.

    If `decompress` is `None`, attempt decompression and silently fallback to an uncompressed string
    if decompression fails.
    """

    bs = base64.b85decode(s.encode())

    if decompress or decompress is None:
        try:
            bs = zlib.decompress(bs)
        except zlib.error:
            if decompress:
                raise
    
    if len(bs) % 4 != 0:
        raise ValueError(f"Byte length {len(bs)} is not a multiple of 4, cannot unpack into floats")
    return list(struct.unpack(f'<{len(bs) // 4}f', bs))


def get_expanded_embedding_vector(embedding_input: Any, embedding_model: str) -> Optional[list[float]]:
    """
    Takes a single raw embeddings input, which might have embeddings from multiple models. 
    Extracts the correct compressed embedding for the given model.
    Then decompresses it and unpacks it into a list of floats.
    """
    compressed_embedding = _extract_compressed_embedding_vector_from_struct(embedding_input, embedding_model)
    if compressed_embedding is None:
        return None
    return _decompress_and_unpack_embedding(compressed_embedding, decompress=True)


# ----------------------------------------
# Input data shape helpers
# ----------------------------------------

def get_padded_embedding_history_and_mask(
    history_embeddings: Any,
    max_history_len: int, 
    embed_dim: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad/truncate a variable-length history of embedding vectors and build a mask.

    This helper is used when a model expects a fixed-length sequence input
    (e.g. a transformer-style user-history encoder), but the available user
    history is variable length.

    Args:
        history:
            Either a 2D numpy array with shape ``[T, embed_dim]`` or a sequence
            (e.g. list) of length ``T`` containing 1D arrays/lists of length
            ``embed_dim``.
        max_history_len:
            The fixed sequence length to emit. If ``T > max_history_len``,
            the history is truncated.
        embed_dim:
            Embedding dimension (width) for each history vector.

    Returns:
        padded:
            A float32 numpy array of shape ``[max_history_len, embed_dim]``.
            Entries beyond the available history are zero-padded.
        mask:
            A boolean numpy array of shape ``[max_history_len]`` where ``True``
            indicates a real (non-padding) history position.
    """
    hist_len = len(history_embeddings)

    # validate input data 
    if hist_len > 0:
        for h in history_embeddings:
            if len(h) != embed_dim:
                raise ValueError(
                    f"History embedding length ({len(h)}) and embed_dim ({embed_dim}) do not match"
                )
            
    seq_len = min(hist_len, max_history_len)
    
    # Initialize padded array
    padded = np.zeros((max_history_len, embed_dim), dtype=np.float32)
    mask = np.zeros(max_history_len, dtype=bool)

    if seq_len > 0:
        # Truncate to max_history_len if needed, load from memmap
        padded[:seq_len] = history_embeddings[: max_history_len]
        mask[:seq_len] = True

    return padded, mask


def get_padded_embedding_history_and_mask_batched(
    history_embeddings: Any,
    max_history_len: int, 
    embed_dim: int,
) -> Tuple[List[List[List[float]]], List[List[float]]]:
    
    if not isinstance(history_embeddings, list):
        raise ValueError("Invalid input: history_embeddings must be a list")

    batch_padded_history_embeddings = []
    batch_history_mask = []

    if len(history_embeddings) == 0:
        padded_history_embeddings, history_mask = get_padded_embedding_history_and_mask(
            history_embeddings=[],
            max_history_len=max_history_len,
            embed_dim=embed_dim,
        )
        return [padded_history_embeddings.tolist()], [history_mask.tolist()]

    first_non_none = next((x for x in history_embeddings if x is not None), None)
    if first_non_none is None:
        padded_history_embeddings, history_mask = get_padded_embedding_history_and_mask(
            history_embeddings=[],
            max_history_len=max_history_len,
            embed_dim=embed_dim,
        )
        return [padded_history_embeddings.tolist()], [history_mask.tolist()]

    if _is_numeric_vector(first_non_none):
        # Single input: history_embeddings has shape [T, D].
        padded_history_embeddings, history_mask = get_padded_embedding_history_and_mask(
            history_embeddings=history_embeddings,
            max_history_len=max_history_len,
            embed_dim=embed_dim,
        )
        batch_padded_history_embeddings.append(padded_history_embeddings.tolist())
        batch_history_mask.append(history_mask.tolist())
    else:
        # Batched input: history_embeddings has shape [B, T, D].
        for single_history_embeddings in history_embeddings:  # type: ignore[assignment]
            if single_history_embeddings is None:
                single_history_embeddings = []
            
            if not isinstance(single_history_embeddings, list):
                raise ValueError("Invalid batched input: each batch element must be a list")

            inner_first_non_none = next((x for x in single_history_embeddings if x is not None), None)
            if inner_first_non_none is not None and not _is_numeric_vector(inner_first_non_none):
                raise ValueError(
                    "Invalid batched input: expected each batch element to have shape [T, D]"
                )

            padded_history_embeddings, history_mask = get_padded_embedding_history_and_mask(
                history_embeddings=single_history_embeddings,
                max_history_len=max_history_len,
                embed_dim=embed_dim,
            )

            batch_padded_history_embeddings.append(padded_history_embeddings.tolist())
            batch_history_mask.append(history_mask.tolist())
    
    return batch_padded_history_embeddings, batch_history_mask


def _is_numeric_vector(x: Any) -> bool:
    return (
        isinstance(x, list)
        and len(x) > 0
        and all(isinstance(v, (int, float)) for v in x)
    )
