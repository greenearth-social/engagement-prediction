"""Content-embedding model metadata and Ingex payload decoding.

Ingex stores model vectors in an embeddings key/value collection. Each value
is a base85-encoded, zlib-compressed sequence of little-endian float32 values;
Stage 7 expands only the configured model before writing ``embeddings.npy``.
"""

from __future__ import annotations

import base64
import struct
import zlib
from typing import Any, Optional


EMBEDDING_MODEL_DIMS: dict[str, int] = {
    "all_MiniLM_L6_v2": 384,
    "all_MiniLM_L12_v2": 384,
    "all-MiniLM-L6-v2": 384,
    "all-MiniLM-L12-v2": 384,
    "paraphrase-MiniLM-L6-v2": 384,
    "multi-qa-MiniLM-L6-cos-v1": 384,
}


def get_embedding_dim_for_known_model(embedding_model: str) -> int:
    """Return the configured vector dimension for a known embedding model."""

    if embedding_model not in EMBEDDING_MODEL_DIMS:
        known_models = ", ".join(sorted(EMBEDDING_MODEL_DIMS))
        raise ValueError(
            f"Unknown embedding model '{embedding_model}'. "
            f"Known models: {known_models}. "
            "Add new models to EMBEDDING_MODEL_DIMS in "
            "engagement_prediction.data.embeddings."
        )
    return EMBEDDING_MODEL_DIMS[embedding_model]


def _extract_compressed_embedding_vector_from_struct(
    embeddings: Any,
    embedding_model: str,
) -> Optional[str]:
    """Extract one model's base85-encoded vector from a raw embeddings value."""

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


def _decompress_and_unpack_embedding(
    encoded: str,
    decompress: Optional[bool] = None,
) -> list[float]:
    """Decode a base85 vector, optionally decompress it, and unpack float32s."""

    payload = base64.b85decode(encoded.encode())

    # ``decompress=None`` is retained for callers that accept historical
    # uncompressed payloads. Stage 7 passes ``True`` and therefore fails fast
    # if a supposedly compressed source value is malformed.
    if decompress or decompress is None:
        try:
            payload = zlib.decompress(payload)
        except zlib.error:
            if decompress:
                raise

    if len(payload) % 4 != 0:
        raise ValueError(
            f"Byte length {len(payload)} is not a multiple of 4, "
            "cannot unpack into floats"
        )
    return list(struct.unpack(f"<{len(payload) // 4}f", payload))


def get_expanded_embedding_vector(
    embedding_input: Any,
    embedding_model: str,
) -> Optional[list[float]]:
    """Extract and decode one model's vector from a raw embeddings value."""

    compressed_embedding = _extract_compressed_embedding_vector_from_struct(
        embedding_input,
        embedding_model,
    )
    if compressed_embedding is None:
        return None
    return _decompress_and_unpack_embedding(
        compressed_embedding,
        decompress=True,
    )
