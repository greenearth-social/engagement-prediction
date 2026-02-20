"""
Compatibility re-export for serving helpers.

Prefer importing the standalone package:
    import model_serving

This `utils.model_serving` package is kept as a convenience for in-repo imports.
"""

from model_serving import (  # noqa: F401
    get_padded_vector_and_mask
)

__all__ = [
    "get_padded_vector_and_mask",
]
