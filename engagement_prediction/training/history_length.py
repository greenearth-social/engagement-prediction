"""History-length bucket definitions shared by configuration and evaluation."""

from __future__ import annotations

from typing import Any


def validate_history_length_bucket_boundaries(boundaries: Any) -> list[int]:
    """Validate inclusive upper bounds, keeping zero history in its own bucket."""

    if (
        not isinstance(boundaries, list)
        or not boundaries
        or any(type(value) is not int for value in boundaries)
        or boundaries[0] != 0
        or any(left >= right for left, right in zip(boundaries, boundaries[1:]))
    ):
        raise ValueError(
            "history_length_bucket_boundaries must be a nonempty list of "
            "strictly increasing integers starting at zero"
        )
    return list(boundaries)


def history_length_bucket_specs(
    boundaries: list[int],
) -> list[tuple[str, int, int | None]]:
    """Return ordered labels and inclusive bounds, including the overflow bucket."""

    boundaries = validate_history_length_bucket_boundaries(boundaries)
    buckets: list[tuple[str, int, int | None]] = []
    lower_bound = 0
    for upper_bound in boundaries:
        label = (
            str(lower_bound)
            if lower_bound == upper_bound
            else f"{lower_bound}–{upper_bound}"
        )
        buckets.append((label, lower_bound, upper_bound))
        lower_bound = upper_bound + 1
    buckets.append((f">{boundaries[-1]}", lower_bound, None))
    return buckets
