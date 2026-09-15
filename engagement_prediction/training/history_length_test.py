"""Tests for history-length configuration and inclusive bucket definitions."""

import pytest

from engagement_prediction.training.history_length import (
    history_length_bucket_specs,
    validate_history_length_bucket_boundaries,
)


@pytest.mark.parametrize(
    "boundaries",
    [None, [], (), [1, 2], [0, 0], [0, 2, 1], [0, -1], [0, True], [False, 1], [0, 1.0], [0, "1"]],
)
def test_rejects_invalid_history_length_boundaries(boundaries):
    with pytest.raises(ValueError, match="history_length_bucket_boundaries"):
        validate_history_length_bucket_boundaries(boundaries)


def test_valid_boundaries_are_copied_and_define_inclusive_buckets():
    boundaries = [0, 1, 2, 4, 8, 16, 32]
    validated = validate_history_length_bucket_boundaries(boundaries)
    assert validated == boundaries
    assert validated is not boundaries
    assert history_length_bucket_specs(boundaries) == [
        ("0", 0, 0),
        ("1", 1, 1),
        ("2", 2, 2),
        ("3–4", 3, 4),
        ("5–8", 5, 8),
        ("9–16", 9, 16),
        ("17–32", 17, 32),
        (">32", 33, None),
    ]


def test_custom_and_zero_only_boundaries():
    assert history_length_bucket_specs([0, 3]) == [
        ("0", 0, 0),
        ("1–3", 1, 3),
        (">3", 4, None),
    ]
    assert history_length_bucket_specs([0]) == [("0", 0, 0), (">0", 1, None)]
