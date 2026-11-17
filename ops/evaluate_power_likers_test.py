"""Standalone power-liker evaluation interface tests."""

from pathlib import Path

import pytest

from ops.evaluate_power_likers import DEFAULT_POOL_SIZE, DEFAULT_TOP_K, build_parser


def test_defaults_match_paper_settings():
    args = build_parser().parse_args(["--model", "runs/model"])
    assert args.model == Path("runs/model")
    assert (args.pool_size, args.top_k, args.min_user_likes) == (DEFAULT_POOL_SIZE, DEFAULT_TOP_K, 5)
    assert args.device is None


@pytest.mark.parametrize("flag", ["--pool-size", "--top-k", "--min-user-likes", "--batch-size"])
def test_rejects_non_positive_sizes(flag):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "runs/model", flag, "0"])


def test_requires_model():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
