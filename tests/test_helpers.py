from __future__ import annotations

from typing import List

import polars as pl

from utils.helpers import validate_dataframe_schema


def test_validate_dataframe_schema_accepts_typing_list_for_polars_columns():
    df = pl.DataFrame(
        {
            "neg_uri": [["at://p1", "at://p2"]],
            "neg_emb_idx": [[1, 2]],
            "neg_author_did": [["did:plc:a", "did:plc:b"]],
        }
    )

    validate_dataframe_schema(
        df,
        {
            "neg_uri": List[str],
            "neg_emb_idx": List[int],
            "neg_author_did": List[str],
        },
    )


def test_validate_dataframe_schema_accepts_builtin_list_generic_for_polars_columns():
    df = pl.DataFrame(
        {
            "names": [["a", "b"]],
            "ids": [[1, 2]],
        }
    )

    validate_dataframe_schema(
        df,
        {
            "names": list[str],
            "ids": list[int],
        },
    )
