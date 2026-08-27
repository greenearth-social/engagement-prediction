"""Build temporary Stage 7 embedding-to-model-author mappings."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Iterator

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from engagement_prediction.data.dataset_hydration import (
    AUTHOR_PAD_IDX,
    AUTHOR_UNK_IDX,
)
from engagement_prediction.training.bst_publication import (
    RANKER_AUTHOR_MAP_SCHEMA,
)


AUTHOR_INDEX_DTYPE = np.dtype("<u4")
_UNFILLED_AUTHOR_IDX = np.iinfo(AUTHOR_INDEX_DTYPE).max


@dataclass(frozen=True)
class AuthorIndexOverride:
    """A temporary read-only-compatible post-author array and its coverage."""

    path: Path
    author_table_num_rows: int
    coverage: dict[str, Any]


def load_model_author_map(
    path: Path,
    *,
    author_table_num_rows: int,
    allow_extra_columns: bool = False,
) -> tuple[dict[str, int], dict[str, int]]:
    """Load and validate a Stage 8 or explicitly allowed legacy author map.

    Canonical maps must contain exactly the two serving columns. Legacy maps may
    carry historical statistics such as ``author_train_count``; callers must opt
    into that compatibility behavior, and only the two serving columns are used.
    """

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Model author map does not exist: {path}")
    if author_table_num_rows < 2:
        raise ValueError("Model author table must reserve PAD=0 and UNK=1")
    try:
        author_map = pl.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Could not read model author map {path}: {exc}") from exc
    expected_schema = pl.Schema(RANKER_AUTHOR_MAP_SCHEMA)
    if allow_extra_columns:
        missing_columns = [
            column for column in expected_schema.names() if column not in author_map.columns
        ]
        invalid_types = {
            column: author_map.schema[column]
            for column, dtype in expected_schema.items()
            if column in author_map.columns and author_map.schema[column] != dtype
        }
        if missing_columns or invalid_types:
            raise ValueError(
                "Model author map is missing required columns or has unexpected "
                "required-column types: "
                f"missing={missing_columns} invalid_types={invalid_types}"
            )
    elif author_map.schema != expected_schema or author_map.columns != list(
        RANKER_AUTHOR_MAP_SCHEMA
    ):
        raise ValueError(
            f"Model author map has an unexpected schema: {author_map.schema}"
        )
    author_count = author_map.height
    if author_map.get_column("author_did").null_count():
        raise ValueError("Model author map contains a null author DID")
    if author_map.get_column("author_idx").null_count():
        raise ValueError("Model author map contains a null author index")
    if author_map.get_column("author_did").n_unique() != author_count:
        raise ValueError("Model author map contains duplicate author DIDs")
    if author_map.get_column("author_idx").n_unique() != author_count:
        raise ValueError("Model author map contains duplicate author indices")
    if author_table_num_rows != author_count + 2:
        raise ValueError("Model author map size does not match the model author table")
    if author_count:
        author_indices = author_map.get_column("author_idx")
        if int(author_indices.min()) != 2:
            raise ValueError("Model author indices must begin at 2 after PAD and UNK")
        if int(author_indices.max()) != author_table_num_rows - 1:
            raise ValueError("Model author indices must be dense through the table size")
        expected_indices = pl.Series(
            "author_idx",
            range(2, author_table_num_rows),
            dtype=pl.UInt32,
        )
        if not author_indices.sort().equals(expected_indices):
            raise ValueError("Model author indices must be dense from 2")
    if not author_map.equals(author_map.sort("author_did")):
        raise ValueError("Model author map must be sorted by author DID")
    mapping_columns = author_map.select("author_did", "author_idx")
    mapping = {
        str(author_did): int(author_idx)
        for author_did, author_idx in mapping_columns.iter_rows()
    }
    return mapping, {
        "author_count": author_count,
        "author_table_num_rows": author_table_num_rows,
        "file_size_bytes": path.stat().st_size,
    }


def validate_model_author_map(
    path: Path,
    *,
    author_table_num_rows: int,
    allow_extra_columns: bool = False,
) -> dict[str, int]:
    """Validate a model map without retaining its eager dictionary."""

    _, statistics = load_model_author_map(
        path,
        author_table_num_rows=author_table_num_rows,
        allow_extra_columns=allow_extra_columns,
    )
    return statistics


def build_author_index_override(
    *,
    stage7_bundle_path: Path,
    model_author_map_path: Path,
    author_table_num_rows: int,
    embedding_count: int,
    output_path: Path,
    allow_extra_columns: bool = False,
) -> dict[str, Any]:
    """Stream Stage 7 posts into an exact ``emb_idx -> model author_idx`` mmap."""

    stage7_bundle_path = Path(stage7_bundle_path)
    posts_path = stage7_bundle_path / "posts"
    post_parts = sorted(posts_path.rglob("*.parquet"))
    if not post_parts:
        raise FileNotFoundError(f"Stage 7 posts artifact has no Parquet parts: {posts_path}")
    if embedding_count < 0:
        raise ValueError("embedding_count must be nonnegative")
    model_author_indices, model_author_stats = load_model_author_map(
        model_author_map_path,
        author_table_num_rows=author_table_num_rows,
        allow_extra_columns=allow_extra_columns,
    )
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite author-index override: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    override = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=AUTHOR_INDEX_DTYPE,
        shape=(embedding_count,),
    )
    override.fill(_UNFILLED_AUTHOR_IDX)
    known_post_count = 0
    scanned_post_count = 0
    used_model_indices = np.zeros(author_table_num_rows, dtype=np.bool_)
    try:
        for part in post_parts:
            try:
                parquet_file = pq.ParquetFile(part)
                batches = parquet_file.iter_batches(
                    columns=["emb_idx", "author_did"]
                )
                for batch in batches:
                    emb_column = batch.column("emb_idx")
                    author_column = batch.column("author_did")
                    if emb_column.null_count:
                        raise ValueError("Stage 7 posts contain a null emb_idx")
                    if author_column.null_count:
                        raise ValueError("Stage 7 posts contain a null author_did")
                    emb_indices = np.asarray(
                        emb_column.to_numpy(zero_copy_only=False),
                        dtype=np.int64,
                    )
                    if emb_indices.size and (
                        int(emb_indices.min()) < 0
                        or int(emb_indices.max()) >= embedding_count
                    ):
                        raise ValueError("Stage 7 post emb_idx is outside the embedding range")
                    if np.unique(emb_indices).size != emb_indices.size:
                        raise ValueError("Stage 7 posts contain duplicate emb_idx values")
                    if emb_indices.size and np.any(
                        override[emb_indices] != _UNFILLED_AUTHOR_IDX
                    ):
                        raise ValueError("Stage 7 posts contain duplicate emb_idx values")
                    author_dids = author_column.to_pylist()
                    mapped_indices = np.fromiter(
                        (
                            model_author_indices.get(str(author_did), AUTHOR_UNK_IDX)
                            for author_did in author_dids
                        ),
                        dtype=AUTHOR_INDEX_DTYPE,
                        count=len(author_dids),
                    )
                    known_mask = mapped_indices != AUTHOR_UNK_IDX
                    known_post_count += int(known_mask.sum())
                    if np.any(known_mask):
                        used_model_indices[mapped_indices[known_mask]] = True
                    override[emb_indices] = mapped_indices
                    scanned_post_count += int(emb_indices.size)
            except ValueError:
                raise
            except Exception as exc:
                raise ValueError(f"Could not map Stage 7 post part {part}: {exc}") from exc
        if scanned_post_count != embedding_count:
            raise ValueError(
                "Stage 7 posts do not contain exactly one row per embedding: "
                f"posts={scanned_post_count} embeddings={embedding_count}"
            )
        missing_count = int(np.count_nonzero(override == _UNFILLED_AUTHOR_IDX))
        if missing_count:
            raise ValueError(
                f"Stage 7 posts are missing {missing_count} embedding indices"
            )
        if override.size and (
            int(override.min()) < AUTHOR_UNK_IDX
            or int(override.max()) >= author_table_num_rows
        ):
            raise ValueError("Remapped author index is outside the model author table")
        override.flush()
    except Exception:
        del override
        output_path.unlink(missing_ok=True)
        raise
    del override

    reopened = np.load(output_path, mmap_mode="r", allow_pickle=False)
    try:
        if (
            not isinstance(reopened, np.memmap)
            or reopened.dtype != AUTHOR_INDEX_DTYPE
            or reopened.shape != (embedding_count,)
        ):
            raise RuntimeError("Published author-index override is not an exact <u4 mmap")
    finally:
        mmap = getattr(reopened, "_mmap", None)
        if mmap is not None:
            mmap.close()
    unknown_post_count = embedding_count - known_post_count
    model_author_count = model_author_stats["author_count"]
    used_model_author_count = int(used_model_indices[2:].sum())
    return {
        "stage7_post_count": embedding_count,
        "model_known_post_count": known_post_count,
        "model_unknown_post_count": unknown_post_count,
        "model_known_post_fraction": (
            known_post_count / embedding_count if embedding_count else 0.0
        ),
        "model_author_count": model_author_count,
        "model_used_author_count": used_model_author_count,
        "model_unused_author_count": model_author_count - used_model_author_count,
        "author_table_num_rows": author_table_num_rows,
        "author_pad_idx": AUTHOR_PAD_IDX,
        "author_unk_idx": AUTHOR_UNK_IDX,
    }


@contextmanager
def temporary_author_index_override(
    *,
    stage7_bundle_path: Path,
    model_author_map_path: Path,
    author_table_num_rows: int,
    embedding_count: int,
    temporary_dir: Path,
    allow_extra_columns: bool = False,
) -> Iterator[AuthorIndexOverride]:
    """Create one model-specific mapping and remove it on every exit path."""

    temporary_dir = Path(temporary_dir)
    temporary_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="author-index-override-",
        dir=temporary_dir,
    ) as work_dir:
        output_path = Path(work_dir) / "post_author_idx.npy"
        coverage = build_author_index_override(
            stage7_bundle_path=stage7_bundle_path,
            model_author_map_path=model_author_map_path,
            author_table_num_rows=author_table_num_rows,
            embedding_count=embedding_count,
            output_path=output_path,
            allow_extra_columns=allow_extra_columns,
        )
        yield AuthorIndexOverride(
            path=output_path,
            author_table_num_rows=author_table_num_rows,
            coverage=coverage,
        )
