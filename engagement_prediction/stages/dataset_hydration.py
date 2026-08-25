"""Stage 7: hydrate the permanent model-training dataset contract.

Stages 00-6 deliberately keep selection, history construction, popularity,
and author support separate. This stage joins those decisions into the tables
read directly by the native dataloader. It is also the only stage that decodes
content embeddings and creates the final NumPy memmap.

The implementation uses several disk-backed layouts because different joins
need different keys: Stage 00 URI partitions for post hydration, Stage 5 URI
partitions for as-of liker counts, and Stage 2 DID partitions for rebuilding
query histories. All temporary layouts live under the stage's partial staging
directory and are removed before the public bundle is atomically published.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict

import polars as pl

from engagement_prediction.data import author_statistics
from engagement_prediction.data import dataset_hydration_artifacts
from engagement_prediction.data import ingex
from engagement_prediction.data import source_manifests
from engagement_prediction.data.parquet import find_artifact_path, load_parquet_from_prior
from engagement_prediction.data.source_metadata_artifacts import (
    SourceMetadataArtifact,
    load_source_metadata_artifact,
)
from engagement_prediction.pipeline.core import Context
from engagement_prediction.pipeline.lineage import resolve_recorded_stage_lineage
from shared.input_data_helpers import get_embedding_dim_for_known_model
from utils.helpers import get_stage_logger


def _load_parameter(stage_dir: Path, name: str) -> int:
    """Load a positive physical partition count from an upstream summary.

    A downstream disk join must use the same hash layout as its producer. The
    count is therefore artifact metadata, not a Stage 7 tuning decision.
    """

    try:
        summary = json.loads((stage_dir / "summary.json").read_text())
        value = int(summary["parameters"][name])
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{stage_dir} does not record a valid {name}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _validate_and_load_snapshots(
    *,
    author_bundle: Path,
    post_bundle: Path,
    post_liker_bundle: Path,
    source_artifact: SourceMetadataArtifact,
) -> tuple[
    source_manifests.SourceSnapshot,
    source_manifests.SourceSnapshot,
    source_manifests.SourceSnapshot,
]:
    """Verify Stage 3, 5, and 6 retained one aligned source snapshot.

    Stage 7 combines artifacts produced at different times. These equality
    checks prevent silent training-data corruption from mixing snapshots even
    when each individual artifact is otherwise schema-valid.
    """

    author_post = source_manifests.load_source_snapshot(
        author_bundle,
        manifest_prefix="post_sources_",
        expected_blob_prefix="bsky_posts",
    )
    author_reply = source_manifests.load_source_snapshot(
        author_bundle,
        manifest_prefix="reply_sources_",
        expected_blob_prefix="bsky_replies",
    )
    author_like = source_manifests.load_source_snapshot(
        author_bundle,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    stage3_post = source_manifests.load_source_snapshot(
        post_bundle,
        manifest_prefix="post_sources_",
        expected_blob_prefix="bsky_posts",
    )
    stage3_reply = source_manifests.load_source_snapshot(
        post_bundle,
        manifest_prefix="reply_sources_",
        expected_blob_prefix="bsky_replies",
    )
    stage5_like = source_manifests.load_source_snapshot(
        post_liker_bundle,
        manifest_prefix="like_sources_",
        expected_blob_prefix="bsky_likes",
    )
    if stage3_post.manifest != source_artifact.post_snapshot.manifest:
        raise ValueError("Stage 3 post snapshot does not match Stage 00")
    if stage3_reply.manifest != source_artifact.reply_snapshot.manifest:
        raise ValueError("Stage 3 reply snapshot does not match Stage 00")
    if author_post.manifest != source_artifact.post_snapshot.manifest:
        raise ValueError("Stage 6 post snapshot does not match Stage 00")
    if author_reply.manifest != source_artifact.reply_snapshot.manifest:
        raise ValueError("Stage 6 reply snapshot does not match Stage 00")
    if author_like.manifest != stage5_like.manifest:
        raise ValueError("Stage 6 like snapshot does not match Stage 5")
    source_manifests.validate_aligned_source_snapshots(
        (author_post, author_reply, author_like),
        description="Stage 7 post, reply, and like snapshots",
    )
    return author_post, author_reply, author_like


def run(context: Context, args: argparse.Namespace) -> Dict[str, Any]:
    """Run Stage 7 and publish the permanent model-training data contract."""

    out_dir = context.new_stage_dir("07_dataset_hydration")
    logger = get_stage_logger("07_DATASET_HYDRATION", log_file=out_dir / "stage.log")
    started_at = time.time()
    embedding_model = str(args.embedding_model)
    embedding_dim = get_embedding_dim_for_known_model(embedding_model)
    embedding_source_batch_size = int(args.embedding_source_batch_size)
    min_author_training_feature_count = int(args.min_author_training_feature_count)
    if embedding_source_batch_size <= 0:
        raise ValueError("embedding_source_batch_size must be positive")
    if min_author_training_feature_count < 1:
        raise ValueError("min_author_training_feature_count must be at least 1")

    logger.info("Phase 1/9: resolving and validating Stage 00-6 lineage")
    lineage = resolve_recorded_stage_lineage(
        context,
        terminal_stage_folder="06_author_statistics",
        ancestor_stage_folders=(
            "00_source_metadata",
            "01_query_selection",
            "02_user_history",
            "03_post_selection",
            "04_negative_selection",
            "05_post_liker_history",
        ),
    )
    source_metadata_dir = lineage["00_source_metadata"]
    author_statistics_dir = lineage["06_author_statistics"]
    post_liker_history_dir = lineage["05_post_liker_history"]
    negative_selection_dir = lineage["04_negative_selection"]
    post_selection_dir = lineage["03_post_selection"]
    user_history_dir = lineage["02_user_history"]
    query_selection_dir = lineage["01_query_selection"]
    source_artifact = load_source_metadata_artifact(source_metadata_dir)
    stage3_partition_count = source_artifact.partition_count
    stage5_partition_count = _load_parameter(
        post_liker_history_dir,
        "post_liker_history_partition_count",
    )
    stage2_partition_count = _load_parameter(
        user_history_dir,
        "user_history_partition_count",
    )
    author_partition_count = _load_parameter(
        author_statistics_dir,
        "author_statistics_partition_count",
    )

    post_bundle = find_artifact_path(post_selection_dir, "post_universe_")
    negative_bundle = find_artifact_path(negative_selection_dir, "negative_candidates_")
    post_liker_bundle = find_artifact_path(post_liker_history_dir, "post_liker_histories_")
    author_bundle = find_artifact_path(author_statistics_dir, "author_statistics_")
    post_snapshot, reply_snapshot, like_snapshot = _validate_and_load_snapshots(
        author_bundle=author_bundle,
        post_bundle=post_bundle,
        post_liker_bundle=post_liker_bundle,
        source_artifact=source_artifact,
    )
    source_start = post_snapshot.start
    source_end = post_snapshot.end
    logger.info(
        "Starting dataset hydration: source_window=[%s, %s) model=%s dim=%s "
        "post_partitions=%s liker_partitions=%s user_partitions=%s "
        "author_partitions=%s embedding_source_batch_size=%s "
        "min_author_training_feature_count=%s",
        source_start.isoformat(),
        source_end.isoformat(),
        embedding_model,
        embedding_dim,
        stage3_partition_count,
        stage5_partition_count,
        stage2_partition_count,
        author_partition_count,
        embedding_source_batch_size,
        min_author_training_feature_count,
    )

    # Stage 3 owns canonical selected-post metadata; Stage 4 owns the final
    # hourly negative choices; Stage 5 owns the selected role universe and all
    # liker events; Stage 6 owns model-independent global author statistics.
    # The final author vocabulary is derived later from Stage 7's surviving
    # training features.
    stage3_posts_path = post_bundle / "posts"
    hourly_candidates_path = negative_bundle / "hourly_candidates"
    post_liker_events_path = post_liker_bundle / "post_liker_events"
    post_liker_posts_path = post_liker_bundle / "post_liker_posts"
    upstream_author_statistics_path = author_bundle / "author_statistics"
    for path in (
        stage3_posts_path,
        hourly_candidates_path,
        post_liker_events_path,
        post_liker_posts_path,
        upstream_author_statistics_path,
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"Required Stage 7 input dataset is missing: {path}")
    upstream_author_statistics_lf = pl.scan_parquet(
        sorted(upstream_author_statistics_path.rglob("*.parquet"))
    )
    if upstream_author_statistics_lf.collect_schema() != pl.Schema(
        author_statistics.AUTHOR_STAT_SCHEMA
    ):
        raise ValueError(
            "Stage 6 uses the obsolete indexed/filtered author schema; "
            "rerun Stages 6-8"
        )

    # Queries and aligned histories stay lazy here. They are not expanded and
    # eagerly loaded until a bounded URI or DID partition needs them.
    queries_lf = load_parquet_from_prior(query_selection_dir, "queries_")
    query_positives_lf = load_parquet_from_prior(query_selection_dir, "query_positives_")
    query_histories_lf = load_parquet_from_prior(user_history_dir, "query_histories_")
    hourly_candidates_lf = pl.scan_parquet(sorted(hourly_candidates_path.rglob("*.parquet")))

    artifact_suffix = out_dir.name
    # ``partial_bundle_path`` is the future public artifact. It contains only
    # final-contract files. All disposable shuffles live under the separate
    # staging root below, so they cannot leak into the published bundle.
    bundle_path = out_dir / f"hydrated_training_data_{artifact_suffix}"
    partial_bundle_path = out_dir / f"hydrated_training_data_{artifact_suffix}.partial"
    partial_bundle_path.mkdir(parents=True, exist_ok=False)
    embeddings_path = partial_bundle_path / "embeddings.npy"
    posts_path = partial_bundle_path / "posts"
    queries_path = partial_bundle_path / "queries"
    positives_path = partial_bundle_path / "query_positives"
    histories_path = partial_bundle_path / "query_histories"
    negatives_path = partial_bundle_path / "hourly_negative_candidates"
    authors_path = partial_bundle_path / "authors"
    copied_manifests = []
    for prefix, manifest in (
        ("post_sources", post_snapshot.manifest),
        ("reply_sources", reply_snapshot.manifest),
        ("like_sources", like_snapshot.manifest),
    ):
        path = partial_bundle_path / f"{prefix}_{artifact_suffix}.json"
        ingex.write_source_manifest(path, manifest)
        copied_manifests.append(path)
    manifest_output_paths = {
        path.name.split(f"_{artifact_suffix}")[0] + "_path": bundle_path / path.name
        for path in copied_manifests
    }

    staging_root = out_dir / f"_dataset_hydration_staging_{artifact_suffix}.partial"
    staging_root.mkdir(parents=True, exist_ok=False)
    # Stage 5 role rows are first routed to Stage 00/3 URI partitions and
    # joined with Stage 3 metadata. ``selected_metadata`` is the authoritative
    # selected URI/role/creation/author table for the rest of this stage.
    selected_routes_path = staging_root / "selected_post_routes"
    selected_metadata_path = staging_root / "selected_metadata"
    # Phase 3 keeps encoded raw payload rows, including duplicate source rows.
    # Phase 4 reduces them to one valid vector per URI in NumPy shards.
    selected_embedding_rows_path = staging_root / "selected_embedding_rows"
    source_scan_staging_path = staging_root / "embedding_source_scans"
    valid_embedding_rows_path = staging_root / "valid_embedding_rows"
    embedding_shards_path = staging_root / "embedding_shards"

    # Stage 5 defines the exact selected URI universe and role flags; Stage 3
    # supplies authoritative creation, author, and root/reply metadata.
    logger.info("Phase 2/9: preparing Stage 5 selected-post metadata")
    dataset_hydration_artifacts.route_selected_posts(
        post_liker_posts_path=post_liker_posts_path,
        output_path=selected_routes_path,
        partition_count=stage3_partition_count,
    )
    selected_stats = dataset_hydration_artifacts.build_selected_metadata(
        stage3_posts_path=stage3_posts_path,
        selected_post_routes_path=selected_routes_path,
        output_path=selected_metadata_path,
        partition_count=stage3_partition_count,
        logger=logger,
    )

    # Load selected URI keys once, then discover and read matching payloads in
    # bounded multi-file batches. Full selected metadata remains partitioned
    # for the URI-aligned memmap publication in Phase 4.
    logger.info(
        "Phase 3/9: batched two-pass filtering of exact root/reply snapshots "
        "to selected embedding rows"
    )
    source_stats = dataset_hydration_artifacts.materialize_selected_embedding_rows(
        post_paths=list(source_artifact.post_snapshot.file_uris),
        reply_paths=list(source_artifact.reply_snapshot.file_uris),
        posts_start=source_start,
        posts_end=source_end,
        selected_metadata_path=selected_metadata_path,
        output_path=selected_embedding_rows_path,
        temporary_routes_root=source_scan_staging_path,
        partition_count=stage3_partition_count,
        source_batch_size=embedding_source_batch_size,
        logger=logger,
    )

    # Each URI partition selects one latest valid payload and writes a NumPy
    # shard. Partition-order concatenation makes emb_idx dense and aligned with
    # the public posts rows.
    logger.info("Phase 4/9: selecting vectors and publishing the exact memmap/post index")
    embedding_stats = dataset_hydration_artifacts.write_embedding_shards(
        selected_embedding_rows_path=selected_embedding_rows_path,
        valid_embedding_rows_path=valid_embedding_rows_path,
        embedding_shards_path=embedding_shards_path,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        partition_count=stage3_partition_count,
        logger=logger,
    )
    # From this point onward the selected payload Parquet is redundant: every
    # surviving URI has an aligned key row and decoded NumPy shard entry.
    shutil.rmtree(selected_embedding_rows_path)
    logger.info(
        "Released filtered embedding source rows before allocating the final memmap"
    )
    hydrated_post_metadata_path = staging_root / "hydrated_post_metadata"
    post_metadata_stats = dataset_hydration_artifacts.publish_embeddings_and_post_metadata(
        selected_metadata_path=selected_metadata_path,
        valid_embedding_rows_path=valid_embedding_rows_path,
        embedding_shards_path=embedding_shards_path,
        embeddings_path=embeddings_path,
        hydrated_post_metadata_path=hydrated_post_metadata_path,
        embedding_dim=embedding_dim,
        partition_count=stage3_partition_count,
    )

    # Missing embeddings remove individual uses without replacement. Queries
    # are dropped later only if every positive use disappeared.
    logger.info("Phase 5/9: filtering positive, history, and negative uses by hydrated posts")
    positive_routes_path = staging_root / "positive_routes"
    history_routes_path = staging_root / "history_routes"
    negative_routes_path = staging_root / "negative_routes"
    dataset_hydration_artifacts.materialize_usage_routes(
        query_positives_lf=query_positives_lf,
        query_histories_lf=query_histories_lf,
        hourly_candidates_lf=hourly_candidates_lf,
        positive_routes_path=positive_routes_path,
        history_routes_path=history_routes_path,
        negative_routes_path=negative_routes_path,
        partition_count=stage3_partition_count,
    )
    hydrated_positives_path = staging_root / "hydrated_positives"
    hydrated_histories_path = staging_root / "hydrated_histories"
    hydrated_negatives_path = staging_root / "hydrated_negatives"
    usage_stats = dataset_hydration_artifacts.hydrate_usage_partitions(
        hydrated_post_metadata_path=hydrated_post_metadata_path,
        positive_routes_path=positive_routes_path,
        history_routes_path=history_routes_path,
        negative_routes_path=negative_routes_path,
        hydrated_positives_path=hydrated_positives_path,
        hydrated_histories_path=hydrated_histories_path,
        hydrated_negatives_path=hydrated_negatives_path,
        partition_count=stage3_partition_count,
    )

    # Stage 5 events are the source of truth for all roles. Recomputing counts
    # here gives one strict as-of contract and checks Stage 4 negative counts.
    logger.info("Phase 6/9: calculating strict as-of like counts from Stage 5 events")
    liker_positive_routes = staging_root / "liker_positive_routes"
    liker_history_routes = staging_root / "liker_history_routes"
    liker_negative_routes = staging_root / "liker_negative_routes"
    dataset_hydration_artifacts.route_hydrated_usage_for_counts(
        hydrated_paths=(hydrated_positives_path, hydrated_histories_path, hydrated_negatives_path),
        output_paths=(liker_positive_routes, liker_history_routes, liker_negative_routes),
        partition_count=stage5_partition_count,
    )
    counted_positives_path = staging_root / "counted_positives"
    counted_histories_path = staging_root / "counted_histories"
    counted_negatives_path = staging_root / "counted_negatives"
    count_stats = dataset_hydration_artifacts.attach_prior_counts(
        positive_routes_path=liker_positive_routes,
        history_routes_path=liker_history_routes,
        negative_routes_path=liker_negative_routes,
        post_liker_events_path=post_liker_events_path,
        counted_positives_path=counted_positives_path,
        counted_histories_path=counted_histories_path,
        counted_negatives_path=counted_negatives_path,
        partition_count=stage5_partition_count,
        logger=logger,
    )

    logger.info(
        "Phase 7/9: building the training-exposure author vocabulary and applying indices"
    )
    author_exposure_routes_path = staging_root / "author_exposure_routes"
    eligible_author_shards_path = staging_root / "eligible_author_shards"
    vocabulary_stats = dataset_hydration_artifacts.build_author_vocabulary(
        queries_lf=queries_lf,
        counted_positives_path=counted_positives_path,
        counted_histories_path=counted_histories_path,
        counted_negatives_path=counted_negatives_path,
        exposure_routes_path=author_exposure_routes_path,
        eligible_shards_path=eligible_author_shards_path,
        authors_path=authors_path,
        min_training_feature_count=min_author_training_feature_count,
        partition_count=author_partition_count,
        logger=logger,
    )
    authors_df = pl.read_parquet(sorted(authors_path.rglob("*.parquet")))
    post_index_stats = dataset_hydration_artifacts.publish_posts_with_author_indices(
        hydrated_post_metadata_path=hydrated_post_metadata_path,
        authors_df=authors_df,
        posts_path=posts_path,
        partition_count=stage3_partition_count,
    )
    indexed_positives_path = staging_root / "indexed_positives"
    indexed_histories_path = staging_root / "indexed_histories"
    indexed_negatives_path = staging_root / "indexed_negatives"
    author_mapping_stats = dataset_hydration_artifacts.attach_author_indices_to_usage(
        counted_paths=(
            counted_positives_path,
            counted_histories_path,
            counted_negatives_path,
        ),
        indexed_paths=(
            indexed_positives_path,
            indexed_histories_path,
            indexed_negatives_path,
        ),
        authors_df=authors_df,
        partition_count=stage5_partition_count,
    )
    post_stats = {**post_metadata_stats, **post_index_stats}

    # Histories were relationalized for filtering. Rebuild their lists using
    # explicit original positions so all feature arrays remain aligned.
    logger.info("Phase 8/9: rebuilding aligned public query artifacts")
    query_stats = dataset_hydration_artifacts.publish_query_artifacts(
        queries_lf=queries_lf,
        counted_positives_path=indexed_positives_path,
        counted_histories_path=indexed_histories_path,
        queries_path=queries_path,
        query_positives_path=positives_path,
        query_histories_path=histories_path,
        staging_path=staging_root / "query_publication",
        partition_count=stage2_partition_count,
    )
    negative_stats = dataset_hydration_artifacts.publish_negative_candidates(
        counted_negatives_path=indexed_negatives_path,
        retained_query_hours=query_stats.pop("retained_query_hours"),
        output_path=negatives_path,
    )
    author_usage_by_split = (
        dataset_hydration_artifacts.summarize_author_index_usage_by_split(
            queries_path=queries_path,
            query_positives_path=positives_path,
            query_histories_path=histories_path,
            hourly_negative_candidates_path=negatives_path,
        )
    )

    logger.info("Phase 9/9: validating and atomically publishing the hydrated bundle")
    validation_stats = dataset_hydration_artifacts.validate_public_bundle(
        embeddings_path=embeddings_path,
        posts_path=posts_path,
        queries_path=queries_path,
        query_positives_path=positives_path,
        query_histories_path=histories_path,
        hourly_negative_candidates_path=negatives_path,
        authors_path=authors_path,
        min_author_training_feature_count=min_author_training_feature_count,
        embedding_dim=embedding_dim,
    )
    # A successful validation is the only path that removes diagnostic staging
    # and promotes the public partial directory to its final artifact name.
    # Exceptions leave partial state behind and no completed stage manifest.
    shutil.rmtree(staging_root)
    partial_bundle_path.replace(bundle_path)
    runtime_seconds = time.time() - started_at
    summary = {
        "parameters": {
            "embedding_model": embedding_model,
            "embedding_dim": embedding_dim,
            "embedding_source_batch_size": embedding_source_batch_size,
            "min_author_training_feature_count": min_author_training_feature_count,
            "source_metadata_partition_count": stage3_partition_count,
            "post_liker_history_partition_count": stage5_partition_count,
            "user_history_partition_count": stage2_partition_count,
            "author_statistics_partition_count": author_partition_count,
        },
        "input": {
            "source_metadata_dir": str(source_metadata_dir),
            "author_statistics_dir": str(author_statistics_dir),
            "post_liker_history_dir": str(post_liker_history_dir),
            "negative_selection_dir": str(negative_selection_dir),
            "post_selection_dir": str(post_selection_dir),
            "user_history_dir": str(user_history_dir),
            "query_selection_dir": str(query_selection_dir),
            "source_start": source_start.isoformat(),
            "source_end": source_end.isoformat(),
        },
        "selected_posts": selected_stats,
        "embedding_sources": source_stats,
        "embeddings": embedding_stats,
        "posts": post_stats,
        "usage": usage_stats,
        "popularity": count_stats,
        "author_vocabulary": vocabulary_stats,
        "author_mapping": author_mapping_stats,
        "author_index_usage_by_split": author_usage_by_split,
        "queries": query_stats,
        "negatives": negative_stats,
        "public_validation": validation_stats,
        "outputs": {
            "hydrated_training_data_path": bundle_path.name,
            "embeddings_path": str(Path(bundle_path.name) / "embeddings.npy"),
            "posts_path": str(Path(bundle_path.name) / "posts"),
            "queries_path": str(Path(bundle_path.name) / "queries"),
            "query_positives_path": str(Path(bundle_path.name) / "query_positives"),
            "query_histories_path": str(Path(bundle_path.name) / "query_histories"),
            "hourly_negative_candidates_path": str(Path(bundle_path.name) / "hourly_negative_candidates"),
            "authors_path": str(Path(bundle_path.name) / "authors"),
            **{
                name: str(Path(bundle_path.name) / path.name)
                for name, path in manifest_output_paths.items()
            },
        },
        "runtime_seconds": runtime_seconds,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (out_dir / "stage_info.txt").write_text(
        "\n".join([
            "stage: dataset_hydration",
            f"runtime_seconds: {runtime_seconds:.2f}",
            f"embedding_model: {embedding_model}",
            f"embedding_dim: {embedding_dim}",
            f"embedding_source_batch_size: {embedding_source_batch_size}",
            f"min_author_training_feature_count: {min_author_training_feature_count}",
            f"hydrated_post_count: {post_stats['hydrated_post_count']}",
            f"author_vocabulary_count: {vocabulary_stats['eligible_author_count']}",
            f"author_table_num_rows: {vocabulary_stats['public_validation']['author_table_num_rows']}",
            f"retained_query_count: {query_stats['retained_query_count']}",
            f"retained_positive_count: {query_stats['retained_positive_count']}",
            f"retained_history_item_count: {query_stats['retained_history_item_count']}",
            f"retained_negative_count: {negative_stats['retained_negative_count']}",
            f"hydrated_training_data_path: {bundle_path.name}",
            "embeddings_path: embeddings.npy",
            "posts_path: posts",
            "queries_path: queries",
            "query_positives_path: query_positives",
            "query_histories_path: query_histories",
            "hourly_negative_candidates_path: hourly_negative_candidates",
            "authors_path: authors",
            *[
                f"{name}: {path.name}"
                for name, path in sorted(manifest_output_paths.items())
            ],
        ]) + "\n"
    )
    logger.info(
        "Dataset hydration completed in %.2fs: posts=%s queries=%s positives=%s histories=%s negatives=%s authors=%s",
        runtime_seconds,
        f"{post_stats['hydrated_post_count']:,}",
        f"{query_stats['retained_query_count']:,}",
        f"{query_stats['retained_positive_count']:,}",
        f"{query_stats['retained_history_item_count']:,}",
        f"{negative_stats['retained_negative_count']:,}",
        f"{vocabulary_stats['eligible_author_count']:,}",
    )
    return {
        "output_dir": out_dir,
        "artifacts": {
            "hydrated_training_data_path": str(bundle_path),
            "embeddings_path": str(bundle_path / "embeddings.npy"),
            "posts_path": str(bundle_path / "posts"),
            "queries_path": str(bundle_path / "queries"),
            "query_positives_path": str(bundle_path / "query_positives"),
            "query_histories_path": str(bundle_path / "query_histories"),
            "hourly_negative_candidates_path": str(bundle_path / "hourly_negative_candidates"),
            "authors_path": str(bundle_path / "authors"),
            **{
                name: str(path)
                for name, path in manifest_output_paths.items()
            },
        },
    }
