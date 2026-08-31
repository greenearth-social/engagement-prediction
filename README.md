# Engagement Prediction

This repo trains engagement rankers for Bluesky posts using production-shaped user-hour queries:

0. `00_source_metadata`: scan exact root/reply snapshots once and publish canonical URI metadata.
1. `01_query_selection`: load Ingex likes, select bounded `(user, hour)` queries, and retain positives backed by canonical root metadata.
2. `02_user_history`: select the bounded as-of like history for every query.
3. `03_post_selection`: resolve required roots and replies and collect a bounded random root-post candidate reservoir.
4. `04_negative_selection`: calculate as-of candidate popularity and select shared hourly negative pools.
5. `05_post_liker_history`: extract complete timestamped liker-event histories for selected posts.
6. `06_author_statistics`: build unfiltered pre-validation statistics for every author represented by roots or replies.
7. `07_dataset_hydration`: hydrate selected posts, build the training-exposure author vocabulary, write the content-embedding memmap, and publish permanent model-training tables plus a compact loader index.
8. Train a canonical model directly from Stage 7: `08_train_bst_ranker` or `08_train_two_tower`.

Native BST and two-tower training are active through Stage 8. MLP training has been removed. Canonical Stage 8 artifacts can be compared independently against a Stage 7 dataset with `ops/compare_model_performance.py`.

## Setup

Install conda, then create the pinned environment:

```bash
conda-lock install -n eng-pred conda-lock.yml
conda activate eng-pred
python -c "import torch; print(torch.__version__)"
```

If dependencies change, update both `environment.yml` and `environment.ci.yml`, then regenerate lock files:

```bash
conda-lock -f environment.yml -p linux-64 --mamba --lockfile conda-lock.yml
conda-lock -f environment.ci.yml -p linux-64 --mamba --lockfile conda-lock.ci.yml
```

ClearML is the implemented experiment tracker. To use it, run `clearml-init` after activating the environment. For local or test runs without ClearML, pass `--experiment-tracker none`.

## Testing

Run tests from this directory with the project conda environment:

```bash
conda run -n eng-pred pytest -q
```

To keep pytest temporary files inside the repo:

```bash
TMPDIR=$PWD conda run -n eng-pred pytest -q
```

Tests use the `*_test.py` naming convention and live next to the code they cover.

## Repository Layout

- `cli.py`: unified pipeline CLI. `run-all` is implicit, so both invocation forms are accepted.
- `ops/compare_model_performance.py`: standalone comparison of exactly two canonical Stage 8 or standard legacy Stage 3 BST model artifacts on one Stage 7 dataset.
- `engagement_prediction/evaluation/`: reusable artifact validation, author remapping, TorchScript scoring, comparison, and reporting helpers.
- `engagement_prediction/stages/source_metadata.py`: active Stage 00 canonical source-metadata indexing.
- `engagement_prediction/stages/query_selection.py`: active Stage 1 user-hour query selection.
- `engagement_prediction/stages/user_history.py`: active Stage 2 orchestration.
- `engagement_prediction/stages/post_selection.py`: active Stage 3 post-universe orchestration.
- `engagement_prediction/stages/negative_selection.py`: active Stage 4 popularity-aware negative selection orchestration.
- `engagement_prediction/stages/post_liker_history.py`: active Stage 5 post-liker event extraction orchestration.
- `engagement_prediction/stages/author_statistics.py`: active Stage 6 training-only author statistics orchestration.
- `engagement_prediction/stages/dataset_hydration.py`: active Stage 7 dataset-hydration orchestration.
- `engagement_prediction/stages/train_bst_ranker.py`: active Stage 8 native BST-training orchestration.
- `engagement_prediction/stages/train_two_tower.py`: active Stage 8 native two-tower-training orchestration.
- `engagement_prediction/training/bst_ranker.py`: reusable native listwise BST training loop.
- `engagement_prediction/training/two_tower.py`: reusable native listwise two-tower training loop.
- `engagement_prediction/models/two_tower.py`: canonical author-aware cross-attention user and post towers.
- `engagement_prediction/training/ranking.py`: shared matrix-ranking metrics and ranking-row helpers.
- `engagement_prediction/experiment_tracking/`: tracker interface, no-op and ClearML backends, and tracker construction.
- `engagement_prediction/pipeline/logging.py`: canonical stage logging.
- `engagement_prediction/training/runtime.py`: shared training device, CUDA cleanup, and seed helpers.
- `engagement_prediction/data/ingex.py`: reusable Ingex access and exact source-file manifests.
- `engagement_prediction/data/embeddings.py`: content-embedding model dimensions and raw payload decoding.
- `engagement_prediction/data/source_metadata.py`: canonical root/reply normalization, deduplication, precedence, and URI partitioning.
- `engagement_prediction/data/training_index.py`: versioned Stage 7 memory-mapped loader-index construction and validation.
- `engagement_prediction/data/datasets.py`: native bucketed dataset and sampler for the Stage 7 contract.
- `engagement_prediction/data/`: Parquet loading plus reusable like, history, post, and hydration transformations.
- `engagement_prediction/pipeline/{core.py,dependencies.py,registry.py}`: artifact directories, lineage, dependency resolution, and stage registry.
- `legacy/`: archived, unsupported configuration and sweep references, plus the old evaluator under `legacy/evaluation/`.

## Running The Pipeline

The CLI merges defaults, an optional YAML/JSON config, and explicit command-line flags. CLI flags win over config values.

```bash
python cli.py --config config.yml --model-type bst-ranker --stop-after train_bst_ranker
```

For foreground local iteration:

```bash
python cli.py --config config.yml --model-type bst-ranker --stop-after train_bst_ranker --background false --experiment-tracker none
```

### Output Layout

By default, outputs are written under `outputs/` in two coordinated views:

- `artifacts/<stage_folder>/<stage_run_id>/`: canonical stage artifacts.
- `runs/<pipeline_run_id>/<stage_folder>`: symlinks to the canonical artifacts for one pipeline run.

Each stage writes `manifest.json`, `resolved_config.json`, `stage.log`, and `stage_info.txt` when it completes.

### Stage 00: Source Metadata

Stage 00 owns the exact `bsky_posts` and `bsky_replies` snapshots for the common half-open `[posts_start, posts_end)` source window. It normalizes narrow metadata, deduplicates each source by URI using latest creation time and ascending-author tie-breaking, applies root precedence to cross-source collisions, and publishes stable URI-hash partitions.

Common config:

```yaml
gcs_bucket: "greenearth-471522-ingex-extract-prod"
posts_start: "2026-06-20T00:00:00Z"
posts_end: "2026-06-24T00:00:00Z"
source_metadata_partition_count: 16
data_partition_worker_count: 4
```

The atomic `source_metadata_*` bundle contains `post_metadata/` with `subject_uri`, UTC `post_created_at`, `author_did`, and `is_reply`, plus exact `post_sources_*.json` and `reply_sources_*.json` manifests. Each URI appears exactly once. Stages 1, 3, and 6 read this index instead of rescanning raw metadata; Stage 7 reuses the manifests to find embedding payloads in the authoritative raw files.

After each streaming hash-routing pass completes, Stages 00, 2, and 3 process independent partitions with up to `data_partition_worker_count` spawned worker processes. The default is `4`; set it to `1` for the lowest-memory serial path. Partition-derived output filenames and parent-side ordered statistics keep logical artifacts deterministic across worker counts.

### Stage 1: Query Selection

Stage 1 reads Ingex likes and writes production-shaped user-hour queries plus positives found among Stage 00 canonical root rows. It does not list or scan raw posts or replies.

Common config keys:

```yaml
gcs_bucket: "greenearth-471522-ingex-extract-prod"
posts_start: "2026-06-20T00:00:00Z"
posts_end: "2026-06-24T00:00:00Z"
train_start: "2026-06-21T00:00:00Z"
val_start: "2026-06-22T00:00:00Z"
holdout_start: "2026-06-23T00:00:00Z"
holdout_end: null
unseen_user_fraction: 0.10
max_hours_per_user_per_split: 16
max_train_query_hours: null
max_eval_query_hours_per_split: null
max_positives_per_user_hour: 32
source_metadata_partition_count: 16
random_seed: 42
```

Important Stage 1 behavior:

- Users are assigned deterministically to a 90% `trainval` cohort and a 10% `unseen_eval` cohort.
- Seen users produce `train`, `val`, and `holdout_seen_users` queries. Unseen users produce only `val_unseen_users` and `holdout_unseen_users` queries.
- Query-hours are hash-sampled independently of their positive counts. Each user contributes at most 16 query-hours per split by default. The per-user cap and split-wide cap use independent hash namespaces so surviving queries from highly active users do not inherit artificially favorable global ranks.
- Training and each evaluation split can be capped directly with query-hour budgets. The global caps default to no limit for the first slice.
- Sampling caps are applied provisionally using all valid likes. Stage 1 then deduplicates positives within each selected user-hour and retains only URIs present as roots in Stage 00.
- Positive counts are recomputed after post filtering. Selected hours with no retained positives or more than 32 retained positives are discarded without backfill; an hour with exactly 32 is retained.
- There is no minimum-likes eligibility filter.

Primary artifacts are `queries_*.parquet`, keyed by `(did, query_hour)`, and `query_positives_*.parquet`, keyed by `(did, query_hour, subject_uri)`. Stage 1 records the exact Ingex like-file snapshot in `like_sources_*.json`; root/reply snapshot ownership remains with Stage 00. All snapshots use the common half-open `[posts_start, posts_end)` source window.

All source and split boundaries must be UTC and hour-aligned. `posts_start <= train_start < val_start`, and the validation and optional holdout boundaries must remain ordered and fall within the source window. Likes at `posts_start` are available for later history construction, while only likes at or after `train_start` can become targets. Likes and post rows at `posts_end` are excluded. Setting `posts_start == train_start` is valid, but provides no pre-training history warm-up; production datasets should normally use `posts_start < train_start`.

### Stage 2: User History

Stage 2 rescans the exact Stage 1 like-file snapshot and creates the partitioned `query_histories_*` dataset, keyed by `(did, query_hour)`.

The history artifact includes:

- `history_subject_uris`: prior liked post URIs, sorted most-recent first.
- `history_like_created_ats`: aligned UTC like timestamps.

Every Stage 1 query has exactly one row, including explicit empty histories. Only likes strictly before the start of `query_hour` are eligible. Histories use all valid source likes from queried users, not only selected target likes. Exact duplicate `(did, subject_uri, like_created_at)` source events are collapsed within each user partition before the as-of cutoff and history cap; re-likes at different timestamps remain distinct.

Stage 2 also publishes the partitioned `history_post_uris_*` dataset. It contains one globally unique, non-null `subject_uri` for every post retained in at least one query history.

Common config:

```yaml
max_history_posts_per_query: 64
user_history_partition_count: 16
data_partition_worker_count: 4
```

The source history window begins at `posts_start`, using the exact Stage 1 `like_sources_*.json` snapshot. Set `posts_start` earlier than `train_start` when training queries need a warm-up period. Stage 2 performs no post or reply lookup: unresolved URIs remain in its aligned lists for Stage 3 to resolve later.

### Stage 3: Post Selection

Stage 3 combines unique Stage 1 positive URIs with Stage 2 `history_post_uris_*`, then resolves them directly against Stage 00 canonical metadata. Required posts are retained whenever metadata exists, independently of candidate sampling. Stage 3 does not list, scan, normalize, or repartition raw roots and replies.

Common config:

```yaml
random_candidate_sampling_fraction: 0.10
data_partition_worker_count: 4
```

The random reservoir uses a stable URI hash and is approximate rather than exactly sized. Candidates are drawn only from root posts. Replies can enter the universe only when required by at least one retained history; they never become positive labels or candidate posts. If a URI occurs in both source types, root metadata takes precedence. Missing or reply-only positives fail the stage, while unresolved history URIs are reported without rewriting Stage 2 history lists.

Stage 3 atomically publishes `post_universe_*`, containing separate partitioned Parquet datasets:

- `posts/`: `subject_uri`, UTC `post_created_at`, `author_did`, and `is_reply` for every retained universe post.
- `required_posts/`: unique positive/history membership flags.
- `candidate_sources/`: unique `random` reservoir memberships.
- `missing_required_posts/`: required URIs for which valid post metadata was not found.

Exact Stage 00 `bsky_posts` and `bsky_replies` source-file manifests are copied into the same bundle for self-contained provenance. Post-liker histories and non-model feed-policy rules are deferred to later work.

### Stage 4: Popularity-Aware Negative Selection

Stage 4 uses the unique root posts in Stage 3 `candidate_sources/` as its bounded reservoir and constructs one shared negative pool for each distinct Stage 1 query hour. It reuses Stage 1's exact like-file snapshot rather than relisting Ingex data.

Common config:

```yaml
negative_candidates_per_hour: 1000
min_likes_for_popular_candidate: 10
popular_candidate_fraction: 0.50
max_candidate_age_hours: 24
```

For each candidate and query hour, `prior_like_count` counts valid raw like rows strictly before the start of that hour. Likes at the boundary and future likes are excluded, duplicate source rows count independently, and posts without prior likes receive zero. A post is eligible in its UTC creation-hour bucket and the following 23 buckets by default.

Stage 4 first selects the desired popular quota from posts with at least the configured number of prior likes. It then selects uniformly from every remaining eligible post until the hour reaches its target. The random method may therefore select another post that also meets the popularity threshold. Both methods use stable, source-specific hash ranks, and hours with fewer eligible posts report a shortfall rather than sampling with replacement.

The atomically published `negative_candidates_*` bundle contains:

- `hourly_candidates/`: unique `(query_hour, subject_uri)` rows with `selection_source` and `prior_like_count`.
- `negative_post_uris/`: the globally unique selected post URIs needed by later enrichment.
- `like_sources_*.json`: the exact Stage 1 source snapshot used for popularity.

The full candidate-hour popularity matrix is internal staging and is removed after successful publication. These pools are shared across users; later model-ready assembly must treat a user's known positive as positive and omit it from that user's negatives.

### Stage 5: Post-Liker History

Stage 5 constructs the exact selected post universe from resolved Stage 3 positive/history requirements plus Stage 4 final negatives. Unresolved histories and Stage 3 reservoir posts that were not selected as negatives are excluded.

Common config:

```yaml
post_liker_history_partition_count: 16
```

Stage 5 reuses the exact Stage 1 `bsky_likes` snapshot copied into Stage 4; it never relists Ingex files. It scans every valid like in the common source window and retains matching events from all users, independently of query, cohort, target, and user-history sampling. Duplicate source rows remain separate events, and no extraction-time event cap is applied.

The atomically published `post_liker_histories_*` bundle contains:

- `post_liker_events/`: `subject_uri`, raw `liker_did`, and UTC `like_created_at` event rows.
- `post_liker_posts/`: one row per selected post with positive/history/negative role flags, exact event count, and nullable first/last event timestamps.
- `like_sources_*.json`: the exact Stage 1 source snapshot used for extraction.

Later model-ready assembly will enforce `like_created_at < query_hour`, take the most recent configured replay cap, map raw DIDs through a training-supported PAD/UNK vocabulary, and calculate a time-decayed pooled liker embedding. Keeping Stage 5 complete and query-independent allows replay caps and decay settings to change without rescanning Ingex.

### Stage 6: Author Statistics

Stage 6 builds descriptive statistics from all Stage 00 canonical roots/replies and raw received-like events in the half-open `[posts_start, val_start)` window. This includes pre-training warm-up activity but excludes validation and holdout activity. It reads the full Stage 00 metadata index and the exact Stage 1 like snapshot without rescanning raw metadata or restricting itself to sampled Stage 3 posts or Stage 5 selected-post events.

Common config:

```yaml
author_statistics_partition_count: 16
```

Stage 00 has already applied metadata deduplication and root precedence. Stage 6 filters those canonical rows to the support window, while every matching raw like row counts independently. Every in-window author is published; Stage 6 does not decide which authors receive embedding rows.

The atomically published `author_statistics_*` bundle contains:

- `author_statistics/`: one row per author with root/reply post counts, raw received-like counts, liked-post count, mean/median/maximum likes per post, and authored-record timestamp bounds.
- `post_sources_*.json`, `reply_sources_*.json`, and `like_sources_*.json`: the exact aligned source snapshots used by the stage.

These full-support-window statistics are model-independent and remain joinable by `author_did`. They must not be joined directly as query-time model features because an early training query would see later training activity; future author features need as-of construction.

### Stage 7: Dataset Hydration

Stage 7 validates the complete Stage 00-6 lineage and reuses Stage 00's exact post/reply snapshots plus the recorded like snapshot without relisting Ingex. It hydrates exactly the Stage 5 selected universe: positives, resolved histories, and final Stage 4 negatives. For each URI it selects the latest source row containing a finite configured-model embedding of the expected dimension while preserving Stage 3's authoritative creation time, author, and root/reply metadata. An older duplicate source row may supply the embedding when the newer canonical metadata row has no valid configured-model payload.

Stage 7 loads the narrow selected-URI lookup once, then scans raw post and reply files once in bounded batches controlled by `embedding_source_batch_size` (default `64`). Each batch semi-joins the payload rows to the selected keys and stream-writes only selected rows into URI partitions. It validates and decodes independent URI partitions with `embedding_partition_worker_count` worker processes (default `4`), retaining each winning decoded vector so it is written without a second decode. Set the worker count to `1` for the lower-memory serial path. The final memmap is assembled deterministically in URI-partition order.

After missing-embedding filtering and zero-positive query attrition, Stage 7 counts final training-feature exposure by author. Each retained training positive relation, retained history event, and hourly negative row counts once; validation and holdout rows never contribute. Authors with at least `min_author_training_feature_count` occurrences (default `50`) receive deterministic dense indices starting at 2. All remaining authors map to `1=UNK`, while `0=PAD` remains reserved.

The atomically published `hydrated_training_data_*` bundle contains:

- `embeddings.npy`: an exact-sized `Float32[N, embedding_dim]` NumPy memmap.
- `posts/`: the dense embedding index, creation metadata, PAD/UNK-aware author index, and selected-post role flags for each hydrated URI.
- `queries/` and `query_positives/`: surviving Stage 1 queries and their hydrated positive labels.
- `query_histories/`: aligned URI, like-time, embedding-index, author-index, and as-of-like-count lists.
- `hourly_negative_candidates/`: hydrated Stage 4 candidates and their selection source.
- `authors/`: the Stage 7 vocabulary with dense `author_idx` plus total and positive/history/negative training-feature counts.
- `loader_index/`: a versioned training projection with read-only NumPy memmaps for numeric query, history, positive, negative, and post metadata plus memory-mapped Arrow IPC identifier tables.
- exact copied `post_sources_*`, `reply_sources_*`, and `like_sources_*` manifests.

Rows without a valid embedding are removed without backfilling. Their individual history events and negative candidates disappear; positive labels disappear individually, and only queries left with no positive are dropped. Popularity is recomputed from all Stage 5 raw liker events with the strict `like_created_at < query_hour` rule, using the same cumulative hourly counts and backward as-of join as Stage 4. Stage 4 negative counts must agree exactly.

`engagement_prediction.data.datasets.HydratedBucketedEngagementDataset` consumes `loader_index/` directly. Numeric arrays, content embeddings, and Arrow identifier tables are opened read-only and lazily in each process. Histories and positives use flattened values plus offset arrays, and hour offsets drive batching without constructing a Python object per query. Each batch unions its users' positives with shared hourly negatives, deduplicates candidates by dense embedding index, and emits the existing padded tensors, identifiers, and user-by-candidate label matrix. It does not construct legacy `likes_core`, `posts_core`, or `history_posts` frames.

The loader-index format is part of the Stage 7 artifact contract. Stage 8 rejects older Stage 7 bundles without a supported `loader_index/`; regenerate Stage 7 rather than constructing a runtime compatibility cache.
For the current full dataset, expect the uncompressed loader index to add approximately 5.5-6.5 GB to the Stage 7 bundle.

Rerun Stage 7 directly with an aligned Stage 6 artifact:

```bash
python cli.py --config config.yml \
  --start-from dataset_hydration \
  --stop-after dataset_hydration \
  --prior-06-author-statistics 20260816_120000_a7b8c9d0
```

### Stage 8: Native BST Training

Stage 8 consumes the required Stage 7 loader index directly. It trains on `train`, validates on `val`, and selects checkpoints using unseen-user validation NDCG from `val_unseen_users`. Training batches shuffle user-hours and resample their bounded negative pool each epoch; validation is deterministic. The existing train loader switches to deterministic evaluation mode for final metrics instead of creating a second worker pool. Holdout splits are not loaded during training. BST training does not compute or report MAP; NDCG is its ranking metric. Random baselines piggyback on the first train, validation, and unseen-validation passes, then appear in one grouped ClearML histogram at iteration 0 with all-query and zero-history bars; learned scalar metrics begin at iteration 1.

Shared training options:

```yaml
model_type: "bst-ranker"
max_history_len: 64
epochs: 100
batch_size: 128
learning_rate: 3e-4
patience: 10
early_stopping_min_delta: 0.002
metrics_top_ks: [30]
num_dataloader_workers: 4
dataloader_pin_memory: true
dataloader_prefetch_factor: 1
dataloader_persistent_workers: false
```

The BST ranker always fuses content embeddings, Stage 7 author indices, time-delta buckets, optional as-of popularity, and a candidate-aware transformer. Author embeddings are mandatory for the canonical model.

```bash
python cli.py --model-type bst-ranker \
  --prediction-hidden-dims 64 32 16 \
  --stop-after train_bst_ranker
```

BST training uses matrix ranking over same-hour candidate sets with additional sampled negatives. It requires `bst_num_transformer_layers: 1` because it uses the optimized one-layer matrix scorer.

Useful options:

- `--bst-additional-batch-negatives`
- `--content-projection-dim`
- `--author-projection-dim`
- `--bst-model-dim`
- `--bst-time-embedding-dim`
- `--bst-num-attention-heads`
- `--bst-num-transformer-layers`
- `--bst-transformer-ff-dim`
- `--bst-dropout-rate`
- `--bst-time-delta-bucket-boundaries-hours`
- `--bst-max-train-batches-per-epoch`
- `--bst-use-popularity-feature` / `--no-bst-use-popularity-feature`
- `--bst-popularity-projection-dim`

Popularity normalization is fit once from training-only model inputs. Retained history events keep their multiplicity; positive and negative candidates are deduplicated by `(query_hour, subject_uri)`. Stage 8 stores the fitted `log1p` mean/std and observation counts in JSON and in the checkpoint.

The `08_train_bst_ranker/<stage_run_id>/` artifact contains `checkpoints/bst_ranker_best.pth`, the serving-ready TorchScript model at `checkpoints/ranker.pt`, `ranker_author_idx.parquet`, `model_config.json`, `training_config.json`, `popularity_stats.json`, `training_results.json`, an exact copy of `authors/`, and an optional training-history plot. Each new best checkpoint refreshes the local TorchScript model through an atomic write, reload, and exact scoring-parity check. Canonical BST training always writes these local model artifacts. `--no-plots` suppresses the optional plot.

After final evaluation, Stage 8 registers `ranker.pt` once as the ClearML OutputModel named `ranker` and uploads `ranker_author_idx.parquet` as the ordinary `author_idx_mapping` artifact expected by the model-promotion tooling. When both uploads succeed, it writes and uploads `checkpoints/ranker_serving_manifest.json` with `ranker_clearml_model_id`, `ranker_uri`, and `clearml_task_id`. ClearML publication is best-effort: upload failures are reported but leave the validated local model artifacts intact, and Stage 8 does not create an incomplete serving manifest. With `--experiment-tracker none`, the local model and author map are still produced but no serving manifest is written.

### Stage 8: Native Two-Tower Training

The canonical two-tower model consumes the Stage 7 loader index directly. It uses an author-aware, cross-attention-only user encoder and a learned post projection. Both towers always emit L2-normalized vectors, and exact history times, popularity, and candidate ages are not model inputs. Empty histories use a learned token, while history position carries the newest-first recency signal.

```bash
python cli.py --config config.yml \
  --model-type two-tower \
  --output-embedding-dim 128 \
  --stop-after train_two_tower
```

Training uses every Stage 4 hourly negative retained in Stage 7. It uses `batch_size` for training and final train metrics, `eval_batch_size` for both validation splits, and unseen-user validation NDCG for checkpoint selection. Random baselines piggyback on the first split passes and are reported in the same grouped ClearML histogram contract as BST training. The architecture is fixed to the canonical author-aware cross-attention encoder and learned post projection; both towers always L2-normalize their outputs.

Useful options include `--output-embedding-dim` (default `128`), `--user-hidden-dim`, `--post-hidden-dim`, `--content-projection-dim`, `--author-projection-dim`, `--author-embedding-dim`, `--dropout-rate-two-tower`, and `--similarity-temperature`.

The `08_train_two_tower/<stage_run_id>/` artifact contains `checkpoints/two_tower_best.pth`, `checkpoints/engagement_user_tower.pt`, `checkpoints/engagement_post_tower.pt`, `two_tower_author_idx.parquet`, model/training/results JSON, an exact copy of `authors/`, and an optional training plot. Every new best refreshes both local TorchScript towers and verifies eager/script parity, output shapes, finiteness, unit norms, and combined scores. Canonical training always writes these artifacts.

After final evaluation, the towers are registered once as the ClearML OutputModels `engagement_user_tower` and `engagement_post_tower`, and the author map is uploaded as `author_idx_mapping`. A successful complete upload writes `checkpoints/two_tower_serving_manifest.json`, including `output_embedding_dim` and the post-tower model ID as `embedding_space_id`. Remote publication is best-effort and never invalidates the verified local outputs. Deploying a non-128-dimensional model requires a coordinated Elasticsearch index migration and full post re-embedding.

## Compare Model Performance

Use the standalone comparison CLI with one completed Stage 7 artifact and exactly two uniquely named model artifacts. Canonical Stage 8 BST, canonical Stage 8 two-tower, and standard legacy Stage 3 BST artifacts can be mixed in one comparison. Model type is inferred from each artifact's configuration.

```bash
python ops/compare_model_performance.py \
  --dataset /path/to/07_dataset_hydration/<run-id> \
  --model baseline=/path/to/08_train_bst_ranker/<run-id> \
  --model candidate=/path/to/08_train_two_tower/<run-id>
```

To compare a standard legacy BST ranker, pass its completed legacy `03_train` directory. The tool normally resolves the aligned legacy author-index artifact from its recorded lineage. If that input has moved or is otherwise unavailable, provide its exact author map explicitly under the same model name:

```bash
python ops/compare_model_performance.py \
  --dataset /path/to/07_dataset_hydration/<run-id> \
  --model legacy=/path/to/03_train/<run-id> \
  --author-map legacy=/path/to/author_idx_<run-id>.parquet \
  --model canonical=/path/to/08_train_bst_ranker/<run-id>
```

Legacy support is limited to standard BST TorchScript models exposing the same eight-input `score_candidate_matrix` API. Experimental legacy models requiring post-liker or target-user features are rejected. A legacy model is evaluated against the supplied canonical Stage 7 queries, histories, embeddings, and candidate pools; this does not reproduce metrics from its original legacy training dataset.

The dataset argument may also point directly to its `hydrated_training_data_*` bundle. By default the tool evaluates `val`, `val_unseen_users`, `holdout_unseen_users`, and `holdout_seen_users`, using all Stage 7 hourly negatives and each model's configured history length. Results are built atomically under `outputs/comparisons/<run-id>/`; pass `--output-dir` to use another parent directory. Each completed comparison contains `metrics.json`, long-form `metrics.csv`, model-B-minus-model-A `metric_deltas.csv`, `model_specs.json`, `stage_info.txt`, and `comparison.log`. Floating-point result values are rounded to five decimal places while exact model and training configuration metadata remains unchanged. The tool does not create pipeline manifests, tracking tasks, uploads, or ranking-row artifacts.

## Legacy References

`legacy/` contains the old configuration, training sweep, and ranking-row evaluator. These files are retained as historical implementation references only. They are not registered pipeline stages, are not covered by the active test suite, and may use arguments or artifacts that the canonical CLI no longer supports.

## Selective Reruns And Prior Pins

Use `--start-from`, `--stop-after`, and prior pins to reuse artifacts:

```bash
python cli.py --config config.yml \
  --start-from query_selection \
  --stop-after query_selection \
  --prior-00-source-metadata 20260810_120000_00112233
```

Stage 1 validates that the pinned Stage 00 bucket and source window match its configuration. All rewritten downstream stages require Stage 00 lineage; artifacts created before Stage 00 was introduced must be regenerated.

```bash
python cli.py --config config.yml \
  --start-from user_history \
  --stop-after user_history \
  --prior-01-query-selection 20260811_120000_a1b2c3d4
```

Rerun Stage 3 directly from a new Stage 2 artifact:

```bash
python cli.py --config config.yml \
  --start-from post_selection \
  --stop-after post_selection \
  --prior-02-user-history 20260812_120000_d4c3b2a1
```

The Stage 2 manifest supplies and validates the aligned Stage 00 and Stage 1 ancestors.

Rerun Stage 4 directly from an existing Stage 3 artifact:

```bash
python cli.py --config config.yml \
  --start-from negative_selection \
  --stop-after negative_selection \
  --prior-03-post-selection 20260813_120000_c3d4e5f6
```

The Stage 3 manifest supplies and validates the aligned Stage 00 through Stage 2 ancestors.

Rerun Stage 5 directly from an existing Stage 4 artifact:

```bash
python cli.py --config config.yml \
  --start-from post_liker_history \
  --stop-after post_liker_history \
  --prior-04-negative-selection 20260814_120000_e5f6a7b8
```

The Stage 4 manifest supplies and validates the aligned Stage 00 through Stage 3 ancestors.

Rerun Stage 6 directly from an existing Stage 5 artifact:

```bash
python cli.py --config config.yml \
  --start-from author_statistics \
  --stop-after author_statistics \
  --prior-05-post-liker-history 20260815_120000_f6a7b8c9
```

The Stage 5 manifest supplies and validates the aligned Stage 00 through Stage 4 ancestors.

Rerun Stage 7 directly from an existing Stage 6 artifact:

```bash
python cli.py --config config.yml \
  --start-from dataset_hydration \
  --stop-after dataset_hydration \
  --prior-06-author-statistics 20260816_120000_a7b8c9d0
```

The Stage 6 manifest supplies and validates the aligned Stage 00 through Stage 5 ancestors.

Rerun Stage 8 directly from an aligned Stage 7 artifact:

```bash
python cli.py --config config.yml \
  --model-type bst-ranker \
  --start-from train_bst_ranker \
  --stop-after train_bst_ranker \
  --prior-07-dataset-hydration 20260817_120000_b8c9d0e1
```

The Stage 7 manifest supplies and validates the complete Stage 00 through Stage 6 ancestry.

For two-tower training, use the same Stage 7 pin with the two-tower stage:

```bash
python cli.py --config config.yml \
  --model-type two-tower \
  --start-from train_two_tower \
  --stop-after train_two_tower \
  --prior-07-dataset-hydration 20260817_120000_b8c9d0e1
```

Accepted stage aliases:

- `source_metadata`
- `query_selection`
- `user_history`
- `post_selection`
- `negative_selection`
- `post_liker_history`
- `author_statistics`
- `dataset_hydration`
- `train`, `train_two_tower`, `train_bst_ranker`

Prior pins can be stage run ids under `artifacts/<stage_folder>/`, absolute paths, or paths relative to `output_dir`.

## Background Runs

By default, `config.yml` may set `background: true`. In background mode, the CLI writes `run-all.resolved-config.json` and starts a foreground child process with `nohup`.

Run in the foreground while iterating:

```bash
python cli.py --config config.yml --model-type bst-ranker --stop-after train_bst_ranker --background false
```

## Development Notes

- Treat `08_train_bst_ranker` and `08_train_two_tower` as the active pipeline boundaries; use the standalone comparison tool for aggregate evaluation across their canonical artifacts.
- Use `engagement_prediction/training/ranking.py` for matrix ranking metrics and ranking-row writes.
- Avoid adding new training paths without registering them in `engagement_prediction/pipeline/registry.py` and documenting their artifact contract here.

## Contributing

Interested in contributing? Please join the Discord and introduce yourself first: https://discord.com/invite/8bWEyrkrJC.
