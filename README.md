# Engagement Prediction

This repo trains and evaluates engagement rankers for Bluesky posts. The artifact pipeline is being migrated to production-shaped user-hour queries:

1. `01_query_selection`: load Ingex likes, select bounded `(user, hour)` queries, and retain positive posts found in the selected `bsky_posts` snapshot.
2. `02_user_history`: select the bounded as-of like history for every query.
3. `03_post_selection`: resolve required roots and replies and collect a bounded random root-post candidate reservoir.

Post selection is currently the explicit new-pipeline boundary. Run the new data path with `--stop-after post_selection`; it cannot yet continue into unchanged training. Direct legacy training reruns require explicit aligned `01_get_data` and legacy `02_user_history` pins.

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

Tests use the `*_test.py` naming convention and live next to the code they cover as modules are migrated into `engagement_prediction/`. Legacy tests remain under `tests/` during the transition.

## Repository Layout

- `cli.py`: unified pipeline CLI. `run-all` is implicit, so both invocation forms are accepted; the new data path currently requires `--stop-after post_selection`.
- `compare.py`: checkpoint-backed ranker comparison CLI.
- `engagement_prediction/stages/query_selection.py`: active Stage 1 user-hour query selection.
- `engagement_prediction/stages/user_history.py`: active Stage 2 orchestration.
- `engagement_prediction/stages/post_selection.py`: active Stage 3 post-universe orchestration.
- `engagement_prediction/data/ingex.py`: reusable Ingex access and exact source-file manifests.
- `engagement_prediction/data/`: Parquet loading plus reusable like, history, and post transformations.
- `utils/01_get_data/stage_get_data.py`: unregistered legacy Stage 1 retained while downstream stages still consume its artifacts.
- `utils/02_user_history/stage_generate_user_history.py`: unregistered legacy Stage 2 retained for pinned legacy training artifacts.
- `utils/03_train/stage_train_mlp.py`: legacy MLP matrix ranker.
- `utils/03_train/stage_train_two_tower.py`: legacy two-tower matrix ranker.
- `utils/03_train/stage_train_bst_ranker.py`: legacy BST heavy ranker.
- `utils/04_evaluate/stage_evaluate.py`: legacy holdout evaluation from compact ranking-row artifacts.
- `utils/dataloaders.py`: bucketed listwise datasets, samplers, and shared user encoders.
- `utils/matrix_ranking.py`: shared matrix ranking metrics, final metric logging, and ranking-row writers.
- `utils/ranking_adapters.py`: `.pth` checkpoint adapters for compare-rankers.
- `engagement_prediction/pipeline/{core.py,dependencies.py,registry.py}`: artifact directories, lineage, dependency resolution, and stage registry.

## Running The Pipeline

The CLI merges defaults, an optional YAML/JSON config, and explicit command-line flags. CLI flags win over config values.

```bash
python cli.py --config config.yml --stop-after post_selection
```

For foreground local iteration:

```bash
python cli.py --config config.yml --stop-after post_selection --background false --experiment-tracker none
```

### Output Layout

By default, outputs are written under `outputs/` in two coordinated views:

- `artifacts/<stage_folder>/<stage_run_id>/`: canonical stage artifacts.
- `runs/<pipeline_run_id>/<stage_folder>`: symlinks to the canonical artifacts for one pipeline run.

Each stage writes `manifest.json`, `resolved_config.json`, `stage.log`, and `stage_info.txt` when it completes.

### Stage 1: Query Selection

Stage 1 reads Ingex likes and posts from GCS and writes production-shaped user-hour queries plus positive posts found in the selected `bsky_posts` snapshot.

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
max_hours_per_user_per_split: 64
max_train_query_hours: null
max_eval_query_hours_per_split: null
max_positives_per_user_hour: 32
post_selection_partition_count: 256
random_seed: 42
```

Important Stage 1 behavior:

- Users are assigned deterministically to a 90% `trainval` cohort and a 10% `unseen_eval` cohort.
- Seen users produce `train`, `val`, and `holdout_seen_users` queries. Unseen users produce only `val_unseen_users` and `holdout_unseen_users` queries.
- Query-hours are hash-sampled independently of their positive counts. Each user contributes at most 64 query-hours per split by default.
- Training and each evaluation split can be capped directly with query-hour budgets. The global caps default to no limit for the first slice.
- Sampling caps are applied provisionally using all valid likes. Stage 1 then deduplicates positives within each selected user-hour and retains only URIs present in valid `bsky_posts` rows.
- Positive counts are recomputed after post filtering. Selected hours with no retained positives or more than 32 retained positives are discarded without backfill; an hour with exactly 32 is retained.
- There is no minimum-likes eligibility filter.

Primary artifacts are `queries_*.parquet`, keyed by `(did, query_hour)`, and `query_positives_*.parquet`, keyed by `(did, query_hour, subject_uri)`. Stage 1 records the exact Ingex like-file snapshot in `like_sources_*.json` for Stage 2 and the exact membership-check post snapshot in `post_sources_*.json`. Both snapshots use the common half-open `[posts_start, posts_end)` source window.

All source and split boundaries must be UTC and hour-aligned. `posts_start <= train_start < val_start`, and the validation and optional holdout boundaries must remain ordered and fall within the source window. Likes at `posts_start` are available for later history construction, while only likes at or after `train_start` can become targets. Likes and post rows at `posts_end` are excluded. Setting `posts_start == train_start` is valid, but provides no pre-training history warm-up; production datasets should normally use `posts_start < train_start`.

### Stage 2: User History

Stage 2 rescans the exact Stage 1 like-file snapshot and creates the partitioned `query_histories_*` dataset, keyed by `(did, query_hour)`.

The history artifact includes:

- `history_subject_uris`: prior liked post URIs, sorted most-recent first.
- `history_like_created_ats`: aligned UTC like timestamps.

Every Stage 1 query has exactly one row, including explicit empty histories. Only likes strictly before the start of `query_hour` are eligible. Histories use all valid source likes from queried users, not only selected target likes, and preserve duplicate source events.

Stage 2 also publishes the partitioned `history_post_uris_*` dataset. It contains one globally unique, non-null `subject_uri` for every post retained in at least one query history. Duplicate source events remain duplicated in `query_histories_*`; only this compact metadata-lookup artifact is deduplicated.

Common config:

```yaml
max_history_posts_per_query: 64
user_history_partition_count: 256
```

The source history window begins at `posts_start`, using the exact Stage 1 `like_sources_*.json` snapshot. Set `posts_start` earlier than `train_start` when training queries need a warm-up period. Stage 2 performs no post or reply lookup: unresolved URIs and duplicate raw events remain in its aligned lists for Stage 3 to resolve later.

### Stage 3: Post Selection

Stage 3 combines unique Stage 1 positive URIs with Stage 2 `history_post_uris_*`, then resolves them against root posts and replies from the common source window. It reuses Stage 1's exact root-post snapshot and records an exact `bsky_replies` snapshot. Required posts are retained whenever metadata exists, independently of candidate sampling.

Common config:

```yaml
posts_start: "2026-06-20T00:00:00Z"
posts_end: "2026-06-24T00:00:00Z"
random_candidate_sampling_fraction: 0.10
post_selection_partition_count: 256
```

The random reservoir uses a stable URI hash and is approximate rather than exactly sized. Candidates are drawn only from root posts. Replies can enter the universe only when required by at least one retained history; they never become positive labels or candidate posts. If a URI occurs in both source types, root metadata takes precedence. Missing or reply-only positives fail the stage, while unresolved history URIs are reported without rewriting Stage 2 history lists.

Stage 3 atomically publishes `post_universe_*`, containing separate partitioned Parquet datasets:

- `posts/`: `subject_uri`, UTC `post_created_at`, `author_did`, and `is_reply` for every retained universe post.
- `required_posts/`: unique positive/history membership flags.
- `candidate_sources/`: unique `random` reservoir memberships.
- `missing_required_posts/`: required URIs for which valid post metadata was not found.

Exact `bsky_posts` and `bsky_replies` source-file manifests are stored in the same bundle. Popularity, post-liker histories, query-specific negative selection, and non-model feed-policy rules are deferred to later work.

### Legacy Training

The unchanged training stages still use legacy `01_get_data` and legacy `02_user_history` artifacts. They do not yet consume `queries_*` or `query_histories_*`.
The legacy ranker history contract includes `prior_like_age_hours_at_bucket_start`, embedding indices, author indices, and target-hour popularity lists.

Shared training options:

```yaml
model_type: "two-tower" # mlp, two-tower, or bst-ranker
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

#### MLP

The MLP path scores the full user-by-candidate matrix for each hour bucket. It supports `summarized`, `full_transformer`, and `cross_attention` user encoders.

```bash
python cli.py --model-type mlp --user-encoder summarized --stop-after train
```

Useful options:

- `--hidden-dims`
- `--dropout-rate-mlp`
- `--user-summarization mean|ema|linear_recency`
- `--ema-alpha`

#### Two-Tower

The two-tower path independently encodes users and candidate posts, then scores with a dot product over the shared embedding space. It supports `full_transformer` and `cross_attention` user encoders.

```bash
python cli.py --model-type two-tower --user-encoder cross_attention --stop-after train
```

Useful options:

- `--shared-dim`
- `--user-hidden-dim`
- `--post-hidden-dim`
- `--num-attention-heads`
- `--num-attention-layers`
- `--l2-normalize-embeddings`
- `--similarity-temperature`
- `--use-author-embedding-table`

Two-tower training writes checkpoint files, `training_config.json`, `training_results.json`, TorchScript tower artifacts, a serving manifest, and holdout ranking rows under `eval/`.

#### BST Ranker

The BST ranker fuses content embeddings, author embeddings, time-delta buckets, and a candidate-aware transformer. It currently requires author embeddings.

```bash
python cli.py --model-type bst-ranker \
  --use-author-embedding-table \
  --prediction-hidden-dims 64 32 16 \
  --stop-after train
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

Current branch note: BST training writes train/validation metrics and checkpoints, but legacy evaluation expects holdout ranking-row artifacts. Until BST holdout ranking rows are wired in, use `--stop-after train` for BST runs and compare checkpoints with `compare-rankers`.

### Legacy Evaluation

Legacy evaluation consumes holdout ranking rows from legacy training:

```text
03_train/<stage_run_id>/eval/holdout_unseen_users_ranking_rows.parquet
03_train/<stage_run_id>/eval/holdout_seen_users_ranking_rows.parquet
```

Run the full pipeline for MLP or two-tower:

```bash
python cli.py --model-type two-tower --user-encoder cross_attention
```

Or evaluate a pinned training output:

```bash
python cli.py --start-from evaluate --prior-03-train 20260620_120000_train_two_tower
```

Useful options:

- `--eval-holdout-type unseen_users|seen_users`
- `--skip-modules cold_start_curves,performance_inequality`
- `--prior-03-train`

## Compare Rankers

`compare-rankers` evaluates saved `.pth` checkpoints on shared bucketed candidate sets without rerunning training.

```bash
python cli.py compare-rankers \
  --output-dir /mnt/data/dave/outputs \
  --prior-01-get-data 20260617_205310_fec862c8 \
  --prior-02-user-history 20260618_095653_14c6b8fc \
  --model tt:two-tower:/path/to/two_tower.pth \
  --model bst:bst-ranker:/path/to/bst_ranker.pth \
  --splits val val_unseen_users holdout_unseen_users \
  --metrics-top-ks 30 \
  --batch-size 256 \
  --device cuda
```

Compare outputs are written under `artifacts/compare_rankers/<stage_run_id>/`:

- `metrics.json`
- `metrics.csv`
- `model_specs.json`
- `stage_info.txt`
- `stage.log`

Current compare-rankers assumptions:

- Model specs use `name:type:path`.
- Supported types are `two-tower` and `bst-ranker`.
- Compared checkpoints must use author embeddings.
- If compared checkpoints use different `max_history_len` values, pass `--max-history-len` to choose the evaluation history length.
- BST checkpoints are scored with the optimized one-layer matrix scorer.

## Selective Reruns And Prior Pins

Use `--start-from`, `--stop-after`, and prior pins to reuse artifacts:

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

The Stage 2 manifest supplies and validates the aligned `01_query_selection` ancestor.

Direct legacy training requires both aligned legacy pins:

```bash
python cli.py --config config.yml \
  --start-from train \
  --stop-after train \
  --prior-01-get-data 20260617_205310_fec862c8 \
  --prior-02-user-history 20260618_095653_14c6b8fc
```

Accepted stage aliases:

- `query_selection`
- `user_history`
- `post_selection`
- `train`, `train_mlp`, `train_two_tower`, `train_bst_ranker`
- `evaluate`

Prior pins can be stage run ids under `artifacts/<stage_folder>/`, absolute paths, or paths relative to `output_dir`.

## Background Runs

By default, `config.yml` may set `background: true`. In background mode, the CLI writes `run-all.resolved-config.json` and starts a foreground child process with `nohup`.

Run in the foreground while iterating:

```bash
python cli.py --config config.yml --stop-after post_selection --background false
```

## Development Notes

- Treat `03_post_selection` as the explicit new-pipeline boundary until later data stages and training are rewritten for its artifact contract.
- Use `utils/matrix_ranking.py` for matrix ranking metrics and ranking-row writes.
- Use `utils/ranking_adapters.py` when adding checkpoint-backed comparison support.
- Avoid adding new training paths without registering them in `engagement_prediction/pipeline/registry.py` and documenting their artifact contract here.

## Contributing

Interested in contributing? Please join the Discord and introduce yourself first: https://discord.com/invite/8bWEyrkrJC.
