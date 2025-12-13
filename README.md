### Engagement Prediction (Modular Workflow)

This repository implements a six-stage modular pipeline to predict engagement on Bluesky posts with minimal duplication of work and fast iteration for releveling experiments.

New design goals:
- Compute and cache all post embeddings up front (text and images) once per time window.
- Perform releveling and user splitting as a separate, iterative step without recomputing embeddings.
- Reuse the same user-featurization logic in both training and evaluation.

### Repository layout (stages under `utils/`)

- `utils/01_get_data/stage_get_data.py`: Stage 1 — Load most recent parquet dumps from Spaces and save a compact raw bundle.
- `utils/02_featurize/stage_featurize.py`: Stage 2 — Build candidate post set and compute text+image embeddings → save `embedding_bundle_*.pkl`.
- `utils/03_relevel/stage_relevel_uniform.py`: Stage 3 — Discover topics and compute per-user mixtures; optional uniform-mixture-balanced relevel selection.
- `utils/04_split/stage_split_users.py`: Stage 4 — Produce `user_splits.json` (train/val/holdout).
- `utils/05_train/stage_train.py`: Stage 5 — Train model using bundle + splits; saves checkpoint and `training_config.json`.
- `utils/06_evaluate/stage_evaluate.py`: Stage 6 — Consolidated evaluation (pairs, matrix, global_unliked).
- `utils/00_helpers/helpers.py`: Minimal cross-stage helpers (re-exported from existing modules).
- `utils/pipeline/{core.py, registry.py}`: Context, timestamped output dirs, and stage registry.

Shared utilities remain:
- `utils/user_features.py`: Shared user-featurization (topic_mixture, multi_centroid, mean)
- `utils/data_utils_with_images.py`, `utils/train_test_helpers.py`, `utils/visual_helpers.py`: Data I/O, modeling, plotting helpers

Legacy scripts (still available, not recommended for the new workflow): `src/preprocess.py` and old CLI flows; the old run-all flow is defunct.

### Quick start

Below, replace paths with your actual workspace if different.

1) Stage 1 — Get data (creates a run dir)
```bash
python /srv/vox/engagement_prediction/wills_tinkering_folder/cli.py run-all --foreground --use-latest \
  --max-files-per-table 5 --max-posts-per-author 3 --image-mode auto
```
Creates a run directory like `outputs/<timestamp>_run_d<files>_mppa<cap>/` and, at Stage 2, saves `featurize/embedding_bundle_<timestamp>.pkl` with:
- `posts_emb_df` (post_emb_* and image_emb_* columns)
- `likes_df`
- `join_like`, `join_post`, `text_column`, `author_column`
- `embedding_dim`, `image_mode`, metadata

2) Stage 3 — Relevel users (iterate here without recomputing embeddings)
```bash
# via run-all or directly calling the stage script; parameters still accepted
```
Saves under the run directory in `relevel/`:
- `user_topic_mixtures.parquet`, optional `topic_model.pkl` and `topic_pca.pkl`
- optional `retained_users.json` when using uniform-mixture-balanced selection

4) Stage 4 — Split users → `user_splits.json`

5) Stage 5 — Train using the bundle + splits
```bash
# orchestrated via run-all; training dir: <run_dir>/train/<timestamp>/
```
Notes:
- Training allocates each user’s liked posts into embedding vs target sets, builds user features from embedding posts, and creates balanced positive/negative target pairs.
- The model checkpoint saves the feature schema for evaluation to match dimensions.

6) Stage 6 — Evaluation (no embedding recompute)
```bash
# via consolidated stage_evaluate.py (modes: pairs | matrix | global_unliked)
```
Outputs:
- Probability matrix `.npz`, optional balanced eval set `.npz`, metrics JSON, and plots under `outputs/.../evaluate/` or `outputs/full_feed_similarity/<timestamp>/`.

### User feature schemas
- `topic_mixture`: Requires a KMeans topic model (and optional PCA) from Stage 2; user features are per-like topic mixtures.
- `multi_centroid`: Per-user MiniBatchKMeans over embedding-capable liked posts; exports K centroids and weights.
- `mean`: Mean embedding of embedding-capable liked posts.

The shared implementation lives in `utils/user_features.py` and is used consistently in training and evaluation.

### Important flags and tips
- Image embeddings: Ensure the `--image-mode` choice at Stage 1 matches expectations at training/evaluation (dimension consistency is enforced).
- Device: Most scripts accept `--device cuda` when available.
- Reproducibility: Set seeds via `--cap-random-seed` (Stage 1/4) and `--random-seed` (Stage 2/3).
- Outputs:
  - Stage 1: `outputs/precompute/<timestamp>/embedding_bundle_<timestamp>.pkl`
  - Stage 2: `outputs/relevel/<timestamp>/{user_splits.json, topic_model.pkl, topic_pca.pkl}`
  - Stage 3: `outputs/checkpoints/engagement_model_<timestamp>.pth`, plus logs/plots
  - Stage 4: `outputs/.../evaluate/<timestamp>_*` and `outputs/full_feed_similarity/<timestamp>/`

### Legacy workflow
The previous `src/preprocess.py` → `src/train.py` → `src/evaluate_full_feed_similarity.py` flow that recomputed embeddings during evaluation is still supported for compatibility, but the new four-stage workflow above is recommended.

### End-to-end execution

Run all six stages automatically using the CLI:
```bash
python /srv/vox/engagement_prediction/wills_tinkering_folder/cli.py run-all \
  --max-files-per-table 7 --max-posts-per-author 3 --image-mode auto \
  --global-topic-k 20 --relevel-strategy uniform_mixture_balanced --min-likes-per-user 10 \
  --epochs 300 --batch-size 256 --device cuda
```
By default this runs in the background with nohup and writes a log under `outputs/run-all_<ts>.log`, then mirrors it to `<run_dir>/run-all.log` once the run directory is created.

### Testing
Use `pytest` for lightweight tests where available.
```bash
pip install -r requirements.txt pytest
pytest -q
```
