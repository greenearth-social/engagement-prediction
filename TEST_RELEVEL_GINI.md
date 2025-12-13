# Testing stage_relevel_gini.py

This document explains how to test the new `stage_relevel_gini.py` script with your embedding bundle.

## Quick Test (Recommended)

Use the provided test script:

```bash
cd /srv/vox/engagement_prediction/wills_tinkering_folder
python test_relevel_gini.py
```

This will:
1. Load the embedding bundle from the specified path
2. Set up the pipeline context correctly
3. Run the Gini-based relevel stage
4. Save outputs to `outputs/20251105_172556_run_d14_mppa5/03_relevel/<timestamp>/`

## What the Test Script Does

The test script (`test_relevel_gini.py`):
- Sets up a `Context` object pointing to the correct run directory
- Points the `prior_outputs['02_featurize']` to the directory containing your bundle
- Creates a simple `TestArgs` object with default parameters
- Directly calls the `run()` function from `stage_relevel_gini.py`

## Expected Outputs

After running, you should see outputs in:
```
outputs/20251105_172556_run_d14_mppa5/03_relevel/<timestamp>/
├── topic_model.pkl          # Fitted KMeans model with optimal k
├── topic_pca.pkl            # PCA model (if PCA was applied)
├── user_topic_mixtures.parquet  # User topic probability distributions
├── retained_users.json      # Selected users (if Gini selection was applied)
└── stage_info.txt           # Metadata and statistics
```

## Customizing Parameters

You can modify the `TestArgs` class in `test_relevel_gini.py` to adjust parameters:

```python
class TestArgs:
    def __init__(self):
        self.global_topic_k = 20              # Initial number of topics
        self.k_range = (20, 30)               # Range for silhouette optimization
        self.random_seed = 42                  # Random seed
        self.use_pca = True                   # Apply PCA dimensionality reduction
        self.pca_components = 50              # PCA dimensions
        self.relevel_strategy = 'gini_based'   # Use Gini-based selection
        self.target_gini = 0.1                 # Target Gini coefficient
        self.min_likes_per_user = 4           # Minimum likes for eligibility
        self.relevel_min_users_per_topic = 0   # Minimum users per topic
```

## Alternative: Using the Pipeline Registry

If you want to test it through the pipeline system, you can temporarily modify the registry:

1. Edit `utils/pipeline/registry.py`:
   ```python
   'relevel': ("utils/03_relevel/stage_relevel_gini.py", "03_relevel"),
   ```

2. Then use the CLI:
   ```bash
   python cli.py run-all --start-from relevel --stop-after relevel \
     --prior-featurize /srv/vox/engagement_prediction/wills_tinkering_folder/outputs/20251105_172556_run_d14_mppa5/02_featurize/20251105_172757 \
     --output-dir /srv/vox/engagement_prediction/wills_tinkering_folder/outputs/20251105_172556_run_d14_mppa5
   ```

## Verifying the Output

After running, check:

1. **stage_info.txt** - Contains runtime statistics and settings
2. **user_topic_mixtures.parquet** - Should have user DIDs as index and topic columns (0, 1, 2, ...)
3. **retained_users.json** - If Gini selection was applied, contains the selected user list
4. **topic_model.pkl** - Can be loaded to inspect the clustering model

## Troubleshooting

- **FileNotFoundError**: Make sure the bundle path is correct and the file exists
- **KeyError**: The bundle might be missing required keys (posts_emb_df, likes_df, join_like, join_post)
- **RuntimeError**: Check that the bundle contains joinable posts and likes data


