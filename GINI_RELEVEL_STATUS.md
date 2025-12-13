# Gini Relevel Method - Current Status

**Status: Work in Progress - Not Production Ready**

This document describes the current state of `stage_relevel_gini.py`, an alternative releveling procedure that uses Gini coefficient optimization for user selection.

## Summary

The Gini-based releveling method is currently **broken** and should not be used. The uniform method (`stage_relevel_uniform.py`) continues to work correctly and should be used for all production runs.

## Evidence of Issue

Test runs using the same embedding bundle show:

| Method | Users in Mixtures | Users Retained | Retention Rate |
|--------|-------------------|----------------|----------------|
| Uniform | 15,107 | 8,739 | 58% |
| Gini | 15,107 | 1-2 | ~0.01% |

The Gini method retains only 1-2 users instead of thousands, making it unusable.

### Test Run Results

From `outputs/20251105_172556_run_d14_mppa5/03_relevel/`:

**Uniform method (20251105_180653):**
```
stage: relevel
runtime_seconds: 32.87
settings: global_topic_k=20, relevel_strategy=uniform_mixture_balanced, relevel_alpha=0.35
N_posts_emb: 2228528
N_likes_joinable: 1347195
N_users_mixtures: 15107
N_retained_users: 8739
```

**Gini method (20251119_131324):**
```
stage: relevel
method: gini_optimized
runtime_seconds: 130.50
settings: global_topic_k=24, k_range=(20, 30), relevel_strategy=gini_based, target_gini=0.1
N_posts_emb: 2228528
N_likes_joinable: 1347195
N_users_mixtures: 15107
N_retained_users: 2
```

## Root Cause Analysis

The bug is in the `gini_based_user_selection` function (lines 560-812 of `stage_relevel_gini.py`).

### The Early Stopping Bug

The algorithm stops when the Gini coefficient drops below the target:

```python
# Lines 751-755
min_users_for_gini = max(2, min_users_per_topic * global_topic_k) if min_users_per_topic > 0 else 2
if len(selected_users) >= min_users_for_gini and current_gini <= target_gini:
    print(f"✅ Target Gini reached: {current_gini:.4f} (selected {len(selected_users)} users)")
    break
```

**Problem:** With `min_users_per_topic=0` (the default), `min_users_for_gini` is only 2. The Gini coefficient can be artificially low with very few users, causing the algorithm to stop after selecting just 1-2 users.

### Why Gini is Low with Few Users

When only 1-2 users are selected and they happen to fall into different topic clusters, the weighted Gini calculation can return a low value (suggesting "balanced" distribution) even though:
1. Most topic clusters have 0 users
2. The sample size is statistically meaningless
3. The distribution is not actually balanced

## Suggested Fixes

1. **Require minimum users before Gini-based stopping:**
   ```python
   # Require at least 10 users per topic or 10% of eligible users
   min_users_for_gini = max(
       global_topic_k * 10,
       len(eligible_users) // 10,
       100  # absolute minimum
   )
   ```

2. **Add coverage constraint:**
   Only allow stopping when at least 80% of topics have at least one user.

3. **Use cumulative Gini target:**
   Instead of stopping at a fixed Gini target, continue until adding more users stops improving diversity significantly.

## How to Use (Current Recommendation)

Until the Gini method is fixed, use the uniform method:

```bash
# Via CLI (recommended)
python cli.py run-all --relevel-method uniform ...

# The default is already uniform, so this also works:
python cli.py run-all ...
```

## Files Involved

- `utils/03_relevel/stage_relevel_gini.py` - The broken Gini implementation
- `utils/03_relevel/stage_relevel_uniform.py` - The working uniform implementation
- `utils/pipeline/registry.py` - Stage registry (supports `--relevel-method gini`)
- `test_relevel_gini.py` - Test script for the Gini method
- `TEST_RELEVEL_GINI.md` - Testing documentation

## Contact

For questions about this work-in-progress, contact the RA who implemented it.

