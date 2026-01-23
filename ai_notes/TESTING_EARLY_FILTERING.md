# Testing Guide for Early Filtering Improvement

## Automated Test

A simulation test has been created in `test_early_filtering.py` that demonstrates the improvement:

```bash
python3 test_early_filtering.py
```

**Expected output:**
- OLD APPROACH: ~33% retention (loses ~67% of sampled users)
- NEW APPROACH: 100% retention (all sampled users meet criteria)
- Improvement: Additional 66k+ users retained

## Integration Testing (requires conda environment)

To test the actual pipeline with the improvements:

### 1. Unit Tests (if pytest is available)
```bash
conda activate eng-pred
pytest tests/ -v
```

### 2. Small-Scale Integration Test

Run the pipeline with a small data sample to verify the filtering works correctly:

```bash
conda activate eng-pred

python cli.py run-all --foreground \
  --data-source greenearth \
  --posts-start 2026-01-04 --posts-end 2026-01-04T02:00:00 \
  --likes-start 2026-01-04 --likes-end 2026-01-04T02:00:00 \
  --max-liking-users 100000 \
  --min-likes-per-user 2 \
  --max-likes-per-user 100
```

### 3. Verify Log Output

Check the stage log for the new pre-filtering messages:

```bash
# Look for lines like:
# "Pass 1: Scanning files for user like counts..."
# "Pre-filtering: X users meet min-likes threshold (2), excluded Y users with too few likes"
# "Sampled Z liking users (W% of eligible)"
# "Min-likes verification removed 0 likes..." (should be 0 or very small)
```

Key indicators of success:
1. Pre-filtering message appears before sampling
2. Number of sampled users ≈ number of final users (no large drop-off)
3. Min-likes verification removes 0 or very few users
4. Final user count is close to `--max-liking-users` setting

### 4. Compare Before/After

**Before the fix:**
```
Sampled 100,000 liking users (4.0% of total)
...
Final: 67,431 users with 2,804,977 likes
```
→ Loss of 32,569 users (32.6%)

**After the fix (expected):**
```
Pre-filtering: 1,200,000 users meet min-likes threshold (2), excluded 800,000 users
Sampled 100,000 liking users (8.3% of eligible)
...
Final: ~100,000 users with ~3,000,000 likes
```
→ Minimal loss (only edge cases from per-user caps)

## What to Check

1. **Statistics in summary.json:**
   ```json
   {
     "filtering_stats": {
       "likes": {
         "n_users_initial": 2000000,
         "n_users_eligible_for_sampling": 1200000,
         "n_users_excluded_min_likes": 800000,
         "n_users_sampled": 100000,
         "n_users_final": 99500
       }
     }
   }
   ```

2. **Memory usage:** Should be similar or slightly higher (Dict vs Set storage)

3. **Runtime:** Should be similar (same number of passes, just counting instead of collecting)

4. **Output quality:** Final datasets should have more users for the same sampling budget

## Edge Cases to Verify

1. **No user sampling (`max_liking_users=0`):**
   - Should still pre-filter by min_likes_per_user
   - All eligible users should be included

2. **No min-likes filter (`min_likes_per_user=0`):**
   - Should behave like before (no pre-filtering)
   - All users are eligible for sampling

3. **Per-user cap reduces users below threshold:**
   - Example: User has 3 likes, `min_likes_per_user=2`, `max_likes_per_user=1`
   - User passes pre-filter but is removed in verification step
   - Should be rare and logged clearly

4. **Eligible users < requested sample size:**
   - Should sample all eligible users
   - No error, just log that sampling is limited by eligibility

## Performance Expectations

- **Memory:** +10-20% in Pass 1 (Dict[str, int] vs Set[str])
- **Runtime:** ±5% (similar operations, just counting)
- **Output:** +30-50% more users retained for typical workloads

## Rollback Plan

If issues arise, the fix can be reverted by:
1. Restore `utils/helpers.py` from git history
2. The changes are isolated to `load_likes_core_polars()` function
3. No database schema or file format changes were made
