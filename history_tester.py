"""Quick inspector for history_posts parquet files."""
import polars as pl
from pathlib import Path

# ── Point at a history_posts file ──────────────────────────────
RUN = "outputs/20260205_014917_start_to_get_data_mlp_uniform"
FEATURIZE_DIR = Path(RUN) / "02_featurize"

# Find all history_posts files, sorted newest first
history_files = sorted(
    FEATURIZE_DIR.rglob("history_posts_*.parquet"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)
print(f"Found {len(history_files)} history_posts files:\n")
for f in history_files:
    size_mb = f.stat().st_size / 1e6
    print(f"  {f.parent.name}/{f.name}  ({size_mb:.1f} MB)")

# ── Load the latest one ───────────────────────────────────────
latest = history_files[0]
print(f"\n{'='*70}")
print(f"Inspecting: {latest.name}  ({latest.stat().st_size/1e6:.1f} MB)")
print(f"{'='*70}")

df = pl.read_parquet(latest)
print(f"\nSchema: {df.schema}")
print(f"Rows:   {len(df):,}")
print(f"Columns: {df.columns}")

# ── Show random rows ──────────────────────────────────────────
print(f"\n{'─'*70}")
print("10 random rows:")
print(f"{'─'*70}")
sample = df.sample(10, seed=42)
for row in sample.iter_rows(named=True):
    emb_list = row["prior_emb_indices"]
    print(
        f"  did=...{row['target_did'][-12:]}  "
        f"like_uri=...{row['like_uri'][-30:]}  "
        f"n_prior={len(emb_list):>4d}  "
        f"emb_indices={emb_list[:8]}{'...' if len(emb_list) > 8 else ''}"
    )

# ── Distribution of list lengths ──────────────────────────────
print(f"\n{'─'*70}")
print("Distribution of prior_emb_indices list lengths (all targets):")
print(f"{'─'*70}")
lengths = df["prior_emb_indices"].list.len()
print(lengths.describe())

# Value counts for short lists (to see the spike at the cap)
vc = lengths.value_counts().sort("prior_emb_indices")
print(f"\nValue counts (list length -> # targets):")
for row in vc.head(20).iter_rows():
    print(f"  length={row[0]:>5d}  count={row[1]:>10,d}")
if len(vc) > 20:
    print(f"  ... ({len(vc)} distinct lengths total)")

# ── Memory footprint of the DataFrame itself ──────────────────
print(f"\n{'─'*70}")
print("In-memory size of this DataFrame:")
print(f"{'─'*70}")
est = df.estimated_size("mb")
print(f"  estimated_size = {est:.1f} MB")
# Break down by column
for col_name in df.columns:
    col_size = df.select(col_name).estimated_size("mb")
    print(f"    {col_name}: {col_size:.1f} MB")

# ── Compare all history files (different caps) ────────────────
if len(history_files) > 1:
    print(f"\n{'─'*70}")
    print("Comparison across all saved history files:")
    print(f"{'─'*70}")
    for f in history_files:
        meta_df = pl.read_parquet(f)
        lens = meta_df["prior_emb_indices"].list.len()
        max_len = lens.max()
        mean_len = lens.mean()
        mem_mb = meta_df.estimated_size("mb")
        emb_col_mb = meta_df.select("prior_emb_indices").estimated_size("mb")
        print(
            f"  {f.parent.name}  "
            f"disk={f.stat().st_size/1e6:>7.1f}MB  "
            f"ram={mem_mb:>7.1f}MB  "
            f"emb_col={emb_col_mb:>7.1f}MB  "
            f"max_len={max_len:>4d}  "
            f"mean_len={mean_len:>6.1f}"
        )
