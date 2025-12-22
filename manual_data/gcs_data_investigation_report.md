# GCS Data Investigation Report

**Generated:** 2025-12-22  
**Source:** `gs://greenearth-471522-ingex-extract-test/`

**Files analyzed:**
- Posts: `bsky_posts_20251221_223355.parquet` (114,156 rows, 342 MB)
- Likes: `bsky_likes_20251221_223600.parquet` (916,764 rows, 125 MB)

---

## Executive Summary

**Status: ⚠️ ALMOST COMPATIBLE** - One critical issue remains: missing post identifier (`rkey`)

### Progress Since Previous Data (Dec 15)

| Issue | Previous Status | Current Status |
|-------|-----------------|----------------|
| Text content | ❌ 100% empty | ✅ 96.2% non-empty (avg 94 chars) |
| Pre-computed embeddings | ❌ All NULL | ✅ 99.7% populated (2 models) |
| Author overlap | ❌ 0 overlap | ✅ 34,090 overlapping authors |
| Post identifier | ❌ Missing | ❌ Still missing |
| Join capability | ❌ Impossible | ❌ Still impossible |

### Remaining Issue

**Posts table is missing the `rkey` column** (the record key that identifies each post).

The likes table references posts using AT URIs like:
```
at://did:plc:mapygudodvnaxbvwlj6qzdpx/app.bsky.feed.post/3mahloycxa22x
                   └─── author DID ───┘                    └── rkey ──┘
```

The posts table has `did` (author) but NOT `rkey`, so we cannot construct the post URI to join with likes.

---

## Schema Details

### Posts DataFrame (114,156 rows)

| Column | Type | Null% | Notes |
|--------|------|-------|-------|
| `did` | object | 0.0% | ✅ Author identifier |
| `record_text` | object | 0.0% | ✅ 96.2% non-empty, avg 94 chars |
| `embeddings` | object | 0.28% | ✅ Pre-computed (see below) |
| `record_created_at` | object | 0.0% | Timestamp |
| `inserted_at` | object | 0.0% | Ingestion timestamp |
| `reply_parent_uri` | object | 47.4% | Reply metadata |
| `reply_root_uri` | object | 47.4% | Reply metadata |
| `embed_quote_uri` | object | 92.9% | Quote/repost metadata |
| **`rkey`** | - | - | ❌ **MISSING - REQUIRED** |
| **`commit_cid`** | - | - | ❌ Missing (alternative to rkey) |

### Likes DataFrame (916,764 rows)

| Column | Type | Null% | Notes |
|--------|------|-------|-------|
| `DID` | object | 0.0% | ⚠️ Liker ID (uppercase - needs rename to `did`) |
| `SubjectURI` | object | 0.0% | Full AT URI of liked post |
| `InsertedAt` | object | 0.0% | Ingestion timestamp |
| `RecordCreatedAt` | object | 0.0% | Like creation time |

---

## Pre-computed Embeddings Analysis

The `embeddings` column contains **encoded embeddings from 2 models**:

```python
# Format: list of (model_name, base85_encoded_string) tuples
[
    ('all_MiniLM_L12_v2', 'c$_4ZZHSjy8ODE...'),  # ~537 dimensions (float16)
    ('all_MiniLM_L6_v2', 'c$_Shdx+Lm8i1e...')    # ~535 dimensions (float16)
]
```

**Decoding:**
- Encoding: Base85
- Data type: float16 (2 bytes per value)
- Dimensions: ~384 per model (standard for MiniLM)

**To use these embeddings:**
```python
import base64
import numpy as np

def decode_embedding(encoded_str):
    decoded = base64.b85decode(encoded_str)
    return np.frombuffer(decoded, dtype=np.float16)
```

**Note:** The pipeline currently uses `all-MiniLM-L6-v2` which matches one of the provided models.

---

## Join Analysis

### Author Overlap Test

```
Posts unique authors:        55,178
Likes unique liked authors: 131,572
Overlapping authors:         34,090 ✅
```

**There IS significant author overlap** - the data is related, but we cannot join because posts lack identifiers.

### Composite Key Test

Attempted using `(did, record_created_at)` as a composite key:
- Total posts: 114,156
- Unique composite keys: 110,590
- **Not unique** - 5,394 duplicate (author, timestamp) pairs

This means multiple posts from the same author at the same second exist, so timestamps alone cannot identify posts.

---

## Required Fix

**Your colleague needs to add ONE column to the posts export:**

Option A (Preferred): Add `rkey` column
```sql
SELECT 
    did,
    rkey,           -- ADD THIS
    record_text,
    embeddings,
    ...
FROM posts
```

Option B: Add `commit_cid` column (content hash)
```sql
SELECT 
    did,
    commit_cid,     -- OR ADD THIS
    record_text,
    ...
FROM posts
```

Either allows joining with likes' `SubjectURI` (which contains `rkey`) or `subject_cid`.

---

## Pipeline Integration Checklist

Once `rkey` is added:

- [x] Text content available (96.2% non-empty)
- [x] Pre-computed embeddings available (decode from base85)
- [x] Author overlap confirmed
- [ ] **Add `rkey` column to posts** ← BLOCKING
- [ ] Rename likes `DID` → `did` (minor, can be done in adapter)
- [ ] Parse `SubjectURI` to extract join key (can be done in adapter)

---

## Adapter Implementation (Ready when data is fixed)

Once `rkey` is added to posts, implement this adapter:

```python
def adapt_gcs_data(posts_df, likes_df):
    """Transform GCS data to match pipeline schema."""
    import base64
    import numpy as np
    
    # 1. Rename columns
    likes_df = likes_df.rename(columns={'DID': 'did'})
    
    # 2. Parse SubjectURI to get (author_did, rkey) for joining
    def parse_uri(uri):
        parts = uri.replace('at://', '').split('/')
        return parts[0], parts[-1]  # author_did, rkey
    
    likes_df[['_author_did', '_rkey']] = pd.DataFrame(
        likes_df['SubjectURI'].apply(parse_uri).tolist()
    )
    
    # 3. Create post URI for joining
    posts_df['post_uri'] = 'at://' + posts_df['did'] + '/app.bsky.feed.post/' + posts_df['rkey']
    
    # 4. Decode embeddings (if using pre-computed)
    def decode_embedding(emb_list, model='all_MiniLM_L6_v2'):
        for model_name, encoded in emb_list:
            if model_name == model:
                return np.frombuffer(base64.b85decode(encoded), dtype=np.float16)
        return None
    
    posts_df['embedding_vector'] = posts_df['embeddings'].apply(
        lambda x: decode_embedding(x) if x else None
    )
    
    return posts_df, likes_df
```

---

## Files Generated

- `investigate_gcs_data.py` - Investigation script
- `gcs_data_investigation_report.json` - Machine-readable results
- `gcs_data_investigation_report.md` - This report

---

## Next Steps

1. **Request `rkey` column** from your colleague
2. Once available, implement the adapter above
3. Test join and embedding decoding
4. Integrate with pipeline Stage 1 (get_data)
