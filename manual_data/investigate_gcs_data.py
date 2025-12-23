#!/usr/bin/env python3
"""
Investigation: GCS Manual Data Compatibility with Engagement Pipeline

This script analyzes parquet files from Google Cloud Storage to determine
compatibility with our engagement prediction pipeline.

Data Source: gs://greenearth-471522-ingex-extract-test/
Files analyzed:
  - bsky_posts_20251221_223355.parquet
  - bsky_likes_20251221_223600.parquet

Pipeline Requirements (from helpers.py):
  Posts DataFrame needs:
    - 'did' (author DID)
    - 'commit_cid' (post identifier for join with likes)
    - text column (for embedding computation)
    - optional: 'rkey', 'image_url', etc.

  Likes DataFrame needs:
    - 'did' (liker DID)  
    - 'subject_cid' (liked post's commit_cid for join)

  Join: likes.subject_cid <-> posts.commit_cid

Usage:
    python investigate_gcs_data.py

Output:
    - Console report with detailed findings
    - gcs_data_investigation_report.json (machine-readable)
    - gcs_data_investigation_report.md (human-readable summary)
"""

from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Configuration
MANUAL_DATA_DIR = Path(__file__).parent
POSTS_FILE = MANUAL_DATA_DIR / "bsky_posts_20251221_223355.parquet"
LIKES_FILE = MANUAL_DATA_DIR / "bsky_likes_20251221_223600.parquet"
OUTPUT_JSON = MANUAL_DATA_DIR / "gcs_data_investigation_report.json"
OUTPUT_MD = MANUAL_DATA_DIR / "gcs_data_investigation_report.md"

# Expected columns from standard S3 pipeline (for comparison)
EXPECTED_POSTS_COLUMNS = {
    'required': ['did', 'commit_cid'],
    'text': ['record_text', 'text'],  # any of these
    'optional': ['rkey', 'record_created_at', 'embed_quote_uri', 'embed_image_uris', 
                 'reply_parent_uri', 'reply_root_uri', 'inserted_at'],
}
EXPECTED_LIKES_COLUMNS = {
    'required': ['did', 'subject_cid'],
    'optional': ['inserted_at', 'record_created_at'],
}


def section_header(title: str, char: str = "=") -> str:
    """Print a formatted section header."""
    line = char * 70
    return f"\n{line}\n{title}\n{line}"


def analyze_dataframe(df: pd.DataFrame, name: str) -> Dict[str, Any]:
    """Analyze a DataFrame and return detailed schema information."""
    info = {
        'name': name,
        'rows': len(df),
        'columns': len(df.columns),
        'column_names': df.columns.tolist(),
        'column_details': {},
        'memory_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
    }
    
    for col in df.columns:
        col_info = {
            'dtype': str(df[col].dtype),
            'null_count': int(df[col].isnull().sum()),
            'null_pct': round(100 * df[col].isnull().sum() / len(df), 2) if len(df) > 0 else 0,
        }
        
        # Try to get unique count (may fail for unhashable types like lists)
        try:
            col_info['unique_count'] = int(df[col].nunique())
        except TypeError:
            col_info['unique_count'] = -1  # Indicates unhashable type
        
        # Check for empty strings (common issue)
        if df[col].dtype == 'object':
            empty_str_count = int((df[col] == '').sum())
            col_info['empty_string_count'] = empty_str_count
            col_info['empty_string_pct'] = round(100 * empty_str_count / len(df), 2) if len(df) > 0 else 0
            
            # Check if it's an array/list column
            first_val = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
            if isinstance(first_val, (list, tuple, np.ndarray)):
                col_info['is_array'] = True
                col_info['array_length'] = len(first_val) if first_val is not None else 0
        
        # Sample values (first 3 non-null)
        samples = df[col].dropna().head(3).tolist()
        try:
            col_info['samples'] = [str(v)[:100] for v in samples]
        except Exception:
            col_info['samples'] = ['<complex>']
        
        info['column_details'][col] = col_info
    
    return info


def check_pipeline_compatibility(posts_info: Dict, likes_info: Dict) -> Dict[str, Any]:
    """Check compatibility with the engagement prediction pipeline."""
    compat = {
        'posts': {
            'has_did': 'did' in posts_info['column_names'],
            'has_commit_cid': 'commit_cid' in posts_info['column_names'],
            'has_rkey': 'rkey' in posts_info['column_names'],
            'text_column': None,
            'text_has_content': False,
            'embedding_columns': [],
        },
        'likes': {
            'has_did': 'did' in [c.lower() for c in likes_info['column_names']],
            'did_column_name': None,
            'has_subject_cid': 'subject_cid' in likes_info['column_names'],
            'has_subject_uri': any('uri' in c.lower() for c in likes_info['column_names']),
        },
        'join_possible': False,
        'join_method': None,
        'issues': [],
        'warnings': [],
        'recommendations': [],
    }
    
    # Find text column in posts
    for text_col in EXPECTED_POSTS_COLUMNS['text']:
        if text_col in posts_info['column_names']:
            compat['posts']['text_column'] = text_col
            col_details = posts_info['column_details'].get(text_col, {})
            empty_pct = col_details.get('empty_string_pct', 100)
            compat['posts']['text_has_content'] = empty_pct < 50  # At least 50% non-empty
            break
    
    # Find DID column in likes (case-insensitive)
    for col in likes_info['column_names']:
        if col.lower() == 'did':
            compat['likes']['did_column_name'] = col
            break
    
    # Check for embedding columns
    for col in posts_info['column_names']:
        if any(x in col.lower() for x in ['emb', 'embed', 'vector', 'feature']):
            details = posts_info['column_details'].get(col, {})
            if details.get('is_array') or details.get('null_pct', 100) < 100:
                compat['posts']['embedding_columns'].append({
                    'name': col,
                    'null_pct': details.get('null_pct', 0),
                    'array_length': details.get('array_length', 0),
                })
    
    # Determine join compatibility
    if compat['posts']['has_commit_cid'] and compat['likes']['has_subject_cid']:
        compat['join_possible'] = True
        compat['join_method'] = 'CID (commit_cid <-> subject_cid)'
    elif compat['posts']['has_rkey'] and compat['likes']['has_subject_uri']:
        compat['join_possible'] = True
        compat['join_method'] = 'URI/rkey (requires parsing SubjectURI)'
    
    # Identify issues
    if not compat['posts']['has_did']:
        compat['issues'].append("Posts missing 'did' column (author identifier)")
    if not compat['posts']['has_commit_cid'] and not compat['posts']['has_rkey']:
        compat['issues'].append("Posts missing unique identifier ('commit_cid' or 'rkey')")
    if not compat['posts']['text_column']:
        compat['issues'].append("Posts missing text column for embeddings")
    elif not compat['posts']['text_has_content']:
        compat['issues'].append("Posts text column is mostly empty")
    
    if not compat['likes']['has_did']:
        if compat['likes']['did_column_name']:
            compat['warnings'].append(f"Likes uses '{compat['likes']['did_column_name']}' instead of lowercase 'did'")
        else:
            compat['issues'].append("Likes missing 'did' column (liker identifier)")
    
    if not compat['likes']['has_subject_cid']:
        if compat['likes']['has_subject_uri']:
            compat['warnings'].append("Likes uses SubjectURI instead of subject_cid (can be parsed)")
        else:
            compat['issues'].append("Likes missing 'subject_cid' or 'SubjectURI' for joining")
    
    if not compat['join_possible']:
        compat['issues'].append("Cannot join posts and likes - missing compatible keys")
    
    return compat


def test_join(posts_df: pd.DataFrame, likes_df: pd.DataFrame, compat: Dict) -> Dict[str, Any]:
    """Attempt to join posts and likes and report results."""
    join_results = {
        'attempted': False,
        'success': False,
        'method': None,
        'posts_count': len(posts_df),
        'likes_count': len(likes_df),
        'posts_join_key_unique': 0,
        'likes_join_key_unique': 0,
        'overlap_count': 0,
        'joined_rows': 0,
        'join_rate': 0.0,
    }
    
    # Try CID-based join first
    if 'commit_cid' in posts_df.columns and 'subject_cid' in likes_df.columns:
        join_results['attempted'] = True
        join_results['method'] = 'CID (commit_cid <-> subject_cid)'
        
        posts_cids = set(posts_df['commit_cid'].dropna().astype(str))
        likes_cids = set(likes_df['subject_cid'].dropna().astype(str))
        overlap = posts_cids & likes_cids
        
        join_results['posts_join_key_unique'] = len(posts_cids)
        join_results['likes_join_key_unique'] = len(likes_cids)
        join_results['overlap_count'] = len(overlap)
        
        if overlap:
            # Perform actual join
            posts_df_str = posts_df.copy()
            posts_df_str['commit_cid'] = posts_df_str['commit_cid'].astype(str)
            likes_df_str = likes_df.copy()
            likes_df_str['subject_cid'] = likes_df_str['subject_cid'].astype(str)
            
            joined = likes_df_str.merge(
                posts_df_str[['commit_cid']].drop_duplicates(),
                left_on='subject_cid',
                right_on='commit_cid',
                how='inner'
            )
            join_results['joined_rows'] = len(joined)
            join_results['join_rate'] = round(100 * len(joined) / len(likes_df), 2) if len(likes_df) > 0 else 0
            join_results['success'] = len(joined) > 0
    
    # Try URI/rkey-based join
    elif 'rkey' in posts_df.columns and 'did' in posts_df.columns:
        # Check for SubjectURI in likes
        uri_col = None
        for col in likes_df.columns:
            if 'uri' in col.lower() and 'subject' in col.lower():
                uri_col = col
                break
        
        if uri_col:
            join_results['attempted'] = True
            join_results['method'] = f'URI parsing ({uri_col} -> did+rkey)'
            
            # Parse URIs from likes
            def parse_at_uri(uri):
                if not uri or not isinstance(uri, str) or not uri.startswith('at://'):
                    return None, None
                rest = uri[5:]
                parts = rest.split('/')
                if len(parts) >= 3:
                    return parts[0], parts[-1]  # author_did, rkey
                return None, None
            
            likes_parsed = likes_df.copy()
            parsed = likes_parsed[uri_col].apply(parse_at_uri)
            likes_parsed['_author_did'] = [p[0] for p in parsed]
            likes_parsed['_rkey'] = [p[1] for p in parsed]
            
            # Create composite keys
            posts_keys = set(zip(posts_df['did'].astype(str), posts_df['rkey'].astype(str)))
            likes_keys = set(zip(
                likes_parsed['_author_did'].dropna().astype(str),
                likes_parsed['_rkey'].dropna().astype(str)
            ))
            overlap = posts_keys & likes_keys
            
            join_results['posts_join_key_unique'] = len(posts_keys)
            join_results['likes_join_key_unique'] = len(likes_keys)
            join_results['overlap_count'] = len(overlap)
            join_results['success'] = len(overlap) > 0
            join_results['join_rate'] = round(100 * len(overlap) / len(likes_df), 2) if len(likes_df) > 0 else 0
    
    return join_results


def check_text_quality(posts_df: pd.DataFrame, text_col: str) -> Dict[str, Any]:
    """Analyze text column quality."""
    if text_col not in posts_df.columns:
        return {'available': False}
    
    text_series = posts_df[text_col].fillna('')
    
    quality = {
        'available': True,
        'column_name': text_col,
        'total_rows': len(text_series),
        'null_count': int(posts_df[text_col].isnull().sum()),
        'empty_count': int((text_series == '').sum()),
        'non_empty_count': int((text_series != '').sum()),
        'non_empty_pct': round(100 * (text_series != '').sum() / len(text_series), 2) if len(text_series) > 0 else 0,
    }
    
    non_empty = text_series[text_series != '']
    if len(non_empty) > 0:
        lengths = non_empty.str.len()
        quality['avg_length'] = round(lengths.mean(), 1)
        quality['min_length'] = int(lengths.min())
        quality['max_length'] = int(lengths.max())
        quality['sample_texts'] = non_empty.head(5).tolist()
    
    return quality


def check_embedding_quality(posts_df: pd.DataFrame) -> Dict[str, Any]:
    """Check for pre-computed embeddings."""
    import base64
    
    emb_cols = [c for c in posts_df.columns if any(x in c.lower() for x in ['emb', 'embed', 'vector'])]
    # Filter out URI columns
    emb_cols = [c for c in emb_cols if 'uri' not in c.lower()]
    
    if not emb_cols:
        return {'available': False, 'columns': []}
    
    results = {
        'available': True,
        'columns': [],
    }
    
    for col in emb_cols:
        col_info = {
            'name': col,
            'dtype': str(posts_df[col].dtype),
            'null_count': int(posts_df[col].isnull().sum()),
            'null_pct': round(100 * posts_df[col].isnull().sum() / len(posts_df), 2),
        }
        
        # Check if it contains actual embeddings
        non_null = posts_df[col].dropna()
        if len(non_null) > 0:
            first_val = non_null.iloc[0]
            
            # Check for encoded embeddings format: list of (model_name, base85_encoded) tuples
            if isinstance(first_val, list) and len(first_val) > 0:
                if isinstance(first_val[0], tuple) and len(first_val[0]) == 2:
                    col_info['is_encoded'] = True
                    col_info['format'] = 'list of (model_name, base85_encoded) tuples'
                    col_info['models'] = []
                    
                    for model_name, encoded in first_val:
                        model_info = {'name': model_name}
                        try:
                            decoded = base64.b85decode(encoded)
                            # Try float16 (most likely for space efficiency)
                            arr = np.frombuffer(decoded, dtype=np.float16)
                            model_info['decoded_dim'] = len(arr)
                            model_info['sample_values'] = arr[:5].astype(float).tolist()
                            model_info['decode_success'] = True
                        except Exception as e:
                            model_info['decode_error'] = str(e)
                            model_info['decode_success'] = False
                        col_info['models'].append(model_info)
                else:
                    col_info['is_array'] = True
                    col_info['embedding_dim'] = len(first_val)
                    try:
                        col_info['sample_values'] = [float(x) for x in first_val[:5]]
                    except Exception:
                        pass
            elif isinstance(first_val, np.ndarray):
                col_info['is_array'] = True
                col_info['embedding_dim'] = len(first_val)
                col_info['sample_values'] = first_val[:5].tolist()
            else:
                col_info['is_array'] = False
                col_info['sample_value'] = str(first_val)[:100]
        
        results['columns'].append(col_info)
    
    return results


def generate_report(
    posts_info: Dict,
    likes_info: Dict,
    compat: Dict,
    join_results: Dict,
    text_quality: Dict,
    embedding_quality: Dict,
) -> Tuple[str, Dict]:
    """Generate comprehensive report."""
    
    # Build JSON report
    report = {
        'generated_at': datetime.now().isoformat(),
        'files': {
            'posts': str(POSTS_FILE.name),
            'likes': str(LIKES_FILE.name),
        },
        'posts_schema': posts_info,
        'likes_schema': likes_info,
        'pipeline_compatibility': compat,
        'join_test': join_results,
        'text_quality': text_quality,
        'embedding_quality': embedding_quality,
    }
    
    # Build markdown report
    md_lines = [
        "# GCS Data Investigation Report",
        f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Files analyzed:**",
        f"- Posts: `{POSTS_FILE.name}` ({posts_info['rows']:,} rows, {posts_info['memory_mb']:.1f} MB)",
        f"- Likes: `{LIKES_FILE.name}` ({likes_info['rows']:,} rows, {likes_info['memory_mb']:.1f} MB)",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Status determination
    critical_issues = [i for i in compat['issues'] if 'missing' in i.lower() or 'cannot' in i.lower()]
    if not critical_issues and join_results['success']:
        md_lines.append("**Status: ✅ COMPATIBLE** - Data can be used with the pipeline (with minor adaptations)")
    elif join_results['success']:
        md_lines.append("**Status: ⚠️ PARTIALLY COMPATIBLE** - Join works but some issues need attention")
    else:
        md_lines.append("**Status: ❌ NOT COMPATIBLE** - Critical issues prevent pipeline use")
    
    md_lines.extend([
        "",
        "### Key Findings",
        "",
    ])
    
    # Join status
    if join_results['success']:
        md_lines.append(f"- ✅ **Join successful**: {join_results['overlap_count']:,} posts can be matched ({join_results['join_rate']}% of likes)")
        md_lines.append(f"  - Method: {join_results['method']}")
    else:
        md_lines.append(f"- ❌ **Join failed**: {join_results.get('method', 'No compatible join keys found')}")
    
    # Text status
    if text_quality.get('available') and text_quality.get('non_empty_pct', 0) > 50:
        md_lines.append(f"- ✅ **Text content**: {text_quality['non_empty_pct']}% of posts have text (avg {text_quality.get('avg_length', 0):.0f} chars)")
    elif text_quality.get('available'):
        md_lines.append(f"- ⚠️ **Text content**: Only {text_quality.get('non_empty_pct', 0)}% of posts have text")
    else:
        md_lines.append("- ❌ **Text content**: No text column found")
    
    # Embedding status
    if embedding_quality.get('available'):
        for emb in embedding_quality['columns']:
            if emb.get('is_array') and emb.get('null_pct', 100) < 50:
                md_lines.append(f"- ✅ **Pre-computed embeddings**: `{emb['name']}` (dim={emb.get('embedding_dim', '?')}, {100-emb['null_pct']:.1f}% populated)")
            elif emb.get('null_pct', 100) >= 100:
                md_lines.append(f"- ⚠️ **Embeddings column exists but empty**: `{emb['name']}`")
    else:
        md_lines.append("- ℹ️ **No pre-computed embeddings** (will compute from text)")
    
    # Issues
    if compat['issues']:
        md_lines.extend(["", "### Issues", ""])
        for issue in compat['issues']:
            md_lines.append(f"- ❌ {issue}")
    
    if compat['warnings']:
        md_lines.extend(["", "### Warnings", ""])
        for warning in compat['warnings']:
            md_lines.append(f"- ⚠️ {warning}")
    
    # Schema details
    md_lines.extend([
        "",
        "---",
        "",
        "## Schema Details",
        "",
        "### Posts DataFrame",
        "",
        f"| Column | Type | Null% | Notes |",
        f"|--------|------|-------|-------|",
    ])
    
    for col, details in posts_info['column_details'].items():
        notes = []
        if details.get('is_array'):
            notes.append(f"array[{details.get('array_length', '?')}]")
        if details.get('empty_string_pct', 0) > 50:
            notes.append(f"{details['empty_string_pct']}% empty")
        md_lines.append(f"| `{col}` | {details['dtype']} | {details['null_pct']}% | {', '.join(notes) or '-'} |")
    
    md_lines.extend([
        "",
        "### Likes DataFrame",
        "",
        f"| Column | Type | Null% | Notes |",
        f"|--------|------|-------|-------|",
    ])
    
    for col, details in likes_info['column_details'].items():
        notes = []
        if details.get('is_array'):
            notes.append(f"array[{details.get('array_length', '?')}]")
        md_lines.append(f"| `{col}` | {details['dtype']} | {details['null_pct']}% | {', '.join(notes) or '-'} |")
    
    # Join details
    md_lines.extend([
        "",
        "---",
        "",
        "## Join Analysis",
        "",
        f"- **Method**: {join_results.get('method', 'N/A')}",
        f"- **Posts with join key**: {join_results['posts_join_key_unique']:,}",
        f"- **Likes with join key**: {join_results['likes_join_key_unique']:,}",
        f"- **Overlap (joinable)**: {join_results['overlap_count']:,}",
        f"- **Join rate**: {join_results['join_rate']}% of likes match posts",
        "",
    ])
    
    # Text samples
    if text_quality.get('sample_texts'):
        md_lines.extend([
            "---",
            "",
            "## Sample Text Content",
            "",
        ])
        for i, txt in enumerate(text_quality['sample_texts'][:3], 1):
            md_lines.append(f"{i}. {txt[:200]}{'...' if len(txt) > 200 else ''}")
        md_lines.append("")
    
    # Recommendations
    md_lines.extend([
        "---",
        "",
        "## Recommendations for Pipeline Integration",
        "",
    ])
    
    if compat['warnings']:
        for w in compat['warnings']:
            if 'instead of lowercase' in w:
                md_lines.append("1. **Rename columns**: Map uppercase column names to lowercase (e.g., `DID` → `did`)")
            if 'SubjectURI' in w:
                md_lines.append("2. **Parse SubjectURI**: Extract `subject_cid` or use URI-based joining")
    
    if not embedding_quality.get('available') or all(e.get('null_pct', 100) >= 100 for e in embedding_quality.get('columns', [])):
        md_lines.append("3. **Compute embeddings**: No pre-computed embeddings available; pipeline will compute from text")
    
    if join_results['success'] and join_results['join_rate'] < 100:
        md_lines.append(f"4. **Data coverage**: {100 - join_results['join_rate']:.1f}% of likes reference posts not in this dataset")
    
    md_lines.append("")
    
    return '\n'.join(md_lines), report


def main():
    print(section_header("GCS DATA INVESTIGATION"))
    print(f"Posts file: {POSTS_FILE}")
    print(f"Likes file: {LIKES_FILE}")
    
    # Load data
    print(section_header("LOADING DATA", "-"))
    posts_df = pd.read_parquet(POSTS_FILE)
    likes_df = pd.read_parquet(LIKES_FILE)
    print(f"Posts: {len(posts_df):,} rows, {len(posts_df.columns)} columns")
    print(f"Likes: {len(likes_df):,} rows, {len(likes_df.columns)} columns")
    
    # Analyze schemas
    print(section_header("ANALYZING SCHEMAS", "-"))
    posts_info = analyze_dataframe(posts_df, "posts")
    likes_info = analyze_dataframe(likes_df, "likes")
    
    print("\nPosts columns:")
    for col in posts_info['column_names']:
        details = posts_info['column_details'][col]
        print(f"  - {col}: {details['dtype']} (null: {details['null_pct']}%)")
    
    print("\nLikes columns:")
    for col in likes_info['column_names']:
        details = likes_info['column_details'][col]
        print(f"  - {col}: {details['dtype']} (null: {details['null_pct']}%)")
    
    # Check compatibility
    print(section_header("PIPELINE COMPATIBILITY CHECK", "-"))
    compat = check_pipeline_compatibility(posts_info, likes_info)
    
    print(f"\nPosts:")
    print(f"  has 'did': {compat['posts']['has_did']}")
    print(f"  has 'commit_cid': {compat['posts']['has_commit_cid']}")
    print(f"  has 'rkey': {compat['posts']['has_rkey']}")
    print(f"  text column: {compat['posts']['text_column']}")
    print(f"  text has content: {compat['posts']['text_has_content']}")
    
    print(f"\nLikes:")
    print(f"  has 'did': {compat['likes']['has_did']} (column: {compat['likes']['did_column_name']})")
    print(f"  has 'subject_cid': {compat['likes']['has_subject_cid']}")
    print(f"  has SubjectURI: {compat['likes']['has_subject_uri']}")
    
    # Test join
    print(section_header("JOIN TEST", "-"))
    join_results = test_join(posts_df, likes_df, compat)
    
    print(f"\nJoin method: {join_results.get('method', 'N/A')}")
    print(f"Posts join keys: {join_results['posts_join_key_unique']:,}")
    print(f"Likes join keys: {join_results['likes_join_key_unique']:,}")
    print(f"Overlap: {join_results['overlap_count']:,}")
    print(f"Join rate: {join_results['join_rate']}%")
    print(f"SUCCESS: {join_results['success']}")
    
    # Check text quality
    print(section_header("TEXT QUALITY CHECK", "-"))
    text_col = compat['posts']['text_column']
    text_quality = check_text_quality(posts_df, text_col) if text_col else {'available': False}
    
    if text_quality['available']:
        print(f"\nText column: {text_quality['column_name']}")
        print(f"Non-empty: {text_quality['non_empty_count']:,} / {text_quality['total_rows']:,} ({text_quality['non_empty_pct']}%)")
        if 'avg_length' in text_quality:
            print(f"Avg length: {text_quality['avg_length']:.0f} chars")
            print(f"\nSample texts:")
            for txt in text_quality.get('sample_texts', [])[:3]:
                print(f"  - {txt[:80]}...")
    else:
        print("No text column available")
    
    # Check embeddings
    print(section_header("EMBEDDING CHECK", "-"))
    embedding_quality = check_embedding_quality(posts_df)
    
    if embedding_quality['available']:
        for emb in embedding_quality['columns']:
            print(f"\nColumn: {emb['name']}")
            print(f"  Null%: {emb['null_pct']}%")
            if emb.get('is_array'):
                print(f"  Dimension: {emb.get('embedding_dim', '?')}")
                if emb.get('sample_values'):
                    print(f"  Sample: {emb['sample_values']}")
    else:
        print("No embedding columns found")
    
    # Generate reports
    print(section_header("GENERATING REPORTS", "-"))
    md_report, json_report = generate_report(
        posts_info, likes_info, compat, join_results, text_quality, embedding_quality
    )
    
    # Save reports
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(json_report, f, indent=2, default=str)
    print(f"JSON report: {OUTPUT_JSON}")
    
    with open(OUTPUT_MD, 'w') as f:
        f.write(md_report)
    print(f"Markdown report: {OUTPUT_MD}")
    
    # Summary
    print(section_header("SUMMARY"))
    
    if not compat['issues'] and join_results['success']:
        print("\n✅ DATA IS COMPATIBLE WITH PIPELINE")
        print("\nMinor adaptations needed:")
        for w in compat['warnings']:
            print(f"  - {w}")
    elif join_results['success']:
        print("\n⚠️ DATA IS PARTIALLY COMPATIBLE")
        print("\nIssues to address:")
        for issue in compat['issues']:
            print(f"  - {issue}")
    else:
        print("\n❌ DATA IS NOT COMPATIBLE")
        print("\nCritical issues:")
        for issue in compat['issues']:
            print(f"  - {issue}")
    
    print(f"\n{'='*70}")
    print("See detailed reports:")
    print(f"  - {OUTPUT_MD}")
    print(f"  - {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

