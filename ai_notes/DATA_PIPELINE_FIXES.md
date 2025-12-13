# Data Pipeline Fixes - Engagement Prediction Model

## Overview

This document describes the fixes implemented to address data leakage issues in the engagement prediction pipeline. The original code had several subtle but critical issues that could lead to overly optimistic model performance due to improper data splitting.

## Problems Identified

### 1. **Double Splitting Issue**
- **Problem**: The data loader was doing temporal splitting within users, then the model script was doing user-level splitting again
- **Impact**: Created confusion about which data was actually being used for training vs testing
- **Fix**: Unified the splitting logic in the data loader to do proper user-level splitting directly

### 2. **Sample-Level Train/Test Split** 
- **Problem**: Final train/test split was at the sample level (line 957-959 in original `load_bluesky_data`), allowing same users in both train and test
- **Impact**: User leakage between train and test sets, leading to overoptimistic performance estimates
- **Fix**: Implemented proper user-level splitting where train users ≠ test users

### 3. **Complex Temporal Splitting**
- **Problem**: Used temporal splitting (first half of posts for embeddings, second half for prediction) which was unnecessarily complex
- **Impact**: Made the code harder to understand and debug
- **Fix**: Simplified to random splitting while maintaining proper separation between embedding and prediction posts

### 4. **Inconsistent Data Flow**
- **Problem**: Multiple data loading steps and unclear terminology ("training" used for both embedding creation and model training)
- **Impact**: Made it difficult to verify that data leakage wasn't occurring
- **Fix**: Created clear, single-path data flow with explicit separation of concerns

## Fixes Implemented

### 1. **Updated `load_bluesky_data` Function**
**File**: `utils/data_utils_with_images.py`

**Key Changes**:
- **User-Level Splitting First**: Split users into train/test sets before any data processing
- **Random Post Splitting**: For each user, randomly split their posts into embedding vs prediction sets
- **Strict Separation**: Ensure embedding posts never appear as prediction targets
- **Holdout User Support**: Properly exclude holdout users from the training pipeline
- **Verification Checks**: Added overlap detection and validation

**Data Flow**:
```
1. Load raw posts and likes data
2. Filter users with minimum likes (≥4 posts required)
3. Split users into train_users vs test_users (user-level split)
4. For each user: randomly split posts into embedding_posts vs prediction_posts
5. Build user embeddings using ONLY embedding_posts
6. Create prediction targets using ONLY prediction_posts + negative examples
7. Create final train/test split based on user assignment
```

### 2. **Simplified `prepare_datasets` Function**
**File**: `with_image_pipeline_model.py`

**Changes**:
- **Removed Double Splitting**: No longer does user-level splitting since data loader handles it
- **Data Validation**: Verifies no user overlap between train and test sets
- **Feature Extraction**: Extracts features from pre-split data
- **Train/Val Split**: Only splits train data into train/val for model validation

### 3. **Updated Data Preprocessor**
**File**: `data_preprocessor.py`

**Changes**:
- **Simplified Workflow**: Uses the new unified data loading approach
- **Proper Holdout Handling**: Creates holdout users then reloads data with them excluded
- **Clear Reporting**: Better logging of user splits and data validation

### 4. **Added Preprocessed Data Support**
**File**: `with_image_pipeline_model.py`

**New Feature**:
- **`--load-processed` Parameter**: Can load preprocessed data files created by `data_preprocessor.py`
- **Unified Interface**: Same model training code works with both fresh and preprocessed data
- **Efficiency**: Avoids reprocessing data for repeated experiments

### 5. **Test Script Creation**
**File**: `test_data_pipeline.py`

**Purpose**:
- **Validation**: Easy way to test that the fixes are working correctly
- **Documentation**: Shows proper usage of the new pipeline
- **Examples**: Demonstrates both preprocessing and direct training workflows

## Data Splitting Architecture

### User-Level Splits
```
All Users (e.g., 1000 users with ≥4 likes each)
├── Holdout Users (20%, e.g., 200 users) - NEVER used in model training
└── Training Pool (80%, e.g., 800 users)
    ├── Train Users (80% of pool, e.g., 640 users)
    └── Test Users (20% of pool, e.g., 160 users)
```

### Post-Level Splits (within each user)
```
User's Posts (e.g., 10 posts liked by user)
├── Embedding Posts (50%, e.g., 5 posts) - Used to create user embedding
└── Prediction Posts (50%, e.g., 5 posts) - Used as positive examples for training
    ├── Positive Examples (actual likes)
    └── Negative Examples (random posts user didn't like, matched count)
```

### Final Dataset Structure
```
Training Data:
- Users: Only train_users
- Posts: Only prediction_posts (+ negatives)
- User Embeddings: Built from embedding_posts only

Test Data:
- Users: Only test_users  
- Posts: Only prediction_posts (+ negatives)
- User Embeddings: Built from embedding_posts only
```

## Validation Checks

### 1. **No User Overlap**
- Train users and test users are completely disjoint sets
- Holdout users don't appear in either train or test

### 2. **No Post Overlap** 
- Posts used for user embeddings never appear as prediction targets
- Verified by intersection checks with warning messages

### 3. **Balanced Datasets**
- Equal number of positive and negative examples
- Consistent positive rates across train/test splits

### 4. **Data Integrity**
- All users have sufficient data for both embedding creation and prediction
- No missing embeddings or features

## Usage Examples

### 1. **Preprocess Data First (Recommended)**
```bash
# Step 1: Preprocess data with proper splitting
python data_preprocessor.py --days 5 --min-likes 4 --max-samples 50000

# Step 2: Train model with preprocessed data
python with_image_pipeline_model.py --load-processed processed_data/processed_data_YYYYMMDD_HHMMSS.pkl
```

### 2. **Direct Training (For Quick Tests)**
```bash
# Train directly with fresh data loading
python with_image_pipeline_model.py --days 2 --min-likes-per-user 4 --max-samples 10000
```

### 3. **Run Tests**
```bash
# Test the complete pipeline
python test_data_pipeline.py

# Test individual components
python test_data_pipeline.py --preprocess-only
python test_data_pipeline.py --model-only
```

## Key Benefits

### 1. **Eliminates Data Leakage**
- Proper user-level splitting prevents overly optimistic performance estimates
- Clear separation between embedding creation and prediction targets

### 2. **Simplified Architecture**
- Single, clear data flow that's easy to understand and debug
- Removed complex temporal splitting logic

### 3. **Better Validation**
- Multiple verification checks to catch potential issues
- Clear logging of data splits and user statistics

### 4. **Improved Workflow**
- Support for preprocessing large datasets once and reusing them
- Consistent interface for both fresh and preprocessed data

### 5. **Future-Proof Design**
- Easy to extend with additional validation checks
- Clear separation of concerns makes modifications safer

## Notes for Future Development

1. **Minimum Likes Requirement**: Currently set to 4 (2 for embeddings + 2 for prediction). This ensures all users have sufficient data for both tasks.

2. **Random Seed Consistency**: All random operations use the same seed for reproducibility across the entire pipeline.

3. **Holdout User Handling**: Holdout users are completely excluded from model training and can be used for final evaluation without any risk of leakage.

4. **Image Processing**: The pipeline supports both text-only and text+image embeddings, with the `--limit-images` flag for faster processing during development.

5. **Memory Management**: CUDA memory is properly cleared at multiple points to prevent memory issues during long training runs.

This implementation ensures that the engagement prediction model can be trained and evaluated with proper data isolation, giving reliable performance estimates that will generalize to real-world usage. 