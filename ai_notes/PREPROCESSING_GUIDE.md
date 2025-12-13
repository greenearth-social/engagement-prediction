# 🔧 Data Preprocessing Pipeline Guide

This guide explains how to use the new first-stage preprocessing tools for the engagement prediction pipeline.

## 🏗️ **How Holdout Users Work**

**Holdout users** are completely separate from train/test splits and work as follows:

### **Definition Process:**
1. **Find Active Users**: Users with ≥ `min_likes_per_user` likes (default: 3 likes)
2. **Calculate Holdout Size**: 20% of active users (minimum 10 users) 
3. **Random Selection**: Randomly sample holdout users with fixed seed (42)
4. **Complete Separation**: Holdout users are **entirely excluded** from training data
5. **Final Evaluation Only**: Used only for final model evaluation after training

### **Data Flow:**
```
All Users → [Remove Holdout Users] → Remaining Users → [Train/Test Split] → Training
                     ↓
              [Holdout Users] → Final Evaluation Only (generates CSVs)
```

### **Key Points:**
- **No Data Leakage**: Holdout users never appear in training or validation
- **Reproducible**: Same seed = same holdout users across runs
- **Realistic Evaluation**: Tests model on completely unseen users
- **Temporal Safety**: Uses temporal splits to prevent future data leakage

---

## 🚀 **Quick Start**

### **1. Process Data (First Time)**
```bash
conda activate vox
cd /srv/vox/engagement_prediction/wills_tinkering_folder

# Small test run
python data_preprocessor.py --days 1 --min-likes 3 --max-samples 500 --limit-images

# Full production run  
python data_preprocessor.py --days 5 --min-likes 4 --max-samples 50000
```

### **2. Inspect Processed Data**
```bash
# List all processed files
python data_loader.py list

# Inspect latest file (detailed)
python data_loader.py inspect

# Inspect specific file
python data_loader.py inspect --file processed_data/processed_data_20250811_125419.pkl
```

### **3. Use Processed Data in Training**
```python
from data_loader import load_processed_data

# Load processed data
data = load_processed_data("processed_data/processed_data_20250811_125419.pkl")

# Access components
train_df = data['train_df']
test_df = data['test_df'] 
holdout_users = data['holdout_users']
training_user_embeddings = data['training_user_embeddings']
holdout_user_embeddings = data['holdout_user_embeddings']
```

---

## 📋 **Command Reference**

### **Data Preprocessor (`data_preprocessor.py`)**

```bash
python data_preprocessor.py [OPTIONS]
```

#### **Key Options:**
- `--days N` - Days of data to process (default: 5)
- `--min-likes N` - Minimum likes per user (default: 4)  
- `--test-ratio F` - Test set ratio (default: 0.2)
- `--holdout-ratio F` - Holdout user ratio (default: 0.2)
- `--random-seed N` - Random seed (default: 42)
- `--max-samples N` - Maximum samples to process
- `--limit-images` - Disable image processing (faster)
- `--drop-unliked` - Drop posts user didn't like

#### **Examples:**
```bash
# Quick test with 1 day, 500 samples
python data_preprocessor.py --days 1 --max-samples 500 --limit-images

# Production run with 5 days, 50K samples
python data_preprocessor.py --days 5 --max-samples 50000

# High-quality run with images (slower)
python data_preprocessor.py --days 7 --max-samples 100000
```

### **Data Loader (`data_loader.py`)**

```bash
python data_loader.py COMMAND [OPTIONS]
```

#### **Commands:**
- `list` - List all processed data files
- `inspect` - Detailed inspection of a file
- `compare` - Compare two processed files

#### **Options:**
- `--file PATH` - Specific file to inspect
- `--file2 PATH` - Second file for comparison
- `--directory DIR` - Directory to search (default: processed_data)

#### **Examples:**
```bash
# List all files
python data_loader.py list

# Inspect latest file
python data_loader.py inspect

# Inspect specific file
python data_loader.py inspect --file processed_data/processed_data_20250811_125419.pkl

# Compare two files
python data_loader.py compare --file file1.pkl --file2 file2.pkl
```

---

## 📊 **Understanding the Output**

### **Processed Data Structure:**
```python
{
    'train_df': DataFrame,           # Training samples (user-post pairs)
    'test_df': DataFrame,            # Test samples (user-post pairs) 
    'holdout_users': List[str],      # Holdout user IDs
    'training_user_embeddings': DataFrame,  # User embeddings for training users
    'holdout_user_embeddings': DataFrame,   # User embeddings for holdout users
    'posts_emb_df': DataFrame,       # All posts with embeddings
    'embedding_dim': int,            # Embedding dimension
    'join_like': str,                # Join key for likes
    'join_post': str,                # Join key for posts  
    'feature_columns': tuple,        # Feature column info
    'text_column': str,              # Text column name
    'metadata': dict                 # Processing parameters and stats
}
```

### **Data Validation Checks:**
- ✅ **No Data Leakage**: Train/Test vs Holdout overlap = 0
- ✅ **Balanced Labels**: Train/Test positive rates similar
- ✅ **Consistent Shapes**: All DataFrames have matching structures
- ✅ **Complete Data**: All required components present

### **File Naming:**
- Format: `processed_data_YYYYMMDD_HHMMSS.pkl`
- Timestamp: When preprocessing completed
- Summary: Matching `.json` file with metadata

---

## 📊 **Automatic HTML Report Generation**

**NEW FEATURE**: Each data preprocessing run now automatically generates a comprehensive HTML report with:

### **Section 1: Data Overview & Statistics**
- **Processing Parameters**: Days, samples, ratios, seed, etc.
- **Data Size Distribution**: Visual breakdown of train/test/holdout
- **User Distribution**: Training vs holdout user counts
- **Examples per User**: Histograms for embedding and training data
- **Label Balance**: Positive/negative rates in train/test sets
- **Embedding Dimensions**: Text + image embedding breakdown
- **Summary Statistics**: Complete data size table

### **Section 2: Quality Checks & Validation**
- **Data Leakage Check**: Critical validation (train ↔ holdout overlap)
- **User Set Relationships**: Detailed overlap analysis
- **Sample Distribution**: Per-user sample count histograms
- **Feature Completeness**: Missing value analysis
- **Embedding Quality**: Value distribution analysis
- **Data Quality Score**: Overall score out of 100
- **Validation Results**: Pass/fail for critical checks
- **Recommendations**: Actionable suggestions for improvement

### **Files Generated**
Each preprocessing run creates three timestamped files:
- `processed_data_YYYYMMDD_HHMMSS.pkl` - Main data file
- `summary_YYYYMMDD_HHMMSS.json` - Quick summary metadata
- `report_YYYYMMDD_HHMMSS.html` - **Comprehensive visual report**

### **Viewing Reports**
HTML reports automatically adapt to any screen size and can be opened in any web browser:

```bash
# Open latest HTML report in browser
open processed_data/report_20250811_130824.html

# Or on Linux
xdg-open processed_data/report_20250811_130824.html

# List all reports
python data_loader.py list
```

**Benefits of HTML Reports:**
- ✅ **Responsive Design**: Automatically adapts to any screen size
- ✅ **No External Dependencies**: Opens in any web browser
- ✅ **Interactive**: Hover effects and responsive layout
- ✅ **Embedded Images**: All plots embedded directly in the file
- ✅ **Professional Styling**: Clean, modern design
- ✅ **Self-Contained**: Single file with everything included

---

## 🎯 **Best Practices**

### **For Development:**
```bash
# Quick iteration - small dataset
python data_preprocessor.py --days 1 --max-samples 1000 --limit-images
```

### **For Experiments:**
```bash
# Medium dataset - good for testing
python data_preprocessor.py --days 3 --max-samples 10000 --limit-images  
```

### **For Production:**
```bash
# Full dataset - best results
python data_preprocessor.py --days 7 --max-samples 100000
```

### **Parameter Guidelines:**
- **days**: 1-2 for testing, 5-7 for production
- **min-likes**: 3-4 (higher = fewer users, better quality)
- **max-samples**: 500-1K for testing, 10K-100K for production
- **limit-images**: Use for faster iteration, disable for best results

---

## 🔍 **Troubleshooting**

### **Common Issues:**

1. **"No users found with minimum likes"**
   - Lower `--min-likes` parameter
   - Increase `--max-samples` or `--days`

2. **"Processed data file not found"**
   - Check file path with `python data_loader.py list`
   - Use absolute path if needed

3. **Memory issues**
   - Reduce `--max-samples`
   - Use `--limit-images` 
   - Process in smaller chunks

4. **Long processing time**
   - Use `--limit-images` to disable image processing
   - Reduce `--max-samples`
   - Check available CPU cores

### **Performance Tips:**
- **GPU**: Automatically used if available (Tesla T4 detected)
- **CPU**: Uses all available cores for parallel processing
- **Memory**: ~1GB RAM per 10K samples (rough estimate)
- **Storage**: ~10MB per 1K samples in processed files

---

## 📈 **Integration with Training Pipeline**

The processed data files are designed to work seamlessly with your training pipeline:

```python
# Example: Load and use processed data
from data_loader import load_processed_data

def train_with_processed_data(processed_file):
    # Load preprocessed data
    data = load_processed_data(processed_file)
    
    # Extract components
    train_df = data['train_df']
    test_df = data['test_df']
    user_embeddings = data['training_user_embeddings']
    
    # Train model (your existing training code)
    model = train_model(train_df, user_embeddings)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_df)
    
    # Final evaluation on holdout users
    holdout_results = evaluate_holdout_users(
        model=model,
        holdout_users=data['holdout_users'],
        holdout_embeddings=data['holdout_user_embeddings']
    )
    
    return model, test_metrics, holdout_results
```

This preprocessing approach gives you:
- **Faster iteration** (no need to reprocess data each time)
- **Reproducible results** (same data splits across experiments)
- **Data quality assurance** (validation checks prevent errors)
- **Easy experimentation** (try different parameters quickly)

🎉 **Ready to tinker with confidence!** 