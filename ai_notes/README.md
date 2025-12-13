# 🎯 Will's Tinkering Folder - Engagement Prediction Pipeline

This is a **complete, self-contained copy** of the engagement prediction pipeline for experimentation and tinkering.

## 🏗️ **What This Contains**

This folder contains everything needed to run the full engagement prediction pipeline:

### **Core Files**
- `with_image_pipeline_model.py` - Main training pipeline script
- `evaluate_holdout_users.py` - Holdout user evaluation script (generates CSVs)
- `utils/` - All utility modules (data processing, training, visualization)

### **Output Directories**
- `checkpoints/` - Trained model files (`.pth` format)
- `holdout_results/` - CSV files with user predictions 
- `plots/` - Training visualizations and performance plots
- `test/` - Detailed test outputs and logs

## 🚀 **Quick Start**

### **0. Activate Environment**
```bash
# First, activate the conda environment
conda activate vox
cd /srv/vox/engagement_prediction/wills_tinkering_folder
```

### **1. Test Setup**
```bash
# Verify everything is working
python test_setup.py
```

### **2. Basic Training Run**
```bash
# Quick test run (fast, small dataset)
python with_image_pipeline_model.py --days 1 --max-samples 500 --epochs 2 --limit-images

# Full training run 
python with_image_pipeline_model.py --days 5 --min-likes-per-user 4 --epochs 50
```

### **3. Evaluate Holdout Users**
```bash
# After training, evaluate holdout users (generates CSVs)
python evaluate_holdout_users.py

# Use specific model
python evaluate_holdout_users.py --model-path checkpoints/final_engagement_model_20250108_123456.pth
```

## 📊 **Pipeline Overview**

### **Training Flow**
1. **Data Loading**: Downloads recent Bluesky posts, likes, and images from S3
2. **Holdout Creation**: Reserves 20% of users for final evaluation (never seen during training)
3. **Feature Extraction**: Computes text embeddings (384D) and image embeddings (512D)
4. **User Embeddings**: Creates user profiles from their liked posts (temporal split)
5. **Model Training**: Trains neural network to predict user engagement
6. **Evaluation**: Tests on held-out data with user tracking

### **Output Files**
- `checkpoints/final_engagement_model_*.pth` - Trained models with holdout data
- `holdout_results/user_did:plc:*_posts.csv` - Predictions for holdout users
- `plots/training_history_*.png` - Training curves
- `plots/model_performance_*.png` - ROC curves, confusion matrices

## 🛠️ **Key Parameters**

### **Data Parameters**
- `--days N` - Days of data to load (default: 5)
- `--min-likes-per-user N` - Minimum posts per user (default: 4, min required)
- `--max-samples N` - Limit total posts for testing
- `--test-ratio F` - Test set ratio (default: 0.2)

### **Model Parameters**
- `--hidden-dims N M K` - Hidden layer sizes (default: [64, 32, 16])
- `--dropout-rate F` - Dropout rate (default: 0.5)
- `--batch-size N` - Batch size (default: 256)
- `--learning-rate F` - Learning rate (default: 0.001)
- `--epochs N` - Training epochs (default: 300)

### **Feature Options**
- `--limit-images` - Disable image embeddings (faster, text-only)
- `--drop-unliked-posts` - Remove posts with no likes
- `--track-users N` - Number of users to track during training

## 🔬 **Experimentation Ideas**

### **Model Architecture**
```bash
# Larger model
python with_image_pipeline_model.py --hidden-dims 128 64 32 --dropout-rate 0.3

# Smaller model (faster)
python with_image_pipeline_model.py --hidden-dims 32 16 --dropout-rate 0.7

# Text-only model
python with_image_pipeline_model.py --limit-images --hidden-dims 64 32
```

### **Data Variations**
```bash
# More recent data
python with_image_pipeline_model.py --days 1 --max-samples 1000

# Longer time period
python with_image_pipeline_model.py --days 10 --min-likes-per-user 6

# Clean dataset (only liked posts)
python with_image_pipeline_model.py --drop-unliked-posts
```

### **Training Variations**
```bash
# Aggressive training
python with_image_pipeline_model.py --learning-rate 0.01 --epochs 100 --patience 20

# Conservative training
python with_image_pipeline_model.py --learning-rate 0.0001 --epochs 500 --patience 100
```

## 📁 **Directory Structure**

```
wills_tinkering_folder/
├── with_image_pipeline_model.py    # Main training script
├── evaluate_holdout_users.py       # Holdout evaluation script
├── README.md                       # This file
├── utils/                          # Utility modules
│   ├── data_utils_with_images.py   # Data loading and processing
│   ├── train_test_helpers.py       # Training and model utilities  
│   ├── visual_helpers.py           # Plotting and visualization
│   └── __init__.py                 # Python package init
├── checkpoints/                    # Model checkpoints (created)
├── holdout_results/                # CSV predictions (created)
├── plots/                          # Training plots (created)
└── test/                           # Detailed test outputs (created)
    ├── checkpoints/
    ├── plots/
    └── logs/
```

## 🎯 **Understanding the Pipeline**

### **Temporal Split Approach**
The pipeline uses a sophisticated temporal split to prevent data leakage:
1. Each user's posts are split chronologically (first half → user embedding, second half → prediction targets)
2. Holdout users (20%) are completely excluded from training
3. User embeddings are built from past behavior, predictions made on future behavior

### **Feature Engineering**
- **Text Features**: 384D sentence-transformer embeddings of post content
- **Image Features**: 512D ResNet18 embeddings of attached images  
- **User Features**: Average embeddings of posts the user previously liked
- **Final Features**: Concatenation of user + post features

### **Model Architecture**
- **Input**: Combined user + post features (768D text-only, 1792D with images)
- **Hidden**: Configurable fully-connected layers with dropout
- **Output**: Single sigmoid output (engagement probability)
- **Loss**: Binary cross-entropy with class balancing

## 🔍 **Monitoring and Debugging**

### **Training Logs**
The pipeline provides extensive logging:
- 📊 Data loading statistics
- 🔍 Feature dimension analysis  
- 📈 Training progress (loss, AUC)
- 👤 User tracking analysis
- 🧹 CUDA memory management

### **Output Files**
- **Model checkpoints** contain full training metadata and holdout user data
- **CSV files** show actual vs predicted engagement for each holdout user
- **Plot files** visualize training curves and model performance
- **JSON logs** contain detailed metrics and configuration

## ⚡ **Performance Tips**

### **For Faster Experimentation**
```bash
# Quick test run (< 5 minutes)
python with_image_pipeline_model.py --days 1 --max-samples 200 --epochs 2 --limit-images --track-users 1

# Medium run (< 30 minutes)  
python with_image_pipeline_model.py --days 2 --max-samples 2000 --epochs 10 --limit-images
```

### **For Full Experiments**
```bash
# Full run with images (2-4 hours)
python with_image_pipeline_model.py --days 5 --epochs 100 --track-users 3

# Large scale run (4+ hours)
python with_image_pipeline_model.py --days 7 --min-likes-per-user 6 --epochs 200
```

## 🔧 **Customization**

All parameters can be modified directly in the scripts or via command line arguments. The pipeline is designed to be modular and extensible.

### **Key Configuration Constants**
```python
# In with_image_pipeline_model.py
DEFAULT_HIDDEN_DIMS = [64, 32, 16]  # Model architecture
DEFAULT_DROPOUT_RATE = 0.5          # Regularization
DEFAULT_EPOCHS = 300                # Training length
DEFAULT_PATIENCE = 50               # Early stopping
```

## 🎉 **Have Fun Experimenting!**

This self-contained pipeline gives you complete freedom to experiment with:
- Model architectures and hyperparameters
- Feature engineering and data processing
- Training strategies and regularization
- Evaluation metrics and visualization

All outputs are saved with timestamps, so you can safely run multiple experiments without overwriting results.

---

**Note**: This tinkering folder is completely independent of the main pipeline. Changes here won't affect the production system. 