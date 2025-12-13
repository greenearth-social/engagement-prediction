#!/usr/bin/env python3
"""
Test script for stage_relevel_gini.py

This script tests the Gini-based relevel stage with a specific embedding bundle.
Usage:
    python test_relevel_gini.py
"""

import sys
from pathlib import Path
from utils.pipeline.core import Context
from utils.pipeline.core import load_run_callable

# The embedding bundle path provided
BUNDLE_PATH = Path("/srv/vox/engagement_prediction/wills_tinkering_folder/outputs/20251105_172556_run_d14_mppa5/02_featurize/20251105_172757/embedding_bundle_20251105_172757.pkl")

# Infer run_dir from bundle path
RUN_DIR = BUNDLE_PATH.parent.parent.parent  # .../outputs/20251105_172556_run_d14_mppa5/
FEATURIZE_DIR = BUNDLE_PATH.parent  # .../02_featurize/20251105_172757/


class TestArgs:
    """Simple args object for testing"""
    def __init__(self):
        # Default values matching the script's expectations
        self.global_topic_k = 20
        self.k_range = (20, 30)
        self.random_seed = 42
        self.use_pca = True
        self.pca_components = 50
        self.relevel_strategy = 'gini_based'
        self.target_gini = 0.1
        self.min_likes_per_user = 4
        self.relevel_min_users_per_topic = 0


def main():
    print("=" * 80)
    print("Testing stage_relevel_gini.py")
    print("=" * 80)
    print(f"Bundle path: {BUNDLE_PATH}")
    print(f"Run directory: {RUN_DIR}")
    print(f"Featurize directory: {FEATURIZE_DIR}")
    print()
    
    # Verify bundle exists
    if not BUNDLE_PATH.exists():
        print(f"❌ Error: Bundle not found at {BUNDLE_PATH}")
        return 1
    
    # Verify featurize directory exists
    if not FEATURIZE_DIR.exists():
        print(f"❌ Error: Featurize directory not found at {FEATURIZE_DIR}")
        return 1
    
    # Set up context
    context = Context(
        run_dir=RUN_DIR,
        use_latest=True,
        prior_outputs={
            '02_featurize': FEATURIZE_DIR  # Point to the specific featurize output directory
        }
    )
    
    # Set up args
    args = TestArgs()
    
    # Load the run function from stage_relevel_gini.py
    gini_script_path = Path(__file__).parent / "utils" / "03_relevel" / "stage_relevel_gini.py"
    if not gini_script_path.exists():
        print(f"❌ Error: stage_relevel_gini.py not found at {gini_script_path}")
        return 1
    
    print(f"Loading stage script: {gini_script_path}")
    run_fn = load_run_callable(gini_script_path)
    
    print("\n" + "=" * 80)
    print("Running stage_relevel_gini.py...")
    print("=" * 80)
    print()
    
    try:
        # Run the stage
        result = run_fn(context, args)
        
        print("\n" + "=" * 80)
        print("✅ Stage completed successfully!")
        print("=" * 80)
        print(f"\nOutput directory: {result.get('output_dir')}")
        print("\nArtifacts:")
        artifacts = result.get('artifacts', {})
        for key, value in artifacts.items():
            print(f"  - {key}: {value}")
        
        return 0
        
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ Error during stage execution:")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

