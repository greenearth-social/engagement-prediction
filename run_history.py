from pathlib import Path
from utils.pipeline.core import Context
from utils.pipeline import registry
import argparse
run_dir = "outputs/20260205_014917_start_to_get_data_mlp_uniform"

ctx = Context(
    run_dir=run_dir,
    use_latest=True,
    prior_outputs={
        "02_target_posts": Path(run_dir) / "02_target_posts/20260206_184708",
    },
)

args = argparse.Namespace(max_prior_likes=128)

registry.run_stage('user_history', ctx, args)