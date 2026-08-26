# Legacy reference code

This directory preserves selected files from the pre-canonical training
pipeline for historical reference. Nothing here is registered with the active
pipeline, covered by the supported CLI, or maintained as a runnable workflow.

- `config-old.yml` is an old pipeline configuration and contains settings that
  the canonical CLI no longer accepts.
- `run_training_sweep.sh` targets the removed MLP and legacy two-tower stages.
  It is preserved byte-for-byte and is not expected to run.
- `evaluation/` consumes ranking-row artifacts produced by the removed legacy
  `03_train` stage. Its `run(context, args)` entrypoint requires an explicit
  `args.prior_03_train`; it never searches for a latest artifact.

New work should live under `engagement_prediction/` and consume the canonical
Stage 7 and Stage 8 artifact contracts.
