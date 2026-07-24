# Power Likers portable GPU runner

This directory turns the Power Likers training pipeline into a user-controlled,
rebuildable GCP workload. It has three deliberate boundaries:

1. The frozen Stage-1 substrate is the canonical input for retraining. It
   avoids silently re-extracting a moving Bluesky/GCS source.
2. DID-bearing R2 exclusion lists are private research data. They are never
   included unless `PL_EXPORT_PRIVATE=1` is set after the destination bucket's
   IAM has been reviewed.
3. A remedy model is evaluated on the baseline substrate before comparing
   AUC. Native R2 predictions alone cannot answer how the excluded users fare.

## Security decision required before export

The currently configured `gcp-vox` project has an existing `kylix-cloud`
bucket in `US-CENTRAL1`, but its legacy project-wide reader bindings are too
broad for DID-bearing exclusion lists. Do **not** use it for the `private/`
prefix. Create a dedicated bucket with uniform bucket-level access and
least-privilege IAM, then export:

```bash
export PL_GCS_BUCKET=<new-restricted-bucket>
python ops/power_likers_portable/export_to_gcs.py \
  --bucket "$PL_GCS_BUCKET" --run-id 20260724_portability_v1 --dry-run

# Review the dry-run manifest and bucket IAM, then:
PL_EXPORT_PRIVATE=1 python ops/power_likers_portable/export_to_gcs.py \
  --bucket "$PL_GCS_BUCKET" --run-id 20260724_portability_v1
```

The export index (`SHA256SUMS.json`) records every copied file, hash, byte
count, source commit, and sensitivity class. Restore verifies it before any
training begins.

## VM sizing

The original full Stage-1 run used about 76 GiB at peak, and full evaluation
can use 15–20 GiB per worker. Use a GPU VM with at least 100 GiB RAM, one
T4/L4-class GPU, and a 500 GiB persistent disk. The bootstrap default is
`n1-highmem-16` + T4 in `US-CENTRAL1`; change the environment variables if
availability or cost requires another compatible configuration.

## Run order

1. Export and verify the package.
2. Bootstrap the VM using `bootstrap_gcp_vm.sh`.
3. Run `run_targeted_validation.sh`: one MLP baseline cell and one MLP
   R2-drop30 cell, then compare both on the same baseline holdout substrate.
4. Require `fixed_cohort_auc.json` to have nontrivial typical and power-liker
   denominators, sensible AUCs, and a result consistent with historical seed
   noise before committing the full matrix.
5. Only then run the full paper-quality matrix described in
   `260428_like_biases/jobs/0024_full_rerun_paper_quality_emissions.md`.

The full rerun must also produce: an attrition ledger, uncapped concentration
artifacts, per-cell manifests/checksums, D1 typical-user Axis A, and D2
structural controls. Those are scientific deliverables, not optional
post-processing.
