#!/usr/bin/env bash
# Provision a disposable GCP runner, then restore a verified portability export.
#
# This script is intentionally inert until the caller supplies a project,
# bucket, zone, and unique VM name.  It neither guesses an account nor creates
# a bucket, because the export contains restricted research inputs.
#
# Example:
#   PL_GCP_PROJECT=my-project PL_GCS_BUCKET=my-private-bucket \
#   PL_GCP_ZONE=us-central1-a PL_VM_NAME=power-likers-20260724 \
#   PL_EXPORT_RUN=20260724_portability_v1 \
#   bash ops/power_likers_portable/bootstrap_gcp_vm.sh
set -euo pipefail

required=(PL_GCP_PROJECT PL_GCS_BUCKET PL_GCP_ZONE PL_VM_NAME PL_EXPORT_RUN)
for name in "${required[@]}"; do
  [[ -n "${!name:-}" ]] || { echo "Missing required environment variable: $name" >&2; exit 64; }
done

machine_type="${PL_MACHINE_TYPE:-n1-highmem-16}"  # 104 GiB RAM; matches the original substrate's memory class.
accelerator="${PL_ACCELERATOR:-nvidia-tesla-t4}"
disk_size="${PL_BOOT_DISK_GB:-500GB}"
image_family="${PL_IMAGE_FAMILY:-pytorch-latest-gpu}"
image_project="${PL_IMAGE_PROJECT:-deeplearning-platform-release}"
prefix="power_likers/exports/${PL_EXPORT_RUN}"

gcloud config set project "$PL_GCP_PROJECT"
if ! gcloud storage ls "gs://${PL_GCS_BUCKET}/${prefix}/SHA256SUMS.json" >/dev/null; then
  echo "Missing portability index at gs://${PL_GCS_BUCKET}/${prefix}/SHA256SUMS.json" >&2
  exit 66
fi

gcloud compute instances create "$PL_VM_NAME" \
  --zone "$PL_GCP_ZONE" \
  --machine-type "$machine_type" \
  --accelerator "type=${accelerator},count=1" \
  --maintenance-policy TERMINATE \
  --boot-disk-size "$disk_size" \
  --image-family "$image_family" \
  --image-project "$image_project" \
  --scopes cloud-platform \
  --metadata "power_likers_export=${PL_EXPORT_RUN}"

gcloud compute ssh "$PL_VM_NAME" --zone "$PL_GCP_ZONE" --command "
  set -euo pipefail
  mkdir -p ~/power-likers
  gcloud storage rsync --recursive gs://${PL_GCS_BUCKET}/${prefix}/ ~/power-likers/
  cd ~/power-likers/code/engagement-prediction
  conda env create -f environment.yml -n eng-pred || conda env update -f environment.yml -n eng-pred --prune
  source \$(conda info --base)/etc/profile.d/conda.sh
  conda activate eng-pred
  python -m py_compile cli.py scripts/run_holdout_pred.py
  python -c 'import torch; assert torch.cuda.is_available(), \"CUDA unavailable\"; print(torch.cuda.get_device_name(0))'
  python ops/power_likers_portable/verify_export.py \
    --index ~/power-likers/SHA256SUMS.json --root ~/power-likers
"

cat <<EOF
Runner is ready:
  gcloud compute ssh ${PL_VM_NAME} --zone ${PL_GCP_ZONE}

Next: run ops/power_likers_portable/run_targeted_validation.sh remotely.
Delete the VM after harvesting outputs:
  gcloud compute instances delete ${PL_VM_NAME} --zone ${PL_GCP_ZONE}
EOF
