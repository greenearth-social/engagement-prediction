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
# `pytorch-latest-gpu` was retired from deeplearning-platform-release.
# Pin the currently supported family rather than a stale "latest" alias.
image_family="${PL_IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
image_project="${PL_IMAGE_PROJECT:-deeplearning-platform-release}"
prefix="power_likers/exports/${PL_EXPORT_RUN}"

if ! gcloud projects describe "$PL_GCP_PROJECT" >/dev/null; then
  echo "GCP project is inaccessible: ${PL_GCP_PROJECT}" >&2
  exit 65
fi
if ! gcloud billing projects describe "$PL_GCP_PROJECT" \
  --format="value(billingEnabled)" | grep -qx "True"; then
  echo "Billing is not enabled for ${PL_GCP_PROJECT}; refusing to create a VM." >&2
  exit 65
fi
if ! gcloud storage ls "gs://${PL_GCS_BUCKET}/${prefix}/SHA256SUMS.json" >/dev/null; then
  echo "Missing portability index at gs://${PL_GCS_BUCKET}/${prefix}/SHA256SUMS.json" >&2
  exit 66
fi
if gcloud compute instances describe "$PL_VM_NAME" \
  --project "$PL_GCP_PROJECT" --zone "$PL_GCP_ZONE" >/dev/null 2>&1; then
  if [[ "${PL_RESTORE_EXISTING:-0}" != "1" ]]; then
    echo "VM already exists: ${PL_VM_NAME} (${PL_GCP_ZONE}); refusing to reuse it." >&2
    echo "Set PL_RESTORE_EXISTING=1 only to resume a bootstrap that created this exact VM." >&2
    exit 67
  fi
  echo "Resuming restore on existing VM: ${PL_VM_NAME} (${PL_GCP_ZONE})"
else
  gcloud compute instances create "$PL_VM_NAME" \
    --project "$PL_GCP_PROJECT" \
    --zone "$PL_GCP_ZONE" \
    --machine-type "$machine_type" \
    --accelerator "type=${accelerator},count=1" \
    --maintenance-policy TERMINATE \
    --boot-disk-size "$disk_size" \
    --boot-disk-type pd-balanced \
    --image-family "$image_family" \
    --image-project "$image_project" \
    --scopes cloud-platform \
    --metadata "power_likers_export=${PL_EXPORT_RUN}"
fi

gcloud compute ssh "$PL_VM_NAME" --project "$PL_GCP_PROJECT" --zone "$PL_GCP_ZONE" --command "
  set -euo pipefail
  mkdir -p ~/power-likers
  gcloud storage rsync --recursive gs://${PL_GCS_BUCKET}/${prefix}/ ~/power-likers/
  cd ~/power-likers/code/engagement-prediction
  export MAMBA_ROOT_PREFIX=\$HOME/.local/share/micromamba
  export PATH=\$HOME/.local/bin:\$PATH
  if ! command -v micromamba >/dev/null; then
    mkdir -p \$HOME/.local/bin
    curl --fail --location --retry 3 --proto '=https' --tlsv1.2 \
      https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar --extract --bzip2 --file - --strip-components=1 \
        --directory \$HOME/.local/bin bin/micromamba
  fi
  micromamba create --yes --file environment.yml --name eng-pred \
    || micromamba env update --yes --file environment.yml --name eng-pred --prune
  micromamba run --name eng-pred python -m py_compile cli.py scripts/run_holdout_pred.py
  micromamba run --name eng-pred python -c 'import torch; assert torch.cuda.is_available(), \"CUDA unavailable\"; print(torch.cuda.get_device_name(0))'
  micromamba run --name eng-pred python ops/power_likers_portable/verify_export.py \
    --index ~/power-likers/SHA256SUMS.json --root ~/power-likers
"

cat <<EOF
Runner is ready:
  gcloud compute ssh ${PL_VM_NAME} --project ${PL_GCP_PROJECT} --zone ${PL_GCP_ZONE}

Next: run ops/power_likers_portable/run_targeted_validation.sh remotely.
Delete the VM after harvesting outputs:
  gcloud compute instances delete ${PL_VM_NAME} --project ${PL_GCP_PROJECT} --zone ${PL_GCP_ZONE}
EOF
