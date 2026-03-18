#!/bin/bash
curl -sSO https://dl.google.com/cloudagents/add-google-cloud-ops-agent-repo.sh
sudo bash add-google-cloud-ops-agent-repo.sh --also-install



# Setup data disk (idempotent)
DISK=/dev/disk/by-id/google-ge-ml-training-data

if [[ ! -b "$DISK" ]]; then
  echo "Disk not found"
  exit 1
fi

if blkid "$DISK" >/dev/null 2>&1; then
  echo "Existing filesystem detected; not formatting"
else
  echo "No filesystem found; formatting new disk"
  mkfs.ext4 -F "$DISK"
fi

mkdir -p /mnt/data

if ! mountpoint -q /mnt/data; then
  mount "$DISK" /mnt/data
fi

if ! grep -q "^$DISK /mnt/data " /etc/fstab; then
  echo "$DISK /mnt/data ext4 defaults 0 2" >> /etc/fstab
fi



if test -f /opt/google/cuda-installer; then
  exit 0
fi

mkdir -p /opt/google/cuda-installer
cd /opt/google/cuda-installer/ || exit 1

curl -fSsL -O https://storage.googleapis.com/compute-gpu-installation-us/installer/latest/cuda_installer.pyz
python3 cuda_installer.pyz install_cuda
