#!/usr/bin/env bash
set -euo pipefail

: "${SSH_HOST:?Set SSH_HOST, for example vmadmin@1.2.3.4}"
: "${LOCAL_SCRATCH_MOUNT:=/local_nvme}"

ssh_base() {
  if [[ -n "${SSH_PASSWORD:-}" ]]; then
    command -v sshpass >/dev/null 2>&1 || {
      echo "SSH_PASSWORD is set but sshpass is not installed locally" >&2
      exit 2
    }
    SSHPASS="${SSH_PASSWORD}" sshpass -e "$@"
  else
    "$@"
  fi
}

ssh_base ssh -o StrictHostKeyChecking=accept-new "${SSH_HOST}" "bash -se" <<EOF
set -euo pipefail
LOCAL_SCRATCH_MOUNT=$(printf "%q" "${LOCAL_SCRATCH_MOUNT}")

if [ "\$LOCAL_SCRATCH_MOUNT" = "/mnt" ]; then
  echo "Refusing to use /mnt for this Azure AMD deployment environment" >&2
  exit 3
fi

if findmnt -rn "\$LOCAL_SCRATCH_MOUNT" >/dev/null 2>&1; then
  echo "\$LOCAL_SCRATCH_MOUNT is already mounted"
  df -h "\$LOCAL_SCRATCH_MOUNT"
  exit 0
fi

if [ -b /dev/md0 ] && sudo blkid /dev/md0 >/dev/null 2>&1; then
  label=\$(sudo blkid -s LABEL -o value /dev/md0 2>/dev/null || true)
  if [ "\$label" != "LOCAL_NVME" ] && [ -n "\$label" ]; then
    echo "/dev/md0 exists but label is not LOCAL_NVME: \$label" >&2
    exit 3
  fi
  sudo mkdir -p "\$LOCAL_SCRATCH_MOUNT"
  sudo mount /dev/md0 "\$LOCAL_SCRATCH_MOUNT"
  sudo chown "\$(id -u):\$(id -g)" "\$LOCAL_SCRATCH_MOUNT"
  df -h "\$LOCAL_SCRATCH_MOUNT"
  exit 0
fi

mapfile -t candidates < <(python3 - <<'PY'
import shlex
import subprocess

out = subprocess.check_output(
    ["lsblk", "-P", "-dn", "-o", "NAME,TYPE,FSTYPE,MOUNTPOINT,MODEL"],
    text=True,
)
for line in out.splitlines():
    fields = dict(item.split("=", 1) for item in shlex.split(line))
    name = fields.get("NAME", "")
    if (
        fields.get("TYPE") == "disk"
        and name.startswith("nvme")
        and not fields.get("FSTYPE")
        and not fields.get("MOUNTPOINT")
        and "NVMe Direct Disk" in fields.get("MODEL", "")
    ):
        print("/dev/" + name)
PY
)

if [ "\${#candidates[@]}" -lt 2 ]; then
  echo "No mounted scratch and fewer than two unused NVMe Direct Disk candidates" >&2
  lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL >&2
  exit 3
fi

if sudo blkid "\${candidates[@]}" 2>/dev/null | grep -q .; then
  echo "One or more NVMe candidates already has a filesystem signature; inspect manually" >&2
  sudo blkid "\${candidates[@]}" 2>/dev/null || true
  exit 3
fi

echo "Creating RAID0 local scratch from: \${candidates[*]}"
sudo mdadm --create /dev/md0 --level=0 --raid-devices="\${#candidates[@]}" --chunk=1024K "\${candidates[@]}"
sudo mkfs.ext4 -F -L LOCAL_NVME /dev/md0
sudo mkdir -p "\$LOCAL_SCRATCH_MOUNT"
sudo mount /dev/md0 "\$LOCAL_SCRATCH_MOUNT"
sudo chown "\$(id -u):\$(id -g)" "\$LOCAL_SCRATCH_MOUNT"
df -h "\$LOCAL_SCRATCH_MOUNT"
EOF
