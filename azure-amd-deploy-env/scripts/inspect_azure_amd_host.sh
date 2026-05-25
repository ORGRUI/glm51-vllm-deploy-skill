#!/usr/bin/env bash
set -euo pipefail

: "${SSH_HOST:?Set SSH_HOST, for example vmadmin@1.2.3.4}"
: "${REMOTE_ROOT:=/data/amd_profiling}"
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
REMOTE_ROOT=$(printf "%q" "${REMOTE_ROOT}")
LOCAL_SCRATCH_MOUNT=$(printf "%q" "${LOCAL_SCRATCH_MOUNT}")

echo "== host =="
hostname
who -b || true
uname -a

echo "== filesystems =="
df -h / /data /data2 "\$LOCAL_SCRATCH_MOUNT" /mnt 2>/dev/null || true
findmnt -rn / /data /data2 "\$LOCAL_SCRATCH_MOUNT" /mnt 2>/dev/null || true

echo "== block devices =="
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL
sudo blkid 2>/dev/null || true

echo "== root pressure =="
root_avail_kb=\$(df -Pk / | awk 'NR==2 {print \$4}')
echo "root_avail_kb=\${root_avail_kb:-unknown}"
sudo du -shx /var/lib/docker /var/lib/containerd /tmp /var/tmp /home/*/.cache 2>/dev/null || true

echo "== gpu =="
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname --showuse --showmemuse || true
else
  echo "rocm-smi not found"
fi

echo "== docker/containerd =="
if command -v docker >/dev/null 2>&1; then
  docker info --format 'DockerRootDir={{.DockerRootDir}}' || true
  docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  docker images --format 'table {{.Repository}}:{{.Tag}}\t{{.Size}}' | head -50 || true
else
  echo "docker not found"
fi
if command -v containerd >/dev/null 2>&1; then
  containerd config dump 2>/dev/null | grep -E 'root =|state =' | head -10 || true
else
  echo "containerd command not found"
fi

echo "== service ports =="
ss -ltnp 2>/dev/null | grep -E ':(7777|7788|18080|8000)' || true

echo "== deployment roots =="
ls -la "\$REMOTE_ROOT" 2>/dev/null || true
ls -la "\$LOCAL_SCRATCH_MOUNT" 2>/dev/null || true
EOF
