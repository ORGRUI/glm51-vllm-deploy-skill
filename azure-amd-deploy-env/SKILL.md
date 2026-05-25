---
name: azure-amd-deploy-env
description: Prepare and operate the Azure AMD MI300X deployment hosts used for GLM-5.1/ATOM/vLLM serving, including host access, volatile system disk behavior, durable slow disks, local NVMe scratch, Docker data-root, and safety checks. Use when asked to deploy or debug on Azure AMD MI300X hosts, prepare fast disks, move weights, inspect host constraints, or recover after reboot/eviction.
---

# Azure AMD Deploy Env

This skill is self-contained. Do not rely on extra references or helper scripts when using it.

## Host Facts

- Target servers are Azure AMD hosts with 8 AMD MI300X GPUs.
- The local working machine has no GPU; GPU validation must run on a user-provided target host.
- Ask the user for the target IP when one is not provided. Do not discover or guess a host.
- Default login user is `vmadmin`. Use the operator-provided password or temporary key access; do not commit credentials.
- Clean temporary SSH host aliases and temporary authorized keys after the job when you added them.
- A reboot or recycle can discard system-disk state. Treat `/data` and `/data2` as durable data disks.
- `/data` and `/data2` are durable but slow. Use them for logs, scripts, service records, Docker/containerd state, persistent caches, and final model backups.
- Local NVMe is the fast scratch tier. Prefer `/local_nvme` for downloads, extraction, Hugging Face cache, BF16 merge output, FP8 quant output, and the live serving model.
- Do not use `/mnt` for this workflow. It is not the intended fast scratch or durable deployment root in this environment.
- Do not put model work, Docker layers, containerd layers, pip wheels, HF cache, archives, or temporary extraction files under `/`, `/tmp`, `/var/tmp`, `/var/lib/docker`, `/var/lib/containerd`, `/home/*/.cache`, or `/mnt`.

## Standard Layout

```text
/data/amd_profiling/          durable root for scripts, configs, logs, records
/data/amd_profiling/models/   durable backup of final FP8 artifacts
/data/docker                  Docker data-root
/data/containerd              containerd root if containerd is used directly
/local_nvme/amd_profiling/    ephemeral fast scratch and live model path
```

If a host has rebooted, recreate or remount `/local_nvme`, then restore the live model from `/data/amd_profiling/models/...` to `/local_nvme/...` before serving. Serving directly from `/data` is only an emergency manual path after the operator accepts the speed risk.

## Connect

Use the user-provided host:

```bash
export SSH_HOST=vmadmin@IP_ADDRESS
export SSH_PASSWORD='<password-if-needed>'
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme
```

Use `sshpass` only when password auth is required:

```bash
if [ -n "${SSH_PASSWORD:-}" ]; then
  SSHPASS="$SSH_PASSWORD" sshpass -e ssh -o StrictHostKeyChecking=accept-new "$SSH_HOST"
else
  ssh -o StrictHostKeyChecking=accept-new "$SSH_HOST"
fi
```

## Inspect Host

Run this before model work:

```bash
if [ -n "${SSH_PASSWORD:-}" ]; then
  SSH_PREFIX=(sshpass -e ssh)
  export SSHPASS="$SSH_PASSWORD"
else
  SSH_PREFIX=(ssh)
fi
: "${REMOTE_ROOT:=/data/amd_profiling}"
: "${LOCAL_SCRATCH_MOUNT:=/local_nvme}"

"${SSH_PREFIX[@]}" -o StrictHostKeyChecking=accept-new "$SSH_HOST" \
  "env REMOTE_ROOT=$(printf '%q' "$REMOTE_ROOT") LOCAL_SCRATCH_MOUNT=$(printf '%q' "$LOCAL_SCRATCH_MOUNT") bash -se" <<'REMOTE'
set -euo pipefail
: "${REMOTE_ROOT:=/data/amd_profiling}"
: "${LOCAL_SCRATCH_MOUNT:=/local_nvme}"

echo "== host =="
hostname
who -b || true
uname -a

echo "== filesystems =="
df -h / /data /data2 "$LOCAL_SCRATCH_MOUNT" /mnt 2>/dev/null || true
findmnt -rn / /data /data2 "$LOCAL_SCRATCH_MOUNT" /mnt 2>/dev/null || true

echo "== block devices =="
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL
sudo blkid 2>/dev/null || true

echo "== root pressure =="
root_avail_kb=$(df -Pk / | awk 'NR==2 {print $4}')
echo "root_avail_kb=${root_avail_kb:-unknown}"
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
ls -la "$REMOTE_ROOT" 2>/dev/null || true
ls -la "$LOCAL_SCRATCH_MOUNT" 2>/dev/null || true
REMOTE
```

Stop before pulling images if `DockerRootDir` is outside `/data` or `/data2`. Stop before model work if root has less than 20 GiB free.

## Prepare Local NVMe

Use `/local_nvme` as fast scratch. This procedure mounts an existing intended scratch filesystem or creates RAID0 `/dev/md0` only from clearly unused Azure `NVMe Direct Disk` devices:

```bash
if [ -n "${SSH_PASSWORD:-}" ]; then
  SSH_PREFIX=(sshpass -e ssh)
  export SSHPASS="$SSH_PASSWORD"
else
  SSH_PREFIX=(ssh)
fi
: "${LOCAL_SCRATCH_MOUNT:=/local_nvme}"

"${SSH_PREFIX[@]}" -o StrictHostKeyChecking=accept-new "$SSH_HOST" \
  "env LOCAL_SCRATCH_MOUNT=$(printf '%q' "$LOCAL_SCRATCH_MOUNT") bash -se" <<'REMOTE'
set -euo pipefail
: "${LOCAL_SCRATCH_MOUNT:=/local_nvme}"

if [ "$LOCAL_SCRATCH_MOUNT" = "/mnt" ]; then
  echo "Refusing to use /mnt for this Azure AMD deployment environment" >&2
  exit 3
fi

if findmnt -rn "$LOCAL_SCRATCH_MOUNT" >/dev/null 2>&1; then
  echo "$LOCAL_SCRATCH_MOUNT is already mounted"
  df -h "$LOCAL_SCRATCH_MOUNT"
  exit 0
fi

if [ -b /dev/md0 ] && sudo blkid /dev/md0 >/dev/null 2>&1; then
  label=$(sudo blkid -s LABEL -o value /dev/md0 2>/dev/null || true)
  if [ "$label" != "LOCAL_NVME" ] && [ -n "$label" ]; then
    echo "/dev/md0 exists but label is not LOCAL_NVME: $label" >&2
    exit 3
  fi
  sudo mkdir -p "$LOCAL_SCRATCH_MOUNT"
  sudo mount /dev/md0 "$LOCAL_SCRATCH_MOUNT"
  sudo chown "$(id -u):$(id -g)" "$LOCAL_SCRATCH_MOUNT"
  df -h "$LOCAL_SCRATCH_MOUNT"
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

if [ "${#candidates[@]}" -lt 2 ]; then
  echo "No mounted scratch and fewer than two unused NVMe Direct Disk candidates" >&2
  lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL >&2
  exit 3
fi

if sudo blkid "${candidates[@]}" 2>/dev/null | grep -q .; then
  echo "One or more NVMe candidates already has a filesystem signature; inspect manually" >&2
  sudo blkid "${candidates[@]}" 2>/dev/null || true
  exit 3
fi

echo "Creating RAID0 local scratch from: ${candidates[*]}"
sudo mdadm --create /dev/md0 --level=0 --raid-devices="${#candidates[@]}" --chunk=1024K "${candidates[@]}"
sudo mkfs.ext4 -F -L LOCAL_NVME /dev/md0
sudo mkdir -p "$LOCAL_SCRATCH_MOUNT"
sudo mount /dev/md0 "$LOCAL_SCRATCH_MOUNT"
sudo chown "$(id -u):$(id -g)" "$LOCAL_SCRATCH_MOUNT"
df -h "$LOCAL_SCRATCH_MOUNT"
REMOTE
```

## Safety Rules

- Stop if root filesystem free space is under 20 GiB before model work.
- Stop if DockerRootDir is outside `/data` or `/data2` before pulling images.
- Stop if local NVMe cannot be mounted safely.
- Stop if an existing `/dev/md0` or candidate NVMe device is ambiguous.
- Stop if active containers conflict with the requested service; a normal deployment should leave only the current vLLM container and its Caddy container.
- Always verify `torch.version.hip` and `torch.cuda.device_count() >= 8` in the merge/quant venv before GPU merge or quantization.

## Weight Movement

For a fresh boot:

1. Inspect host state.
2. Mount or recreate `/local_nvme`.
3. Recreate scratch directories under `/local_nvme/amd_profiling/<run_slug>`.
4. Prefer direct Hugging Face download into local NVMe for base weights.
5. Restore final FP8 model from durable `/data/.../models` to local NVMe only when a durable copy already exists.
6. Serve from local NVMe.

For a new deployment, download OSS archives, extract adapters, prefetch base shards, merge, and quantize directly under `/local_nvme`; sync only the final FP8 artifact back to `/data` as durable backup.
