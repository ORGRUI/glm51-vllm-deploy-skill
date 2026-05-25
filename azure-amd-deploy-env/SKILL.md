---
name: azure-amd-deploy-env
description: Prepare and operate the Azure AMD MI300X deployment hosts used for GLM-5.1/ATOM/vLLM serving, including host access, volatile system disk behavior, durable slow disks, local NVMe scratch, Docker data-root, and safety checks. Use when asked to deploy or debug on Azure AMD MI300X hosts, prepare fast disks, move weights, inspect host constraints, or recover after reboot/eviction.
---

# Azure AMD Deploy Env

This skill is self-contained. Do not rely on extra references or helper scripts when using it. Keep execution details lightweight: perform the checks below, use the command skeletons as starting points, and adapt only after inspecting the actual host state.

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
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme
ssh -o StrictHostKeyChecking=accept-new "$SSH_HOST"
```

If password auth is required, use `sshpass -e` with `SSHPASS` in the environment. Prefer temporary key access when available, and remove any temporary public key you added before handing off.

## Inspect Host

Run these checks before model work. Capture enough output to justify the deployment decision.

```bash
hostname; who -b || true; uname -a
df -h / /data /data2 "$LOCAL_SCRATCH_MOUNT" /mnt 2>/dev/null || true
findmnt -rn / /data /data2 "$LOCAL_SCRATCH_MOUNT" /mnt 2>/dev/null || true
lsblk -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL
sudo blkid 2>/dev/null || true
sudo du -shx /var/lib/docker /var/lib/containerd /tmp /var/tmp /home/*/.cache 2>/dev/null || true
rocm-smi --showproductname --showuse --showmemuse || true
docker info --format 'DockerRootDir={{.DockerRootDir}}' 2>/dev/null || true
docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' 2>/dev/null || true
ss -ltnp 2>/dev/null | grep -E ':(7777|7788|18080|8000)' || true
ls -la "$REMOTE_ROOT" "$LOCAL_SCRATCH_MOUNT" 2>/dev/null || true
```

Judgment standards:

- GPU path is usable only when ROCm sees 8 MI300X-class devices.
- Root must have at least 20 GiB free before downloads, image pulls, merge, quant, or serve work.
- `DockerRootDir` must be under `/data` or `/data2` before pulling images. Reconfigure first if it points at `/var/lib/docker`.
- `/data` or `/data2` must be present for durable records and final backups.
- `/local_nvme` should be mounted on a fast local disk before any large model operation.
- Active containers or listeners on service ports must be understood before replacing a service.

## Prepare Local NVMe

Use `/local_nvme` as fast scratch. Do not blindly format disks. Inspect first, then choose one of these paths:

```bash
# Already mounted: verify it is not /mnt, has enough capacity, and is writable.
findmnt -rn "$LOCAL_SCRATCH_MOUNT" && df -h "$LOCAL_SCRATCH_MOUNT"
touch "$LOCAL_SCRATCH_MOUNT/.write_test" && rm -f "$LOCAL_SCRATCH_MOUNT/.write_test"

# Existing intended RAID: only mount if /dev/md0 is clearly the LOCAL_NVME scratch volume.
sudo blkid /dev/md0
sudo mkdir -p "$LOCAL_SCRATCH_MOUNT"
sudo mount /dev/md0 "$LOCAL_SCRATCH_MOUNT"
sudo chown "$(id -u):$(id -g)" "$LOCAL_SCRATCH_MOUNT"

# Fresh scratch creation: only from unused Azure "NVMe Direct Disk" devices.
lsblk -P -dn -o NAME,TYPE,SIZE,FSTYPE,MOUNTPOINT,MODEL
sudo blkid /dev/nvme...  # candidates must have no filesystem signatures
sudo mdadm --create /dev/md0 --level=0 --raid-devices=N --chunk=1024K /dev/nvme...
sudo mkfs.ext4 -F -L LOCAL_NVME /dev/md0
sudo mount /dev/md0 "$LOCAL_SCRATCH_MOUNT"
```

Judgment standards:

- Refuse `/mnt` as the scratch mount.
- Prefer an existing mounted `/local_nvme` when it is clearly the fast scratch tier.
- Reuse `/dev/md0` only if its label or prior evidence identifies it as the deployment scratch volume.
- Create RAID0 only from unmounted, filesystem-free `NVMe Direct Disk` devices. Do not include OS, durable `/data`, or ambiguous devices.
- Stop and ask for human confirmation when device identity is unclear, a candidate has a filesystem signature, or fewer than two suitable NVMe direct disks are visible.

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
