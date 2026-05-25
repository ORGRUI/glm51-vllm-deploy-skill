---
name: azure-amd-deploy-env
description: Prepare and operate the Azure AMD MI300X deployment hosts used for GLM-5.1/ATOM/vLLM serving, including host access, volatile system disk behavior, durable slow disks, local NVMe scratch, Docker data-root, and safety checks. Use when asked to deploy or debug on Azure AMD MI300X hosts, prepare fast disks, move weights, inspect host constraints, or recover after reboot/eviction.
---

# Azure AMD Deploy Env

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

## Entrypoints

Run from this skill directory:

```bash
export SSH_HOST=vmadmin@<ip>
export SSH_PASSWORD='<password-if-needed>'
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme

./scripts/inspect_azure_amd_host.sh
./scripts/prepare_local_nvme.sh
```

`inspect_azure_amd_host.sh` prints disks, mounts, GPU visibility, Docker/containerd roots, root filesystem pressure, relevant listeners, and current containers.

`prepare_local_nvme.sh` mounts an existing intended scratch filesystem or creates a RAID0 `/dev/md0` only from clearly unused `NVMe Direct Disk` devices. It refuses to format mounted disks, disks with filesystem signatures, OS disks, `/data`, `/data2`, `/mnt`, or ambiguous devices.

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
