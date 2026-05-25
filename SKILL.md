---
name: glm51-merge-quant-vllm-deploy
description: Standalone workflow for deploying a GLM-5.1 LoRA from either a Tinker checkpoint URL or an OSS HTTP(S) archive download link: optionally convert Tinker to signed OSS, download and extract the archive on the target host, convert PEFT or raw Tinker weights to a PEFT adapter, merge into BF16 shards, quantize to corrected official-partial FP8 block-128, launch vLLM + ATOM, start capture proxy/Caddy, run smoke tests and benchmarks, and preserve launch records.
---

# GLM-5.1 OSS LoRA Merge, Quant, vLLM + ATOM Deploy

> Type: workflow

This skill is self-contained and intentionally contains no fixed host, password, IP, remote root, model path, adapter repo, token, or OSS link. If any required value is missing, ask the user before running commands.

## Standalone Usage Contract

This skill folder can be shared and run without the GPU Lease Manager webapp prompt wrapper. It still depends on external services and machines:

- With `OSS_URL`, no transfer service is needed; the local resolver validates the HTTP(S) archive URL and deployment proceeds to the target host.
- With `TINKER_URL`, provide either `GPU_LEASE_BASE_URL` plus `GPU_LEASE_API_KEY`, or `TRANSFER_JOBS_ENDPOINT`; the bundled resolver creates and polls the transfer job before any target-host SSH work.
- Target deployment requires SSH access to a prepared MI300X/ROCm host with durable `/data`, local NVMe scratch such as `/local_nvme`, Docker state under `/data`, and enough storage for BF16/FP8 intermediates.
- The target workflow creates or repairs its own merge venv, but it needs network access for the OSS archive, Hugging Face base model, Docker image, ATOM checkout, and Python packages unless those are already present.

## Source Policy

This workflow accepts either:

```text
TINKER_URL=tinker://<run_id>/weights/<checkpoint>
OSS_URL=<signed or public HTTP(S) OSS archive URL>
```

If `OSS_URL` is already provided, it must be a signed or public HTTP(S) archive URL and the workflow can proceed directly to target-host deployment. If `TINKER_URL` is provided, first resolve it to a signed HTTP(S) `OSS_URL` using the bundled `resolve_model_source.py` script and either GPU Lease Manager `/api/transfer/jobs` or a compatible direct `/transfer/jobs` endpoint. Do not SSH to the target host, download archives, merge, quantize, or start services until a resolved `OSS_URL` starts with `http://` or `https://`.

Do not accept Hugging Face repos, local adapter paths, raw `oss://` bucket/key URIs, MinT IDs, or existing model directories as deployment sources in this skill. The final deployment source after resolution is always `OSS_URL`. The OSS link should point to an archive such as `.tar.gz`, `.tgz`, `.tar`, or `.zip` that contains either:

- a PEFT LoRA adapter directory with `adapter_config.json` and `adapter_model.safetensors`, or
- a raw Tinker checkpoint directory that can be converted with `tinker_cookbook.weights.build_lora_adapter()`.

Because signed OSS links contain temporary credentials, do not paste the full query string into long-lived records. Record the OSS key or the URL path without sensitive query parameters when possible.

Standard public serving chain:

```text
public :7777 Caddy -> 127.0.0.1:18080 capture proxy -> 127.0.0.1:7788 vLLM frontend + ATOM out-of-tree plugin backend
```

Target hosts are dedicated model-serving machines. Every deployment must first stop and remove all existing Docker containers on the target host, not only previous `vllm-*` containers. After startup, `docker ps -a` must show exactly two containers for the current model:

```text
${CONTAINER_NAME}
${CONTAINER_NAME}-caddy
```

If any other Docker container exists before deployment, remove it directly with `docker rm -f`. If any other Docker container exists after startup, stop and fix the deployment before handoff. Do not leave multiple model deployments or stale Caddy containers on the same host.

## Ask First

Before deployment, collect the minimum values from the user:

```text
SSH_HOST=<ssh alias or user@host>
SSH_PASSWORD=<password if sshpass is needed, otherwise empty>
REMOTE_ROOT=<durable remote workspace root>
LOCAL_SCRATCH_MOUNT=<local NVMe mount, default /local_nvme>
PUBLIC_BASE_URL=<public /v1 base URL; infer http://<server-address>:7777/v1 when the public server address is known>
OSS_URL=<signed or public HTTP(S) OSS archive URL; when launched from GPU Lease Manager, this must come from /api/transfer/jobs result.oss_url>
TINKER_URL=<optional tinker:// checkpoint URL; use this only when OSS_URL is not already available>
GPU_LEASE_BASE_URL=<optional; required with GPU_LEASE_API_KEY to convert TINKER_URL through GPU Lease Manager>
GPU_LEASE_API_KEY=<optional; API key for GPU Lease Manager conversion>
TRANSFER_JOBS_ENDPOINT=<optional direct transfer service /transfer/jobs endpoint for TINKER_URL conversion>
TRANSFER_POLL_INTERVAL=<default 30 seconds>
TRANSFER_TIMEOUT_SECONDS=<default 7200 seconds>
OSS_SHA256=<optional archive sha256 for download verification>
BASE_REPO=<base Hugging Face repo, normally zai-org/GLM-5.1>
DOCKER_IMAGE=<normally rocm/atom-dev:vllm-latest>
TENSOR_PARALLEL_SIZE=<normally 8 on 8-GPU MI300X>
MAX_MODEL_LEN=<context length, default 65536 / 64k if user does not specify>
MAX_NUM_SEQS=<concurrency budget, default 2 if user does not specify>
MAX_NUM_BATCHED_TOKENS=<batch token budget, default 65536 if user does not specify>
GPU_MEMORY_UTILIZATION=<default 0.60 if user does not specify>
VLLM_EXTRA_ARGS=<default --async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching; change only for recorded diagnostics>
FORCE_TEMPERATURE=<proxy rewrite temperature, default 1; set empty to disable>
DEFAULT_MAX_TOKENS=<optional proxy default max_tokens, often 8192>
NORMALIZE_TOOL_CALL_ARGUMENTS=<proxy compatibility normalization, default 1>
MERGE_DEVICES=<default cuda:0,cuda:1,...,cuda:7 for MI300X; set cpu to disable GPU merge>
MERGE_DEVICE=<legacy single-device fallback if MERGE_DEVICES is unset>
QUANT_DEVICES=<default cuda:0,cuda:1,...,cuda:7 for MI300X; set cpu to disable GPU quant>
MERGE_JOBS=<default 8 for 8-GPU merge>
QUANT_WORKERS=<default 8 for 8-GPU quant>
EXPECTED_GPU_COUNT=<default TENSOR_PARALLEL_SIZE, normally 8 on MI300X>
ROCM_TORCH_VERSION=<default 2.9.1+rocm6.4>
ROCM_TORCH_INDEX_URL=<default https://download.pytorch.org/whl/rocm6.4>
```

Infer these values before asking:

```text
DATA_DISK=<first path component of REMOTE_ROOT, then verify with df -h>
RUN_SLUG=<sanitized OSS URL path basename without archive suffix>
SCRATCH_ROOT=${LOCAL_SCRATCH_MOUNT}/amd_profiling/${RUN_SLUG}
HF_CACHE_DIR=${SCRATCH_ROOT}/hf-cache
DURABLE_MODEL_PATH=${REMOTE_ROOT}/models/${RUN_SLUG}-merged-fp8-block128-official-partial
ATOM_SOURCE_DIR=${REMOTE_ROOT}/atom-fork if that directory exists
```

If an inferred value is missing or suspicious, ask the user. After all explicit and inferred parameters are set, show the full parameter table and ask for confirmation before resolving a `TINKER_URL`, running preflight, copying scripts, downloading the OSS archive, merging, quantizing, starting services, or changing env files. Resolving a `TINKER_URL` may create a long-running transfer job that packs, uploads, and signs a large checkpoint.

Command examples below use `sshpass -e` for password login. If `SSH_PASSWORD` is empty because key auth works, remove `export SSHPASS=...` and replace `sshpass -e ssh/scp` with plain `ssh/scp`.

## Bundled References

Set `SKILL_DIR` to the directory containing this `SKILL.md`. Required files live in `$SKILL_DIR/references/`:

```text
prepare_oss_lora_source.py
resolve_model_source.py
prefetch_glm51_base.py
merge_glm51_lora_sharded.py
validate_and_repair_safetensors_shards.py
quantize_glm51_fp8_block128.py
patch_glm51_fp8_qabf16.py (diagnostic only; not part of the default deploy path)
serve_vllm_glm51.sh
capture_proxy.py
serve_capture_proxy.sh
serve_caddy_proxy.sh
benchmark_vllm_glm51.sh
```

Use these as execution entry points. Keep full vLLM parameters visible in env files and launch records; never leave the generated wrapper as the only source of launch truth.

For MI300X/ROCm PyTorch, use `cuda:N` device strings for GPU compute; `cuda:0` maps to the first visible AMD GPU.

## Known amd3 Production Version

For the validated `amd3` GLM-5.1 corrected official-partial FP8 service, start the ATOM/vLLM backend with:

```text
DOCKER_IMAGE=rocm/atom-dev:vllm-latest
ATOM_SOURCE_DIR=/data/amd_profiling/atom-fork
ATOM GitHub URL=https://github.com/san-tian/ATOM/tree/prod/glm51-qabf16-vllm
ATOM production commit=2088bff453392d701a397d9e5008c9a400fc6eb1
VLLM_EXTRA_ARGS=--async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching
```

On a new host, clone or fetch `https://github.com/san-tian/ATOM.git`, then checkout the fixed production commit:

```bash
git clone https://github.com/san-tian/ATOM.git "$ATOM_SOURCE_DIR"
git -C "$ATOM_SOURCE_DIR" fetch origin prod/glm51-qabf16-vllm
git -C "$ATOM_SOURCE_DIR" checkout 2088bff453392d701a397d9e5008c9a400fc6eb1
```

The local branch name is irrelevant; a detached checkout at the production commit is acceptable and avoids per-machine branch maintenance. Keep `VLLM_SOURCE_DIR=$ATOM_SOURCE_DIR` so the container uses that checkout through `PYTHONPATH` and `ATOM_SOURCE_DIR`. Do not use the official image's bundled source for amd3 normal recovery: on 2026-05-18, official non-eager loaded all shards but did not open `7788`, and official `--enforce-eager` was too slow for production throughput.

The OPE-13 recovered service used prefix caching, async scheduling, and `cudagraph_mode=FULL_AND_PIECEWISE` at 64k context with `max_num_seqs=2` and `max_num_batched_tokens=65536`. Earlier diagnostics on another host had used PIECEWISE as a runtime mitigation, but do not substitute that older runtime path when reproducing the current corrected `official-partial` result unless the user asks for an explicit runtime A/B.

The backend launch script writes `source_git` into `*.server_argv.json`. Treat that field, plus the wrapper script and env file, as the startup source of truth.

## Corrected Official-Partial Quant Contract

The default quantized artifact is the corrected `official-partial` model:

```text
${RUN_SLUG}-merged-fp8-block128-official-partial
```

It follows the official GLM-5.1 FP8 coverage: attention projection linears, MLP linears, and MoE expert linears use FP8 e4m3 block-128 weights with `weight_scale_inv`; embeddings, norms, routers/gates, `lm_head`, and indexer compatibility modules stay unconverted. In particular, `self_attn.q_a_proj.weight` and `self_attn.kv_a_proj_with_mqa.weight` must remain in the FP8 quantization contract. Do not add those projection modules to `modules_to_not_convert`, and do not run the q_a BF16 patch as part of the normal deployment path.

`kv_a_proj_with_mqa` has an output dimension of 576, so the quantizer must support ceil/padded block quantization. The expected representative shapes are:

```text
q_a_proj.weight              FP8 [2048, 6144], scale_inv [16, 48]
kv_a_proj_with_mqa.weight    FP8 [576, 6144],  scale_inv [5, 48]
q_b/kv_b/o_proj and MLP/MoE  FP8 block-128, scale_inv [ceil(out/128), ceil(in/128)]
embed_tokens and lm_head     BF16, no scale_inv
```

The older `*-merged-fp8-block128-qabf16` artifact is superseded and diagnostic only. It leaves q_a, or in older broken flows q_a plus kv_a, outside the official FP8 coverage and must not be used as the default deploy artifact or as evidence about corrected `official-partial` quality.

Validation record from OPE-13:

- Corrected `official-partial` static check: `q_a_proj` and `kv_a_proj_with_mqa` are FP8 and have scale tensors; `scale_inv_count=59158`, `quant_weight_count=59158`.
- 100-prompt harness: 100/100 success, judge accuracy 0.98, empty 0.
- 10,000 long Chinese/repeated-prefix requests with prefix caching and proxy-disabled thinking: HTTP 10000/10000, strict clean 9997/10000, mojibake 0, prefix echo 0, reasoning anomaly 0.
- BF16 merge-only did not serve under the tested vLLM/ATOM TP=8 memory envelope, so do not claim a BF16-vs-FP8 same-prompt regression result from that run.

## Compatibility

This workflow is for GLM-5.1-family PEFT LoRA deployment:

- `BASE_REPO` must expose `model.safetensors.index.json` and sharded safetensors.
- `OSS_URL` must be a downloadable HTTP(S) OSS archive.
- If the archive contains PEFT files, the adapter must expose `adapter_config.json` and `adapter_model.safetensors`.
- If the archive contains raw Tinker weights, the remote merge venv must have `tinker-cookbook` available so the raw checkpoint can be converted to a PEFT adapter.
- The workflow venv must use a ROCm/HIP PyTorch build on MI300X. A CUDA/NVIDIA PyTorch wheel is invalid even though the device strings are `cuda:N`. For normal 8-GPU hosts, repair the venv to `torch==2.9.1+rocm6.4`, verify `torch.version.hip` is set, verify `torch.version.cuda` is empty, and verify `torch.cuda.device_count() >= 8` before merge or quantization.
- `capture_proxy.py` runs with the workflow venv and requires `aiohttp`; install it during venv preparation, and verify the capture proxy process stays alive and answers `127.0.0.1:18080` before starting or declaring Caddy ready.
- The capture proxy must normalize historical OpenAI Chat Completions tool calls for vLLM compatibility: when enabled with `CAPTURE_PROXY_NORMALIZE_TOOL_CALL_ARGUMENTS=1`, it converts non-string `messages[*].tool_calls[*].function.arguments` values into compact JSON strings before forwarding to vLLM. Existing string arguments are preserved. Keep this enabled for production unless a diagnostic explicitly needs to replay the raw invalid client payload.
- Adapter keys must follow PEFT names under `base_model.model.*.lora_A.weight` and matching `lora_B.weight`.
- LoRA scale is read from `adapter_config.json` as `lora_alpha / r`.
- If a local adapter directory includes a malformed or tensor-parallel-sharded `lm_head` LoRA pair plus `mp_rank_*_adapter.pt` files from Tinker, the merge script reconstructs `lm_head.lora_A` and/or `lm_head.lora_B` from `output_layer.adapter.linear_in.weight` and `output_layer.adapter.linear_out.weight` before shape validation. The reconstructed tensors are recorded in `merge_manifest.json` under `lm_head_reconstruction`. If reconstruction is needed but the `mp_rank` files are missing or inconsistent, validation fails instead of silently skipping `lm_head`.
- Tinker sparse routed-expert adapters for GLM-5.1 may export only representative expert ids `0,8,16,...248` for each 256-expert routed MLP tensor. The merge script must expand each representative LoRA delta to every expert in its group of 8 before merging. Fully exported adapters with all `0..255` expert ids are merged one-to-one. Any partial or mixed routed-expert coverage must fail validation.
- For non-GLM-5.1 or structurally different adapters, stop after `--validate-only` failure and inspect or modify the merge mapping.

## Base Model Source

The initial base model does not come from the OSS LoRA archive. `BASE_REPO` defaults to the Hugging Face repo `zai-org/GLM-5.1`. For each deployment, download the base model from Hugging Face into the target server's local NVMe scratch cache, normally `HF_CACHE_DIR=${SCRATCH_ROOT}/hf-cache`. The merge script reads `model.safetensors.index.json` from `BASE_REPO` and downloads the required base-model safetensors shards with `huggingface_hub.hf_hub_download()` into that local NVMe `HF_CACHE_DIR`. Do not route this base-model cache through durable `/data`; on the tested Azure MI300 hosts, direct Hugging Face download to local NVMe was faster than copying cached shards from `/data` to local NVMe. `prepare_oss_lora_source.py` also passes `BASE_REPO` to `tinker_cookbook.weights.build_lora_adapter()` when the OSS archive contains a raw Tinker checkpoint that must be converted to a PEFT adapter.

Do not replace this with an unlisted local base-model copy or another repo unless the user explicitly provides and confirms a different `BASE_REPO`.

## Local NVMe Scratch Policy

Use the durable disk for launch records, logs, service state, Docker image/cache state, and the durable model backup, normally `/data` through `REMOTE_ROOT`. All model-work paths must use local NVMe: OSS archive extraction, PEFT adapter staging, Hugging Face base-model cache, BF16 merge output, corrected official-partial FP8 quant output, optional diagnostic q_a BF16 patch output, the serving-time HF cache, and the primary live serving model. Do not run merge or quantization with any of these paths under `/data`; if `/local_nvme` is unavailable, mount or recreate the local NVMe scratch volume first.

Never download model archives, Hugging Face shards, merge outputs, quantization outputs, Docker layers, containerd layers, pip wheels, or temporary extraction files onto the OS root filesystem. Do not use `/`, `/tmp`, `/var/tmp`, `/var/lib/docker`, `/var/lib/containerd`, `/home/<user>/.cache`, or `/mnt` for model work. Use only:

- `/local_nvme` for ephemeral high-throughput scratch, downloads, extraction, HF cache, merge, corrected official-partial quant, optional diagnostic q_a BF16 patch output, and live serving.
- `/data` for durable logs, scripts, launch records, Docker/containerd state, persistent caches, and final model backups.

At preflight and before every large download, require:

```bash
export TMPDIR="${SCRATCH_ROOT}/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export PIP_CACHE_DIR="${REMOTE_ROOT}/pip-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"
```

Then verify `df -h / "$DATA_DISK" "$LOCAL_SCRATCH_MOUNT"` and stop if `/` has less than 20 GiB free or if any large target path resolves outside `/local_nvme` or `/data`.

The preferred live model path is local NVMe:

```text
MODEL_PATH=$LOCAL_MODEL_PATH
LOCAL_MODEL_PATH=${SCRATCH_ROOT}/serve/${RUN_SLUG}-merged-fp8-block128-official-partial
```

After quantization, start vLLM from `LOCAL_MODEL_PATH` and sync the same model to `DURABLE_MODEL_PATH` under `/data` in the background for persistence. If the host has rebooted or local NVMe was recreated, recreate/mount `/local_nvme` and restore `LOCAL_MODEL_PATH` from `DURABLE_MODEL_PATH` before starting vLLM. Do not serve directly from `/data` in this workflow; that is only an emergency manual recovery path after explicit operator acceptance.

If the target server does not have the local scratch mount, initialize and mount it before downloading the OSS archive or touching model files. Do not continue to download, merge, quantize, run optional diagnostics, or start a normal service until this succeeds. Default to:

```text
LOCAL_SCRATCH_MOUNT=/local_nvme
```

The local NVMe devices are ephemeral. It is acceptable to recreate this scratch mount after a Spot eviction or fresh boot, but never format mounted disks, the OS disk, the durable `/data` disk, or any device whose identity is ambiguous. Only create a new RAID0 scratch array from unmounted NVMe direct disks whose `lsblk` output shows no mountpoint and whose filesystem/probe output does not identify them as `/`, `/data`, `/mnt`, or another persistent filesystem. If an existing md device or filesystem is present, inspect it first and mount it when safe instead of overwriting it. Because this workflow intentionally re-downloads the Hugging Face base model into local NVMe for speed, do not rely on local `HF_CACHE_DIR`, `HF_HOME`, `HF_HUB_CACHE`, or `TRANSFORMERS_CACHE` surviving machine recycle. If device detection is ambiguous, stop and ask an administrator.

## Concurrency Policy

Front-load independent work, but separate always-run deployment downloads from conditional environment repair.

Always start these two per-deployment downloads in the background as soon as local NVMe scratch and scripts are ready:

- OSS archive download to local NVMe.
- Hugging Face base-model prefetch to local NVMe using `prefetch_glm51_base.py`.

Probe environment state before installing, pulling, or cloning anything else:

- Docker: check that Docker exists, `DockerRootDir` is under `DATA_DISK`, and `DOCKER_IMAGE` is already present. Install/start Docker only if missing, migrate data-root before any pull if needed, and pull the image only if absent.
- containerd: check `/var/lib/containerd` is not holding stale image/snapshot content on the OS disk. If containerd is used directly, configure its root under `DATA_DISK` before pulling images. If Docker/containerd have no active containers but unused images/layers are consuming `/`, prune unused images/layers before deployment and record the cleanup log under `REMOTE_ROOT/logs`.
- ATOM: check `ATOM_SOURCE_DIR` is a git checkout at production commit `2088bff453392d701a397d9e5008c9a400fc6eb1`. Clone/fetch/checkout only if missing or at the wrong commit; never overwrite a non-git directory without operator confirmation.
- Python: check the workflow venv and imports first. Install only missing packages; do not run blanket `pip install -U` unless a broken version is proven. The venv must contain `safetensors`, `huggingface_hub`, `accelerate`, `peft`, `aiohttp`, and a ROCm PyTorch wheel. If PyTorch is missing, CUDA-flavored, or reports too few GPUs, replace only PyTorch with `torch==${ROCM_TORCH_VERSION}` from `${ROCM_TORCH_INDEX_URL}` and recheck.
- Tools: check `tar`, `pigz`, `rsync`, `curl`, and `git`; install only missing tools if the host package manager and sudo allow it.

Conditional repair jobs may run in parallel with the two always-run downloads. Then join all started jobs before validation/merge. Use multi-threaded CPU tools for decompression when available, especially `tar --use-compress-program='pigz -dc -p N'` for `.tar.gz`/`.tgz`; fall back to Python extraction only when external tools are unavailable. Keep progress logs for every background job under `REMOTE_ROOT/logs`.

## Derived Paths

After collecting the user inputs, derive:

```bash
: "${BASE_REPO:=zai-org/GLM-5.1}"
: "${OSS_URL:=}"
: "${TINKER_URL:=}"
: "${GPU_LEASE_BASE_URL:=}"
: "${GPU_LEASE_API_KEY:=}"
: "${TRANSFER_JOBS_ENDPOINT:=}"
: "${TRANSFER_POLL_INTERVAL:=30}"
: "${TRANSFER_TIMEOUT_SECONDS:=7200}"
: "${OSS_SHA256:=}"
: "${DOCKER_IMAGE:=rocm/atom-dev:vllm-latest}"
: "${TENSOR_PARALLEL_SIZE:=8}"
: "${MAX_MODEL_LEN:=65536}"
: "${MAX_NUM_SEQS:=2}"
: "${MAX_NUM_BATCHED_TOKENS:=65536}"
: "${GPU_MEMORY_UTILIZATION:=0.60}"
DEFAULT_VLLM_EXTRA_ARGS='--async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching'
: "${VLLM_EXTRA_ARGS:=$DEFAULT_VLLM_EXTRA_ARGS}"
: "${FORCE_TEMPERATURE=1}"
: "${DEFAULT_MAX_TOKENS:=8192}"
: "${NORMALIZE_TOOL_CALL_ARGUMENTS:=1}"
: "${MERGE_DEVICES:=${MERGE_DEVICE:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}}"
: "${QUANT_DEVICES:=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
: "${MERGE_JOBS:=8}"
: "${QUANT_WORKERS:=8}"
: "${LOCAL_SCRATCH_MOUNT:=/local_nvme}"
: "${EXPECTED_GPU_COUNT:=$TENSOR_PARALLEL_SIZE}"
: "${ROCM_TORCH_VERSION:=2.9.1+rocm6.4}"
: "${ROCM_TORCH_INDEX_URL:=https://download.pytorch.org/whl/rocm6.4}"

if [ -z "${DATA_DISK:-}" ] && [ -n "${REMOTE_ROOT:-}" ]; then
  DATA_DISK="/$(printf '%s\n' "$REMOTE_ROOT" | cut -d/ -f2)"
fi

SOURCE_FOR_SLUG="${OSS_URL:-$TINKER_URL}"
if [ -z "$SOURCE_FOR_SLUG" ]; then
  echo "Provide OSS_URL or TINKER_URL before deriving paths" >&2
  exit 2
fi

if [ -z "${RUN_SLUG:-}" ]; then
  RUN_SLUG="$(python3 - "$SOURCE_FOR_SLUG" <<'PY'
from urllib.parse import urlparse, unquote
import os, sys
path = unquote(urlparse(sys.argv[1]).path.rstrip("/"))
name = os.path.basename(path) or "oss-lora-source"
for suffix in (".tar.gz", ".tgz", ".tar", ".zip", ".gz"):
    if name.endswith(suffix):
        name = name[:-len(suffix)]
        break
print(name)
PY
)"
  RUN_SLUG="$(printf '%s\n' "$RUN_SLUG" | tr -cs 'A-Za-z0-9._-' '-' | sed 's/^-//; s/-$//')"
fi

if [ -z "${ATOM_SOURCE_DIR:-}" ]; then
  ATOM_SOURCE_DIR="${REMOTE_ROOT}/atom-fork"
fi
ATOM_REPO_URL="${ATOM_REPO_URL:-https://github.com/san-tian/ATOM.git}"
ATOM_BRANCH="${ATOM_BRANCH:-prod/glm51-qabf16-vllm}"
ATOM_PROD_COMMIT="${ATOM_PROD_COMMIT:-2088bff453392d701a397d9e5008c9a400fc6eb1}"

SCRATCH_ROOT="${LOCAL_SCRATCH_MOUNT}/amd_profiling/${RUN_SLUG}"
HF_CACHE_DIR="${SCRATCH_ROOT}/hf-cache"
SERVE_HF_CACHE_DIR="${HF_CACHE_DIR}"
TMPDIR="${SCRATCH_ROOT}/tmp"
XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
PIP_CACHE_DIR="${REMOTE_ROOT}/pip-cache"
OSS_WORK_DIR="${SCRATCH_ROOT}/downloads/${RUN_SLUG}"
PEFT_ADAPTER="${SCRATCH_ROOT}/adapters/${RUN_SLUG}-peft"
BF16_OUT="${SCRATCH_ROOT}/models/${RUN_SLUG}-merged"
FP8_OUT="${SCRATCH_ROOT}/models/${RUN_SLUG}-merged-fp8-block128-official-partial"
QABF16_DIAGNOSTIC_OUT="${SCRATCH_ROOT}/models/${RUN_SLUG}-merged-fp8-block128-qabf16-diagnostic"
LOCAL_MODEL_PATH="${SCRATCH_ROOT}/serve/${RUN_SLUG}-merged-fp8-block128-official-partial"
DURABLE_MODEL_PATH="${REMOTE_ROOT}/models/${RUN_SLUG}-merged-fp8-block128-official-partial"
MODEL_PATH="$LOCAL_MODEL_PATH"
ENV_FILE="${REMOTE_ROOT}/configs/vllm_${RUN_SLUG}_atom_64k_seq2.env"
CONTAINER_NAME="vllm-${RUN_SLUG}-atom"
SERVED_MODEL_NAME="${RUN_SLUG}-fp8-atom"
VENV_PYTHON="${REMOTE_ROOT}/venv-merge/bin/python"
PREFETCH_WORKERS="${PREFETCH_WORKERS:-16}"
EXTRACT_WORKERS="${EXTRACT_WORKERS:-$(nproc 2>/dev/null || echo 16)}"
```

Before proceeding, verify inferred values and ask the user to confirm:

```text
SSH_HOST
REMOTE_ROOT
DATA_DISK
LOCAL_SCRATCH_MOUNT
SCRATCH_ROOT
HF_CACHE_DIR
SERVE_HF_CACHE_DIR
TMPDIR
PIP_CACHE_DIR
LOCAL_MODEL_PATH
DURABLE_MODEL_PATH
PUBLIC_BASE_URL
OSS_URL host/path, with query string redacted in records
TINKER_URL presence and checkpoint name, if provided
GPU_LEASE_BASE_URL / TRANSFER_JOBS_ENDPOINT conversion source, if TINKER_URL is provided
OSS_SHA256 presence
BASE_REPO
RUN_SLUG
OSS_WORK_DIR
PEFT_ADAPTER
FP8_OUT
QABF16_DIAGNOSTIC_OUT, only if an explicit diagnostic A/B is requested
MODEL_PATH
PREFETCH_WORKERS
EXTRACT_WORKERS
ENV_FILE
ATOM_SOURCE_DIR
ATOM_REPO_URL
ATOM_BRANCH
ATOM_PROD_COMMIT
DOCKER_IMAGE
TENSOR_PARALLEL_SIZE
MAX_MODEL_LEN
MAX_NUM_SEQS
MAX_NUM_BATCHED_TOKENS
GPU_MEMORY_UTILIZATION
VLLM_EXTRA_ARGS
FORCE_TEMPERATURE
DEFAULT_MAX_TOKENS
NORMALIZE_TOOL_CALL_ARGUMENTS
MERGE_DEVICES
QUANT_DEVICES
MERGE_JOBS
QUANT_WORKERS
EXPECTED_GPU_COUNT
ROCM_TORCH_VERSION
ROCM_TORCH_INDEX_URL
```

During preflight, probe `ATOM_SOURCE_DIR` and `VENV_PYTHON`; do not fail merely because ATOM, the workflow venv, Docker, an image, or an optional tool is missing. The preparation phase below must install, clone, pull, or repair only the missing pieces. Stop only for unsafe states such as `ATOM_SOURCE_DIR` existing as a non-git directory or Docker already using an OS-disk data root that cannot be migrated under `DATA_DISK`.

## Resolve Model Source

Resolve the model source locally before any target-host SSH, script sync, download, merge, quantization, or service startup. If `OSS_URL` is already set, this validates and echoes it. If only `TINKER_URL` is set, this creates and polls a transfer job until an HTTP(S) OSS archive URL is available. Use one of:

- `GPU_LEASE_BASE_URL` plus `GPU_LEASE_API_KEY`, which calls `$GPU_LEASE_BASE_URL/api/transfer/jobs`.
- `TRANSFER_JOBS_ENDPOINT`, a direct compatible `/transfer/jobs` endpoint such as `http://123.57.26.97:8001/transfer/jobs`.

Do not continue if the resolver returns an empty value, an `oss://` URI, or any non-HTTP URL. The resolver only needs local Python 3 standard-library modules:

```bash
LOCAL_SOURCE_RESOLUTION_JSON="/tmp/source_resolution_${RUN_SLUG}.json"
if [ -n "${OSS_URL:-}" ]; then
  RESOLVED_OSS_URL="$(python3 "$SKILL_DIR/references/resolve_model_source.py" \
    --oss-url "$OSS_URL" \
    --output-json "$LOCAL_SOURCE_RESOLUTION_JSON")"
else
  RESOLVED_OSS_URL="$(python3 "$SKILL_DIR/references/resolve_model_source.py" \
    --tinker-url "$TINKER_URL" \
    --gpu-lease-base-url "$GPU_LEASE_BASE_URL" \
    --gpu-lease-api-key "$GPU_LEASE_API_KEY" \
    --transfer-jobs-endpoint "$TRANSFER_JOBS_ENDPOINT" \
    --poll-interval "$TRANSFER_POLL_INTERVAL" \
    --timeout "$TRANSFER_TIMEOUT_SECONDS" \
    --output-json "$LOCAL_SOURCE_RESOLUTION_JSON")"
fi
case "$RESOLVED_OSS_URL" in
  http://*|https://*) OSS_URL="$RESOLVED_OSS_URL"; export OSS_URL ;;
  *) echo "failed to resolve HTTP(S) OSS_URL: $RESOLVED_OSS_URL" >&2; exit 3 ;;
esac
```

Record only non-sensitive source metadata in long-lived notes. If `LOCAL_SOURCE_RESOLUTION_JSON` contains a signed URL, keep it as a short-lived run artifact and do not paste the full query string into docs or chat.

## Preflight And Script Sync

Check host, disk, processes, Docker, and GPUs:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  df -h '$DATA_DISK'
  df -h / '$DATA_DISK' '$LOCAL_SCRATCH_MOUNT' 2>/dev/null || true
  root_avail_kb=\$(df -Pk / | awk 'NR==2 {print \$4}')
  if [ -n \"\$root_avail_kb\" ] && [ \"\$root_avail_kb\" -lt 20971520 ]; then
    echo \"OS root filesystem has less than 20 GiB free; clean root-disk caches before model work\" >&2
    exit 3
  fi
  mkdir -p '$REMOTE_ROOT'/{configs,logs,results,scripts,models,hf-cache,request_captures}
  if [ -d '$ATOM_SOURCE_DIR/.git' ]; then
    git -C '$ATOM_SOURCE_DIR' rev-parse --short HEAD || true
  elif [ -e '$ATOM_SOURCE_DIR' ]; then
    echo 'ATOM_SOURCE_DIR exists but is not a git checkout: $ATOM_SOURCE_DIR' >&2
  else
    echo 'ATOM_SOURCE_DIR missing; conditional prep can clone it'
  fi
  if [ -x '$VENV_PYTHON' ]; then
    '$VENV_PYTHON' --version
  else
    echo 'VENV_PYTHON missing; conditional prep can create it'
  fi
  who -b || true
  ss -ltnp 2>/dev/null | grep -E ':(7788|18080|7777)' || true
  if command -v docker >/dev/null 2>&1; then
    docker info --format 'DockerRootDir={{.DockerRootDir}}' || true
    docker ps -a --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' || true
  fi
  sudo -n du -shx /var/lib/containerd /var/lib/docker /home/\$(id -un)/.cache 2>/dev/null || true
  if command -v containerd >/dev/null 2>&1; then
    containerd config dump 2>/dev/null | grep -E 'root =|state =' | head -5 || true
  fi
  if command -v rocm-smi >/dev/null 2>&1; then rocm-smi --showuse --showmemuse | tail -80 || true; fi
"
```

Ensure local NVMe scratch is mounted before model work. If it is already mounted, reuse it. If `/dev/md0` or another local NVMe filesystem already exists but is not mounted, inspect and mount it when it is clearly the intended scratch volume. If no scratch filesystem exists and the host exposes unused NVMe direct disks, create a RAID0 scratch volume and mount it at `LOCAL_SCRATCH_MOUNT`. This may require passwordless sudo on the target host:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  if findmnt -rn '$LOCAL_SCRATCH_MOUNT' >/dev/null 2>&1; then
    echo 'local scratch already mounted'
  elif [ -b /dev/md0 ] && sudo blkid /dev/md0 >/dev/null 2>&1; then
    sudo mkdir -p '$LOCAL_SCRATCH_MOUNT'
    sudo mount /dev/md0 '$LOCAL_SCRATCH_MOUNT'
    sudo chown "\$(id -u):\$(id -g)" '$LOCAL_SCRATCH_MOUNT'
  else
    mapfile -t candidates < <(python3 - <<'PY'
import shlex
import subprocess

out = subprocess.check_output(
    ["lsblk", "-P", "-dn", "-o", "NAME,TYPE,FSTYPE,MOUNTPOINT,MODEL"],
    text=True,
)
for line in out.splitlines():
    fields = dict(item.split("=", 1) for item in shlex.split(line))
    if (
        fields.get("TYPE") == "disk"
        and fields.get("NAME", "").startswith("nvme")
        and not fields.get("FSTYPE")
        and not fields.get("MOUNTPOINT")
        and "NVMe Direct Disk" in fields.get("MODEL", "")
    ):
        print("/dev/" + fields["NAME"])
PY
    )
    if [ \"\${#candidates[@]}\" -lt 2 ]; then
      echo 'No mounted local scratch and not enough unused NVMe disks to create RAID0 safely' >&2
      exit 3
    fi
    if sudo blkid \"\${candidates[@]}\" 2>/dev/null | grep -q .; then
      echo 'One or more candidate NVMe disks already has a filesystem signature; inspect manually before formatting.' >&2
      sudo blkid \"\${candidates[@]}\" 2>/dev/null || true
      exit 3
    fi
    sudo mdadm --create /dev/md0 --level=0 --raid-devices=\"\${#candidates[@]}\" --chunk=1024K \"\${candidates[@]}\"
    sudo mkfs.ext4 -F -L LOCAL_NVME /dev/md0
    sudo mkdir -p '$LOCAL_SCRATCH_MOUNT'
    sudo mount /dev/md0 '$LOCAL_SCRATCH_MOUNT'
    sudo chown \"\$(id -u):\$(id -g)\" '$LOCAL_SCRATCH_MOUNT'
  fi
  mkdir -p '$SCRATCH_ROOT'/{downloads,adapters,models,hf-cache}
  df -h '$LOCAL_SCRATCH_MOUNT' '$DATA_DISK'
"
```

After this step, fail the deployment unless `findmnt -rn "$LOCAL_SCRATCH_MOUNT"` succeeds and the filesystem has enough space for BF16, corrected FP8, optional diagnostics, and HF cache intermediates. Do not overwrite an existing md device or filesystem unless it is clearly the intended ephemeral local NVMe scratch array.

Docker must not store images/layers on the OS disk. During preflight, probe Docker and the target image only; do not pull images yet. If Docker is missing, let the conditional preparation phase install/start it when possible. If Docker exists but `DockerRootDir` is outside `DATA_DISK`, record that it must be migrated before any image pull or service start:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  docker_root=\$({ docker info --format '{{.DockerRootDir}}' 2>/dev/null || sudo docker info --format '{{.DockerRootDir}}' 2>/dev/null || true; } | awk 'NF { print; exit }')
  if [ -z \"\$docker_root\" ]; then
    echo 'Docker is not available; conditional prep can install/start it if sudo/package manager allow'
  fi
  if [ -n \"\$docker_root\" ]; then
    case \"\$docker_root\" in
      '$DATA_DISK'/*) echo \"DockerRootDir ok: \$docker_root\" ;;
      *)
        echo \"DockerRootDir must be moved under $DATA_DISK before pulling images; current: \$docker_root\" >&2
        echo 'Conditional prep will try to set /etc/docker/daemon.json {\"data-root\":\"$DATA_DISK/docker\"} and restart Docker.' >&2
        ;;
    esac
    docker image inspect '$DOCKER_IMAGE' >/dev/null 2>&1 || sudo docker image inspect '$DOCKER_IMAGE' >/dev/null 2>&1 || echo 'Docker image missing; conditional prep can pull it'
  fi
"
```

If containerd is used directly on the host, keep its `root` under `DATA_DISK` as well, normally `/data/containerd`. Do not put Docker/containerd state under `/`, `/var/lib/docker`, `/var/lib/containerd`, `/mnt`, or local NVMe. On 2026-05-19, `mi300-04` had 78 GiB of stale containerd content under `/var/lib/containerd`; pruning unused Docker/containerd objects reduced `/` from 94% used to 27% used. Treat `/var/lib/containerd` growth as a deployment blocker until it is moved or cleaned.

If migration is needed and the host uses ordinary Docker, perform the migration explicitly before any image pull:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  sudo mkdir -p '$DATA_DISK/docker'
  tmp=\$(mktemp)
  if [ -f /etc/docker/daemon.json ]; then
    if command -v jq >/dev/null 2>&1; then
      jq '. + {\"data-root\":\"$DATA_DISK/docker\"}' /etc/docker/daemon.json > \"\$tmp\"
    else
      python3 - <<'PY' > \"\$tmp\"
import json
from pathlib import Path
p = Path('/etc/docker/daemon.json')
data = json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}
data['data-root'] = '$DATA_DISK/docker'
print(json.dumps(data, indent=2))
PY
    fi
  else
    printf '{\n  \"data-root\": \"%s/docker\"\n}\n' '$DATA_DISK' > \"\$tmp\"
  fi
  sudo install -m 0644 \"\$tmp\" /etc/docker/daemon.json
  rm -f \"\$tmp\"
  sudo systemctl restart docker
  docker_root=\$({ docker info --format '{{.DockerRootDir}}' 2>/dev/null || sudo docker info --format '{{.DockerRootDir}}'; } | awk 'NF { print; exit }')
  test \"\$docker_root\" = '$DATA_DISK/docker'
"
```

If `/var/lib/docker` already contains a large image cache and the host has enough `/data` space, an administrator may copy it to `/data/docker` before restart. Do not delete `/var/lib/docker` during this workflow.

Sync bundled scripts:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e scp "$SKILL_DIR"/references/{resolve_model_source.py,prepare_oss_lora_source.py,merge_glm51_lora_sharded.py,validate_and_repair_safetensors_shards.py,quantize_glm51_fp8_block128.py,patch_glm51_fp8_qabf16.py,serve_vllm_glm51.sh,capture_proxy.py,serve_capture_proxy.sh,serve_caddy_proxy.sh,benchmark_vllm_glm51.sh} \
  "$SSH_HOST:$REMOTE_ROOT/scripts/"
sshpass -e scp "$SKILL_DIR"/references/prefetch_glm51_base.py "$SSH_HOST:$REMOTE_ROOT/scripts/"
sshpass -e ssh "$SSH_HOST" "chmod +x '$REMOTE_ROOT'/scripts/{resolve_model_source.py,prepare_oss_lora_source.py,prefetch_glm51_base.py,serve_vllm_glm51.sh,serve_capture_proxy.sh,serve_caddy_proxy.sh,benchmark_vllm_glm51.sh} 2>/dev/null || true"
```

After `$REMOTE_ROOT/logs` exists, copy the local source-resolution record into the remote run logs if it was created:

```bash
export SSHPASS="$SSH_PASSWORD"
if [ -f "${LOCAL_SOURCE_RESOLUTION_JSON:-}" ]; then
  sshpass -e scp "$LOCAL_SOURCE_RESOLUTION_JSON" "$SSH_HOST:$REMOTE_ROOT/logs/source_resolution_${RUN_SLUG}.json"
fi
```

If a host rebooted, missing containers/listeners usually means runtime state was lost. Do not assume model files, env files, or dependencies disappeared until the durable disk is checked.

## Prepare OSS Source, Merge, And Quantize

Start per-deployment downloads and conditional environment preparation as early as possible. The OSS archive download and Hugging Face base-model prefetch are always started. Docker, ATOM, Python deps, and host tools are first probed, then only missing or incorrect pieces are repaired in background jobs:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" \
  "REMOTE_ROOT='$REMOTE_ROOT' SCRATCH_ROOT='$SCRATCH_ROOT' HF_CACHE_DIR='$HF_CACHE_DIR' OSS_WORK_DIR='$OSS_WORK_DIR' PEFT_ADAPTER='$PEFT_ADAPTER' OSS_URL='$OSS_URL' OSS_SHA256='$OSS_SHA256' BASE_REPO='$BASE_REPO' VENV_PYTHON='$VENV_PYTHON' DATA_DISK='$DATA_DISK' DOCKER_IMAGE='$DOCKER_IMAGE' ATOM_SOURCE_DIR='$ATOM_SOURCE_DIR' ATOM_REPO_URL='$ATOM_REPO_URL' ATOM_BRANCH='$ATOM_BRANCH' ATOM_PROD_COMMIT='$ATOM_PROD_COMMIT' PREFETCH_WORKERS='$PREFETCH_WORKERS' RUN_SLUG='$RUN_SLUG' EXPECTED_GPU_COUNT='$EXPECTED_GPU_COUNT' ROCM_TORCH_VERSION='$ROCM_TORCH_VERSION' ROCM_TORCH_INDEX_URL='$ROCM_TORCH_INDEX_URL' bash -s" <<'REMOTE_PREP'
set -euo pipefail
mkdir -p "$REMOTE_ROOT/logs" "$SCRATCH_ROOT"/{downloads,adapters,models,hf-cache,serve}
export TMPDIR="${SCRATCH_ROOT}/tmp"
export TEMP="$TMPDIR"
export TMP="$TMPDIR"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
export HF_HOME="$HF_CACHE_DIR"
export HF_HUB_CACHE="$HF_CACHE_DIR/hub"
export TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers"
export PIP_CACHE_DIR="${REMOTE_ROOT}/pip-cache"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"
case "$SCRATCH_ROOT" in /local_nvme/*) ;; *) echo "SCRATCH_ROOT must be under /local_nvme: $SCRATCH_ROOT" >&2; exit 3 ;; esac
case "$HF_CACHE_DIR" in /local_nvme/*) ;; *) echo "HF_CACHE_DIR must be under /local_nvme: $HF_CACHE_DIR" >&2; exit 3 ;; esac
case "$REMOTE_ROOT" in /data/*|/data) ;; *) echo "REMOTE_ROOT must be under /data: $REMOTE_ROOT" >&2; exit 3 ;; esac
if ! findmnt -rn /local_nvme >/dev/null 2>&1; then
  echo "/local_nvme is not mounted; mount local NVMe scratch before model work" >&2
  exit 3
fi
python3 - <<'PY'
import os
import sys
from pathlib import Path

required_local = [
    "SCRATCH_ROOT",
    "HF_CACHE_DIR",
    "OSS_WORK_DIR",
    "PEFT_ADAPTER",
]
for name in required_local:
    value = os.environ.get(name)
    if not value:
        print(f"{name} is empty", file=sys.stderr)
        sys.exit(3)
    try:
        resolved = Path(value).resolve()
    except FileNotFoundError:
        resolved = Path(value).absolute()
    if not str(resolved).startswith("/local_nvme/"):
        print(f"{name} must resolve under /local_nvme, got {resolved}", file=sys.stderr)
        sys.exit(3)
PY
root_avail_kb="$(df -Pk / | awk 'NR==2 {print $4}')"
if [ -n "$root_avail_kb" ] && [ "$root_avail_kb" -lt 20971520 ]; then
  echo "OS root filesystem has less than 20 GiB free; clean /var/lib/containerd, /var/lib/docker, /tmp, /var/tmp, or user caches before model work" >&2
  exit 3
fi

jobs_file="$REMOTE_ROOT/logs/prep_jobs_${RUN_SLUG}.list"
probe_file="$REMOTE_ROOT/logs/env_probe_${RUN_SLUG}.env"
: > "$jobs_file"
: > "$probe_file"

status_path() { printf '%s/logs/%s_%s.status' "$REMOTE_ROOT" "$1" "$RUN_SLUG"; }
log_path() { printf '%s/logs/%s_%s.log' "$REMOTE_ROOT" "$1" "$RUN_SLUG"; }

start_job() {
  local name="$1"
  local fn="$2"
  echo "$name" >> "$jobs_file"
  rm -f "$(status_path "$name")"
  (
    set +e
    "$fn"
    code=$?
    echo "$code" > "$(status_path "$name")"
  ) > "$(log_path "$name")" 2>&1 &
  echo $! > "$REMOTE_ROOT/logs/${name}_${RUN_SLUG}.pid"
}

wait_status() {
  local name="$1"
  local status_file
  status_file="$(status_path "$name")"
  while [ ! -f "$status_file" ]; do
    sleep 10
  done
  test "$(cat "$status_file")" = 0
}

docker_cmd() {
  docker "$@" 2>/dev/null || sudo -n docker "$@"
}

need_pip_deps=0
need_torch_repair=0
if [ ! -x "$VENV_PYTHON" ]; then
  need_pip_deps=1
  need_torch_repair=1
else
  "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1 || need_pip_deps=1
import importlib.util
for module in ("safetensors", "huggingface_hub", "accelerate", "peft", "aiohttp"):
    assert importlib.util.find_spec(module), module
PY
  "$VENV_PYTHON" - "$EXPECTED_GPU_COUNT" <<'PY' >/dev/null 2>&1 || need_torch_repair=1
import sys
import torch

expected = int(sys.argv[1] or "0")
version = str(getattr(torch, "__version__", ""))
hip = getattr(torch.version, "hip", None)
cuda_version = getattr(torch.version, "cuda", None)
if not hip or cuda_version:
    raise SystemExit(f"invalid torch build for ROCm: version={version} hip={hip} cuda={cuda_version}")
count = torch.cuda.device_count()
if expected and count < expected:
    raise SystemExit(f"torch sees {count} GPUs, expected at least {expected}")
PY
fi

missing_tools=()
for tool in tar pigz rsync curl git; do
  command -v "$tool" >/dev/null 2>&1 || missing_tools+=("$tool")
done

need_atom=0
atom_blocked=0
atom_current=""
if [ -d "$ATOM_SOURCE_DIR/.git" ] && command -v git >/dev/null 2>&1; then
  atom_current="$(git -C "$ATOM_SOURCE_DIR" rev-parse HEAD 2>/dev/null || true)"
  [ "$atom_current" = "$ATOM_PROD_COMMIT" ] || need_atom=1
elif [ -e "$ATOM_SOURCE_DIR" ] && [ ! -d "$ATOM_SOURCE_DIR/.git" ]; then
  atom_blocked=1
else
  need_atom=1
fi

need_docker_prep=0
docker_root=""
docker_image_present=0
if ! command -v docker >/dev/null 2>&1; then
  need_docker_prep=1
else
  docker_root="$(docker_cmd info --format '{{.DockerRootDir}}' 2>/dev/null | awk 'NF { print; exit }' || true)"
  case "$docker_root" in
    "$DATA_DISK"/*) ;;
    *) need_docker_prep=1 ;;
  esac
  if docker_cmd image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    docker_image_present=1
  else
    need_docker_prep=1
  fi
fi

{
  printf 'VENV_PYTHON=%s\n' "$VENV_PYTHON"
  printf 'NEED_PIP_DEPS=%s\n' "$need_pip_deps"
  printf 'NEED_TORCH_REPAIR=%s\n' "$need_torch_repair"
  printf 'EXPECTED_GPU_COUNT=%s\n' "$EXPECTED_GPU_COUNT"
  printf 'ROCM_TORCH_VERSION=%s\n' "$ROCM_TORCH_VERSION"
  printf 'ROCM_TORCH_INDEX_URL=%s\n' "$ROCM_TORCH_INDEX_URL"
  printf 'MISSING_TOOLS=%s\n' "${missing_tools[*]:-}"
  printf 'ATOM_SOURCE_DIR=%s\n' "$ATOM_SOURCE_DIR"
  printf 'ATOM_CURRENT=%s\n' "$atom_current"
  printf 'ATOM_PROD_COMMIT=%s\n' "$ATOM_PROD_COMMIT"
  printf 'NEED_ATOM=%s\n' "$need_atom"
  printf 'ATOM_BLOCKED=%s\n' "$atom_blocked"
  printf 'DOCKER_ROOT=%s\n' "$docker_root"
  printf 'DOCKER_IMAGE_PRESENT=%s\n' "$docker_image_present"
  printf 'NEED_DOCKER_PREP=%s\n' "$need_docker_prep"
} > "$probe_file"

if [ "$atom_blocked" = 1 ]; then
  echo "ATOM_SOURCE_DIR exists but is not a git checkout: $ATOM_SOURCE_DIR" >&2
  exit 3
fi

download_oss() {
  local py="$VENV_PYTHON"
  if [ ! -x "$py" ]; then
    py="$(command -v python3)"
  fi
  args=(
    --url "$OSS_URL"
    --work-dir "$OSS_WORK_DIR"
    --base-repo "$BASE_REPO"
    --out "$PEFT_ADAPTER"
    --download-only
  )
  if [ -n "$OSS_SHA256" ]; then
    args+=(--sha256 "$OSS_SHA256")
  fi
  "$py" "$REMOTE_ROOT/scripts/prepare_oss_lora_source.py" "${args[@]}"
}

python_deps() {
  local venv_dir
  venv_dir="$(dirname "$(dirname "$VENV_PYTHON")")"
  if [ ! -x "$VENV_PYTHON" ]; then
    python3 -m venv "$venv_dir"
  fi
  "$VENV_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || true
  if [ "$need_torch_repair" = 1 ]; then
    "$VENV_PYTHON" -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    "$VENV_PYTHON" -m pip install --cache-dir "$PIP_CACHE_DIR" \
      --index-url "$ROCM_TORCH_INDEX_URL" \
      "torch==$ROCM_TORCH_VERSION"
  fi
  missing="$("$VENV_PYTHON" - <<'PY'
import importlib.util
packages = {
    "safetensors": "safetensors",
    "huggingface_hub": "huggingface_hub",
    "accelerate": "accelerate",
    "peft": "peft",
    "aiohttp": "aiohttp",
}
print(" ".join(pkg for module, pkg in packages.items() if importlib.util.find_spec(module) is None))
PY
)"
  if [ -n "$missing" ]; then
    "$VENV_PYTHON" -m pip install --cache-dir "$PIP_CACHE_DIR" $missing
  fi
  "$VENV_PYTHON" - "$EXPECTED_GPU_COUNT" <<'PY'
import sys
import torch

expected = int(sys.argv[1] or "0")
version = str(torch.__version__)
hip = getattr(torch.version, "hip", None)
cuda_version = getattr(torch.version, "cuda", None)
count = torch.cuda.device_count()
print(f"torch={version} hip={hip} cuda={cuda_version} device_count={count}", flush=True)
if not hip or cuda_version:
    raise SystemExit("workflow venv still does not have a ROCm PyTorch build")
if expected and count < expected:
    raise SystemExit(f"workflow venv sees {count} GPUs, expected at least {expected}")
PY
}

tool_install() {
  if [ "${#missing_tools[@]}" -eq 0 ]; then
    return 0
  fi
  if command -v apt-get >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "${missing_tools[@]}" || true
  fi
  if [ "$need_atom" = 1 ]; then
    command -v git >/dev/null 2>&1
  fi
}

atom_checkout() {
  if printf '%s\n' "${missing_tools[@]}" | grep -qx git; then
    wait_status tool_install || return 3
  fi
  if [ -e "$ATOM_SOURCE_DIR" ] && [ ! -d "$ATOM_SOURCE_DIR/.git" ]; then
    echo "Refusing to overwrite non-git ATOM_SOURCE_DIR: $ATOM_SOURCE_DIR" >&2
    return 3
  fi
  mkdir -p "$(dirname "$ATOM_SOURCE_DIR")"
  if [ ! -d "$ATOM_SOURCE_DIR/.git" ]; then
    git clone "$ATOM_REPO_URL" "$ATOM_SOURCE_DIR"
  fi
  git -C "$ATOM_SOURCE_DIR" fetch origin "$ATOM_BRANCH"
  git -C "$ATOM_SOURCE_DIR" checkout "$ATOM_PROD_COMMIT"
  test "$(git -C "$ATOM_SOURCE_DIR" rev-parse HEAD)" = "$ATOM_PROD_COMMIT"
}

docker_prepare() {
  if ! command -v docker >/dev/null 2>&1; then
    if command -v apt-get >/dev/null 2>&1 && sudo -n true >/dev/null 2>&1; then
      sudo apt-get update || return 3
      sudo apt-get install -y docker.io || return 3
      sudo systemctl enable --now docker || sudo service docker start || return 3
    else
      echo "Docker is missing and cannot be installed non-interactively" >&2
      return 3
    fi
  fi

  docker_root="$(docker_cmd info --format '{{.DockerRootDir}}' 2>/dev/null | awk 'NF { print; exit }' || true)"
  case "$docker_root" in
    "$DATA_DISK"/*) ;;
    *)
      sudo -n mkdir -p "$DATA_DISK/docker" || return 3
      tmp="$(mktemp)"
      python3 - "$DATA_DISK/docker" > "$tmp" <<'PY'
import json
import sys
from pathlib import Path
target = sys.argv[1]
p = Path("/etc/docker/daemon.json")
data = json.loads(p.read_text()) if p.exists() and p.read_text().strip() else {}
data["data-root"] = target
print(json.dumps(data, indent=2))
PY
      sudo -n install -m 0644 "$tmp" /etc/docker/daemon.json || return 3
      rm -f "$tmp"
      sudo -n systemctl restart docker || sudo -n service docker restart || return 3
      docker_root="$(docker_cmd info --format '{{.DockerRootDir}}' | awk 'NF { print; exit }')"
      case "$docker_root" in "$DATA_DISK"/*) ;; *) echo "DockerRootDir still outside $DATA_DISK: $docker_root" >&2; return 3 ;; esac
      ;;
  esac

  if ! docker_cmd image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
    docker_cmd pull "$DOCKER_IMAGE" || return 3
  fi
}

prefetch_base() {
  if [ "$need_pip_deps" = 1 ] || [ "$need_torch_repair" = 1 ]; then
    wait_status pip_deps || return 3
  fi
  env HF_HOME="$HF_CACHE_DIR" "$VENV_PYTHON" "$REMOTE_ROOT/scripts/prefetch_glm51_base.py" \
    --base-repo "$BASE_REPO" \
    --cache-dir "$HF_CACHE_DIR" \
    --workers "$PREFETCH_WORKERS"
}

start_job download download_oss
[ "$need_pip_deps" = 1 ] && start_job pip_deps python_deps
[ "$need_torch_repair" = 1 ] && [ "$need_pip_deps" = 0 ] && start_job pip_deps python_deps
[ "${#missing_tools[@]}" -gt 0 ] && start_job tool_install tool_install
[ "$need_atom" = 1 ] && start_job atom_checkout atom_checkout
[ "$need_docker_prep" = 1 ] && start_job docker_prepare docker_prepare
start_job prefetch_base prefetch_base

cat "$probe_file"
printf 'started_jobs=%s\n' "$(tr '\n' ' ' < "$jobs_file")"
REMOTE_PREP
```

Join the background jobs before adapter validation:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" \
  "REMOTE_ROOT='$REMOTE_ROOT' RUN_SLUG='$RUN_SLUG' bash -s" <<'REMOTE_JOIN'
set -euo pipefail
jobs_file="$REMOTE_ROOT/logs/prep_jobs_${RUN_SLUG}.list"
test -f "$jobs_file"
while read -r name; do
  [ -n "$name" ] || continue
  log_file="$REMOTE_ROOT/logs/${name}_${RUN_SLUG}.log"
  status_file="$REMOTE_ROOT/logs/${name}_${RUN_SLUG}.status"
  echo "waiting for $name"
  while [ ! -f "$status_file" ]; do
    tail -5 "$log_file" 2>/dev/null || true
    sleep 30
  done
  status="$(cat "$status_file")"
  tail -20 "$log_file" || true
  if [ "$status" != 0 ]; then
    echo "$name failed with exit $status" >&2
    exit 3
  fi
done < "$jobs_file"
REMOTE_JOIN
```

Do not run unconditional `docker pull`, `pip install -U`, or ATOM clone/fetch on every deployment. The only unconditional jobs in this phase are the two model-source downloads. The environment probe and job list in `env_probe_${RUN_SLUG}.env` and `prep_jobs_${RUN_SLUG}.list` are part of the deployment record.

After the archive download is complete, extract it with multi-threaded CPU decompression when possible and resolve it to a local PEFT adapter directory:

```bash
export SSHPASS="$SSH_PASSWORD"
RESOLVED_ADAPTER="$(sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  args=(
    --url '$OSS_URL'
    --work-dir '$OSS_WORK_DIR'
    --base-repo '$BASE_REPO'
    --out '$PEFT_ADAPTER'
    --extract-workers '$EXTRACT_WORKERS'
  )
  if [ -n '$OSS_SHA256' ]; then
    args+=(--sha256 '$OSS_SHA256')
  fi
  '$VENV_PYTHON' '$REMOTE_ROOT/scripts/prepare_oss_lora_source.py' \"\${args[@]}\"
" | tail -1)"
export RESOLVED_ADAPTER
printf 'RESOLVED_ADAPTER=%s\n' "$RESOLVED_ADAPTER"
```

If this exits because `tinker-cookbook` is missing and the archive is a raw Tinker checkpoint, install that package into the workflow venv and rerun the same command:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "'$VENV_PYTHON' -m pip install -U tinker-cookbook"
```

Validate adapter mapping:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  HF_HOME='$HF_CACHE_DIR' '$VENV_PYTHON' '$REMOTE_ROOT/scripts/merge_glm51_lora_sharded.py' \
    --base-repo '$BASE_REPO' \
    --adapter-repo '$RESOLVED_ADAPTER' \
    --cache-dir '$HF_CACHE_DIR' \
    --out '$BF16_OUT' \
    --devices '$MERGE_DEVICES' \
    --validate-only
"
```

Before merge, quantization, optional diagnostics, and service env creation, assert that every heavy model path resolves under local NVMe:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  findmnt -rn '$LOCAL_SCRATCH_MOUNT' >/dev/null
  for p in '$HF_CACHE_DIR' '$OSS_WORK_DIR' '$PEFT_ADAPTER' '$BF16_OUT' '$FP8_OUT' '$QABF16_DIAGNOSTIC_OUT' '$LOCAL_MODEL_PATH'; do
    resolved=\$(python3 -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve(strict=False))' \"\$p\")
    case \"\$resolved\" in '$LOCAL_SCRATCH_MOUNT'/*) ;;
      *) echo \"heavy model path is not under $LOCAL_SCRATCH_MOUNT: \$p -> \$resolved\" >&2; exit 3 ;;
    esac
  done
"
```

Remove stale intermediate outputs, then run a fresh merge. Do not use resume for merge or quantization; these stages are fast enough and stale shards can preserve old merge bugs when `RUN_SLUG` is reused.

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  rm -rf '$BF16_OUT' '$FP8_OUT' '$QABF16_DIAGNOSTIC_OUT' '$LOCAL_MODEL_PATH'
  HF_HOME='$HF_CACHE_DIR' '$VENV_PYTHON' '$REMOTE_ROOT/scripts/merge_glm51_lora_sharded.py' \
    --base-repo '$BASE_REPO' \
    --adapter-repo '$RESOLVED_ADAPTER' \
    --cache-dir '$HF_CACHE_DIR' \
    --out '$BF16_OUT' \
    --jobs '$MERGE_JOBS' \
    --devices '$MERGE_DEVICES' \
    --copy-untouched hardlink
"
```

Validate BF16 shards before quantizing:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  test -f '$BF16_OUT/model.safetensors.index.json'
  test -f '$BF16_OUT/merge_manifest.json'
  '$VENV_PYTHON' '$REMOTE_ROOT/scripts/validate_and_repair_safetensors_shards.py' \
    --model '$BF16_OUT' \
    --cache-dir '$HF_CACHE_DIR' \
    --expected-shards 282 \
    --remove-temp-artifacts \
    --repair-dangling-links
"
```

Expected BF16 output: exactly the formal sharded safetensors referenced by `model.safetensors.index.json`, `merge_manifest.json`, tokenizer/config side files, and `merged_lora_info/`. Hidden temporary artifacts such as `.model-*.safetensors.tmp` are not official model shards and must be removed before quantization; `validate_and_repair_safetensors_shards.py --remove-temp-artifacts` and `quantize_glm51_fp8_block128.py` both enforce this cleanup.

Quantize to FP8 block-128:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  HF_HOME='$HF_CACHE_DIR' '$VENV_PYTHON' '$REMOTE_ROOT/scripts/quantize_glm51_fp8_block128.py' \
    --src '$BF16_OUT' \
    --out '$FP8_OUT' \
    --workers '$QUANT_WORKERS' \
    --cache-dir '$HF_CACHE_DIR' \
    --devices '$QUANT_DEVICES'
"
```

This is the default serving artifact. It must include FP8 `q_a_proj` and `kv_a_proj_with_mqa` tensors and their `weight_scale_inv` entries.

Optional diagnostic only: restore selected tensors such as `q_a_proj.weight` back to BF16 for a controlled A/B run. Do not run this in the normal deployment path, and do not serve the diagnostic output unless the user explicitly asks for this experiment:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  HF_HOME='$HF_CACHE_DIR' '$VENV_PYTHON' '$REMOTE_ROOT/scripts/patch_glm51_fp8_qabf16.py' \
    --fp8-src '$FP8_OUT' \
    --bf16-src '$BF16_OUT' \
    --out '$QABF16_DIAGNOSTIC_OUT' \
    --copy-mode hardlink
"
```

For the known GLM-5.1 shape, the diagnostic patch manifest should report `q_a_proj_restored=79` and `affected_shards=78` when patching only q_a. Treat different counts as a signal to inspect the model structure. This diagnostic artifact is superseded for production and must not replace `FP8_OUT`.

Prepare the local live model path before creating the serving env file. Prefer hardlinks from `FP8_OUT` when both paths are on local NVMe:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete --link-dest='$FP8_OUT' '$FP8_OUT/' '$LOCAL_MODEL_PATH/'
  else
    rm -rf '$LOCAL_MODEL_PATH'
    cp -al '$FP8_OUT' '$LOCAL_MODEL_PATH'
  fi
  test -f '$LOCAL_MODEL_PATH/model.safetensors.index.json'
"
```

Start the durable `/data` sync in the background. Do not wait for it before starting vLLM unless the user specifically asks for durable copy completion before serving. Serving reads `LOCAL_MODEL_PATH`, so durable sync is a persistence backup and does not affect the public API once vLLM is ready:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  mkdir -p '$REMOTE_ROOT/models' '$REMOTE_ROOT/logs'
  (
    set +e
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete --info=progress2 '$LOCAL_MODEL_PATH/' '$DURABLE_MODEL_PATH/'
    else
      rm -rf '$DURABLE_MODEL_PATH'
      cp -a '$LOCAL_MODEL_PATH' '$DURABLE_MODEL_PATH'
    fi
    test -f '$DURABLE_MODEL_PATH/model.safetensors.index.json'
    echo \$? > '$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.status'
  ) > '$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.log' 2>&1 &
  echo \$! > '$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.pid'
"
```

If the host has rebooted and `LOCAL_MODEL_PATH` is missing but `DURABLE_MODEL_PATH` exists, restore local NVMe before starting vLLM:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  if [ ! -f '$LOCAL_MODEL_PATH/model.safetensors.index.json' ] && [ -f '$DURABLE_MODEL_PATH/model.safetensors.index.json' ]; then
    mkdir -p \$(dirname '$LOCAL_MODEL_PATH')
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete --info=progress2 '$DURABLE_MODEL_PATH/' '$LOCAL_MODEL_PATH/'
    else
      rm -rf '$LOCAL_MODEL_PATH'
      cp -a '$DURABLE_MODEL_PATH' '$LOCAL_MODEL_PATH'
    fi
  fi
  test -f '$LOCAL_MODEL_PATH/model.safetensors.index.json'
"
```

Serving env files and vLLM launch records must refer to `MODEL_PATH=$LOCAL_MODEL_PATH`. Handoff records must also include `DURABLE_MODEL_PATH` and the durable sync status/log.

## Tensor Checksum Comparison

Use this for transfer validation or artifact comparison:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  MODEL='$MODEL_PATH' '$VENV_PYTHON' - <<'PY'
import hashlib, json, os, torch
from pathlib import Path
from safetensors import safe_open
model = Path(os.environ['MODEL'])
index = json.load(open(model / 'model.safetensors.index.json'))
keys = [
  'model.layers.0.self_attn.q_a_proj.weight',
  'model.layers.0.self_attn.q_b_proj.weight',
  'model.layers.39.self_attn.q_a_proj.weight',
  'model.layers.39.self_attn.o_proj.weight',
  'model.layers.78.mlp.experts.gate_up_proj.weight',
]
for k in keys:
    shard = index['weight_map'].get(k)
    if not shard:
        print(k, 'MISSING')
        continue
    with safe_open(str(model / shard), framework='pt', device='cpu') as f:
        t = f.get_tensor(k).detach().cpu().contiguous()
    raw = t.view(torch.uint8).numpy().tobytes()
    print(json.dumps({'key': k, 'shard': shard, 'dtype': str(t.dtype), 'shape': list(t.shape), 'sha256': hashlib.sha256(raw).hexdigest()}, ensure_ascii=False))
PY
"
```

Run the same key list against both paths; compare `dtype`, `shape`, and `sha256`.

## Deployment Env File

Create the env file from collected values:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "cat > '$ENV_FILE' <<EOF
AMD_PROFILING_ROOT=$REMOTE_ROOT
VLLM_IMAGE=$DOCKER_IMAGE
VLLM_CONTAINER_NAME=$CONTAINER_NAME
VLLM_MODEL=$MODEL_PATH
VLLM_SERVED_MODEL_NAME=$SERVED_MODEL_NAME
VLLM_PORT=7788
VLLM_HOST=0.0.0.0
VLLM_TP=$TENSOR_PARALLEL_SIZE
VLLM_DTYPE=bfloat16
VLLM_KV_CACHE_DTYPE=fp8
VLLM_SOURCE_DIR=$ATOM_SOURCE_DIR
VLLM_HOST_DATA_ROOT=$DATA_DISK
VLLM_EXTRA_MOUNTS=$LOCAL_SCRATCH_MOUNT
REQUIRE_DOCKER_DATA_ROOT_PREFIX=$DATA_DISK
VLLM_MAX_MODEL_LEN=$MAX_MODEL_LEN
VLLM_MAX_NUM_SEQS=$MAX_NUM_SEQS
VLLM_MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
VLLM_GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
VLLM_ENABLE_AUTO_TOOL_CHOICE=1
VLLM_TOOL_CALL_PARSER=glm47
VLLM_REASONING_PARSER=glm45
VLLM_CHAT_TEMPLATE_CONTENT_FORMAT=string
VLLM_EXTRA_ARGS='$VLLM_EXTRA_ARGS'
HF_HOME=$SERVE_HF_CACHE_DIR
HF_HUB_CACHE=$SERVE_HF_CACHE_DIR/hub
TRANSFORMERS_CACHE=$SERVE_HF_CACHE_DIR/transformers
CAPTURE_PROXY_HOST=127.0.0.1
CAPTURE_PROXY_PORT=18080
CAPTURE_PROXY_UPSTREAM=http://127.0.0.1:7788
CAPTURE_PROXY_DIR=$REMOTE_ROOT/request_captures
CAPTURE_PROXY_FORCE_TEMPERATURE=$FORCE_TEMPERATURE
CAPTURE_PROXY_DEFAULT_MAX_TOKENS=$DEFAULT_MAX_TOKENS
CAPTURE_PROXY_MASK_REPLACEMENT_CHAR=1
CAPTURE_PROXY_NORMALIZE_TOOL_CALL_ARGUMENTS=$NORMALIZE_TOOL_CALL_ARGUMENTS
CADDY_LISTEN=:7777
CADDY_BIND=0.0.0.0
CADDY_UPSTREAM=127.0.0.1:18080
CADDY_CONTAINER_NAME=${CONTAINER_NAME}-caddy
EOF"
```

Do not replace `VLLM_EXTRA_ARGS=--async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching` with `--enforce-eager` for normal throughput unless a specific diagnostic requires it and the tradeoff is recorded. If diagnostic vLLM args are needed, append them carefully and preserve which runtime path is being tested.

Effective vLLM command:

```text
vllm serve <LOCAL_MODEL_PATH>
  --host 0.0.0.0
  --port 7788
  --served-model-name <SERVED_MODEL_NAME>
  --tensor-parallel-size <TENSOR_PARALLEL_SIZE>
  --max-model-len <MAX_MODEL_LEN>
  --max-num-seqs <MAX_NUM_SEQS>
  --max-num-batched-tokens <MAX_NUM_BATCHED_TOKENS>
  --gpu-memory-utilization <GPU_MEMORY_UTILIZATION>
  --dtype bfloat16
  --kv-cache-dtype fp8
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
  --chat-template-content-format string
  --trust-remote-code
  --async-scheduling
  --enable-prefix-caching
  --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"}
```

## Start Backend, Proxy, And Caddy

Before starting the backend, clear all existing Docker containers. This is a hard requirement for every deployment on these dedicated model hosts; do not keep old stopped containers, old Caddy containers, or old model containers:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  mapfile -t containers < <({ docker ps -a --format '{{.ID}}' 2>/dev/null || sudo docker ps -a --format '{{.ID}}' 2>/dev/null || true; } | awk 'NF')
  if [ \"\${#containers[@]}\" -gt 0 ]; then
    { docker rm -f \"\${containers[@]}\" 2>/dev/null || sudo docker rm -f \"\${containers[@]}\"; }
  fi
  remaining=\$({ docker ps -a --format '{{.Names}}' 2>/dev/null || sudo docker ps -a --format '{{.Names}}' 2>/dev/null || true; } | awk 'NF')
  if [ -n \"\$remaining\" ]; then
    echo 'Docker containers remain after cleanup:' >&2
    printf '%s\n' \"\$remaining\" >&2
    exit 3
  fi
"
```

Start backend:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  test -f '$LOCAL_MODEL_PATH/model.safetensors.index.json'
  docker_root=\$({ docker info --format '{{.DockerRootDir}}' 2>/dev/null || sudo docker info --format '{{.DockerRootDir}}' 2>/dev/null || true; } | awk 'NF { print; exit }')
  case \"\$docker_root\" in '$DATA_DISK'/*) ;; *) echo \"DockerRootDir is not under $DATA_DISK: \$docker_root\" >&2; exit 3 ;; esac
  AMD_PROFILING_ROOT='$REMOTE_ROOT' VLLM_ENV_FILE='$ENV_FILE' \
    bash '$REMOTE_ROOT/scripts/serve_vllm_glm51.sh'
"
```

Poll readiness before starting ingress:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  LOG=\$(ls -t '$REMOTE_ROOT'/logs/vllm_glm51_*.log | head -1)
  for i in \$(seq 1 100); do
    code=\$(curl -sS --connect-timeout 2 --max-time 5 -o /tmp/vllm_models_check.json -w '%{http_code}' http://127.0.0.1:7788/v1/models 2>/dev/null || true)
    marker=\$(tr '\r' '\n' < \"\$LOG\" | grep -E 'Loading safetensors shards|GPU KV cache size|Maximum concurrency|Graph capturing finished|Application startup complete|ERROR|Traceback|Exception|OOM|Killed' | tail -1 || true)
    printf '%s code=%s marker=%s\n' \"\$(date -Is)\" \"\$code\" \"\$marker\"
    if [ \"\$code\" = '200' ]; then cat /tmp/vllm_models_check.json; exit 0; fi
    if printf '%s' \"\$marker\" | grep -qE 'ERROR|Traceback|Exception|OOM|Killed'; then exit 2; fi
    sleep 30
  done
  exit 1
"
```

Start capture proxy and Caddy:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  AMD_PROFILING_ROOT='$REMOTE_ROOT' ATOM_ENV_FILE='$ENV_FILE' bash '$REMOTE_ROOT/scripts/serve_capture_proxy.sh'
  for i in \$(seq 1 20); do
    code=\$(curl -sS --connect-timeout 2 --max-time 5 -o /tmp/capture_proxy_models_check.json -w '%{http_code}' http://127.0.0.1:18080/v1/models 2>/dev/null || true)
    if [ \"\$code\" = '200' ]; then
      cat /tmp/capture_proxy_models_check.json
      break
    fi
    if [ \"\$i\" = 20 ]; then
      echo 'capture proxy did not become ready on 127.0.0.1:18080' >&2
      tail -80 \"\$(ls -t '$REMOTE_ROOT'/logs/capture_proxy_*.log | head -1)\" >&2 || true
      exit 3
    fi
    sleep 3
  done
  AMD_PROFILING_ROOT='$REMOTE_ROOT' ATOM_ENV_FILE='$ENV_FILE' bash '$REMOTE_ROOT/scripts/serve_caddy_proxy.sh'
  mapfile -t containers < <({ docker ps -a --format '{{.Names}}' 2>/dev/null || sudo docker ps -a --format '{{.Names}}' 2>/dev/null || true; } | awk 'NF' | sort)
  if [ \"\${#containers[@]}\" -ne 2 ]; then
    printf 'Expected exactly two Docker containers after startup, found %s:\n' \"\${#containers[@]}\" >&2
    printf '  %s\n' \"\${containers[@]}\" >&2
    exit 3
  fi
  printf '%s\n' \"\${containers[@]}\" | grep -Fx '$CONTAINER_NAME' >/dev/null
  printf '%s\n' \"\${containers[@]}\" | grep -Fx '${CONTAINER_NAME}-caddy' >/dev/null
"
```

Expected listeners: `0.0.0.0:7788`, `127.0.0.1:18080`, and `:7777` with Caddy `bind 0.0.0.0`. Verify the generated Caddyfile includes `bind 0.0.0.0`.

## Service Ready Announcement

As soon as vLLM, capture proxy, and Caddy pass `/v1/models` checks, print the callable endpoint information in the operator console and send the same concise message to the user. Do this immediately; do not wait for durable `/data` sync. Explicitly say that the durable sync is running in the background and does not affect serving because vLLM is reading from local NVMe.

Use this template after substituting actual values:

```text
Service is ready.
Public OpenAI base URL: <PUBLIC_BASE_URL>
Model name: <SERVED_MODEL_NAME>
Internal checks: vLLM http://127.0.0.1:7788/v1, capture proxy http://127.0.0.1:18080/v1
Durable sync: background copy from <LOCAL_MODEL_PATH> to <DURABLE_MODEL_PATH>; serving is not blocked. I will check sync progress every 5 minutes.
```

Also print ready-to-use curls:

```bash
printf '%s\n' "Public OpenAI base URL: $PUBLIC_BASE_URL"
printf '%s\n' "Model name: $SERVED_MODEL_NAME"
cat <<EOF
curl -sS "$PUBLIC_BASE_URL/models"

curl -sS "$PUBLIC_BASE_URL/chat/completions" \\
  -H 'Content-Type: application/json' \\
  -d '{"model":"$SERVED_MODEL_NAME","messages":[{"role":"user","content":"2+3等于几？只回答数字。"}],"temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
EOF
```

If `PUBLIC_BASE_URL` was not explicitly provided, infer it from the public server address and port `7777` when the user supplied a server IP or DNS name:

```text
PUBLIC_BASE_URL=http://<server-address>:7777/v1
```

If no public address is known, use `http://127.0.0.1:7777/v1` only for remote-local checks and state that the external caller URL is still unknown.

## Validation

Run all layers on the remote host:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  for url in http://127.0.0.1:7788 http://127.0.0.1:18080 http://127.0.0.1:7777; do
    echo \"== \$url /health ==\"
    curl -sS --connect-timeout 3 --max-time 10 -w '\nHTTP_STATUS=%{http_code}\n' \"\$url/health\"
    echo \"== \$url /v1/models ==\"
    curl -sS --connect-timeout 3 --max-time 10 -w '\nHTTP_STATUS=%{http_code}\n' \"\$url/v1/models\"
    echo \"== \$url completion ==\"
    curl -sS --connect-timeout 3 --max-time 60 -w '\nHTTP_STATUS=%{http_code}\n' \"\$url/v1/completions\" \
      -H 'Content-Type: application/json' \
      -d '{\"model\":\"$SERVED_MODEL_NAME\",\"prompt\":\"2+3=\",\"max_tokens\":8,\"temperature\":0}'
  done
  ss -ltnp 2>/dev/null | grep -E '0.0.0.0:7788|:7777|127.0.0.1:18080'
  mapfile -t containers < <({ docker ps -a --format '{{.Names}}' 2>/dev/null || sudo docker ps -a --format '{{.Names}}' 2>/dev/null || true; } | awk 'NF' | sort)
  if [ \"\${#containers[@]}\" -ne 2 ]; then
    printf 'Expected exactly two Docker containers after startup, found %s:\n' \"\${#containers[@]}\" >&2
    printf '  %s\n' \"\${containers[@]}\" >&2
    exit 3
  fi
  printf '%s\n' \"\${containers[@]}\" | grep -Fx '$CONTAINER_NAME' >/dev/null
  printf '%s\n' \"\${containers[@]}\" | grep -Fx '${CONTAINER_NAME}-caddy' >/dev/null
  grep -q 'bind 0.0.0.0' '$REMOTE_ROOT/configs/Caddyfile.capture-proxy'
  grep -q '^VLLM_EXTRA_ARGS=.*--enable-prefix-caching' '$ENV_FILE'
  grep -q '^VLLM_EXTRA_ARGS=.*--async-scheduling' '$ENV_FILE'
  grep -q '^VLLM_EXTRA_ARGS=.*cudagraph_mode.*FULL_AND_PIECEWISE' '$ENV_FILE'
  LOG=\$(ls -t '$REMOTE_ROOT'/logs/vllm_glm51_*.log | head -1)
  grep -E 'enable_prefix_caching=True|async_scheduling=True|Asynchronous scheduling is enabled|FULL_AND_PIECEWISE|cudagraph_mode.*FULL_AND_PIECEWISE' \"\$LOG\" | tail -10 || true
  test -f '$LOCAL_MODEL_PATH/model.safetensors.index.json'
  if [ -f '$DURABLE_MODEL_PATH/model.safetensors.index.json' ]; then
    echo 'durable model copy exists'
  else
    echo 'durable model sync still running or failed; inspect sync_durable log/status' >&2
  fi
"
```

Start a background durable-sync progress monitor after service readiness. It must check every 5 minutes, append to a log, and exit when the sync status file reports success or failure:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  set -euo pipefail
  monitor_log='$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.monitor.log'
  monitor_pid='$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.monitor.pid'
  (
    while true; do
      ts=\$(date -Is)
      status='running'
      if [ -f '$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.status' ]; then
        status=\$(cat '$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.status')
      fi
      src_size=\$(du -sh '$LOCAL_MODEL_PATH' 2>/dev/null | awk '{print \$1}' || true)
      dst_size=\$(du -sh '$DURABLE_MODEL_PATH' 2>/dev/null | awk '{print \$1}' || true)
      last_progress=\$(tail -n 5 '$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.log' 2>/dev/null | tr '\n' ' ' | cut -c1-500 || true)
      printf '%s status=%s src=%s dst=%s last=\"%s\"\n' \"\$ts\" \"\$status\" \"\$src_size\" \"\$dst_size\" \"\$last_progress\"
      case \"\$status\" in
        0) echo \"\$ts durable sync complete\"; exit 0 ;;
        running) sleep 300 ;;
        *) echo \"\$ts durable sync failed with status=\$status\"; exit \"\$status\" ;;
      esac
    done
  ) >> \"\$monitor_log\" 2>&1 &
  echo \$! > \"\$monitor_pid\"
  echo \"durable sync monitor started: pid=\$(cat \"\$monitor_pid\") log=\$monitor_log\"
"
```

When the user asks for progress, report the latest monitor line and make clear that public serving remains available while this background copy continues.

If `PUBLIC_BASE_URL` is provided, run caller-side public check:

```bash
curl -sS --connect-timeout 5 --max-time 15 -w '\nHTTP_STATUS=%{http_code}\n' "$PUBLIC_BASE_URL/models"
curl -sS --connect-timeout 5 --max-time 15 -w '\nHTTP_STATUS=%{http_code}\n' "${PUBLIC_BASE_URL%/v1}/health"
```

Public curl failure with remote-local success is an ingress/cloud-network problem, not a model failure, but the deployment handoff is incomplete until this is stated clearly. If the user expects public access, do not say deployment is complete until `PUBLIC_BASE_URL` succeeds or the exact cloud firewall/NSG blocker is identified.

Proxy policy check:

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  ps -eo pid,args | grep '[c]apture_proxy.py' || true
  grep -E 'CAPTURE_PROXY_(FORCE_TEMPERATURE|DEFAULT_MAX_TOKENS|MASK_REPLACEMENT_CHAR|NORMALIZE_TOOL_CALL_ARGUMENTS)' '$ENV_FILE' || true
"
```

Explicit caller `max_tokens` must not be overwritten by a default-max-tokens policy.
Keep `CAPTURE_PROXY_MASK_REPLACEMENT_CHAR=1` unless a diagnostic explicitly needs raw `U+FFFD` output. This removes literal replacement characters from forwarded chat history and from responses returned through the capture proxy.
Keep `CAPTURE_PROXY_NORMALIZE_TOOL_CALL_ARGUMENTS=1` unless replaying raw invalid client payloads for diagnostics. vLLM expects historical `tool_calls[*].function.arguments` to be JSON strings, so object or array values must be stringified before forwarding.

## Benchmark

Run only after smoke tests pass. Use input lengths such as 10k/20k and concurrency up to `32`; do not benchmark a request whose input length exceeds live `MAX_MODEL_LEN`.

```bash
export SSHPASS="$SSH_PASSWORD"
sshpass -e ssh "$SSH_HOST" "
  AMD_PROFILING_ROOT='$REMOTE_ROOT' VLLM_ENV_FILE='$ENV_FILE' \
    bash '$REMOTE_ROOT/scripts/benchmark_vllm_glm51.sh'
"
```

Required benchmark artifacts: `deployment.env`, `deployment.wrapper.sh`, `deployment.server_argv.json`, `benchmark_context.json`, `summary.json`, and `summary.md`.

## Live Logs

After deployment, report these commands to the user:

```bash
ssh "$SSH_HOST" "tail -f \"\$(ls -t '$REMOTE_ROOT'/logs/vllm_glm51_*.log | head -1)\""
ssh "$SSH_HOST" "tail -f \"\$(ls -t '$REMOTE_ROOT'/logs/capture_proxy_*.log | head -1)\""
ssh "$SSH_HOST" "docker logs -f '${CONTAINER_NAME}-caddy'"
ssh "$SSH_HOST" "tail -f '$REMOTE_ROOT/logs/sync_durable_${RUN_SLUG}.monitor.log'"
```

## Records To Preserve

Every OSS source preparation, merge, quantization, deployment, recovery, or benchmark must leave a human-readable record. If no project ledger exists, create `Records/STARTUPS.md` and `Records/WORK_RECORDS.md`.

Each startup entry must include: host, Beijing-time heading if the user prefers Chinese records, OSS URL host/path with sensitive query parameters redacted, env file, wrapper script, server argv JSON, backend log, capture proxy log, live local model path, durable model path, durable sync log/status, ATOM source checkout path, ATOM git remote/branch/upstream/commit/dirty state from `source_git`, Docker image, DockerRootDir, container name, full vLLM serve parameters, capture proxy policy, Caddy `bind 0.0.0.0` state, ingress topology, local and public smoke results, benchmark result directory if run, and reboot/cold-cache notes if relevant.

## Final Handoff

Deployment is callable once vLLM, capture proxy, and Caddy pass checks on `:7777`. Do not delay the first user-facing handoff for durable `/data` sync if the service is already reading from `LOCAL_MODEL_PATH`; tell the user that sync is a background persistence copy and that progress is checked every 5 minutes.

When deployment is complete enough to call, include ready-to-run caller curl examples in the final response. Use `PUBLIC_BASE_URL` when provided or inferred; otherwise use the reachable Caddy base URL and append `/v1` as needed. Include at least:

```bash
curl -sS "$PUBLIC_BASE_URL/models"

curl -sS "$PUBLIC_BASE_URL/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$SERVED_MODEL_NAME"'","prompt":"2+3=","max_tokens":32,"temperature":0}'

curl -sS "$PUBLIC_BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"'"$SERVED_MODEL_NAME"'","messages":[{"role":"user","content":"2+3等于几？"}],"max_tokens":64,"temperature":0,"chat_template_kwargs":{"enable_thinking":false}}'
```

If the capture proxy forces temperature, fills missing `max_tokens`, masks replacement characters, or normalizes historical tool-call arguments, explicitly mention that caller-visible behavior in the handoff.

## Failure Rules

- Existing shard count is not enough; validate safetensors headers.
- `/local_nvme` is a hard prerequisite for this workflow. If it is missing, mount or recreate the local NVMe scratch volume before downloads, merge, quantization, optional diagnostics, or normal serving. Do not move these stages to `/data` for convenience; `/data` is only for logs, scripts, Docker/containerd state, and durable model backup.
- `/v1/models` is not enough; run semantic completion smoke through the full chain.
- Loader-clean block-128 FP8 may still fail semantically; smoke-test language output through the full chain. Use q_a BF16 patching only as an explicitly requested diagnostic A/B artifact, not as the default path.
- Keep `--async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching` in `VLLM_EXTRA_ARGS` for the corrected OPE-13 ROCm/ATOM FP8 GLM-5.1 launch. Only change this for an explicit runtime A/B test, and record whether the test changes async scheduling, prefix caching, CUDAGraph mode, or all three.
- Do not use `--enforce-eager` for normal throughput unless a specific diagnostic requires it and the tradeoff is recorded.
- `MAX_NUM_BATCHED_TOKENS` below live input length can fail or queue poorly for long-input short-output workloads.
- Host reboot clears runtime containers/processes and possibly image cache; durable model/env files should remain under `REMOTE_ROOT` on `DATA_DISK`.
- If host reboot or Spot recycle clears local NVMe, recreate/mount local scratch, rsync `DURABLE_MODEL_PATH` back to `LOCAL_MODEL_PATH`, and start vLLM from local NVMe. Do not start from `/data` in this workflow; that is an emergency manual recovery path after explicit operator acceptance.
- Docker image/layer state must live under `DATA_DISK`, not the OS disk. If `DockerRootDir` is outside `DATA_DISK`, migrate Docker before `docker pull` or service start.
- Caddy must expose the final public endpoint on `:7777` with `bind 0.0.0.0`; local-only success is not enough when public access is requested.
- Background durable sync can continue after vLLM starts, but the final handoff must state whether `sync_durable_${RUN_SLUG}.status` is complete or still running.
- Once public `:7777` passes `/v1/models`, immediately print and send the public base URL plus served model name. Do not make users wait for durable sync; keep the 5-minute background sync monitor running and report progress separately.
- Signed OSS URLs expire. If download fails with 403 or signature errors, ask the user for a fresh OSS link.
- If a reference script starts but exits on a missing Python dependency, install that dependency into the workflow venv and rerun the same script.
