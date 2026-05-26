#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-}"
SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat >&2 <<'EOF'
Usage: scripts/run_stage.sh <stage>

Stages:
  derive resolve-source sync-scripts preflight prepare-env fetch-source
  prefetch-base merge validate-bf16 quantize stage-model write-serve-env
  serve-backend serve-proxy serve-observability serve-caddy smoke benchmark deploy-all

Required env for remote stages:
  SSH_HOST, REMOTE_ROOT, and either OSS_URL or TINKER_URL.
Optional:
  SSH_PASSWORD for sshpass, LOCAL_SCRATCH_MOUNT, PUBLIC_BASE_URL,
  BASE_REPO, DOCKER_IMAGE, ATOM_SOURCE_DIR, ATOM_PROD_COMMIT.
EOF
}

if [[ -z "${STAGE}" || "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
  usage
  exit 2
fi

require() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "Missing required environment variable: ${name}" >&2
      exit 2
    fi
  done
}

shell_quote() {
  printf "%q" "$1"
}

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

ssh_exec() {
  require SSH_HOST
  ssh_base ssh -o StrictHostKeyChecking=accept-new "${SSH_HOST}" "$@"
}

scp_to_remote() {
  require SSH_HOST
  ssh_base scp -o StrictHostKeyChecking=accept-new "$@"
}

derive() {
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
  : "${VLLM_ENABLE_MTP:=1}"
  if [[ -z "${VLLM_SPECULATIVE_CONFIG+x}" ]]; then
    VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
  fi
  local default_vllm_extra_args='--async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching'
  if [[ -z "${VLLM_EXTRA_ARGS+x}" ]]; then
    VLLM_EXTRA_ARGS="${default_vllm_extra_args}"
    if [[ "${VLLM_ENABLE_MTP}" != "0" ]]; then
      VLLM_EXTRA_ARGS+=" --speculative-config=${VLLM_SPECULATIVE_CONFIG}"
    fi
  fi
  : "${FORCE_TEMPERATURE:=1}"
  : "${DEFAULT_MAX_TOKENS:=8192}"
  : "${NORMALIZE_TOOL_CALL_ARGUMENTS:=1}"
  : "${DISABLE_THINKING:=1}"
  : "${MERGE_DEVICES:=${MERGE_DEVICE:-cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}}"
  : "${QUANT_DEVICES:=cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7}"
  : "${MERGE_JOBS:=8}"
  : "${QUANT_WORKERS:=8}"
  : "${LOCAL_SCRATCH_MOUNT:=/local_nvme}"
  : "${EXPECTED_GPU_COUNT:=${TENSOR_PARALLEL_SIZE}}"
  : "${ROCM_TORCH_VERSION:=2.9.1+rocm6.4}"
  : "${ROCM_TORCH_INDEX_URL:=https://download.pytorch.org/whl/rocm6.4}"
  : "${ATOM_REPO_URL:=https://github.com/san-tian/ATOM.git}"
  : "${ATOM_BRANCH:=prod/glm51-qabf16-vllm}"
  : "${ATOM_PROD_COMMIT:=2088bff453392d701a397d9e5008c9a400fc6eb1}"
  : "${PREFETCH_WORKERS:=16}"
  : "${EXTRACT_WORKERS:=$(nproc 2>/dev/null || echo 16)}"
  : "${OBSERVABILITY_ENABLED:=1}"
  : "${PROMETHEUS_IMAGE:=prom/prometheus:v2.55.1}"
  : "${GRAFANA_IMAGE:=grafana/grafana:11.3.1}"
  : "${CADDY_IMAGE:=caddy:2.8.4-alpine}"
  : "${PUBLIC_ROOT_URL:=}"
  : "${SUDO_PASSWORD:=}"

  require REMOTE_ROOT
  if [[ -z "${OSS_URL}" && -z "${TINKER_URL}" ]]; then
    echo "Set OSS_URL or TINKER_URL" >&2
    exit 2
  fi

  if [[ -z "${DATA_DISK:-}" ]]; then
    DATA_DISK="/$(printf "%s\n" "${REMOTE_ROOT}" | cut -d/ -f2)"
  fi

  local source_for_slug="${OSS_URL:-${TINKER_URL}}"
  if [[ -z "${RUN_SLUG:-}" ]]; then
    RUN_SLUG="$(
      python3 - "${source_for_slug}" <<'PY'
from urllib.parse import urlparse, unquote
import os
import sys

path = unquote(urlparse(sys.argv[1]).path.rstrip("/"))
name = os.path.basename(path) or "oss-lora-source"
for suffix in (".tar.gz", ".tgz", ".tar", ".zip", ".gz"):
    if name.endswith(suffix):
        name = name[: -len(suffix)]
        break
print(name)
PY
    )"
    RUN_SLUG="$(printf "%s\n" "${RUN_SLUG}" | tr -cs "A-Za-z0-9._-" "-" | sed "s/^-//; s/-$//")"
  fi

  : "${ATOM_SOURCE_DIR:=${REMOTE_ROOT}/atom-fork}"
  SCRATCH_ROOT="${LOCAL_SCRATCH_MOUNT}/amd_profiling/${RUN_SLUG}"
  HF_CACHE_DIR="${SCRATCH_ROOT}/hf-cache"
  SERVE_HF_CACHE_DIR="${HF_CACHE_DIR}"
  TMPDIR="${SCRATCH_ROOT}/tmp"
  XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg-cache"
  PIP_CACHE_DIR="${REMOTE_ROOT}/pip-cache"
  OSS_WORK_DIR="${SCRATCH_ROOT}/downloads/${RUN_SLUG}"
  PEFT_ADAPTER="${SCRATCH_ROOT}/adapters/${RUN_SLUG}-peft"
  BF16_OUT="${SCRATCH_ROOT}/models/${RUN_SLUG}-merged"
  FP8_OUT="${SCRATCH_ROOT}/models/${RUN_SLUG}-merged-fp8-finegrained-block128"
  LOCAL_MODEL_PATH="${SCRATCH_ROOT}/serve/${RUN_SLUG}-merged-fp8-finegrained-block128"
  DURABLE_MODEL_PATH="${REMOTE_ROOT}/models/${RUN_SLUG}-merged-fp8-finegrained-block128"
  MODEL_PATH="${LOCAL_MODEL_PATH}"
  ENV_FILE="${REMOTE_ROOT}/configs/vllm_${RUN_SLUG}_atom_64k_seq2.env"
  CONTAINER_NAME="vllm-${RUN_SLUG}-atom"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${RUN_SLUG}-fp8-atom}"
  VENV_PYTHON="${VENV_PYTHON:-${REMOTE_ROOT}/venv-merge/bin/python}"
  PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-}"
  if [[ -z "${PUBLIC_ROOT_URL}" && -n "${PUBLIC_BASE_URL}" ]]; then
    PUBLIC_ROOT_URL="${PUBLIC_BASE_URL%/v1}"
  fi
  if [[ -z "${PUBLIC_ROOT_URL}" ]]; then
    PUBLIC_ROOT_URL="http://127.0.0.1:7777"
  fi

  export BASE_REPO OSS_URL TINKER_URL GPU_LEASE_BASE_URL GPU_LEASE_API_KEY
  export TRANSFER_JOBS_ENDPOINT TRANSFER_POLL_INTERVAL TRANSFER_TIMEOUT_SECONDS OSS_SHA256
  export DOCKER_IMAGE TENSOR_PARALLEL_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS
  export GPU_MEMORY_UTILIZATION VLLM_ENABLE_MTP VLLM_SPECULATIVE_CONFIG VLLM_EXTRA_ARGS
  export FORCE_TEMPERATURE DEFAULT_MAX_TOKENS
  export NORMALIZE_TOOL_CALL_ARGUMENTS DISABLE_THINKING MERGE_DEVICES QUANT_DEVICES MERGE_JOBS
  export QUANT_WORKERS
  export LOCAL_SCRATCH_MOUNT EXPECTED_GPU_COUNT ROCM_TORCH_VERSION ROCM_TORCH_INDEX_URL
  export DATA_DISK RUN_SLUG ATOM_SOURCE_DIR ATOM_REPO_URL ATOM_BRANCH ATOM_PROD_COMMIT
  export SCRATCH_ROOT HF_CACHE_DIR SERVE_HF_CACHE_DIR TMPDIR XDG_CACHE_HOME PIP_CACHE_DIR
  export OSS_WORK_DIR PEFT_ADAPTER BF16_OUT FP8_OUT LOCAL_MODEL_PATH DURABLE_MODEL_PATH
  export MODEL_PATH ENV_FILE CONTAINER_NAME SERVED_MODEL_NAME VENV_PYTHON PREFETCH_WORKERS EXTRACT_WORKERS
  export PUBLIC_BASE_URL OBSERVABILITY_ENABLED PROMETHEUS_IMAGE GRAFANA_IMAGE CADDY_IMAGE PUBLIC_ROOT_URL
}

print_derived() {
  derive
  for name in SSH_HOST REMOTE_ROOT DATA_DISK LOCAL_SCRATCH_MOUNT RUN_SLUG SCRATCH_ROOT HF_CACHE_DIR TMPDIR PIP_CACHE_DIR OSS_WORK_DIR PEFT_ADAPTER BF16_OUT FP8_OUT LOCAL_MODEL_PATH DURABLE_MODEL_PATH MODEL_PATH ENV_FILE ATOM_SOURCE_DIR DOCKER_IMAGE TENSOR_PARALLEL_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION VLLM_ENABLE_MTP VLLM_SPECULATIVE_CONFIG VLLM_EXTRA_ARGS FORCE_TEMPERATURE DEFAULT_MAX_TOKENS NORMALIZE_TOOL_CALL_ARGUMENTS DISABLE_THINKING MERGE_DEVICES QUANT_DEVICES MERGE_JOBS QUANT_WORKERS EXPECTED_GPU_COUNT PUBLIC_BASE_URL PUBLIC_ROOT_URL OBSERVABILITY_ENABLED PROMETHEUS_IMAGE GRAFANA_IMAGE CADDY_IMAGE; do
    printf "%s=%s\n" "${name}" "${!name:-}"
  done
}

remote_exports() {
  derive
  local name
  for name in REMOTE_ROOT DATA_DISK LOCAL_SCRATCH_MOUNT RUN_SLUG SCRATCH_ROOT HF_CACHE_DIR SERVE_HF_CACHE_DIR TMPDIR XDG_CACHE_HOME PIP_CACHE_DIR OSS_URL OSS_SHA256 BASE_REPO OSS_WORK_DIR PEFT_ADAPTER BF16_OUT FP8_OUT LOCAL_MODEL_PATH DURABLE_MODEL_PATH MODEL_PATH ENV_FILE CONTAINER_NAME SERVED_MODEL_NAME VENV_PYTHON DOCKER_IMAGE TENSOR_PARALLEL_SIZE MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION VLLM_ENABLE_MTP VLLM_SPECULATIVE_CONFIG VLLM_EXTRA_ARGS FORCE_TEMPERATURE DEFAULT_MAX_TOKENS NORMALIZE_TOOL_CALL_ARGUMENTS DISABLE_THINKING MERGE_DEVICES QUANT_DEVICES MERGE_JOBS QUANT_WORKERS EXPECTED_GPU_COUNT ROCM_TORCH_VERSION ROCM_TORCH_INDEX_URL ATOM_SOURCE_DIR ATOM_REPO_URL ATOM_BRANCH ATOM_PROD_COMMIT PREFETCH_WORKERS EXTRACT_WORKERS OBSERVABILITY_ENABLED PROMETHEUS_IMAGE GRAFANA_IMAGE CADDY_IMAGE PUBLIC_ROOT_URL SUDO_PASSWORD; do
    printf "export %s=%s\n" "${name}" "$(shell_quote "${!name:-}")"
  done
}

resolve_source() {
  derive
  local out_json="${SOURCE_RESOLUTION_JSON:-/tmp/source_resolution_${RUN_SLUG}.json}"
  if [[ -n "${OSS_URL}" ]]; then
    python3 "${SKILL_DIR}/scripts/resolve_model_source.py" --oss-url "${OSS_URL}" --output-json "${out_json}"
  else
    python3 "${SKILL_DIR}/scripts/resolve_model_source.py" \
      --tinker-url "${TINKER_URL}" \
      --gpu-lease-base-url "${GPU_LEASE_BASE_URL}" \
      --gpu-lease-api-key "${GPU_LEASE_API_KEY}" \
      --transfer-jobs-endpoint "${TRANSFER_JOBS_ENDPOINT}" \
      --poll-interval "${TRANSFER_POLL_INTERVAL}" \
      --timeout "${TRANSFER_TIMEOUT_SECONDS}" \
      --output-json "${out_json}"
  fi
}

sync_scripts() {
  derive
  require SSH_HOST
  ssh_exec "mkdir -p $(shell_quote "${REMOTE_ROOT}")/scripts $(shell_quote "${REMOTE_ROOT}")/logs $(shell_quote "${REMOTE_ROOT}")/configs $(shell_quote "${REMOTE_ROOT}")/observability-skill"
  scp_to_remote "${SKILL_DIR}"/scripts/* "${SSH_HOST}:$(shell_quote "${REMOTE_ROOT}")/scripts/"
  scp_to_remote -r "${SKILL_DIR}"/observability/* "${SSH_HOST}:$(shell_quote "${REMOTE_ROOT}")/observability-skill/"
  ssh_exec "chmod +x $(shell_quote "${REMOTE_ROOT}")/scripts/"'*.sh'" $(shell_quote "${REMOTE_ROOT}")/scripts/"'*.py 2>/dev/null || true'
}

remote_stage() {
  local body="$1"
  derive
  require SSH_HOST
  {
    remote_exports
    printf "%s\n" "${body}"
  } | ssh_base ssh -o StrictHostKeyChecking=accept-new "${SSH_HOST}" "bash -se"
}

case "${STAGE}" in
  derive)
    print_derived
    ;;
  resolve-source)
    resolve_source
    ;;
  sync-scripts)
    sync_scripts
    ;;
  preflight)
    remote_stage '
set -euo pipefail
df -h / "$DATA_DISK" "$LOCAL_SCRATCH_MOUNT" 2>/dev/null || true
mkdir -p "$REMOTE_ROOT"/{configs,logs,results,scripts,models,hf-cache,request_captures}
root_avail_kb=$(df -Pk / | awk "NR==2 {print \$4}")
if [ -n "$root_avail_kb" ] && [ "$root_avail_kb" -lt 20971520 ]; then
  echo "OS root has less than 20 GiB free" >&2
  exit 3
fi
if command -v rocm-smi >/dev/null 2>&1; then rocm-smi --showuse --showmemuse | tail -80 || true; fi
if command -v docker >/dev/null 2>&1; then
  docker info --format "DockerRootDir={{.DockerRootDir}}" || true
  docker ps -a --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" || true
fi
ss -ltnp 2>/dev/null | grep -E ":(7788|18080|7777)" || true
'
    ;;
  prepare-env)
    remote_stage '
set -euo pipefail
mkdir -p "$SCRATCH_ROOT"/{downloads,adapters,models,serve,hf-cache,tmp,xdg-cache} "$PIP_CACHE_DIR" "$REMOTE_ROOT"/{configs,logs,models,results,scripts,request_captures}
if ! findmnt -rn "$LOCAL_SCRATCH_MOUNT" >/dev/null 2>&1; then
  echo "$LOCAL_SCRATCH_MOUNT is not mounted; prepare local NVMe with the azure-amd-deploy-env skill before model work" >&2
  exit 3
fi
if command -v docker >/dev/null 2>&1; then
  if docker compose version >/dev/null 2>&1; then
    docker compose version
  else
    echo "docker compose v2 is not available; bundled serve scripts will use docker run fallback"
  fi
  if docker ps >/dev/null 2>&1; then
    echo "docker socket access: direct"
    docker_root=$(docker info --format "{{.DockerRootDir}}" 2>/dev/null || true)
  elif sudo -n docker ps >/dev/null 2>&1; then
    echo "docker socket access: sudo"
    docker_root=$(sudo -n docker info --format "{{.DockerRootDir}}" 2>/dev/null || true)
  elif [ -n "${SUDO_PASSWORD:-}" ] && printf "%s\n" "$SUDO_PASSWORD" | sudo -S docker ps >/dev/null 2>&1; then
    echo "docker socket access: sudo password"
    docker_root=$(printf "%s\n" "$SUDO_PASSWORD" | sudo -S docker info --format "{{.DockerRootDir}}" 2>/dev/null || true)
  else
    echo "docker socket access requires sudo password or Docker group membership; set SUDO_PASSWORD if sudo prompts" >&2
    exit 3
  fi
  case "$docker_root" in
    "$DATA_DISK"/*|"") ;;
    *) echo "DockerRootDir must be under $DATA_DISK before pulling images: $docker_root" >&2; exit 3 ;;
  esac
fi
if [ ! -d "$ATOM_SOURCE_DIR/.git" ]; then
  git clone "$ATOM_REPO_URL" "$ATOM_SOURCE_DIR"
fi
git -C "$ATOM_SOURCE_DIR" fetch origin "$ATOM_BRANCH"
git -C "$ATOM_SOURCE_DIR" checkout "$ATOM_PROD_COMMIT"
if [ ! -x "$VENV_PYTHON" ]; then
  python3 -m venv "$(dirname "$(dirname "$VENV_PYTHON")")"
fi
"$VENV_PYTHON" -m pip install -U pip
"$VENV_PYTHON" -m pip install safetensors huggingface_hub accelerate peft aiohttp tqdm transformers
set +e
"$VENV_PYTHON" - <<PY
import sys
try:
    import torch
except Exception:
    sys.exit(10)
if not getattr(torch.version, "hip", None) or torch.cuda.device_count() < int("$EXPECTED_GPU_COUNT"):
    sys.exit(11)
PY
torch_status="$?"
set -e
case "$torch_status" in
  0) ;;
  10|11) "$VENV_PYTHON" -m pip install --index-url "$ROCM_TORCH_INDEX_URL" "torch==$ROCM_TORCH_VERSION" ;;
  *) exit "$torch_status" ;;
esac
'
    ;;
  fetch-source)
    remote_stage '
set -euo pipefail
export TMPDIR TEMP="$TMPDIR" TMP="$TMPDIR" XDG_CACHE_HOME HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR/hub" TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers" PIP_CACHE_DIR
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$PIP_CACHE_DIR"
sha_args=()
if [ -n "$OSS_SHA256" ]; then
  sha_args=(--sha256 "$OSS_SHA256")
fi
"$VENV_PYTHON" "$REMOTE_ROOT/scripts/prepare_oss_lora_source.py" --url "$OSS_URL" --work-dir "$OSS_WORK_DIR" --out "$PEFT_ADAPTER" --base-repo "$BASE_REPO" --extract-workers "$EXTRACT_WORKERS" "${sha_args[@]}"
'
    ;;
  prefetch-base)
    remote_stage '
set -euo pipefail
export HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR/hub" TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers" PIP_CACHE_DIR
"$VENV_PYTHON" "$REMOTE_ROOT/scripts/prefetch_glm51_base.py" --base-repo "$BASE_REPO" --cache-dir "$HF_CACHE_DIR" --workers "$PREFETCH_WORKERS" --include-side-files
'
    ;;
  merge)
    remote_stage '
set -euo pipefail
export HF_HOME="$HF_CACHE_DIR" HF_HUB_CACHE="$HF_CACHE_DIR/hub" TRANSFORMERS_CACHE="$HF_CACHE_DIR/transformers" PIP_CACHE_DIR
"$VENV_PYTHON" "$REMOTE_ROOT/scripts/merge_glm51_lora_sharded.py" --base-repo "$BASE_REPO" --adapter-repo "$PEFT_ADAPTER" --out "$BF16_OUT" --cache-dir "$HF_CACHE_DIR" --dtype bfloat16 --devices "$MERGE_DEVICES" --jobs "$MERGE_JOBS" --copy-untouched symlink
'
    ;;
  validate-bf16)
    remote_stage '
set -euo pipefail
"$VENV_PYTHON" "$REMOTE_ROOT/scripts/validate_and_repair_safetensors_shards.py" --model "$BF16_OUT" --cache-dir "$HF_CACHE_DIR" --repair-dangling-links --remove-temp-artifacts
'
    ;;
  quantize)
    remote_stage '
set -euo pipefail
"$VENV_PYTHON" "$REMOTE_ROOT/scripts/quantize_glm51_fp8_block128.py" --base-model-path "$BF16_OUT" --export-dir "$FP8_OUT" --cache-dir "$HF_CACHE_DIR" --devices "$QUANT_DEVICES" --workers "$QUANT_WORKERS" --trust-remote-code
'
    ;;
  stage-model)
    remote_stage '
set -euo pipefail
mkdir -p "$(dirname "$LOCAL_MODEL_PATH")" "$(dirname "$DURABLE_MODEL_PATH")"
rsync -a --delete "$FP8_OUT"/ "$LOCAL_MODEL_PATH"/
(rsync -a --delete "$LOCAL_MODEL_PATH"/ "$DURABLE_MODEL_PATH"/ >"$REMOTE_ROOT/logs/rsync_${RUN_SLUG}_durable.log" 2>&1 &)
'
    ;;
  write-serve-env)
    remote_stage '
set -euo pipefail
mkdir -p "$(dirname "$ENV_FILE")"
printf -v VLLM_SPECULATIVE_CONFIG_QUOTED "%q" "$VLLM_SPECULATIVE_CONFIG"
printf -v VLLM_EXTRA_ARGS_QUOTED "%q" "$VLLM_EXTRA_ARGS"
cat >"$ENV_FILE" <<EOF
AMD_PROFILING_ROOT=$REMOTE_ROOT
VLLM_ENV_FILE=$ENV_FILE
VLLM_IMAGE=$DOCKER_IMAGE
VLLM_MODEL=$MODEL_PATH
VLLM_TP=$TENSOR_PARALLEL_SIZE
VLLM_DTYPE=bfloat16
VLLM_KV_CACHE_DTYPE=bfloat16
VLLM_SOURCE_DIR=$ATOM_SOURCE_DIR
VLLM_HOST=127.0.0.1
VLLM_PORT=7788
VLLM_CONTAINER_NAME=$CONTAINER_NAME
VLLM_SERVED_MODEL_NAME=$SERVED_MODEL_NAME
VLLM_MAX_MODEL_LEN=$MAX_MODEL_LEN
VLLM_MAX_NUM_SEQS=$MAX_NUM_SEQS
VLLM_MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS
VLLM_GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION
VLLM_ENABLE_MTP=$VLLM_ENABLE_MTP
VLLM_SPECULATIVE_CONFIG=$VLLM_SPECULATIVE_CONFIG_QUOTED
VLLM_EXTRA_ARGS=$VLLM_EXTRA_ARGS_QUOTED
HF_HOME=$SERVE_HF_CACHE_DIR
HF_HUB_CACHE=$SERVE_HF_CACHE_DIR/hub
TRANSFORMERS_CACHE=$SERVE_HF_CACHE_DIR/transformers
CAPTURE_PROXY_UPSTREAM=http://127.0.0.1:7788
CAPTURE_PROXY_PORT=18080
CAPTURE_PROXY_FORCE_TEMPERATURE=$FORCE_TEMPERATURE
CAPTURE_PROXY_DEFAULT_MAX_TOKENS=$DEFAULT_MAX_TOKENS
CAPTURE_PROXY_NORMALIZE_TOOL_CALL_ARGUMENTS=$NORMALIZE_TOOL_CALL_ARGUMENTS
CAPTURE_PROXY_DISABLE_THINKING=$DISABLE_THINKING
CADDY_LISTEN=:7777
CADDY_UPSTREAM=127.0.0.1:18080
CADDY_METRICS_UPSTREAM=127.0.0.1:7788
CADDY_GRAFANA_UPSTREAM=127.0.0.1:3000
CADDY_PROMETHEUS_UPSTREAM=127.0.0.1:9090
OBSERVABILITY_ENABLED=$OBSERVABILITY_ENABLED
PROMETHEUS_IMAGE=$PROMETHEUS_IMAGE
PROMETHEUS_PORT=9090
GRAFANA_IMAGE=$GRAFANA_IMAGE
GRAFANA_PORT=3000
PUBLIC_ROOT_URL=$PUBLIC_ROOT_URL
GRAFANA_ROOT_URL=${PUBLIC_ROOT_URL%/}/grafana/
PROMETHEUS_EXTERNAL_URL=${PUBLIC_ROOT_URL%/}/prometheus/
SKILL_OBSERVABILITY_ROOT=$REMOTE_ROOT/observability-skill
EOF
'
    ;;
  serve-backend)
    remote_stage '
set -euo pipefail
ATOM_ENV_FILE="$ENV_FILE" VLLM_ENV_FILE="$ENV_FILE" "$REMOTE_ROOT/scripts/serve_vllm_glm51.sh"
'
    ;;
  serve-proxy)
    remote_stage 'AMD_PROFILING_ROOT="$REMOTE_ROOT" ATOM_ENV_FILE="$ENV_FILE" "$REMOTE_ROOT/scripts/serve_capture_proxy.sh"'
    ;;
  serve-observability)
    remote_stage 'AMD_PROFILING_ROOT="$REMOTE_ROOT" ATOM_ENV_FILE="$ENV_FILE" "$REMOTE_ROOT/scripts/serve_observability.sh"'
    ;;
  serve-caddy)
    remote_stage 'AMD_PROFILING_ROOT="$REMOTE_ROOT" ATOM_ENV_FILE="$ENV_FILE" "$REMOTE_ROOT/scripts/serve_caddy_proxy.sh"'
    ;;
  smoke)
    derive
    require PUBLIC_BASE_URL
    curl -fsS "${PUBLIC_BASE_URL%/}/models"
    curl -fsS -H "Content-Type: application/json" "${PUBLIC_BASE_URL%/}/chat/completions" \
      -d "{\"model\":\"${SERVED_MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"请直接给最终答案，不要展示推理过程。问题：1+1等于几？\"}],\"max_tokens\":64,\"temperature\":0}"
    ;;
  benchmark)
    remote_stage 'ATOM_ENV_FILE="$ENV_FILE" VLLM_ENV_FILE="$ENV_FILE" "$REMOTE_ROOT/scripts/benchmark_vllm_glm51.sh"'
    ;;
  deploy-all)
    derive
    if [[ -z "${OSS_URL}" ]]; then
      OSS_URL="$("${BASH_SOURCE[0]}" resolve-source)"
      export OSS_URL
    fi
    for next_stage in sync-scripts preflight prepare-env fetch-source prefetch-base merge validate-bf16 quantize stage-model write-serve-env serve-backend serve-proxy serve-observability serve-caddy smoke; do
      "${BASH_SOURCE[0]}" "${next_stage}"
    done
    ;;
  *)
    usage
    exit 2
    ;;
esac
