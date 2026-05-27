#!/usr/bin/env bash
set -euo pipefail

ROOT="${AMD_PROFILING_ROOT:-/data/amd_profiling}"
ENV_FILE="${VLLM_ENV_FILE:-${ROOT}/configs/vllm_glm51_amd2.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

ROOT="${AMD_PROFILING_ROOT:-${ROOT}}"

IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
MODEL="${VLLM_MODEL:-/data/sft_aug_v1_from_0429_retry10_state_r16_no_unembed_32k_lr1e5_batch32_20260501_075124_final_fp8}"
TP="${VLLM_TP:-8}"
DTYPE="${VLLM_DTYPE:-bfloat16}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-bfloat16}"
SOURCE_DIR="${VLLM_SOURCE_DIR:-}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"
CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-glm51-local-64k-seq2}"
SERVER_NAME="${VLLM_SERVED_MODEL_NAME:-glm51-local-fp8}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-2}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
HF_HOME="${HF_HOME:-${ROOT}/hf-cache}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
SUDO_PASSWORD="${SUDO_PASSWORD:-}"
VLLM_ENABLE_MTP="${VLLM_ENABLE_MTP:-1}"
VLLM_SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG:-}"
if [[ -z "${VLLM_SPECULATIVE_CONFIG}" ]]; then
  VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":1}'
fi
DEFAULT_VLLM_EXTRA_ARGS='--async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching'
if [[ -z "${VLLM_EXTRA_ARGS+x}" ]]; then
  VLLM_EXTRA_ARGS="${DEFAULT_VLLM_EXTRA_ARGS}"
  if [[ "${VLLM_ENABLE_MTP}" != "0" ]]; then
    VLLM_EXTRA_ARGS+=" --speculative-config=${VLLM_SPECULATIVE_CONFIG}"
  fi
fi
VLLM_TARGET_DEVICE="${VLLM_TARGET_DEVICE:-rocm}"
VLLM_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE:-1}"
VLLM_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-1}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-glm47}"
VLLM_REASONING_PARSER="${VLLM_REASONING_PARSER:-glm45}"
VLLM_CHAT_TEMPLATE_CONTENT_FORMAT="${VLLM_CHAT_TEMPLATE_CONTENT_FORMAT:-string}"
VLLM_HOST_DATA_ROOT="${VLLM_HOST_DATA_ROOT:-/data}"
VLLM_EXTRA_MOUNTS="${VLLM_EXTRA_MOUNTS:-}"
REQUIRE_DOCKER_DATA_ROOT_PREFIX="${REQUIRE_DOCKER_DATA_ROOT_PREFIX:-}"

DOCKER_RUN=(docker run)
DOCKER_RM=(docker rm)
DOCKER_PS=(docker ps)
if ! docker ps >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    DOCKER_RUN=(sudo -S docker run)
    DOCKER_RM=(sudo -S docker rm)
    DOCKER_PS=(sudo -S docker ps)
  fi
fi

run_docker_ps() {
  if [[ "${DOCKER_PS[0]}" == "sudo" && -n "${SUDO_PASSWORD}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | "${DOCKER_PS[@]}" "$@"
  else
    "${DOCKER_PS[@]}" "$@"
  fi
}

run_docker_rm() {
  if [[ "${DOCKER_RM[0]}" == "sudo" && -n "${SUDO_PASSWORD}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | "${DOCKER_RM[@]}" "$@"
  else
    "${DOCKER_RM[@]}" "$@"
  fi
}

remove_existing_backend_container() {
  mapfile -t existing_containers < <(run_docker_ps -a --filter "name=^/${CONTAINER_NAME}$" --format '{{.ID}}' 2>/dev/null | awk 'NF')
  if [[ "${#existing_containers[@]}" -gt 0 ]]; then
    echo "Removing existing backend Docker container: ${CONTAINER_NAME}"
    run_docker_rm -f "${existing_containers[@]}" >/dev/null 2>&1 || true
  fi
}

mkdir -p "${ROOT}/logs" "${ROOT}/configs" "${ROOT}/results" "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}"

if [[ "${MODEL}" == /* && ! -e "${MODEL}" ]]; then
  echo "ERROR: model path does not exist: ${MODEL}" >&2
  exit 1
fi

if [[ -n "${REQUIRE_DOCKER_DATA_ROOT_PREFIX}" ]]; then
  docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null | awk 'NF { print; exit }' || true)"
  if [[ -z "${docker_root}" && "${DOCKER_RUN[0]}" == "sudo" ]]; then
    if [[ -n "${SUDO_PASSWORD}" ]]; then
      docker_root="$(printf '%s\n' "${SUDO_PASSWORD}" | sudo -S docker info --format '{{.DockerRootDir}}' 2>/dev/null | awk 'NF { print; exit }' || true)"
    else
      docker_root="$(sudo -n docker info --format '{{.DockerRootDir}}' 2>/dev/null | awk 'NF { print; exit }' || true)"
    fi
  fi
  if [[ -z "${docker_root}" ]]; then
    echo "ERROR: cannot determine DockerRootDir" >&2
    exit 1
  fi
  if [[ "${docker_root}" != "${REQUIRE_DOCKER_DATA_ROOT_PREFIX}"* ]]; then
    echo "ERROR: DockerRootDir=${docker_root} is not under ${REQUIRE_DOCKER_DATA_ROOT_PREFIX}" >&2
    exit 1
  fi
fi

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${ROOT}/logs/vllm_glm51_${timestamp}.log"
cmd_file="${ROOT}/configs/vllm_glm51_${timestamp}.sh"
argv_file="${ROOT}/configs/vllm_glm51_${timestamp}.server_argv.json"

remove_existing_backend_container

server_cmd=(
  vllm serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVER_NAME}"
  --tensor-parallel-size "${TP}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
)

if [[ "${VLLM_ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  server_cmd+=(--enable-auto-tool-choice)
  server_cmd+=(--tool-call-parser "${VLLM_TOOL_CALL_PARSER}")
  server_cmd+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
  server_cmd+=(--chat-template-content-format "${VLLM_CHAT_TEMPLATE_CONTENT_FORMAT}")
fi

if [[ "${VLLM_TRUST_REMOTE_CODE}" == "1" ]]; then
  server_cmd+=(--trust-remote-code)
fi

if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
  read -r -a extra_args <<<"${VLLM_EXTRA_ARGS}"
  server_cmd+=("${extra_args[@]}")
fi

pythonpath_arg=""
atom_source_arg=""
if [[ -n "${SOURCE_DIR}" ]]; then
  if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "ERROR: VLLM_SOURCE_DIR does not exist: ${SOURCE_DIR}" >&2
    exit 1
  fi
  pythonpath_arg="${SOURCE_DIR}"
  atom_source_arg="${SOURCE_DIR}"
fi

SOURCE_GIT_REMOTE=""
SOURCE_GIT_REMOTE_URL=""
SOURCE_GIT_BRANCH=""
SOURCE_GIT_UPSTREAM=""
SOURCE_GIT_COMMIT=""
SOURCE_GIT_DIRTY="false"
SOURCE_GIT_DIRTY_FILES=""
if [[ -n "${SOURCE_DIR}" && -d "${SOURCE_DIR}/.git" ]] && command -v git >/dev/null 2>&1; then
  SOURCE_GIT_BRANCH="$(git -C "${SOURCE_DIR}" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  SOURCE_GIT_UPSTREAM="$(git -C "${SOURCE_DIR}" rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null || true)"
  SOURCE_GIT_COMMIT="$(git -C "${SOURCE_DIR}" rev-parse HEAD 2>/dev/null || true)"
  SOURCE_GIT_DIRTY_FILES="$(git -C "${SOURCE_DIR}" status --porcelain 2>/dev/null || true)"
  if [[ -n "${SOURCE_GIT_DIRTY_FILES}" ]]; then
    SOURCE_GIT_DIRTY="true"
  fi
  if [[ "${SOURCE_GIT_UPSTREAM}" == */* ]]; then
    SOURCE_GIT_REMOTE="${SOURCE_GIT_UPSTREAM%%/*}"
  elif git -C "${SOURCE_DIR}" remote get-url fork >/dev/null 2>&1; then
    SOURCE_GIT_REMOTE="fork"
  elif git -C "${SOURCE_DIR}" remote get-url origin >/dev/null 2>&1; then
    SOURCE_GIT_REMOTE="origin"
  fi
  if [[ -n "${SOURCE_GIT_REMOTE}" ]]; then
    SOURCE_GIT_REMOTE_URL="$(git -C "${SOURCE_DIR}" remote get-url "${SOURCE_GIT_REMOTE}" 2>/dev/null || true)"
  fi
fi

VLLM_ARGV_TIMESTAMP="${timestamp}" \
VLLM_ARGV_ENV_FILE="${ENV_FILE}" \
VLLM_ARGV_IMAGE="${IMAGE}" \
VLLM_ARGV_CONTAINER_NAME="${CONTAINER_NAME}" \
VLLM_ARGV_MODEL="${MODEL}" \
VLLM_ARGV_TP="${TP}" \
VLLM_ARGV_DTYPE="${DTYPE}" \
VLLM_ARGV_KV_CACHE_DTYPE="${KV_CACHE_DTYPE}" \
VLLM_ARGV_SOURCE_DIR="${SOURCE_DIR}" \
VLLM_ARGV_HOST="${HOST}" \
VLLM_ARGV_PORT="${PORT}" \
VLLM_ARGV_SERVER_NAME="${SERVER_NAME}" \
VLLM_ARGV_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
VLLM_ARGV_MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
VLLM_ARGV_MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
VLLM_ARGV_GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
VLLM_ARGV_TARGET_DEVICE="${VLLM_TARGET_DEVICE}" \
VLLM_ARGV_ENABLE_MTP="${VLLM_ENABLE_MTP}" \
VLLM_ARGV_SPECULATIVE_CONFIG="${VLLM_SPECULATIVE_CONFIG}" \
VLLM_ARGV_EXTRA_ARGS="${VLLM_EXTRA_ARGS}" \
VLLM_ARGV_TRUST_REMOTE_CODE="${VLLM_TRUST_REMOTE_CODE}" \
VLLM_ARGV_ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE}" \
VLLM_ARGV_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER}" \
VLLM_ARGV_REASONING_PARSER="${VLLM_REASONING_PARSER}" \
VLLM_ARGV_CHAT_TEMPLATE_CONTENT_FORMAT="${VLLM_CHAT_TEMPLATE_CONTENT_FORMAT}" \
VLLM_ARGV_HOST_DATA_ROOT="${VLLM_HOST_DATA_ROOT}" \
VLLM_ARGV_EXTRA_MOUNTS="${VLLM_EXTRA_MOUNTS}" \
VLLM_ARGV_REQUIRE_DOCKER_DATA_ROOT_PREFIX="${REQUIRE_DOCKER_DATA_ROOT_PREFIX}" \
VLLM_ARGV_SOURCE_GIT_REMOTE="${SOURCE_GIT_REMOTE}" \
VLLM_ARGV_SOURCE_GIT_REMOTE_URL="${SOURCE_GIT_REMOTE_URL}" \
VLLM_ARGV_SOURCE_GIT_BRANCH="${SOURCE_GIT_BRANCH}" \
VLLM_ARGV_SOURCE_GIT_UPSTREAM="${SOURCE_GIT_UPSTREAM}" \
VLLM_ARGV_SOURCE_GIT_COMMIT="${SOURCE_GIT_COMMIT}" \
VLLM_ARGV_SOURCE_GIT_DIRTY="${SOURCE_GIT_DIRTY}" \
VLLM_ARGV_SOURCE_GIT_DIRTY_FILES="${SOURCE_GIT_DIRTY_FILES}" \
python3 - "$argv_file" "${server_cmd[@]}" <<'PY'
import json
import os
import sys

source_git = None
if os.environ["VLLM_ARGV_SOURCE_GIT_COMMIT"] or os.environ["VLLM_ARGV_SOURCE_GIT_REMOTE_URL"]:
    source_git = {
        "remote": os.environ["VLLM_ARGV_SOURCE_GIT_REMOTE"],
        "remote_url": os.environ["VLLM_ARGV_SOURCE_GIT_REMOTE_URL"],
        "branch": os.environ["VLLM_ARGV_SOURCE_GIT_BRANCH"],
        "upstream": os.environ["VLLM_ARGV_SOURCE_GIT_UPSTREAM"],
        "commit": os.environ["VLLM_ARGV_SOURCE_GIT_COMMIT"],
        "dirty": os.environ["VLLM_ARGV_SOURCE_GIT_DIRTY"] == "true",
        "dirty_files": [
            line
            for line in os.environ["VLLM_ARGV_SOURCE_GIT_DIRTY_FILES"].splitlines()
            if line
        ],
    }

data = {
    "timestamp_utc": os.environ["VLLM_ARGV_TIMESTAMP"],
    "env_file": os.environ["VLLM_ARGV_ENV_FILE"],
    "image": os.environ["VLLM_ARGV_IMAGE"],
    "container_name": os.environ["VLLM_ARGV_CONTAINER_NAME"],
    "model": os.environ["VLLM_ARGV_MODEL"],
    "tensor_parallel": os.environ["VLLM_ARGV_TP"],
    "dtype": os.environ["VLLM_ARGV_DTYPE"],
    "kv_cache_dtype": os.environ["VLLM_ARGV_KV_CACHE_DTYPE"],
    "source_dir": os.environ["VLLM_ARGV_SOURCE_DIR"],
    "host": os.environ["VLLM_ARGV_HOST"],
    "port": os.environ["VLLM_ARGV_PORT"],
    "served_model_name": os.environ["VLLM_ARGV_SERVER_NAME"],
    "max_model_len": os.environ["VLLM_ARGV_MAX_MODEL_LEN"],
    "max_num_seqs": os.environ["VLLM_ARGV_MAX_NUM_SEQS"],
    "max_num_batched_tokens": os.environ["VLLM_ARGV_MAX_NUM_BATCHED_TOKENS"],
    "gpu_memory_utilization": os.environ["VLLM_ARGV_GPU_MEMORY_UTILIZATION"],
    "target_device": os.environ["VLLM_ARGV_TARGET_DEVICE"],
    "enable_mtp": os.environ["VLLM_ARGV_ENABLE_MTP"],
    "speculative_config": os.environ["VLLM_ARGV_SPECULATIVE_CONFIG"],
    "extra_args": os.environ["VLLM_ARGV_EXTRA_ARGS"],
    "trust_remote_code": os.environ["VLLM_ARGV_TRUST_REMOTE_CODE"],
    "enable_auto_tool_choice": os.environ["VLLM_ARGV_ENABLE_AUTO_TOOL_CHOICE"],
    "tool_call_parser": os.environ["VLLM_ARGV_TOOL_CALL_PARSER"],
    "reasoning_parser": os.environ["VLLM_ARGV_REASONING_PARSER"],
    "chat_template_content_format": os.environ["VLLM_ARGV_CHAT_TEMPLATE_CONTENT_FORMAT"],
    "host_data_root": os.environ["VLLM_ARGV_HOST_DATA_ROOT"],
    "extra_mounts": os.environ["VLLM_ARGV_EXTRA_MOUNTS"],
    "require_docker_data_root_prefix": os.environ["VLLM_ARGV_REQUIRE_DOCKER_DATA_ROOT_PREFIX"],
    "source_git": source_git,
    "server_argv": sys.argv[2:],
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  if [[ "${DOCKER_RUN[0]}" == "sudo" && -n "${SUDO_PASSWORD}" ]]; then
    printf 'printf %%s\\\\n %q | sudo -S docker run --rm --name %q \\\n' "${SUDO_PASSWORD}" "${CONTAINER_NAME}"
  elif [[ "${DOCKER_RUN[0]}" == "sudo" ]]; then
    printf 'sudo -S docker run --rm --name %q \\\n' "${CONTAINER_NAME}"
  else
    printf 'docker run --rm --name %q \\\n' "${CONTAINER_NAME}"
  fi
  printf '  --network host \\\n'
  printf '  --ipc host \\\n'
  printf '  --group-add video \\\n'
  printf '  --cap-add SYS_PTRACE \\\n'
  printf '  --security-opt seccomp=unconfined \\\n'
  printf '  --device /dev/kfd \\\n'
  printf '  --device /dev/dri \\\n'
  printf '  -e HF_HOME=%q \\\n' "${HF_HOME}"
  printf '  -e HF_HUB_CACHE=%q \\\n' "${HF_HUB_CACHE}"
  printf '  -e TRANSFORMERS_CACHE=%q \\\n' "${TRANSFORMERS_CACHE}"
  printf '  -e VLLM_TARGET_DEVICE=%q \\\n' "${VLLM_TARGET_DEVICE}"
  if [[ -n "${pythonpath_arg}" ]]; then
    printf '  -e PYTHONPATH=%q \\\n' "${pythonpath_arg}"
    printf '  -e ATOM_SOURCE_DIR=%q \\\n' "${atom_source_arg}"
  fi
  if [[ -n "${VLLM_HOST_DATA_ROOT}" && -d "${VLLM_HOST_DATA_ROOT}" && "${VLLM_HOST_DATA_ROOT}" != "${ROOT}" ]]; then
    printf '  -v %q:%q \\\n' "${VLLM_HOST_DATA_ROOT}" "${VLLM_HOST_DATA_ROOT}"
  fi
  model_parent=""
  if [[ "${MODEL}" == /* ]]; then
    model_parent="$(dirname "${MODEL}")"
    while [[ -n "${model_parent}" && "${model_parent}" != "/" && ! -d "${model_parent}" ]]; do
      model_parent="$(dirname "${model_parent}")"
    done
  fi
  for mount_path in ${VLLM_EXTRA_MOUNTS}; do
    if [[ -n "${mount_path}" && -d "${mount_path}" ]]; then
      printf '  -v %q:%q \\\n' "${mount_path}" "${mount_path}"
    fi
  done
  if [[ -n "${model_parent}" && -d "${model_parent}" && "${model_parent}" != "${ROOT}" && "${model_parent}" != "${VLLM_HOST_DATA_ROOT}" ]]; then
    printf '  -v %q:%q \\\n' "${model_parent}" "${model_parent}"
  fi
  printf '  -v %q:%q \\\n' "${ROOT}" "${ROOT}"
  printf '  %q \\\n' "${IMAGE}"
  printf '  %q' "${server_cmd[@]}"
  echo
} >"${cmd_file}"
chmod +x "${cmd_file}"

if [[ "${ATOM_DRY_RUN:-0}" == "1" || "${VLLM_DRY_RUN:-0}" == "1" ]]; then
  echo "Dry run only. Saved launch command: ${cmd_file}"
  sed -n '1,120p' "${cmd_file}"
  exit 0
fi

echo "Starting vLLM server. Logs: ${log_file}"
echo "Saved launch command: ${cmd_file}"
echo "Saved vLLM server argv: ${argv_file}"

nohup "${cmd_file}" >"${log_file}" 2>&1 &
pid="$!"
echo "${pid}" >"${ROOT}/logs/vllm_glm51_server.pid"
echo "Host wrapper PID: ${pid}"
echo "OpenAI-compatible API port: ${PORT}"
