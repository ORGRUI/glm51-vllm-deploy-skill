#!/usr/bin/env bash
set -euo pipefail

ROOT="${AMD_PROFILING_ROOT:-/data2/amd_profiling}"
ENV_FILE="${ATOM_ENV_FILE:-${ROOT}/configs/atom_glm5_engine.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

HOST="${CAPTURE_PROXY_HOST:-127.0.0.1}"
PORT="${CAPTURE_PROXY_PORT:-18080}"
UPSTREAM="${CAPTURE_PROXY_UPSTREAM:-http://127.0.0.1:7778}"
CAPTURE_DIR="${CAPTURE_PROXY_DIR:-${ROOT}/request_captures}"
FORCE_TEMPERATURE="${CAPTURE_PROXY_FORCE_TEMPERATURE:-}"
DEFAULT_MAX_TOKENS="${CAPTURE_PROXY_DEFAULT_MAX_TOKENS:-}"
MASK_REPLACEMENT_CHAR="${CAPTURE_PROXY_MASK_REPLACEMENT_CHAR:-1}"
NORMALIZE_TOOL_CALL_ARGUMENTS="${CAPTURE_PROXY_NORMALIZE_TOOL_CALL_ARGUMENTS:-1}"
DISABLE_THINKING="${CAPTURE_PROXY_DISABLE_THINKING:-1}"
CAPTURE_PROXY_PYTHON="${CAPTURE_PROXY_PYTHON:-${ROOT}/venv-merge/bin/python}"
if [[ ! -x "${CAPTURE_PROXY_PYTHON}" ]]; then
  CAPTURE_PROXY_PYTHON="python3"
fi
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${ROOT}/pip-cache}"

mkdir -p "${ROOT}/logs" "${CAPTURE_DIR}" "${PIP_CACHE_DIR}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${ROOT}/logs/capture_proxy_${timestamp}.log"

if ! "${CAPTURE_PROXY_PYTHON}" - <<'PY' >/dev/null 2>&1
import aiohttp  # noqa: F401
PY
then
  "${CAPTURE_PROXY_PYTHON}" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "${CAPTURE_PROXY_PYTHON}" -m pip install --cache-dir "${PIP_CACHE_DIR}" aiohttp
fi

old_pid_file="${ROOT}/logs/capture_proxy.pid"
if [[ -f "${old_pid_file}" ]]; then
  old_pid="$(cat "${old_pid_file}")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" >/dev/null 2>&1; then
    kill "${old_pid}" || true
  fi
fi

cmd=("${CAPTURE_PROXY_PYTHON}" "${ROOT}/scripts/capture_proxy.py" \
  --host "${HOST}" \
  --port "${PORT}" \
  --upstream "${UPSTREAM}" \
  --capture-dir "${CAPTURE_DIR}")

if [[ -n "${FORCE_TEMPERATURE}" ]]; then
  cmd+=(--force-temperature "${FORCE_TEMPERATURE}")
fi
if [[ -n "${DEFAULT_MAX_TOKENS}" ]]; then
  cmd+=(--default-max-tokens "${DEFAULT_MAX_TOKENS}")
fi
case "${MASK_REPLACEMENT_CHAR}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    cmd+=(--no-mask-replacement-char)
    ;;
  *)
    cmd+=(--mask-replacement-char)
    ;;
esac
case "${NORMALIZE_TOOL_CALL_ARGUMENTS}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    cmd+=(--no-normalize-tool-call-arguments)
    ;;
  *)
    cmd+=(--normalize-tool-call-arguments)
    ;;
esac
case "${DISABLE_THINKING}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    cmd+=(--no-disable-thinking)
    ;;
  *)
    cmd+=(--disable-thinking)
    ;;
esac

nohup "${cmd[@]}" >"${log_file}" 2>&1 &

pid="$!"
echo "${pid}" >"${old_pid_file}"
sleep 2
if ! kill -0 "${pid}" >/dev/null 2>&1; then
  echo "Capture proxy failed to stay running. Log: ${log_file}" >&2
  tail -80 "${log_file}" >&2 || true
  exit 1
fi
echo "Capture proxy started: http://${HOST}:${PORT} -> ${UPSTREAM}"
echo "Capture dir: ${CAPTURE_DIR}"
if [[ -n "${FORCE_TEMPERATURE}" ]]; then
  echo "Force temperature: ${FORCE_TEMPERATURE}"
fi
if [[ -n "${DEFAULT_MAX_TOKENS}" ]]; then
  echo "Default max tokens: ${DEFAULT_MAX_TOKENS}"
fi
echo "Mask replacement char: ${MASK_REPLACEMENT_CHAR}"
echo "Normalize tool-call arguments: ${NORMALIZE_TOOL_CALL_ARGUMENTS}"
echo "Disable thinking: ${DISABLE_THINKING}"
echo "Log: ${log_file}"
