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

ROOT="${AMD_PROFILING_ROOT:-${ROOT}}"

CADDY_IMAGE="${CADDY_IMAGE:-caddy:2-alpine}"
CADDY_CONTAINER_NAME="${CADDY_CONTAINER_NAME:-amd-profiling-caddy}"
CADDY_LISTEN="${CADDY_LISTEN:-:7777}"
CADDY_BIND="${CADDY_BIND:-}"
CADDY_UPSTREAM="${CADDY_UPSTREAM:-127.0.0.1:18080}"
CADDY_METRICS_UPSTREAM="${CADDY_METRICS_UPSTREAM:-127.0.0.1:${VLLM_PORT:-7788}}"
CADDY_GRAFANA_UPSTREAM="${CADDY_GRAFANA_UPSTREAM:-127.0.0.1:${GRAFANA_PORT:-3000}}"
CADDY_PROMETHEUS_UPSTREAM="${CADDY_PROMETHEUS_UPSTREAM:-127.0.0.1:${PROMETHEUS_PORT:-9090}}"
CADDYFILE="${CADDYFILE:-${ROOT}/configs/Caddyfile.capture-proxy}"
SUDO_PASSWORD="${SUDO_PASSWORD:-}"

mkdir -p "${ROOT}/logs" "${ROOT}/configs" "${ROOT}/caddy/data" "${ROOT}/caddy/config"

cat >"${CADDYFILE}" <<EOF
{
  auto_https off
  admin off
}

${CADDY_LISTEN} {
EOF

if [[ -n "${CADDY_BIND}" ]]; then
  printf '  bind %s\n' "${CADDY_BIND}" >>"${CADDYFILE}"
fi

cat >>"${CADDYFILE}" <<EOF
  redir /grafana /grafana/ 308
  redir /prometheus /prometheus/ 308

  @health {
    method GET
    path /health
  }
  @api {
    path /v1 /v1/*
  }
  @metrics {
    path /metrics
  }
  handle /grafana/* {
    reverse_proxy ${CADDY_GRAFANA_UPSTREAM}
  }
  handle /prometheus/* {
    reverse_proxy ${CADDY_PROMETHEUS_UPSTREAM}
  }
  handle @health {
    respond "OK" 200
  }
  handle @api {
    reverse_proxy ${CADDY_UPSTREAM}
  }
  handle @metrics {
    reverse_proxy ${CADDY_METRICS_UPSTREAM}
  }
  respond 404
}
EOF

DOCKER_RUN=(docker run)
DOCKER_RM=(docker rm)
if ! docker ps >/dev/null 2>&1; then
  DOCKER_RUN=(sudo -S docker run)
  DOCKER_RM=(sudo -S docker rm)
fi

run_docker_rm() {
  if [[ "${DOCKER_RM[0]}" == "sudo" && -n "${SUDO_PASSWORD}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | "${DOCKER_RM[@]}" "$@"
  else
    "${DOCKER_RM[@]}" "$@"
  fi
}

run_docker() {
  if [[ "${DOCKER_RUN[0]}" == "sudo" && -n "${SUDO_PASSWORD}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | "${DOCKER_RUN[@]}" "$@"
  else
    "${DOCKER_RUN[@]}" "$@"
  fi
}

run_docker_rm -f "${CADDY_CONTAINER_NAME}" >/dev/null 2>&1 || true

run_docker -d --name "${CADDY_CONTAINER_NAME}" \
  --network host \
  -v "${CADDYFILE}:/etc/caddy/Caddyfile:ro" \
  -v "${ROOT}/caddy/data:/data" \
  -v "${ROOT}/caddy/config:/config" \
  "${CADDY_IMAGE}" >/dev/null

echo "Caddy proxy started on ${CADDY_LISTEN} -> ${CADDY_UPSTREAM}"
echo "Caddy metrics upstream: ${CADDY_METRICS_UPSTREAM}"
echo "Caddy Grafana upstream: ${CADDY_GRAFANA_UPSTREAM}"
echo "Caddy Prometheus upstream: ${CADDY_PROMETHEUS_UPSTREAM}"
if [[ -n "${CADDY_BIND}" ]]; then
  echo "Caddy bind: ${CADDY_BIND}"
fi
echo "Caddyfile: ${CADDYFILE}"
