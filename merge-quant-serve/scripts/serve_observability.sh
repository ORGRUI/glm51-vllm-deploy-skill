#!/usr/bin/env bash
set -euo pipefail

ROOT="${AMD_PROFILING_ROOT:-/data/amd_profiling}"
ENV_FILE="${ATOM_ENV_FILE:-${ROOT}/configs/atom_glm5_engine.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

ROOT="${AMD_PROFILING_ROOT:-${ROOT}}"

OBSERVABILITY_ENABLED="${OBSERVABILITY_ENABLED:-1}"
case "${OBSERVABILITY_ENABLED}" in
  0|false|False|FALSE|no|No|NO|off|Off|OFF)
    echo "Observability disabled by OBSERVABILITY_ENABLED=${OBSERVABILITY_ENABLED}"
    exit 0
    ;;
esac

PROMETHEUS_IMAGE="${PROMETHEUS_IMAGE:-prom/prometheus:v2.55.1}"
GRAFANA_IMAGE="${GRAFANA_IMAGE:-grafana/grafana:11.3.1}"
PROMETHEUS_CONTAINER_NAME="${PROMETHEUS_CONTAINER_NAME:-amd-profiling-prometheus}"
GRAFANA_CONTAINER_NAME="${GRAFANA_CONTAINER_NAME:-amd-profiling-grafana}"
PROMETHEUS_PORT="${PROMETHEUS_PORT:-9090}"
GRAFANA_PORT="${GRAFANA_PORT:-3000}"
VLLM_METRICS_HOST="${VLLM_METRICS_HOST:-127.0.0.1}"
VLLM_METRICS_PORT="${VLLM_METRICS_PORT:-${VLLM_PORT:-7788}}"
VLLM_METRICS_PATH="${VLLM_METRICS_PATH:-/metrics}"
VLLM_SCRAPE_INTERVAL="${VLLM_SCRAPE_INTERVAL:-5s}"
GRAFANA_ADMIN_USER="${GRAFANA_ADMIN_USER:-admin}"
GRAFANA_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD:-admin}"
PUBLIC_ROOT_URL="${PUBLIC_ROOT_URL:-http://127.0.0.1:7777}"
GRAFANA_ROOT_URL="${GRAFANA_ROOT_URL:-${PUBLIC_ROOT_URL%/}/grafana/}"
PROMETHEUS_EXTERNAL_URL="${PROMETHEUS_EXTERNAL_URL:-${PUBLIC_ROOT_URL%/}/prometheus/}"
SUDO_PASSWORD="${SUDO_PASSWORD:-}"

OBS_ROOT="${OBSERVABILITY_ROOT:-${ROOT}/observability}"
PROM_CONFIG="${OBS_ROOT}/prometheus/prometheus.yml"
GRAFANA_PROVISIONING="${OBS_ROOT}/grafana/provisioning"
GRAFANA_DASHBOARDS="${OBS_ROOT}/grafana/dashboards"
SKILL_OBS_ROOT="${SKILL_OBSERVABILITY_ROOT:-${ROOT}/observability-skill}"

mkdir -p \
  "${ROOT}/logs" \
  "${OBS_ROOT}/prometheus/data" \
  "${OBS_ROOT}/grafana/data" \
  "${GRAFANA_PROVISIONING}/dashboards" \
  "${GRAFANA_PROVISIONING}/datasources" \
  "${GRAFANA_DASHBOARDS}"

if [[ -d "${SKILL_OBS_ROOT}/grafana/provisioning" ]]; then
  cp -R "${SKILL_OBS_ROOT}/grafana/provisioning/." "${GRAFANA_PROVISIONING}/"
fi
if [[ -d "${SKILL_OBS_ROOT}/grafana/dashboards" ]]; then
  cp -R "${SKILL_OBS_ROOT}/grafana/dashboards/." "${GRAFANA_DASHBOARDS}/"
fi

cat >"${PROM_CONFIG}" <<EOF
global:
  scrape_interval: ${VLLM_SCRAPE_INTERVAL}
  evaluation_interval: ${VLLM_SCRAPE_INTERVAL}

scrape_configs:
  - job_name: vllm
    metrics_path: ${VLLM_METRICS_PATH}
    static_configs:
      - targets:
          - ${VLLM_METRICS_HOST}:${VLLM_METRICS_PORT}
        labels:
          service: vllm

  - job_name: prometheus
    metrics_path: /prometheus/metrics
    static_configs:
      - targets:
          - 127.0.0.1:${PROMETHEUS_PORT}
EOF

PROMETHEUS_DATASOURCE_URL="http://127.0.0.1:${PROMETHEUS_PORT}/prometheus"
export PROMETHEUS_DATASOURCE_URL
if command -v envsubst >/dev/null 2>&1 && [[ -f "${GRAFANA_PROVISIONING}/datasources/prometheus.yml" ]]; then
  envsubst <"${GRAFANA_PROVISIONING}/datasources/prometheus.yml" >"${GRAFANA_PROVISIONING}/datasources/prometheus.yml.rendered"
  mv "${GRAFANA_PROVISIONING}/datasources/prometheus.yml.rendered" "${GRAFANA_PROVISIONING}/datasources/prometheus.yml"
else
  cat >"${GRAFANA_PROVISIONING}/datasources/prometheus.yml" <<EOF
apiVersion: 1

datasources:
  - name: Prometheus
    uid: Prometheus
    type: prometheus
    access: proxy
    url: ${PROMETHEUS_DATASOURCE_URL}
    isDefault: true
    editable: true
EOF
fi

DOCKER=(docker)
if ! docker ps >/dev/null 2>&1; then
  DOCKER=(sudo -S docker)
fi

run_docker() {
  if [[ "${DOCKER[0]}" == "sudo" && -n "${SUDO_PASSWORD}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | "${DOCKER[@]}" "$@"
  else
    "${DOCKER[@]}" "$@"
  fi
}

run_docker rm -f "${PROMETHEUS_CONTAINER_NAME}" "${GRAFANA_CONTAINER_NAME}" >/dev/null 2>&1 || true
run_docker run --rm \
  --user 0 \
  --entrypoint sh \
  -v "${OBS_ROOT}/prometheus/data:/prometheus" \
  -v "${OBS_ROOT}/grafana/data:/var/lib/grafana" \
  "${GRAFANA_IMAGE}" \
  -c 'chown -R 65534:65534 /prometheus && chown -R 472:472 /var/lib/grafana' >/dev/null

run_docker run -d --name "${PROMETHEUS_CONTAINER_NAME}" \
  --network host \
  -v "${PROM_CONFIG}:/etc/prometheus/prometheus.yml:ro" \
  -v "${OBS_ROOT}/prometheus/data:/prometheus" \
  "${PROMETHEUS_IMAGE}" \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --storage.tsdb.retention.time=15d \
  --web.enable-lifecycle \
  --web.listen-address=127.0.0.1:${PROMETHEUS_PORT} \
  --web.external-url="${PROMETHEUS_EXTERNAL_URL}" \
  --web.route-prefix=/prometheus >/dev/null

run_docker run -d --name "${GRAFANA_CONTAINER_NAME}" \
  --network host \
  -e GF_SECURITY_ADMIN_USER="${GRAFANA_ADMIN_USER}" \
  -e GF_SECURITY_ADMIN_PASSWORD="${GRAFANA_ADMIN_PASSWORD}" \
  -e GF_USERS_DEFAULT_THEME=light \
  -e GF_ANALYTICS_REPORTING_ENABLED=false \
  -e GF_ANALYTICS_CHECK_FOR_UPDATES=false \
  -e GF_SERVER_HTTP_ADDR=127.0.0.1 \
  -e GF_SERVER_HTTP_PORT="${GRAFANA_PORT}" \
  -e GF_SERVER_ROOT_URL="${GRAFANA_ROOT_URL}" \
  -e GF_SERVER_SERVE_FROM_SUB_PATH=true \
  -v "${OBS_ROOT}/grafana/data:/var/lib/grafana" \
  -v "${GRAFANA_PROVISIONING}:/etc/grafana/provisioning:ro" \
  -v "${GRAFANA_DASHBOARDS}:/var/lib/grafana/dashboards:ro" \
  "${GRAFANA_IMAGE}" >/dev/null

echo "Prometheus started: http://127.0.0.1:${PROMETHEUS_PORT}/prometheus/"
echo "Grafana started: http://127.0.0.1:${GRAFANA_PORT}/grafana/"
echo "Public routes expected through Caddy: ${PUBLIC_ROOT_URL%/}/prometheus/ and ${PUBLIC_ROOT_URL%/}/grafana/"
