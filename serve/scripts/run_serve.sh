#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGE="${1:-serve-all}"

if [[ "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage: scripts/run_serve.sh [stage]

Default stage: serve-all
Common stages: derive sync-scripts write-serve-env serve-backend serve-proxy
               serve-observability serve-caddy smoke benchmark plan doctor serve-all
EOF
  exit 0
fi

exec "${ROOT_DIR}/shared/scripts/run_stage.sh" "${STAGE}"
