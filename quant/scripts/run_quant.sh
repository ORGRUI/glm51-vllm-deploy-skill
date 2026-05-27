#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGE="${1:-quant-all}"

if [[ "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage: scripts/run_quant.sh [stage]

Default stage: quant-all
Common stages: derive sync-scripts preflight prepare-env quantize stage-model quant-all
EOF
  exit 0
fi

exec "${ROOT_DIR}/merge-quant-serve/scripts/run_stage.sh" "${STAGE}"
