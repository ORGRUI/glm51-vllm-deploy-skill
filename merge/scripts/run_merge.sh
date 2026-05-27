#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STAGE="${1:-merge-all}"

if [[ "${STAGE}" == "-h" || "${STAGE}" == "--help" ]]; then
  cat >&2 <<'EOF'
Usage: scripts/run_merge.sh [stage]

Default stage: merge-all
Common stages: derive resolve-source sync-scripts preflight prepare-env
               fetch-source prefetch-base merge validate-bf16 plan doctor merge-all
EOF
  exit 0
fi

exec "${ROOT_DIR}/merge-quant-serve/scripts/run_stage.sh" "${STAGE}"
