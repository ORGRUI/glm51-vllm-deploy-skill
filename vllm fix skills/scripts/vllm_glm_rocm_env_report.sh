#!/usr/bin/env bash
set -euo pipefail

container="${1:-${CONTAINER_NAME:-glm51-vllm}}"
model_path="${MODEL_PATH:-}"

section() {
  printf '\n## %s\n' "$1"
}

section "Host"
hostnamectl || true
lscpu | sed -n '1,35p' || true
free -h || true
df -h / /data /local_nvme 2>/dev/null || true

section "GPU"
rocm-smi --showproductname --showdriverversion --showvbios --showmeminfo vram --showbus --json || true

section "Host ROCm"
hipcc --version || true
rocm-smi --version || true

section "Docker"
docker version --format 'Client={{.Client.Version}} Server={{.Server.Version}} API={{.Server.APIVersion}}' || true
docker info --format 'Root={{.DockerRootDir}} Driver={{.Driver}} Runtime={{.DefaultRuntime}} Cgroup={{.CgroupVersion}} Images={{.Images}} Containers={{.Containers}}' || true
docker images --format '{{.Repository}}:{{.Tag}}	{{.ID}}	{{.Size}}	{{.CreatedSince}}' | grep -E 'vllm|rocm' || true
docker ps --format '{{.Names}}	{{.Image}}	{{.Status}}	{{.Ports}}' || true

section "Container Dependencies"
if docker ps --format '{{.Names}}' | grep -qx "${container}"; then
  docker exec "${container}" bash -lc "/opt/python/bin/python - <<'PY'
import importlib.metadata as md
mods = ['vllm', 'torch', 'triton', 'transformers', 'tokenizers', 'safetensors', 'numpy', 'aiter']
for mod in mods:
    try:
        print(f'{mod}={md.version(mod)}')
    except Exception as exc:
        print(f'{mod}=<not found: {exc}>')
import torch
print('torch.version.hip=' + str(torch.version.hip))
print('torch.cuda.is_available=' + str(torch.cuda.is_available()))
print('torch.cuda.device_count=' + str(torch.cuda.device_count()))
PY"
  docker exec "${container}" bash -lc 'env | sort | grep -E "^(VLLM|AITER|HIP|ROCM|PYTORCH|CUDA|NCCL|HSA|TORCH)"' || true
else
  echo "container not running: ${container}"
fi

section "Model"
if [[ -n "${model_path}" ]]; then
  du -sh "${model_path}" || true
  find "${model_path}" -maxdepth 1 -name '*.safetensors' | wc -l || true
else
  echo "MODEL_PATH is not set"
fi
