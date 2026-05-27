# Diagnostics

Run these commands when the host, image, container, model, or performance changes.

## Host

```bash
hostnamectl
lscpu
free -h
df -h / /data /local_nvme 2>/dev/null || true
rocm-smi --showproductname --showdriverversion --showvbios --showmeminfo vram --showbus --json
hipcc --version
docker version --format 'Client={{.Client.Version}} Server={{.Server.Version}} API={{.Server.APIVersion}}'
docker info --format 'Root={{.DockerRootDir}} Driver={{.Driver}} Runtime={{.DefaultRuntime}} Cgroup={{.CgroupVersion}} Images={{.Images}} Containers={{.Containers}}'
```

## Docker Images and Service

```bash
docker images --format '{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}' | grep -E 'vllm|rocm'
docker ps --format '{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}'
docker logs --tail 120 "$CONTAINER_NAME"
curl -fsS "${BASE_URL:-http://127.0.0.1:7804/v1}/models"
```

## Container Dependencies

```bash
docker exec "$CONTAINER_NAME" bash -lc "/opt/python/bin/python - <<'PY'
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
docker exec "$CONTAINER_NAME" bash -lc 'env | sort | grep -E "^(VLLM|AITER|HIP|ROCM|PYTORCH|CUDA|NCCL|HSA|TORCH)"'
```

## Model

```bash
du -sh "$MODEL_PATH"
find "$MODEL_PATH" -maxdepth 1 -name '*.safetensors' | wc -l
```

## Logs

```bash
docker logs --since 10m "$CONTAINER_NAME" 2>&1 | grep -Ei 'error|exception|traceback|nan|failed' | tail -50 || true
```
