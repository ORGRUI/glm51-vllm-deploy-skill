---
name: ajith-vllm-recipe
description: Reproduce Ajith's GLM-5/GLM-5.1 AMD ROCm vLLM recipe, or translate its official native-FP8 serve settings into this repo's merge/quant/serve pipeline.
---

# Ajith GLM-5.1 vLLM Recipe

Use this skill when the requested source of truth is:

```text
https://github.com/ajith-sirra-amd/recipes/blob/amd_glm5_support/GLM/GLM5.md
```

## Official AMD Recipe Shape

The recipe serves the native FP8 Hugging Face model directly on 8x MI300X/MI355X:

```bash
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export VLLM_ROCM_USE_AITER_RMSNORM=0

vllm serve zai-org/GLM-5.1-FP8 \
  --tensor-parallel-size 8 \
  --speculative-config.method mtp \
  --speculative-config.num_speculative_tokens 3 \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --enable-auto-tool-choice \
  --chat-template-content-format=string \
  --block-size=1 \
  --served-model-name glm-5.1-fp8
```

The Docker form uses:

```bash
docker run --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined \
  --group-add video \
  --ipc=host \
  -p 8000:8000 \
  -e VLLM_ROCM_USE_AITER=1 \
  -e VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4 \
  -e VLLM_ROCM_USE_AITER_RMSNORM=0 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai-rocm:latest \
  zai-org/GLM-5.1-FP8 \
    --tensor-parallel-size 8 \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --enable-auto-tool-choice \
    --chat-template-content-format=string \
    --block-size=1 \
    --enable-prefix-caching \
    --served-model-name glm-5.1-fp8
```

For thinking-off validation, pass:

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

## Using This Repo's Pipeline

This repo adds Tinker/OSS adapter resolution, BF16 merge, FP8 quantization, capture
proxy defaults, Caddy, and observability. For a full adapter deployment, enter:

```bash
cd merge-quant-serve
export SSH_HOST=vmadmin@<ip>
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme
export PUBLIC_BASE_URL=http://<ip>:7777/v1
export TINKER_URL='tinker://...'
./scripts/run_stage.sh deploy-all
```

The branch defaults already translate the recipe serve settings into the pipeline:

```bash
VLLM_ROCM_USE_AITER=1
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4
VLLM_ROCM_USE_AITER_RMSNORM=0
VLLM_ENABLE_MTP=1
VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'
VLLM_EXTRA_ARGS='--async-scheduling --compilation-config={"cudagraph_mode":"FULL_AND_PIECEWISE"} --enable-prefix-caching --block-size=1 --speculative-config={"method":"mtp","num_speculative_tokens":3}'
```

## Intentional Differences From The Recipe

- Native recipe model: `zai-org/GLM-5.1-FP8`; this pipeline serves a local
  merged and quantized adapter artifact when `TINKER_URL` or `OSS_URL` is used.
- Native recipe image: `vllm/vllm-openai-rocm:latest`; this pipeline keeps
  `rocm/atom-dev:vllm-latest` by default because the adapter path depends on the
  pinned ATOM/vLLM runtime recorded in `merge-quant-serve/SKILL.md`.
- Native recipe endpoint: direct vLLM `:8000`; this pipeline exposes Caddy
  `:7777`, forwards through the capture proxy, and keeps vLLM internal.
- Native recipe leaves thinking on unless the client disables it; this pipeline's
  capture proxy injects `chat_template_kwargs.enable_thinking=false` by default
  unless overridden.

If a task asks for the official recipe with no adapter pipeline, use the official
Docker command above. If it asks for a Tinker adapter deployment, use
`merge-quant-serve/scripts/run_stage.sh deploy-all` and record the differences
above in the handoff.
