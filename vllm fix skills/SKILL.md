---
name: vllm-fix-skills
description: Deploy, patch, validate, and debug GLM-5.1 FP8/MLA models on AMD ROCm with pure vLLM. Use when applying the ROCm MLA fixes, custom allreduce correctness fix, GLM Responses/tool-call parser fix, starting a patched vLLM service, checking NaN or garbled-token failures, or tuning throughput only after correctness is confirmed.
---

# vLLM Fix Skills

## Contract

This skill is host-agnostic. Do not hard-code host IPs, passwords, signed model URLs, access tokens, private model names, or local-only absolute paths into committed artifacts. Collect them from the user or environment variables at runtime.

Required runtime inputs:

```bash
export MODEL_PATH=/path/to/merged-fp8-block128-glm51
export SERVED_MODEL_NAME=glm51-fp8-vllm
export VLLM_IMAGE=<patched-vllm-image>
export VLLM_PORT=7804
```

Use this skill only for pure vLLM on ROCm. If the deployment is ATOM-based, keep the ATOM-specific merge/quant/serve flow separate.

## Workflow

1. Confirm the environment baseline:
   - 8 AMD MI300X-class GPUs or equivalent ROCm target.
   - Containerized vLLM with Python 3.13 site-packages at `/opt/python/lib/python3.13/site-packages`, unless overridden.
   - GLM-5.1 merged FP8 block-128 model with sparse MLA support.
   - vLLM version compatible with GLM tool parser patch PR 39253 when tool calling is enabled.

2. Ensure required correctness fixes are present before serving:
   - ROCm sparse MLA/indexer patch set.
   - AITER MLA GQA8 stage2-reduce patch for TP=8 decode.
   - PA MQA logits tile-count patch.
   - custom allreduce warmup/replay correctness patch.
   - GLM Responses/tool-call parser compatibility patch when `/v1/responses` and tools are used.

3. Start the service with correctness-first settings from `references/runtime.md`.
   - Keep MLA, custom allreduce, compiled decode, and CUDAGraph enabled when validating the intended production path.
   - Do not work around failures by disabling these features unless the user explicitly asks for a temporary isolation run.

4. Validate in this order:
   - Run patch sentinel checks with `scripts/vllm_glm_rocm_fix_helper.py verify-image`.
   - Start the container and watch logs until `/v1/models` responds.
   - Run deterministic Chat Completions sanity checks.
   - Run Chat Completions tool-call checks.
   - Run Responses tool-call checks, including a request that carries `text.format.name = "tool_calling_response"`.
   - Run a short garbled-output/NaN smoke test before any long stress or TPS benchmark.

5. Tune performance only after correctness is stable.
   - Change one thing at a time.
   - After every change, rerun correctness checks before TPS.
   - If TPS does not improve or correctness regresses, roll the change back before trying another option.

## Failure Signatures

- `TileQCount` or PA MQA logits failure: PA MQA logits patch is missing or mismatched.
- Sparse MLA/indexer parameter errors: ROCm sparse MLA/indexer patch set is missing or incompatible with the vLLM version.
- TP=8 decode garbage with GQA8: AITER MLA stage2-reduce patch is missing.
- `!0,0,...`, repeated token id 0, or all-NaN logits under compiled decode: inspect custom allreduce warmup/replay first.
- `/v1/responses` returns reasoning-only output and no `function_call` for `tool_choice="required"`: Responses tools are not being converted for the GLM tool parser, or structured tool JSON schema is overriding the model-native tool syntax.
- `/v1/responses` returns `{"{city": ...}`-style arguments: GLM tool parser is not normalizing JSON-like argument-key fragments against the selected tool schema.

## References

- `references/runtime.md`: stable runtime environment and `vllm serve` arguments.
- `references/patch-map.md`: patch files, target paths, and sentinel strings.
- `references/validation.md`: curl payloads for correctness, tool-call, and streaming checks.
- `references/diagnostics.md`: low-risk commands for collecting host/container state.

## Scripts

```bash
python3 scripts/vllm_glm_rocm_fix_helper.py dockerfile-snippet
python3 scripts/vllm_glm_rocm_fix_helper.py verify-image --image "$VLLM_IMAGE"
bash scripts/vllm_glm_rocm_env_report.sh "$CONTAINER_NAME"
```

The helper scripts use placeholders and environment variables. They should not contain deployment-specific secrets or private model identifiers.
