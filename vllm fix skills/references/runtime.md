# Runtime

## Stable Environment

Use these as a correctness-first baseline for pure vLLM on ROCm:

```bash
-e VLLM_TARGET_DEVICE=rocm
-e VLLM_ROCM_USE_AITER=1
-e VLLM_ROCM_USE_AITER_LINEAR=0
-e VLLM_ROCM_USE_AITER_MOE=1
-e VLLM_ROCM_USE_AITER_RMSNORM=0
-e VLLM_ROCM_USE_AITER_FP8BMM=0
-e VLLM_ROCM_USE_AITER_TRITON_GEMM=0
-e VLLM_ROCM_AITER_MLA_SPARSE_REFERENCE_FALLBACK=1
```

## Stable vLLM Arguments

```bash
vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT:-7804}" \
  --served-model-name "${SERVED_MODEL_NAME:-glm51-fp8-vllm}" \
  --tensor-parallel-size "${TP_SIZE:-8}" \
  --dtype bfloat16 \
  --trust-remote-code \
  --max-model-len "${MAX_MODEL_LEN:-65536}" \
  --max-num-seqs "${MAX_NUM_SEQS:-16}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-32768}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.70}" \
  --generation-config vllm \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --chat-template-content-format string \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'
```

## Correctness Notes

- Keep custom allreduce enabled when validating the production path. The fix should repair warmup/replay behavior rather than avoiding the path.
- Keep MLA enabled when validating the production path. Disabling MLA can hide the original compatibility issue.
- Keep compiled decode and CUDAGraph enabled when validating NaN or garbled-output fixes.
- Use `kv_cache_dtype=auto` unless the model and kernel path have already been validated with FP8 KV cache.
- Use deterministic short requests for correctness before running long prompts, high concurrency, or TPS tests.
