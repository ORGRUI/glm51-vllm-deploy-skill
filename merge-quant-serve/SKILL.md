---
name: merge-quant-serve
description: Deploy a GLM-5.1 LoRA from a Tinker checkpoint URL or signed OSS archive by resolving the source, preparing PEFT, merging BF16 shards, quantizing corrected official-partial FP8 block-128, and serving through vLLM + ATOM with capture proxy and Caddy. Use when asked for merge/quant/serve, GLM-5.1 FP8 block-128 deployment, OPE-13 corrected official-partial quantization, or one-command staged deployment.
---

# Merge Quant Serve

## Contract

This skill is host-agnostic. Do not hard-code host IPs, passwords, model URLs, tokens, or signed OSS query strings. Collect missing values first, show the derived parameter table, and get confirmation before modifying a remote host.

Accepted model sources:

- `OSS_URL`: signed or public HTTP(S) archive containing a PEFT adapter or raw Tinker checkpoint.
- `TINKER_URL`: `tinker://...` checkpoint URL, resolved locally with `scripts/resolve_model_source.py` through `GPU_LEASE_BASE_URL + GPU_LEASE_API_KEY` or `TRANSFER_JOBS_ENDPOINT`.

Do not proceed to SSH, download, merge, quantize, or serve until the final `OSS_URL` starts with `http://` or `https://`.

## Required Inputs

Set these as environment variables before running stage scripts:

```bash
export SSH_HOST=vmadmin@<ip>
export SSH_PASSWORD='<password-if-needed>'
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme
export PUBLIC_BASE_URL=http://<ip>:7777/v1
export OSS_URL='<signed-or-public-http-archive>'
# or: export TINKER_URL='tinker://...'
```

Useful defaults are built into `scripts/run_stage.sh`: `BASE_REPO=zai-org/GLM-5.1`, `DOCKER_IMAGE=rocm/atom-dev:vllm-latest`, TP=8, 64k context, seq2, batch tokens 65536, GPU memory utilization 0.60, `--async-scheduling`, `FULL_AND_PIECEWISE`, and prefix caching.

## One-Command Stages

From this skill directory:

```bash
./scripts/run_stage.sh derive
./scripts/run_stage.sh resolve-source
./scripts/run_stage.sh sync-scripts
./scripts/run_stage.sh preflight
./scripts/run_stage.sh prepare-env
./scripts/run_stage.sh fetch-source
./scripts/run_stage.sh prefetch-base
./scripts/run_stage.sh merge
./scripts/run_stage.sh validate-bf16
./scripts/run_stage.sh quantize
./scripts/run_stage.sh stage-model
./scripts/run_stage.sh write-serve-env
./scripts/run_stage.sh serve-backend
./scripts/run_stage.sh serve-proxy
./scripts/run_stage.sh serve-caddy
./scripts/run_stage.sh smoke
./scripts/run_stage.sh benchmark
```

For a full run after confirmation:

```bash
./scripts/run_stage.sh deploy-all
```

If only `TINKER_URL` is set, `deploy-all` resolves it first and exports the resolved `OSS_URL` for later stages.

## Bundled Script Entrypoints

- `scripts/run_stage.sh`: local orchestrator for one-command stages over SSH.
- `scripts/resolve_model_source.py`: validate `OSS_URL` or convert `TINKER_URL` to signed HTTP(S) archive.
- `scripts/prepare_oss_lora_source.py`: download/extract archive and produce a PEFT adapter.
- `scripts/prefetch_glm51_base.py`: prefetch GLM-5.1 base shards into local NVMe HF cache.
- `scripts/merge_glm51_lora_sharded.py`: merge LoRA into BF16 model shards.
- `scripts/validate_and_repair_safetensors_shards.py`: validate merged shard integrity and repair dangling links.
- `scripts/quantize_glm51_fp8_block128.py`: produce corrected official-partial FP8 block-128 artifact.
- `scripts/patch_glm51_fp8_qabf16.py`: diagnostic-only q_a BF16 patch; do not use in the default path.
- `scripts/serve_vllm_glm51.sh`: launch vLLM + ATOM backend.
- `scripts/capture_proxy.py` and `scripts/serve_capture_proxy.sh`: OpenAI-compatible capture/rewrite proxy.
- `scripts/serve_caddy_proxy.sh`: public `:7777` Caddy proxy.
- `scripts/benchmark_vllm_glm51.sh`: throughput benchmark wrapper.

## Quantization Contract

The default artifact is `${RUN_SLUG}-merged-fp8-block128-official-partial`. It follows the official GLM-5.1 FP8 coverage: attention projection linears, MLP linears, and MoE expert linears are FP8 e4m3 block-128 with `weight_scale_inv`; embeddings, norms, routers/gates, `lm_head`, and indexer compatibility modules stay unconverted.

Keep `self_attn.q_a_proj.weight` and `self_attn.kv_a_proj_with_mqa.weight` in the FP8 contract. Expected representative shapes:

```text
q_a_proj.weight              FP8 [2048, 6144], scale_inv [16, 48]
kv_a_proj_with_mqa.weight    FP8 [576, 6144],  scale_inv [5, 48]
```

The older `*-qabf16` artifact is diagnostic only and must not be the default deployment artifact.

## Runtime Checks

After serving, require:

```bash
curl -fsS "$PUBLIC_BASE_URL/models"
curl -fsS -H 'Content-Type: application/json' "$PUBLIC_BASE_URL/chat/completions" \
  -d '{"model":"'"${SERVED_MODEL_NAME:-${RUN_SLUG}-fp8-atom}"'","messages":[{"role":"user","content":"请直接给最终答案，不要展示推理过程。问题：1+1等于几？"}],"max_tokens":64,"temperature":0}'
```

Record launch truth in the env file, wrapper logs, and `*.server_argv.json`.
