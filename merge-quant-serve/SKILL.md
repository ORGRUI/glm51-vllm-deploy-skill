---
name: merge-quant-serve
description: Deploy a GLM-5.1 LoRA from a Tinker checkpoint URL or signed OSS archive by resolving the source, preparing PEFT, merging BF16 shards, quantizing with Transformers FineGrainedFP8 block-128 plus explicit MoE expert rewrite, and serving through vLLM + ATOM with capture proxy and Caddy. Use when asked for merge/quant/serve, GLM-5.1 FP8 block-128 deployment, OPE-13 attachment-style quantization, or one-command staged deployment.
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

Useful defaults are built into `scripts/run_stage.sh`: `BASE_REPO=zai-org/GLM-5.1`, `DOCKER_IMAGE=rocm/atom-dev:vllm-latest`, TP=8, 64k context, seq2, batch tokens 65536, GPU memory utilization 0.60, `--async-scheduling`, `FULL_AND_PIECEWISE`, prefix caching enabled, merge untouched shards as symlinks, capture proxy `max_tokens=8192` when omitted, and request-side `chat_template_kwargs.enable_thinking=false` when omitted.

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

Treat `serve-backend -> serve-proxy -> serve-caddy -> smoke` as the standard service restart sequence. `serve-backend` only replaces the vLLM backend container; `serve-caddy` always restarts the public `:7777` Caddy container so a runtime restart leaves the public endpoint in the expected state.

## Bundled Script Entrypoints

- `scripts/run_stage.sh`: local orchestrator for one-command stages over SSH.
- `scripts/resolve_model_source.py`: validate `OSS_URL` or convert `TINKER_URL` to signed HTTP(S) archive.
- `scripts/prepare_oss_lora_source.py`: download/extract archive and produce a PEFT adapter.
- `scripts/prefetch_glm51_base.py`: prefetch GLM-5.1 base shards into local NVMe HF cache.
- `scripts/merge_glm51_lora_sharded.py`: merge LoRA into BF16 model shards, expand sparse expert representatives, reconstruct lm_head shards when possible, and emit `merge_summary.json`.
- `scripts/validate_and_repair_safetensors_shards.py`: validate merged shard integrity and repair dangling links.
- `scripts/quantize_glm51_fp8_block128.py`: produce attachment-style `FineGrainedFP8Config` block-128 artifact by streaming safetensors shards and writing `weight_scale_inv` tensors directly, including MoE experts.
- `scripts/serve_vllm_glm51.sh`: launch vLLM + ATOM backend.
- `scripts/capture_proxy.py` and `scripts/serve_capture_proxy.sh`: OpenAI-compatible capture/rewrite proxy.
- `scripts/serve_caddy_proxy.sh`: public `:7777` Caddy proxy.
- `scripts/benchmark_vllm_glm51.sh`: throughput benchmark wrapper.

## Quantization Contract

The default artifact is `${RUN_SLUG}-merged-fp8-finegrained-block128`. It follows the attachment flow semantics without constructing the full BF16 model in GPU memory: the quantizer streams source safetensors shards, writes block-128 FP8 e4m3 Linear weights plus `weight_scale_inv`, and leaves embeddings, norms, routers/gates, `lm_head`, q_a / kv_a compatibility modules, and indexer compatibility modules unconverted. Sparse MoE expert `gate_proj` / `up_proj` / `down_proj` weights are emitted with their scale tensors during the same streaming pass.

Keep `self_attn.indexer.{wq_b,wk,weights_proj}.weight` in BF16 in this attachment-style path. If `indexer.wq_b` or `indexer.wk` is absent from `modules_to_not_convert`, ATOM/vLLM will allocate FP8 `weight_scale` parameters for those layers while the streamed checkpoint has no matching scale tensors, leaving the scale params at init values and causing load warnings such as `indexer.{wk,wq_b}.weight_scale` not loaded.

Keep `self_attn.q_a_proj.weight` and `self_attn.kv_a_proj_with_mqa.weight` in BF16 in this attachment-style path. Expected representative unconverted shapes:

```text
q_a_proj.weight              BF16 [2048, 6144]
kv_a_proj_with_mqa.weight    BF16 [576, 6144]
```

## Runtime Checks

After serving, require:

```bash
curl -fsS "$PUBLIC_BASE_URL/models"
curl -fsS -H 'Content-Type: application/json' "$PUBLIC_BASE_URL/chat/completions" \
  -d '{"model":"'"${SERVED_MODEL_NAME:-${RUN_SLUG}-fp8-atom}"'","messages":[{"role":"user","content":"请直接给最终答案，不要展示推理过程。问题：1+1等于几？"}],"max_tokens":64,"temperature":0}'
```

Minimal default rewrite check:

```bash
curl -fsS -H 'Content-Type: application/json' "$PUBLIC_BASE_URL/chat/completions" \
  -d '{"model":"'"${SERVED_MODEL_NAME:-${RUN_SLUG}-fp8-atom}"'","messages":[{"role":"user","content":"请直接给最终答案。问题：1+1等于几？"}],"temperature":0}'
tail -1 "$REMOTE_ROOT/request_captures/index.jsonl" | python3 -c 'import json,pathlib,sys; row=json.loads(sys.stdin.read()); body=json.loads(pathlib.Path(row["forwarded_body_path"]).read_text()); assert body["max_tokens"] == 8192; assert body["chat_template_kwargs"]["enable_thinking"] is False; print("proxy defaults ok")'
```

Record launch truth in the env file, wrapper logs, and `*.server_argv.json`.
