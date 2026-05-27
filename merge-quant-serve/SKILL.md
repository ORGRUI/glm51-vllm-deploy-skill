---
name: merge-quant-serve
description: Compatibility umbrella for the GLM-5.1 merge, quant, and serve pipeline. Use when asked for a full one-command deployment or for the underlying shared stage runner; prefer the stage-specific merge/, quant/, and serve/ skills when only one part changed.
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

When resuming from quant or serve without the original source URL, set `RUN_SLUG`
or explicit artifact paths instead:

```bash
export RUN_SLUG=<stable-run-name>
export BF16_OUT=/local_nvme/amd_profiling/<run>/models/<run>-merged
export FP8_OUT=/local_nvme/amd_profiling/<run>/models/<run>-merged-fp8-finegrained-block128
export MODEL_PATH=/local_nvme/amd_profiling/<run>/serve/<run>-merged-fp8-finegrained-block128
```

Useful defaults are built into `scripts/run_stage.sh`: `BASE_REPO=zai-org/GLM-5.1`, `DOCKER_IMAGE=rocm/atom-dev:vllm-latest`, TP=8, 64k context, seq2, batch tokens 65536, GPU memory utilization 0.60, `--async-scheduling`, `FULL_AND_PIECEWISE`, prefix caching enabled, `--block-size=1`, MTP off by default, ROCm recipe env `VLLM_ROCM_USE_AITER=1`, `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4`, `VLLM_ROCM_USE_AITER_RMSNORM=0`, merge untouched shards as symlinks, merge on `cuda:0..cuda:7` with `MERGE_JOBS=8`, quantization on `cuda:0..cuda:7` with `QUANT_WORKERS=8`, capture proxy `max_tokens=8192` when omitted, no default temperature override, request-side `chat_template_kwargs.enable_thinking=false` when omitted, and single-port observability enabled by default.

MTP is an explicit canary path controlled by `VLLM_ENABLE_MTP=1` and `VLLM_SPECULATIVE_CONFIG='{"method":"mtp","num_speculative_tokens":3}'`, matching Ajith's GLM-5.1 AMD recipe. Leave `VLLM_ENABLE_MTP` unset or set it to `0` for the accepted public deployment path, or set `VLLM_EXTRA_ARGS` explicitly to fully override the generated vLLM extra args. Do not switch the public endpoint to MTP until the canary reaches API readiness and passes E2E.

The recipe-first reference lives in `../ajith-vllm-recipe/`. Use it when the task is to reproduce the official native FP8 flow rather than this repo's Tinker adapter merge, quantize, capture proxy, and observability pipeline.

Temperature passthrough is the default. Leave `FORCE_TEMPERATURE` unset or empty to forward the client's `temperature` unchanged; set `FORCE_TEMPERATURE=<float>` only when a deployment intentionally needs the capture proxy to override request temperatures.

## Pinned Runtime Versions

The MTP canary runtime is pinned and must be preserved in deployment summaries when MTP is enabled:

- vLLM package: `0.19.1rc1.dev90+g5af684c31`.
- vLLM GLM tool parser patch: PR 39253, `[Bugfix] Fix GLM tool parser streaming with MTP or stream interval`, `refs/pull/39253/head`, commit `920af3c7a1b29847fb237fa9a9aaedacf48e8bbd`.
- ATOM repo: `https://github.com/san-tian/ATOM.git`, branch `fix/mtp-arange-buffer-token-capacity`, commit `d5f9a49bb2b6f3e82fda35e411d3cd962c19bf15`.

Install and patch order:

1. Start from the serving image `rocm/atom-dev:vllm-latest`.
2. Ensure the image contains vLLM `0.19.1rc1.dev90+g5af684c31`.
3. Apply vLLM PR 39253's GLM tool parser patch on top of that vLLM package.
4. Clone ATOM from `https://github.com/san-tian/ATOM.git`, fetch `fix/mtp-arange-buffer-token-capacity`, and checkout `d5f9a49bb2b6f3e82fda35e411d3cd962c19bf15`.
5. Write the serve env with `VLLM_SOURCE_DIR=$ATOM_SOURCE_DIR`, MTP disabled for public deploys, `--block-size=1`, and the ROCm AITER env values above. For an MTP canary, set `VLLM_ENABLE_MTP=1` and record the speculative config above.

`scripts/run_stage.sh write-serve-env` records `VLLM_EXPECTED_VERSION`, `VLLM_TOOL_PARSER_PATCH_PR`, `VLLM_TOOL_PARSER_PATCH_REF`, and `VLLM_TOOL_PARSER_PATCH_COMMIT`. `scripts/serve_vllm_glm51.sh` writes those values into `*.server_argv.json` and, before starting `vllm serve`, fails fast if the container vLLM version does not match or the PR 39253 parser marker is absent.

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
./scripts/run_stage.sh serve-observability
./scripts/run_stage.sh serve-caddy
./scripts/run_stage.sh smoke
./scripts/run_stage.sh benchmark
```

Stage-group entrypoints:

```bash
./scripts/run_stage.sh merge-all
./scripts/run_stage.sh quant-all
./scripts/run_stage.sh serve-all
```

For a full run after confirmation:

```bash
./scripts/run_stage.sh deploy-all
```

If only `TINKER_URL` is set, `deploy-all` resolves it first and exports the resolved `OSS_URL` for later stages.

For independent versioning and cleaner resumes, prefer the split skills:

- `../merge/`: source resolution through reusable BF16 `BF16_OUT`.
- `../quant/`: `BF16_OUT` through reusable FP8 `FP8_OUT` and staged `LOCAL_MODEL_PATH`.
- `../serve/`: `MODEL_PATH` through runtime restart and smoke checks.

Treat `serve-backend -> serve-proxy -> serve-observability -> serve-caddy -> smoke` as the standard service restart sequence. `serve-backend` only replaces the vLLM backend container; `serve-observability` starts Prometheus and Grafana bound to localhost; `serve-caddy` always restarts the public `:7777` Caddy container so a runtime restart leaves the public endpoint in the expected state.

## Default Single-Port Observability

`deploy-all` now starts vLLM/ATOM, the capture proxy, Prometheus, Grafana, and Caddy by default. Only Caddy should listen on the external interface:

- `PUBLIC_BASE_URL=http://<ip>:7777/v1` for OpenAI-compatible clients.
- `PUBLIC_ROOT_URL=http://<ip>:7777` for generated Grafana and Prometheus subpath URLs. If omitted, it is derived from `PUBLIC_BASE_URL` when possible.
- `/v1/*` routes through the capture proxy to vLLM/ATOM.
- `/metrics` routes to the vLLM/ATOM backend metrics endpoint.
- `/grafana/` routes to Grafana, provisioned with the bundled `ATOM / ATOM vLLM Overview` dashboard.
- `/prometheus/` routes to Prometheus.

Do not publish `7791`/`7788`, `18080`, `9090`, or `3000` externally in the default flow. Prometheus and Grafana are launched with host networking but bind to `127.0.0.1`; Caddy is the only public listener on `:7777`.

Use `OBSERVABILITY_ENABLED=0` only when the user explicitly wants inference without the dashboard stack. Useful overrides are `PROMETHEUS_IMAGE`, `GRAFANA_IMAGE`, `CADDY_IMAGE`, `GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`, `PROMETHEUS_PORT`, `GRAFANA_PORT`, and `VLLM_SCRAPE_INTERVAL`.

The default implementation intentionally uses `docker run` rather than requiring `docker compose`. On the verified target host, Compose v2 was missing and `vmadmin` could not access `/var/run/docker.sock` directly. The serve scripts therefore:

- try plain `docker` first;
- fall back to `sudo -S docker` when the socket is not accessible;
- use `SUDO_PASSWORD` only if the remote account requires a password for sudo;
- keep the stack on host networking with localhost-bound internals so the external exposure remains `7777` only.

During preflight, record whether `docker compose version` is available and whether `docker ps` works without sudo. If Compose is absent, continue with the bundled `docker run` path. If both direct Docker and non-interactive sudo fail, stop and ask for Docker group membership, passwordless sudo for Docker, or a usable `SUDO_PASSWORD`.

## Bundled Script Entrypoints

- `scripts/run_stage.sh`: local orchestrator for one-command stages over SSH.
- `scripts/resolve_model_source.py`: validate `OSS_URL` or convert `TINKER_URL` to signed HTTP(S) archive.
- `scripts/prepare_oss_lora_source.py`: download/extract archive and produce a PEFT adapter.
- `scripts/prefetch_glm51_base.py`: prefetch GLM-5.1 base shards into local NVMe HF cache.
- `scripts/merge_glm51_lora_sharded.py`: merge LoRA into BF16 model shards, expand sparse expert representatives, reconstruct lm_head shards when possible, and emit `merge_summary.json`.
- `scripts/validate_and_repair_safetensors_shards.py`: validate merged shard integrity and repair dangling links.
- `scripts/quantize_glm51_fp8_block128.py`: produce attachment-style `FineGrainedFP8Config` block-128 artifact by streaming safetensors shards and writing `weight_scale_inv` tensors directly, including MoE experts. Use `--devices` and `--workers` to quantize shards concurrently across multiple GPUs.
- `scripts/serve_vllm_glm51.sh`: launch vLLM + ATOM backend.
- `scripts/capture_proxy.py` and `scripts/serve_capture_proxy.sh`: OpenAI-compatible capture/rewrite proxy.
- `scripts/serve_observability.sh`: launch Prometheus and Grafana on localhost with the bundled vLLM dashboard.
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
  -d '{"model":"'"${SERVED_MODEL_NAME:-${RUN_SLUG}-fp8-atom}"'","messages":[{"role":"user","content":"请直接给最终答案。问题：1+1等于几？"}],"temperature":0.7}'
tail -1 "$REMOTE_ROOT/request_captures/index.jsonl" | python3 -c 'import json,pathlib,sys; row=json.loads(sys.stdin.read()); body=json.loads(pathlib.Path(row["forwarded_body_path"]).read_text()); assert body["max_tokens"] == 8192; assert body["temperature"] == 0.7; assert body["chat_template_kwargs"]["enable_thinking"] is False; print("proxy defaults ok")'
```

Also verify the single-port observability routes when `OBSERVABILITY_ENABLED` is not disabled:

```bash
curl -fsS "${PUBLIC_ROOT_URL%/}/metrics" | head
curl -fsS "${PUBLIC_ROOT_URL%/}/prometheus/-/ready"
curl -fsS "${PUBLIC_ROOT_URL%/}/grafana/api/health"
```

From the target host, Prometheus should show `up{job="vllm"} == 1`, and Grafana should list the `ATOM / ATOM vLLM Overview` dashboard.

Record launch truth in the env file, wrapper logs, and `*.server_argv.json`.
