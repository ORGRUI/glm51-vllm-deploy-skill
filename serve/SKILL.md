---
name: glm51-serve
description: Serve an existing GLM-5.1 FP8 artifact with vLLM + ATOM, capture proxy, observability, Caddy, smoke test, and benchmark entrypoints. Use when only runtime serving code or launch settings changed.
---

# GLM-5.1 Serve

## Contract

This skill consumes an already quantized FP8 artifact. It writes the serve env, starts vLLM + ATOM, starts the capture proxy, observability, and Caddy, then runs smoke checks. It does not fetch, merge, or quantize model weights.

Set the host, public endpoint, and artifact identity before running:

```bash
export SSH_HOST=vmadmin@<ip>
export SSH_PASSWORD='<password-if-needed>'
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme
export PUBLIC_BASE_URL=http://<ip>:7777/v1
export RUN_SLUG=<stable-run-name>
```

Use the default staged model path derived from `RUN_SLUG`, or point serving at an existing artifact:

```bash
export MODEL_PATH=/local_nvme/amd_profiling/<run>/serve/<run>-merged-fp8-finegrained-block128
export ENV_FILE=/data/amd_profiling/configs/vllm_<run>_atom_64k_seq2.env
```

## Entry

From this skill directory:

```bash
./scripts/run_serve.sh
```

This delegates to `../shared/scripts/run_stage.sh serve-all`, which runs:

```text
sync-scripts -> write-serve-env -> serve-backend -> serve-proxy -> serve-observability -> serve-caddy -> smoke
```

For runtime-only changes, rerun `./scripts/run_serve.sh` with the same `RUN_SLUG` and `MODEL_PATH`.
