---
name: glm51-quant
description: Quantize reusable GLM-5.1 BF16 merged shards into the FP8 block-128 artifact and stage it for serving. Use when quantization code or quant settings changed, or when resuming after merge.
---

# GLM-5.1 Quant

## Contract

This skill consumes an existing BF16 merged checkpoint and produces the reusable FP8 `FineGrainedFP8Config` block-128 artifact. It does not fetch the source adapter, merge LoRA weights, or start serving.

Set the host and artifact inputs before running:

```bash
export SSH_HOST=vmadmin@<ip>
export SSH_PASSWORD='<password-if-needed>'
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme
export RUN_SLUG=<stable-run-name>
```

Use the default artifact paths derived from `RUN_SLUG`, or override them explicitly:

```bash
export BF16_OUT=/local_nvme/amd_profiling/<run>/models/<run>-merged
export FP8_OUT=/local_nvme/amd_profiling/<run>/models/<run>-merged-fp8-finegrained-block128
export LOCAL_MODEL_PATH=/local_nvme/amd_profiling/<run>/serve/<run>-merged-fp8-finegrained-block128
export DURABLE_MODEL_PATH=/data/amd_profiling/models/<run>-merged-fp8-finegrained-block128
```

## Entry

From this skill directory:

```bash
./scripts/run_quant.sh
```

This delegates to `../merge-quant-serve/scripts/run_stage.sh quant-all`, which runs:

```text
sync-scripts -> preflight -> prepare-env -> quantize -> stage-model
```

The handoff artifact for the serve skill is `MODEL_PATH`, which defaults to `LOCAL_MODEL_PATH`.
