---
name: glm51-merge
description: Merge a GLM-5.1 LoRA source into reusable BF16 shards. Use when a Tinker/OSS source changed, when a BF16 merged checkpoint must be regenerated, or when resuming the deployment pipeline from merge.
---

# GLM-5.1 Merge

## Contract

This skill owns source resolution, remote environment preparation, adapter extraction, base prefetch, LoRA merge, and BF16 shard validation. It stops at the reusable BF16 artifact and does not quantize or serve.

Set the source and host inputs before running:

```bash
export SSH_HOST=vmadmin@<ip>
export SSH_PASSWORD='<password-if-needed>'
export REMOTE_ROOT=/data/amd_profiling
export LOCAL_SCRATCH_MOUNT=/local_nvme
export OSS_URL='<signed-or-public-http-archive>'
# or: export TINKER_URL='tinker://...'
```

Optional resume/versioning inputs:

```bash
export RUN_SLUG=<stable-run-name>
export BF16_OUT=/local_nvme/amd_profiling/<run>/models/<run>-merged
```

## Entry

From this skill directory:

```bash
./scripts/run_merge.sh
```

This delegates to `../merge-quant-serve/scripts/run_stage.sh merge-all`, which runs:

```text
sync-scripts -> preflight -> prepare-env -> fetch-source -> prefetch-base -> merge -> validate-bf16
```

For inspection without remote changes:

```bash
../merge-quant-serve/scripts/run_stage.sh derive
```

The handoff artifact for the quant skill is `BF16_OUT`.
