# GLM-5.1 Deployment Skills

This branch exposes three stage-specific GLM deployment skills:

- `merge/`: GLM-5.1 source resolution, adapter preparation, BF16 merge, and merged shard validation. Produces the reusable `BF16_OUT` artifact.
- `quant/`: consumes `BF16_OUT`, writes the FP8 block-128 artifact, and stages it as `LOCAL_MODEL_PATH` / `DURABLE_MODEL_PATH`.
- `serve/`: consumes `MODEL_PATH`, writes the vLLM + ATOM serve env, restarts backend/proxy/observability/Caddy, and runs smoke tests.
- `azure-amd-deploy-env/`: self-contained Azure AMD MI300X deployment environment constraints, command skeletons, and judgment standards for durable/ephemeral disk policy and local NVMe setup.

The shared runner and helper scripts live under `shared/` as internal
implementation, not as a public skill entrypoint. Each stage skill has its own
`SKILL.md` and provides a clear resume entrypoint:

```bash
cd merge && ./scripts/run_merge.sh
cd ../quant && ./scripts/run_quant.sh
cd ../serve && ./scripts/run_serve.sh
```

Set `RUN_SLUG` to reuse the default intermediate paths, or set `BF16_OUT`,
`FP8_OUT`, `LOCAL_MODEL_PATH`, `DURABLE_MODEL_PATH`, and `MODEL_PATH`
explicitly when reusing artifacts from a previous run.

Each completed stage writes a compact `stage_manifest.json` beside its reusable
artifact, plus a serve manifest next to the generated env file. The manifest
records the repo commit for traceability and the stage-specific hash used for
reuse decisions. Use `plan` or `doctor` through any stage wrapper to compare the
current merge / quant / serve fingerprints and parameters against existing
manifests:

```bash
./merge/scripts/run_merge.sh plan
```

The decision is conservative: missing manifests, changed stage hashes, changed
upstream manifest hashes, or changed relevant parameters rerun from the earliest
uncertain stage.
