# GLM-5.1 Deployment Skills

This branch contains two independent agent skills:

- `merge-quant-serve/`: GLM-5.1 LoRA source resolution, merge, corrected official-partial FP8 block-128 quantization, vLLM + ATOM serving, proxy, smoke test, and benchmark workflow.
- `azure-amd-deploy-env/`: Azure AMD MI300X deployment environment constraints and host preparation workflow, including durable/ephemeral disk policy and local NVMe setup.

Each skill has its own `SKILL.md` and executable `scripts/` entries.
