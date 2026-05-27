# Patch Map

Default target root inside the image:

```text
/opt/python/lib/python3.13/site-packages
```

Override with `--site-packages` when the image layout differs.

## ROCm MLA and AITER Patches

| Patch source | Target under site-packages | Sentinel |
| --- | --- | --- |
| `rocm_aiter_mla_sparse.py` | `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py` | `VLLM_ROCM_AITER_MLA_SPARSE_REFERENCE_FALLBACK` |
| `ops_rocm_aiter_mla_sparse.py` | `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py` | `rocm_aiter` |
| `_aiter_ops.py` | `vllm/_aiter_ops.py` | `sparse` |
| `sparse_attn_indexer.py` | `vllm/model_executor/layers/sparse_attn_indexer.py` | `topk` |
| `mla.py` | `vllm/model_executor/layers/mla.py` | `sparse` |
| `mla_attention.py` | `vllm/model_executor/layers/attention/mla_attention.py` | `mla` |
| `pa_mqa_logits.py` | `aiter/ops/triton/attention/pa_mqa_logits.py` | `TileQCount = max(1` |
| `aiter/mla.py` | `aiter/mla.py` | `nhead in (8, 16)` |
| `aiter/ops/topk.py` | `aiter/ops/topk.py` | `experts_per_group > 32` |
| `aiter_meta/hsa/gfx942/mla/` | `aiter_meta/hsa/gfx942/mla/` | `mla_asm.csv` exists |
| `aiter_meta/csrc/py_itfs_cu/asm_mla.cu` | `aiter_meta/csrc/py_itfs_cu/asm_mla.cu` | `asm` |

## Custom Allreduce Patch

| Target under site-packages | Required behavior |
| --- | --- |
| `vllm/distributed/device_communicators/custom_all_reduce.py` | CUDAGraph warmup must return `self.all_reduce(input, registered=False)` instead of uninitialized `torch.empty_like(input)` data. |

Recommended sentinel strings:

```text
Returning uninitialized data here can poison that state
return self.all_reduce(input, registered=False)
```

## Responses Tool-Call Parser Patch

| Target under site-packages or source tree | Required behavior |
| --- | --- |
| `vllm/entrypoints/openai/responses/utils.py` | Convert Responses flat tools into Chat Completions-style tools before constructing GLM tool parsers. |
| `vllm/entrypoints/openai/responses/serving.py` | Use converted parser tools and clear `structured_outputs` for GLM native required tool calls when the SDK sends `tool_calling_response`. |
| `vllm/entrypoints/openai/parser/responses_parser.py` | Construct the parser with converted Responses tools. |
| `vllm/parser/abstract_parser.py` | For `tool_choice="required"`, try the model-specific tool parser before falling back to JSON-list parsing. Keep raw output available if the reasoning parser returns reasoning-only content. |
| `vllm/tool_parsers/glm4_moe_tool_parser.py` | Normalize JSON-like argument-key fragments, such as `{city`, against the selected tool schema for both non-streaming and streaming paths. |

Recommended sentinel strings:

```text
construct_tool_parser_tools
tool_calling_response
tool_parse_content
_normalize_arg_key
```

## Why These Are Required

- Sparse MLA/indexer patches allow pure vLLM to run GLM sparse MLA on ROCm.
- PA MQA logits patch prevents invalid tile-count behavior in the ROCm attention path.
- AITER GQA8 MLA patch handles TP=8 where per-rank decode heads can become `nhead=8`.
- Custom allreduce patch prevents CUDAGraph warmup from returning uninitialized data that can later surface as all-NaN logits or token id 0 garbage.
- Responses tool-call parser patch makes `/v1/responses` use GLM's native `<tool_call>` syntax instead of forcing JSON schema output that the GLM parser does not consume.
