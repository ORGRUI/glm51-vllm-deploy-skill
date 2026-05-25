#!/usr/bin/env python3
"""Quantize a merged GLM-5.1 checkpoint with Transformers FineGrainedFP8.

This exports a HF-compatible `FineGrainedFP8Config` checkpoint without loading
the full merged BF16 model onto GPUs. The export streams safetensors
shard-by-shard, quantizes only GLM-5.1 Linear weights that should become FP8,
and writes explicit block-128 `weight_scale_inv` tensors alongside them.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, FineGrainedFP8Config, GenerationConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated CUDA devices, e.g. cuda:0,cuda:1 or 0,1.",
    )
    return parser.parse_args()


def configure_visible_devices(raw: str | None) -> None:
    if not raw or os.environ.get("CUDA_VISIBLE_DEVICES"):
        return
    devices = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if item.startswith("cuda:"):
            item = item.split(":", 1)[1]
        devices.append(item)
    if devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)


def resolve_config_source(base_model_path: str) -> str:
    config_path = Path(base_model_path) / "config.json"
    if config_path.is_file():
        return base_model_path

    for summary_name in ("merge_summary.json", "merge_manifest.json"):
        summary_path = Path(base_model_path) / summary_name
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text())
        source = summary.get("base_model_path") or summary.get("base_repo")
        if not isinstance(source, str):
            continue
        if (Path(source) / "config.json").is_file():
            return source
    raise RuntimeError(
        f"missing readable config.json or merge summary under {base_model_path}"
    )


def materialize_merged_model_metadata(
    base_model_path: str, config_source: str
) -> list[str]:
    copied: list[str] = []
    source_root = Path(config_source)
    target_root = Path(base_model_path)
    for filename in (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "chat_template.jinja",
    ):
        source_path = source_root / filename
        if not source_path.is_file():
            continue
        target_path = target_root / filename
        if target_path.is_file():
            continue
        if target_path.is_symlink():
            target_path.unlink()
        target_path.write_bytes(source_path.read_bytes())
        copied.append(filename)
    return copied


def repair_missing_merged_shards(base_model_path: str, config_source: str) -> list[str]:
    source_root = Path(config_source)
    target_root = Path(base_model_path)
    index_path = target_root / "model.safetensors.index.json"
    if not index_path.is_file():
        return []

    summary_path = target_root / "merge_summary.json"
    if not summary_path.is_file():
        return []
    summary = json.loads(summary_path.read_text())
    unchanged_shards = set(summary.get("unchanged_shards", []))
    if not unchanged_shards:
        unchanged_shards = {
            shard_info["shard"]
            for worker in summary.get("worker_results", [])
            for shard_info in worker.get("shards", [])
            if shard_info.get("changed_params") == 0
        }

    shard_names = sorted(set(json.loads(index_path.read_text())["weight_map"].values()))
    repaired: list[str] = []
    for shard_name in shard_names:
        target_path = target_root / shard_name
        if target_path.exists():
            continue
        if shard_name not in unchanged_shards:
            raise RuntimeError(
                "missing merged shard is not marked unchanged in "
                f"merge_summary.json: {shard_name}"
            )
        source_path = source_root / shard_name
        if not source_path.exists():
            continue
        if target_path.is_symlink():
            target_path.unlink()
        target_path.symlink_to(source_path)
        repaired.append(shard_name)
    return repaired


def copy_tokenizer_artifacts(config_source: str, export_dir: Path) -> list[str]:
    source_root = Path(config_source)
    copied: list[str] = []
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "spiece.model",
        "sentencepiece.bpe.model",
    ):
        source_path = source_root / filename
        if not source_path.is_file():
            continue
        target_path = export_dir / filename
        target_path.write_bytes(source_path.read_bytes())
        copied.append(filename)
    return copied


def build_fp32_skip_modules(config) -> list[str]:
    layer_ids = range(int(config.num_hidden_layers))
    return [
        "model.embed_tokens",
        "lm_head",
        *[
            f"model.layers.{layer_idx}.self_attn.indexer.weights_proj"
            for layer_idx in layer_ids
        ],
        *[f"model.layers.{layer_idx}.self_attn.q_a_proj" for layer_idx in layer_ids],
        *[
            f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa"
            for layer_idx in layer_ids
        ],
    ]


def normalize_glm5_config_for_fp8(config) -> None:
    if (
        getattr(config, "model_type", None) == "glm_moe_dsa"
        and getattr(config, "num_experts", None) is None
    ):
        routed = getattr(config, "n_routed_experts", None)
        if routed is not None:
            config.num_experts = int(routed)


def sanitize_generation_config(generation_config) -> list[str]:
    fixed: list[str] = []
    if bool(getattr(generation_config, "do_sample", False)):
        return fixed

    sample_only_defaults = {
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 1.0,
        "min_p": None,
        "typical_p": 1.0,
        "epsilon_cutoff": 0.0,
        "eta_cutoff": 0.0,
    }
    for field_name, neutral_value in sample_only_defaults.items():
        current = getattr(generation_config, field_name, None)
        if current != neutral_value:
            setattr(generation_config, field_name, neutral_value)
            fixed.append(field_name)
    return fixed


def sanitize_generation_config_file(config_source: str, export_dir: Path) -> list[str]:
    try:
        generation_config = GenerationConfig.from_pretrained(config_source)
    except Exception:
        return []
    fixed = sanitize_generation_config(generation_config)
    generation_config.save_pretrained(export_dir)
    return fixed


def is_skipped_module(module_name: str, modules_to_not_convert: list[str]) -> bool:
    return any(
        module_name == skipped or module_name.startswith(f"{skipped}.")
        for skipped in modules_to_not_convert
    )


def is_quantizable_glm51_weight(
    tensor_name: str,
    *,
    tensor: torch.Tensor,
    modules_to_not_convert: list[str],
    weight_block_size: tuple[int, int],
) -> bool:
    if not tensor_name.endswith(".weight"):
        return False
    if tensor.ndim != 2:
        return False

    module_name = tensor_name[: -len(".weight")]
    if is_skipped_module(module_name, modules_to_not_convert):
        return False

    rows, cols = tensor.shape
    block_m, block_n = weight_block_size
    if rows % block_m != 0 or cols % block_n != 0:
        return False

    attention_suffixes = (
        ".self_attn.q_b_proj.weight",
        ".self_attn.kv_b_proj.weight",
        ".self_attn.o_proj.weight",
    )
    mlp_suffixes = (
        ".mlp.gate_proj.weight",
        ".mlp.up_proj.weight",
        ".mlp.down_proj.weight",
    )
    expert_parts = (
        ".mlp.experts.",
        ".mlp.shared_experts.",
    )
    expert_suffixes = (
        ".gate_proj.weight",
        ".up_proj.weight",
        ".down_proj.weight",
    )
    return (
        tensor_name.endswith(attention_suffixes)
        or tensor_name.endswith(mlp_suffixes)
        or (
            any(part in tensor_name for part in expert_parts)
            and tensor_name.endswith(expert_suffixes)
        )
    )


def read_safetensors_index(model_path: Path) -> dict:
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        shard_paths = sorted(model_path.glob("*.safetensors"))
        if not shard_paths:
            raise RuntimeError(f"no safetensors shards found under {model_path}")
        if len(shard_paths) != 1:
            raise RuntimeError(
                f"missing safetensors index for multi-shard checkpoint: {model_path}"
            )
        with safe_open(str(shard_paths[0]), framework="pt", device="cpu") as handle:
            return {
                "metadata": {},
                "weight_map": {key: shard_paths[0].name for key in handle.keys()},
            }
    return json.loads(index_path.read_text())


def export_streaming_fp8_checkpoint(
    *,
    base_model_path: str,
    export_dir: Path,
    config_source: str,
    config,
    quant_cfg: FineGrainedFP8Config,
    modules_to_not_convert: list[str],
) -> dict:
    source_root = Path(base_model_path)
    index_payload = read_safetensors_index(source_root)
    source_weight_map = index_payload["weight_map"]
    source_shards = sorted(set(source_weight_map.values()))
    weight_block_size = tuple(quant_cfg.weight_block_size)
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config.quantization_config = quant_cfg.to_dict()
    config.save_pretrained(export_dir)
    fixed_generation_fields = sanitize_generation_config_file(config_source, export_dir)

    export_weight_map: dict[str, str] = {}
    quantized_weight_count = 0
    copied_tensor_count = 0
    scale_tensor_count = 0
    shard_summaries: list[dict] = []

    for shard_idx, shard_name in enumerate(source_shards, start=1):
        shard_path = source_root / shard_name
        out_shard_name = shard_name
        shard_tensors: dict[str, torch.Tensor] = {}
        shard_quantized = 0
        shard_copied = 0
        with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
            for tensor_name in sorted(handle.keys()):
                tensor = handle.get_tensor(tensor_name)
                if is_quantizable_glm51_weight(
                    tensor_name,
                    tensor=tensor,
                    modules_to_not_convert=modules_to_not_convert,
                    weight_block_size=weight_block_size,
                ):
                    quantized, scale_inv = quantize_block_fp8_tensor(
                        tensor,
                        block_size=weight_block_size,
                        device=quant_device,
                    )
                    scale_name = tensor_name[: -len(".weight")] + ".weight_scale_inv"
                    shard_tensors[tensor_name] = quantized.contiguous()
                    shard_tensors[scale_name] = scale_inv.contiguous()
                    export_weight_map[tensor_name] = out_shard_name
                    export_weight_map[scale_name] = out_shard_name
                    quantized_weight_count += 1
                    scale_tensor_count += 1
                    shard_quantized += 1
                else:
                    shard_tensors[tensor_name] = tensor.contiguous()
                    export_weight_map[tensor_name] = out_shard_name
                    copied_tensor_count += 1
                    shard_copied += 1
                del tensor

        save_file(
            shard_tensors, str(export_dir / out_shard_name), metadata={"format": "pt"}
        )
        shard_summaries.append(
            {
                "shard": out_shard_name,
                "quantized_weight_count": shard_quantized,
                "copied_tensor_count": shard_copied,
            }
        )
        print(
            json.dumps(
                {
                    "phase": "streaming_shard_done",
                    "shard_index": shard_idx,
                    "shard_count": len(source_shards),
                    **shard_summaries[-1],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        del shard_tensors
        if quant_device.type == "cuda":
            torch.cuda.empty_cache()

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    export_index_payload = {
        "metadata": dict(index_payload.get("metadata") or {}),
        "weight_map": export_weight_map,
    }
    export_index_payload["metadata"]["total_size"] = total_size
    (export_dir / "model.safetensors.index.json").write_text(
        json.dumps(export_index_payload, ensure_ascii=True, indent=2)
    )
    copied_tokenizer_artifacts = copy_tokenizer_artifacts(config_source, export_dir)

    return {
        "quantized_weight_count": quantized_weight_count,
        "copied_tensor_count": copied_tensor_count,
        "scale_tensor_count": scale_tensor_count,
        "source_shard_count": len(source_shards),
        "total_safetensors_bytes": total_size,
        "generation_config_sanitized_fields": fixed_generation_fields,
        "tokenizer_artifacts_copied": copied_tokenizer_artifacts,
        "shard_summaries": shard_summaries,
    }


def quantize_block_fp8_tensor(
    tensor: torch.Tensor,
    *,
    block_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, cols = tensor.shape
    block_m, block_n = block_size
    if rows % block_m != 0 or cols % block_n != 0:
        raise RuntimeError(
            f"tensor shape {tuple(tensor.shape)} is not divisible by block size {block_size}"
        )
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    weight_fp32 = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    reshaped = weight_fp32.reshape(rows // block_m, block_m, cols // block_n, block_n)
    max_abs = reshaped.abs().amax(dim=(1, 3))
    safe_max = torch.where(max_abs > 0, max_abs, torch.ones_like(max_abs))
    scales = fp8_max / safe_max
    scales = torch.where(max_abs > 0, scales, torch.ones_like(scales))
    quantized = torch.clamp(
        reshaped * scales.unsqueeze(1).unsqueeze(3),
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=fp8_max,
    ).to(torch.float8_e4m3fn)
    quantized = quantized.reshape(rows, cols).to("cpu")
    scale_inv = scales.reciprocal().to(dtype=torch.float32, device="cpu")
    del weight_fp32, reshaped, max_abs, safe_max, scales
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return quantized, scale_inv


def rewrite_moe_expert_shards(
    *,
    base_model_path: str,
    config,
    export_dir: Path,
    weight_block_size: tuple[int, int],
) -> list[str]:
    """Compatibility no-op for exports that already stream expert scales."""
    source_root = Path(base_model_path)
    source_index_path = source_root / "model.safetensors.index.json"
    if not source_index_path.is_file():
        raise RuntimeError(f"missing source safetensors index: {source_index_path}")
    export_index_path = export_dir / "model.safetensors.index.json"
    if not export_index_path.is_file():
        raise RuntimeError(f"missing export safetensors index: {export_index_path}")

    source_weight_map = json.loads(source_index_path.read_text())["weight_map"]
    index_payload = json.loads(export_index_path.read_text())
    weight_map = index_payload["weight_map"]
    already_quantized = [
        name
        for name in weight_map
        if ".mlp.experts." in name and name.endswith(".weight_scale_inv")
    ]
    if already_quantized:
        print(
            json.dumps(
                {
                    "phase": "moe_fix_skipped",
                    "reason": "streaming_export_already_wrote_expert_scales",
                    "expert_scale_count": len(already_quantized),
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        return []
    mlp_layer_types = list(getattr(config, "mlp_layer_types", []))
    if not mlp_layer_types:
        raise RuntimeError("config.mlp_layer_types is empty")

    open_handles: dict[str, object] = {}
    fixed_shards: list[str] = []
    quant_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def load_tensor(name: str) -> torch.Tensor:
        shard_name = source_weight_map.get(name)
        if shard_name is None:
            raise KeyError(f"missing source tensor in merged checkpoint: {name}")
        handle = open_handles.get(shard_name)
        if handle is None:
            handle = safe_open(
                str(source_root / shard_name), framework="pt", device="cpu"
            )
            open_handles[shard_name] = handle
        return handle.get_tensor(name)

    experts_per_chunk = 32
    num_experts = int(getattr(config, "n_routed_experts"))
    for layer_idx, layer_type in enumerate(mlp_layer_types):
        if layer_type != "sparse":
            continue
        for chunk_start in range(0, num_experts, experts_per_chunk):
            chunk_stop = min(chunk_start + experts_per_chunk, num_experts)
            shard_name = (
                f"model-experts-fix-layer-{layer_idx:03d}-chunk-{chunk_start:03d}"
                ".safetensors"
            )
            shard_tensors: dict[str, torch.Tensor] = {}
            for expert_idx in range(chunk_start, chunk_stop):
                expert_prefix = f"model.layers.{layer_idx}.mlp.experts.{expert_idx}"
                for proj_name in ("gate_proj", "up_proj", "down_proj"):
                    weight_key = f"{expert_prefix}.{proj_name}.weight"
                    quantized, scale_inv = quantize_block_fp8_tensor(
                        load_tensor(weight_key),
                        block_size=weight_block_size,
                        device=quant_device,
                    )
                    shard_tensors[weight_key] = quantized.contiguous()
                    shard_tensors[f"{weight_key}_scale_inv"] = scale_inv.contiguous()
                    weight_map[weight_key] = shard_name
                    weight_map[f"{weight_key}_scale_inv"] = shard_name
            save_file(
                shard_tensors, str(export_dir / shard_name), metadata={"format": "pt"}
            )
            fixed_shards.append(shard_name)
            del shard_tensors
            if quant_device.type == "cuda":
                torch.cuda.empty_cache()

    total_size = sum(path.stat().st_size for path in export_dir.glob("*.safetensors"))
    index_payload.setdefault("metadata", {})["total_size"] = total_size
    export_index_path.write_text(json.dumps(index_payload, ensure_ascii=True, indent=2))
    print(
        json.dumps(
            {
                "phase": "moe_fix_done",
                "fixed_shard_count": len(fixed_shards),
                "fixed_sparse_layer_count": sum(
                    1 for x in mlp_layer_types if x == "sparse"
                ),
                "total_safetensors_bytes": total_size,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return fixed_shards


def main() -> None:
    args = parse_args()
    configure_visible_devices(args.devices)
    export_dir = Path(args.export_dir)
    if export_dir.exists() and any(export_dir.iterdir()):
        raise RuntimeError(f"export_dir already exists and is non-empty: {export_dir}")
    export_dir.mkdir(parents=True, exist_ok=True)

    cuda_count = torch.cuda.device_count()
    if cuda_count < 1:
        raise RuntimeError("FP8 quantization requires at least one visible CUDA device")

    config_source = resolve_config_source(args.base_model_path)
    print(
        json.dumps(
            {"phase": "config_source", "config_source": config_source},
            ensure_ascii=True,
        ),
        flush=True,
    )
    copied_metadata = materialize_merged_model_metadata(
        args.base_model_path, config_source
    )
    if copied_metadata:
        print(
            json.dumps(
                {"phase": "metadata_repaired", "copied_files": copied_metadata},
                ensure_ascii=True,
            ),
            flush=True,
        )
    repaired_shards = repair_missing_merged_shards(args.base_model_path, config_source)
    if repaired_shards:
        print(
            json.dumps(
                {
                    "phase": "shards_repaired",
                    "repaired_shards_count": len(repaired_shards),
                    "repaired_shards_sample": repaired_shards[:8],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    config = AutoConfig.from_pretrained(
        config_source,
        cache_dir=args.cache_dir,
        trust_remote_code=args.trust_remote_code,
    )
    normalize_glm5_config_for_fp8(config)
    modules_to_not_convert = build_fp32_skip_modules(config)

    quant_cfg = FineGrainedFP8Config(modules_to_not_convert=modules_to_not_convert)
    print(
        json.dumps(
            {
                "phase": "streaming_export_start",
                "base_model_path": args.base_model_path,
                "export_dir": str(export_dir),
                "cuda_count": cuda_count,
                "quant_method": "fp8",
                "quantizer": "FineGrainedFP8Config",
                "strategy": "safetensors_streaming",
                "modules_to_not_convert_count": len(modules_to_not_convert),
                "modules_to_not_convert_sample": modules_to_not_convert[:4],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    export_summary = export_streaming_fp8_checkpoint(
        base_model_path=args.base_model_path,
        export_dir=export_dir,
        config_source=config_source,
        config=config,
        quant_cfg=quant_cfg,
        modules_to_not_convert=modules_to_not_convert,
    )
    print(
        json.dumps(
            {
                "phase": "streaming_export_done",
                **{
                    key: value
                    for key, value in export_summary.items()
                    if key != "shard_summaries"
                },
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    if export_summary["generation_config_sanitized_fields"]:
        print(
            json.dumps(
                {
                    "phase": "generation_config_sanitized",
                    "fields": export_summary["generation_config_sanitized_fields"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    if export_summary["tokenizer_artifacts_copied"]:
        print(
            json.dumps(
                {
                    "phase": "tokenizer_artifacts_copied",
                    "files": export_summary["tokenizer_artifacts_copied"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    fixed_shards = rewrite_moe_expert_shards(
        base_model_path=args.base_model_path,
        config=config,
        export_dir=export_dir,
        weight_block_size=tuple(quant_cfg.weight_block_size),
    )

    meta = {
        "base_model_path": args.base_model_path,
        "export_dir": str(export_dir),
        "cuda_count": cuda_count,
        "quantization_method": "fp8",
        "is_quantized": True,
        "quantizer": "FineGrainedFP8Config",
        "strategy": "safetensors_streaming",
        "modules_to_not_convert_count": len(modules_to_not_convert),
        "modules_to_not_convert": modules_to_not_convert,
        **{
            key: value
            for key, value in export_summary.items()
            if key != "shard_summaries"
        },
        "moe_fix_shard_count": len(fixed_shards),
    }
    (export_dir / "fp8_quant_meta.json").write_text(
        json.dumps(meta, ensure_ascii=True, indent=2)
    )
    print(json.dumps({"phase": "done", **meta}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
