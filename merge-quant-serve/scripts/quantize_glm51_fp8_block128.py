#!/usr/bin/env python3
"""Convert merged GLM-5.1 shards to ATOM-compatible FP8 block weights.

The generic compressed-tensors FP8_DYNAMIC path produces per-channel
`weight_scale` tensors and a `compressed-tensors` quantization config. ATOM's
GLM-5 recipe expects the GLM-5.1-FP8 layout instead: `quant_method=fp8`,
`weight_block_size=[128, 128]`, and per-block `weight_scale_inv` tensors.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable

import torch
import tqdm
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file

DEFAULT_SRC = "/data2/amd_profiling/models/glm51-sft_aug_v1_merged"
DEFAULT_OUT = "/data2/amd_profiling/models/glm51-sft_aug_v1_merged-fp8-block128"
DEFAULT_FP8_REPO = "zai-org/GLM-5.1-FP8"
BLOCK = 128
FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
QUANTIZED_WEIGHT_PATTERNS = [
    re.compile(r"^model\.layers\.\d+\.mlp\.down_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.gate_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.up_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.shared_experts\.down_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.shared_experts\.gate_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.shared_experts\.up_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.down_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.gate_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.mlp\.experts\.\d+\.up_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.indexer\.wk\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.indexer\.wq_b\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.q_a_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.kv_b_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.kv_a_proj_with_mqa\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$"),
    re.compile(r"^model\.layers\.\d+\.self_attn\.q_b_proj\.weight$"),
]


def is_temp_model_artifact(path: Path) -> bool:
    name = path.name
    return name.endswith(".safetensors.tmp") and (
        name.startswith(".model-") or name.startswith("model-")
    )


def cleanup_temp_model_artifacts(model_dir: Path) -> list[str]:
    removed: list[str] = []
    for path in model_dir.iterdir():
        if not is_temp_model_artifact(path):
            continue
        if path.is_dir() and not path.is_symlink():
            continue
        path.unlink()
        removed.append(path.name)
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=DEFAULT_SRC)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-shards", type=int, default=None, help="Debug limit")
    parser.add_argument("--cache-dir", default="/data2/amd_profiling/hf-cache")
    parser.add_argument("--fp8-reference-repo", default=DEFAULT_FP8_REPO)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device, e.g. cpu or cuda:0 on ROCm PyTorch",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for shard-level parallel quantization, e.g. cuda:0,cuda:1",
    )
    return parser.parse_args()


def copy_side_files(src: Path, out: Path) -> None:
    skip = {"model.safetensors.index.json"}
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir():
            continue
        if is_temp_model_artifact(rel):
            continue
        if rel.name.endswith(".safetensors") or str(rel) in skip:
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def load_reference_quant_config(repo: str, cache_dir: str | None) -> dict:
    path = hf_hub_download(repo, "config.json", cache_dir=cache_dir)
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    quant = cfg.get("quantization_config")
    if not isinstance(quant, dict):
        raise RuntimeError(f"{repo} has no quantization_config")
    required = {"quant_method": "fp8", "weight_block_size": [BLOCK, BLOCK]}
    for key, expected in required.items():
        if quant.get(key) != expected:
            raise RuntimeError(
                f"Unexpected reference quant config {key}={quant.get(key)!r}"
            )
    return quant


def expand_glm_ignore(ignore: Iterable[str]) -> list[str]:
    expanded: list[str] = []
    for item in ignore:
        expanded.append(item)
        if "indexers_proj" in item:
            expanded.append(item.replace("indexers_proj", "indexer.weights_proj"))
    return expanded


def should_ignore_tensor(name: str, ignore: Iterable[str]) -> bool:
    return any(name == item or name.startswith(item + ".") for item in ignore)


def should_quantize_by_contract(name: str) -> bool:
    return any(pattern.match(name) for pattern in QUANTIZED_WEIGHT_PATTERNS)


def is_quantizable_weight(
    name: str, tensor: torch.Tensor, ignore: Iterable[str]
) -> bool:
    if not name.endswith(".weight"):
        return False
    if not should_quantize_by_contract(name):
        return False
    if tensor.ndim != 2:
        return False
    if not tensor.dtype.is_floating_point:
        return False
    module = name[: -len(".weight")]
    if should_ignore_tensor(module, ignore) or should_ignore_tensor(name, ignore):
        return False
    return True


def quantize_block128(
    weight: torch.Tensor, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if weight.ndim != 2:
        raise ValueError(f"expected 2D weight, got {tuple(weight.shape)}")
    out_dim, in_dim = weight.shape
    out_blocks = (out_dim + BLOCK - 1) // BLOCK
    in_blocks = (in_dim + BLOCK - 1) // BLOCK
    padded_out = out_blocks * BLOCK
    padded_in = in_blocks * BLOCK
    weight = weight.to(device=device).float()
    if padded_out != out_dim or padded_in != in_dim:
        padded = weight.new_zeros((padded_out, padded_in))
        padded[:out_dim, :in_dim] = weight
        weight = padded
    blocks = weight.reshape(out_blocks, BLOCK, in_blocks, BLOCK)
    amax = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8)
    scale = amax / FP8_MAX
    qweight = (blocks / scale).clamp(min=-FP8_MAX, max=FP8_MAX).to(torch.float8_e4m3fn)
    qweight = qweight.reshape(padded_out, padded_in)[:out_dim, :in_dim].contiguous()
    scale = scale.squeeze(3).squeeze(1).to(torch.float32).contiguous()
    return qweight.cpu(), scale.cpu()


def process_shard_task(
    args: tuple[str, str, str, list[str], str],
) -> tuple[str, int, dict[str, str], int]:
    shard_name, src_dir, out_dir, ignore, device = args
    in_path = Path(src_dir) / shard_name
    out_path = Path(out_dir) / shard_name
    tmp_path = out_path.with_name("." + out_path.name + ".tmp")
    tensors = load_file(str(in_path), device="cpu")
    quantized = 0
    for name in list(tensors):
        tensor = tensors[name]
        if not is_quantizable_weight(name, tensor, ignore):
            continue
        qweight, scale = quantize_block128(tensor, device)
        tensors[name] = qweight
        tensors[name + "_scale_inv"] = scale
        quantized += 1
    save_file(tensors, str(tmp_path))
    os.replace(tmp_path, out_path)
    total_size = sum(t.nbytes for t in tensors.values())
    weight_map = {key: shard_name for key in tensors}
    return shard_name, total_size, weight_map, quantized


def rebuild_index(out: Path, shards: Iterable[str]) -> tuple[int, dict[str, str]]:
    total_size = 0
    weight_map: dict[str, str] = {}
    for shard in shards:
        path = out / shard
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_size += tensor.nbytes
                weight_map[key] = shard
    return total_size, weight_map


def update_config(out: Path, quant_config: dict) -> None:
    path = out / "config.json"
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg["quantization_config"] = quant_config
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")


def attention_layer_ids(weight_map: dict[str, str]) -> list[int]:
    layer_ids: set[int] = set()
    pattern = re.compile(
        r"^model\.layers\.(\d+)\.self_attn\."
        r"(?:q_a_proj|kv_a_proj_with_mqa|indexer\.weights_proj)\.weight$"
    )
    for key in weight_map:
        match = pattern.match(key)
        if match:
            layer_ids.add(int(match.group(1)))
    return sorted(layer_ids)


def merge_modules_to_not_convert(
    quant_config: dict, weight_map: dict[str, str]
) -> dict:
    quant_config = dict(quant_config)
    modules = list(quant_config.get("modules_to_not_convert", []))
    seen = set(modules)

    def add(module: str) -> None:
        if module not in seen:
            modules.append(module)
            seen.add(module)

    add("lm_head")
    for layer_idx in attention_layer_ids(weight_map):
        add(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj")
    quant_config["modules_to_not_convert"] = modules
    return quant_config


def write_index(out: Path, total_size: int, weight_map: dict[str, str]) -> None:
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": dict(sorted(weight_map.items())),
    }
    with open(out / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    devices = [d.strip() for d in (args.devices or args.device).split(",") if d.strip()]
    if not devices:
        devices = ["cpu"]
    if args.workers is None:
        args.workers = len(devices) if devices != ["cpu"] else 4
    if len(devices) == 1 and devices[0] != "cpu" and args.workers != 1:
        print(
            f"single GPU device {devices[0]} requested; forcing --workers 1 to avoid GPU memory contention",
            flush=True,
        )
        args.workers = 1
    if len(devices) > 1 and args.workers > len(devices):
        print(
            f"limiting workers from {args.workers} to device count {len(devices)}",
            flush=True,
        )
        args.workers = len(devices)
    src = Path(args.src)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"output directory already exists and is non-empty: {out}. "
            "Remove it before running a fresh quantization; resume mode is intentionally disabled."
        )
    removed_temp = cleanup_temp_model_artifacts(src)
    if removed_temp:
        print(
            "removed temporary model artifacts before quantization: "
            + ", ".join(removed_temp[:20])
            + (" ..." if len(removed_temp) > 20 else ""),
            flush=True,
        )
    out.mkdir(parents=True, exist_ok=True)
    copy_side_files(src, out)

    with open(src / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        src_index = json.load(f)
    quant_config = load_reference_quant_config(args.fp8_reference_repo, args.cache_dir)
    quant_config = merge_modules_to_not_convert(quant_config, src_index["weight_map"])
    ignore = expand_glm_ignore(quant_config.get("modules_to_not_convert", []))
    shards = sorted(set(src_index["weight_map"].values()))
    if args.max_shards is not None:
        shards = shards[: args.max_shards]

    jobs = []
    for idx, shard in enumerate(shards):
        jobs.append((shard, str(src), str(out), ignore, devices[idx % len(devices)]))

    total_size = 0
    weight_map: dict[str, str] = {}
    quantized_total = 0
    if jobs:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=get_context("spawn")
        ) as ex:
            futures = {ex.submit(process_shard_task, job): job[0] for job in jobs}
            for fut in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Quantizing"
            ):
                shard, shard_size, shard_map, quantized = fut.result()
                total_size += shard_size
                weight_map.update(shard_map)
                quantized_total += quantized
                print(f"done {shard} quantized={quantized}", flush=True)

    if args.max_shards is not None:
        total_size, weight_map = rebuild_index(out, shards)

    update_config(out, quant_config)
    write_index(out, total_size, weight_map)
    manifest = {
        "source": str(src),
        "fp8_reference_repo": args.fp8_reference_repo,
        "format": "fp8_block128_weight_scale_inv",
        "workers": args.workers,
        "device": args.device,
        "devices": devices,
        "shards": len(shards),
        "quantized_tensors_this_run": quantized_total,
        "ignore_count": len(ignore),
        "removed_temp_artifacts": removed_temp,
    }
    with open(out / "fp8_block128_quant_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print("quant complete", flush=True)


if __name__ == "__main__":
    main()
