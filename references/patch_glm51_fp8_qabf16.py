#!/usr/bin/env python3
"""Patch GLM-5.1 FP8 block-128 weights by restoring selected tensors to BF16.

Input is the ATOM-compatible FP8 block-128 model produced by
quantize_glm51_fp8_block128.py. The output keeps the FP8 layout for most
2D weights, replaces selected tensors from the BF16 merged source, removes
their weight_scale_inv tensors, and marks those
modules as not converted in config.json.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import load_file, save_file


DEFAULT_FP8_SRC = "/data2/amd_profiling/models/glm51-sft_aug_v1_merged-fp8-block128"
DEFAULT_BF16_SRC = "/data2/amd_profiling/models/glm51-sft_aug_v1_merged"
DEFAULT_OUT = "/data2/amd_profiling/models/glm51-sft_aug_v1_merged-fp8-block128-qabf16"
DEFAULT_TARGET_SUFFIXES = ("self_attn.q_a_proj.weight",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp8-src", default=DEFAULT_FP8_SRC)
    parser.add_argument("--bf16-src", default=DEFAULT_BF16_SRC)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument(
        "--target-suffix",
        action="append",
        dest="target_suffixes",
        help=(
            "Tensor suffix to restore from BF16. Can be repeated. "
            "Defaults to self_attn.q_a_proj.weight."
        ),
    )
    parser.add_argument(
        "--copy-mode",
        choices=["hardlink", "copy"],
        default="hardlink",
        help="How to place unmodified shards in the output directory.",
    )
    return parser.parse_args()


def target_suffixes(args: argparse.Namespace) -> tuple[str, ...]:
    suffixes = tuple(args.target_suffixes or DEFAULT_TARGET_SUFFIXES)
    invalid = [suffix for suffix in suffixes if not suffix.endswith(".weight")]
    if invalid:
        raise ValueError(f"target suffixes must end with '.weight': {invalid}")
    return suffixes


def load_index(model_dir: Path) -> dict:
    with open(model_dir / "model.safetensors.index.json", "r", encoding="utf-8") as f:
        return json.load(f)


def copy_side_files(src: Path, out: Path) -> None:
    skip = {"model.safetensors.index.json"}
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if path.is_dir() or rel.name.endswith(".safetensors") or str(rel) in skip:
            continue
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)


def place_unmodified(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        dest.unlink()
    if mode == "hardlink":
        os.link(src, dest)
    else:
        shutil.copy2(src, dest)


def rebuild_index(out: Path, shards: list[str]) -> tuple[int, dict[str, str]]:
    total_size = 0
    weight_map: dict[str, str] = {}
    for shard in shards:
        with safe_open(str(out / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                total_size += tensor.nbytes
                weight_map[key] = shard
    return total_size, weight_map


def patch_config(out: Path, restored_modules: list[str]) -> int:
    config_path = out / "config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    quant = cfg.setdefault("quantization_config", {})
    modules = list(quant.get("modules_to_not_convert", []))
    merged = sorted(set(modules).union(restored_modules))
    quant["modules_to_not_convert"] = merged
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write("\n")
    return len(merged)


def main() -> None:
    args = parse_args()
    fp8_src = Path(args.fp8_src)
    bf16_src = Path(args.bf16_src)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise RuntimeError(
            f"output directory already exists and is non-empty: {out}. "
            "Remove it before running a fresh q_a BF16 patch."
        )
    out.mkdir(parents=True, exist_ok=True)

    fp8_index = load_index(fp8_src)
    bf16_index = load_index(bf16_src)
    fp8_map = fp8_index["weight_map"]
    bf16_map = bf16_index["weight_map"]
    shards = sorted(set(fp8_map.values()))

    suffixes = target_suffixes(args)
    targets = sorted(key for key in fp8_map if any(key.endswith(suffix) for suffix in suffixes))
    if not targets:
        raise RuntimeError(f"No targets ending with {suffixes!r} found")
    missing = [key for key in targets if key not in bf16_map]
    if missing:
        raise RuntimeError(f"{len(missing)} target tensors missing from BF16 source: {missing[:5]}")

    copy_side_files(fp8_src, out)

    targets_by_fp8_shard: dict[str, list[str]] = {}
    for key in targets:
        targets_by_fp8_shard.setdefault(fp8_map[key], []).append(key)

    for shard in shards:
        out_path = out / shard
        shard_targets = targets_by_fp8_shard.get(shard, [])
        if not shard_targets:
            place_unmodified(fp8_src / shard, out_path, args.copy_mode)
            continue

        tensors = load_file(str(fp8_src / shard), device="cpu")
        bf16_cache: dict[str, dict] = {}
        for key in shard_targets:
            bf16_shard = bf16_map[key]
            if bf16_shard not in bf16_cache:
                bf16_cache[bf16_shard] = load_file(str(bf16_src / bf16_shard), device="cpu")
            tensors[key] = bf16_cache[bf16_shard][key]
            tensors.pop(key + "_scale_inv", None)
        tmp_path = out_path.with_name("." + out_path.name + ".tmp")
        save_file(tensors, str(tmp_path))
        os.replace(tmp_path, out_path)
        print(f"patched {shard} targets={len(shard_targets)}", flush=True)

    total_size, weight_map = rebuild_index(out, shards)
    with open(out / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump({"metadata": {"total_size": total_size}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
        f.write("\n")

    restored_modules = sorted(key[: -len(".weight")] for key in targets)
    modules_count = patch_config(out, restored_modules)
    manifest = {
        "source": str(fp8_src),
        "base_bf16_source": str(bf16_src),
        "change": "restore selected weights to BF16 and remove corresponding weight_scale_inv tensors",
        "target_suffixes": list(suffixes),
        "restored": len(targets),
        "q_a_proj_restored": sum(key.endswith("self_attn.q_a_proj.weight") for key in targets),
        "affected_shards": len(targets_by_fp8_shard),
        "modules_to_not_convert_count": modules_count,
        "copy_mode": args.copy_mode,
    }
    with open(out / "qabf16_patch_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    print("patch complete", flush=True)


if __name__ == "__main__":
    main()
