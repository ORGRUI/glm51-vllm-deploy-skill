#!/usr/bin/env python3
"""Shard-by-shard LoRA merge for zai-org/GLM-5.1.

This avoids loading the 1.5TB base model at once. It downloads each base
Safetensors shard, applies any LoRA deltas targeting tensors in that shard,
and writes a merged shard to the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file, save_file

DEFAULT_BASE_REPO = "zai-org/GLM-5.1"
DEFAULT_ADAPTER_REPO = (
    "mindlab-research/"
    "sft_aug_v1_from_0429_retry10_state_r16_no_unembed_32k_lr1e5_batch32_20260501_075124_final"
)
ROUTED_EXPERT_RE = re.compile(
    r"^(?P<prefix>model\.layers\.\d+\.mlp\.experts\.)(?P<expert>\d+)(?P<suffix>\..+\.weight)$"
)
SPARSE_EXPERT_GROUP_SIZE = 8
LM_HEAD_A_KEY = "base_model.model.lm_head.lora_A.weight"
LM_HEAD_B_KEY = "base_model.model.lm_head.lora_B.weight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-repo", default=DEFAULT_BASE_REPO)
    parser.add_argument("--adapter-repo", default=DEFAULT_ADAPTER_REPO)
    parser.add_argument(
        "--out", required=True, help="Output directory for merged model"
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--max-shards", type=int, default=None, help="Debug limit")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    parser.add_argument(
        "--jobs", type=int, default=1, help="Number of shard merge workers"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Compute device, e.g. cpu or cuda:0 on ROCm PyTorch",
    )
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated compute devices for shard workers, e.g. cuda:0,cuda:1. Overrides --device.",
    )
    parser.add_argument(
        "--copy-untouched",
        default="symlink",
        choices=["symlink", "hardlink", "copy", "none"],
        help=(
            "For shards without LoRA targets, symlink/hardlink/copy the base shard "
            "instead of loading and rewriting it"
        ),
    )
    return parser.parse_args()


def parse_devices(device: str, devices: str | None) -> list[str]:
    if devices is None:
        return [device]
    parsed = [item.strip() for item in devices.split(",") if item.strip()]
    if not parsed:
        raise ValueError("--devices must contain at least one device")
    if "cpu" in parsed and len(parsed) > 1:
        raise ValueError("--devices cannot mix cpu with other devices")
    return parsed


def effective_jobs(requested_jobs: int, devices: list[str]) -> int:
    if requested_jobs < 1:
        raise ValueError("--jobs must be >= 1")
    if devices == ["cpu"]:
        return requested_jobs
    if len(devices) == 1:
        if requested_jobs != 1:
            print(
                f"single GPU device {devices[0]} requested; forcing --jobs 1 to avoid GPU memory contention",
                flush=True,
            )
        return 1
    if requested_jobs > len(devices):
        print(
            f"capping --jobs {requested_jobs} to {len(devices)} so each GPU has at most one worker",
            flush=True,
        )
        return len(devices)
    return requested_jobs


def adapter_base_key(adapter_a_key: str) -> str:
    key = adapter_a_key
    prefix = "base_model.model."
    if not key.startswith(prefix):
        raise ValueError(f"Unexpected adapter key prefix: {key}")
    key = key[len(prefix) :]
    key = key.replace(".lora_A.weight", ".weight")
    # This adapter was exported with singular shared_expert, while the HF base
    # checkpoint uses shared_experts.
    key = key.replace(".shared_expert.", ".shared_experts.")
    return key


def is_local_source(source: str) -> bool:
    return Path(source).exists()


def resolve_file(source: str, filename: str, cache_dir: str | None) -> str:
    if is_local_source(source):
        path = Path(source) / filename
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist")
        return str(path)
    return hf_hub_download(source, filename, cache_dir=cache_dir)


def load_json(repo: str, filename: str, cache_dir: str | None) -> dict:
    path = resolve_file(repo, filename, cache_dir)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tensor_shape(
    repo: str, weight_map: dict[str, str], key: str, cache_dir: str | None
) -> tuple[int, ...]:
    from safetensors import safe_open

    path = resolve_file(repo, weight_map[key], cache_dir)
    with safe_open(path, framework="pt", device="cpu") as f:
        try:
            return tuple(f.get_slice(key).get_shape())
        except AttributeError:
            return tuple(f.get_tensor(key).shape)


def local_source_path(source: str) -> Path | None:
    path = Path(source)
    return path if path.exists() else None


def tp_adapter_files(adapter_path: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(adapter_path.glob("mp_rank_*_adapter.pt")):
        match = re.search(r"mp_rank_(\d+)_(\d+)_adapter\.pt$", path.name)
        if match:
            files.setdefault(int(match.group(1)), path)
    return files


def load_adapter_state(path: Path) -> dict:
    data = torch.load(path, map_location="cpu")
    if not isinstance(data, dict):
        raise RuntimeError(
            f"bad adapter checkpoint payload in {path}: {type(data).__name__}"
        )
    state = data.get("adapter_state_dict", data)
    if not isinstance(state, dict):
        raise RuntimeError(
            f"bad adapter_state_dict payload in {path}: {type(state).__name__}"
        )
    return state


def reconstruct_lm_head_b(
    *,
    adapter_path: Path,
    rank: int,
    expected_rows: int,
    original_b: torch.Tensor,
) -> torch.Tensor | None:
    files = tp_adapter_files(adapter_path)
    if not files:
        return None

    parts = []
    for tp_rank in sorted(files):
        state = load_adapter_state(files[tp_rank])
        tensor = state.get("output_layer.adapter.linear_out.weight")
        if tensor is None:
            return None
        if tensor.ndim != 2 or tensor.shape[1] < rank:
            raise RuntimeError(
                f"bad lm_head shard shape in {files[tp_rank]}: {tuple(tensor.shape)}"
            )
        parts.append(tensor[:, :rank].contiguous())

    full = torch.cat(parts, dim=0)
    if tuple(full.shape) != (expected_rows, original_b.shape[1]):
        raise RuntimeError(
            f"bad reconstructed lm_head LoRA-B shape: got={tuple(full.shape)} "
            f"expected={(expected_rows, original_b.shape[1])}"
        )
    if not torch.equal(full[: original_b.shape[0]].to(original_b.dtype), original_b):
        max_diff = float(
            (full[: original_b.shape[0]].float() - original_b.float()).abs().max()
        )
        raise RuntimeError(f"rank0 lm_head LoRA-B shard mismatch; max_diff={max_diff}")
    return full.to(dtype=original_b.dtype)


def reconstruct_lm_head_pair(
    *,
    adapter_path: Path,
    rank: int,
    expected_rows: int,
    original_a: torch.Tensor,
    original_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    files = tp_adapter_files(adapter_path)
    if not files:
        return None

    a_parts = []
    b_parts = []
    for tp_rank in sorted(files):
        state = load_adapter_state(files[tp_rank])
        a_tensor = state.get("output_layer.adapter.linear_in.weight")
        b_tensor = state.get("output_layer.adapter.linear_out.weight")
        if a_tensor is None or b_tensor is None:
            return None
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise RuntimeError(
                f"bad lm_head LoRA shard rank in {files[tp_rank]}: "
                f"A={None if a_tensor is None else tuple(a_tensor.shape)} "
                f"B={None if b_tensor is None else tuple(b_tensor.shape)}"
            )
        a_parts.append(a_tensor.contiguous())
        b_parts.append(b_tensor[:, :rank].contiguous())

    full_a = torch.cat(a_parts, dim=0)
    full_b = torch.cat(b_parts, dim=0)
    if tuple(full_a.shape) != (rank, original_a.shape[1]):
        raise RuntimeError(
            f"bad reconstructed lm_head LoRA-A shape: got={tuple(full_a.shape)} "
            f"expected={(rank, original_a.shape[1])}"
        )
    if tuple(full_b.shape) != (expected_rows, rank):
        raise RuntimeError(
            f"bad reconstructed lm_head LoRA-B shape: got={tuple(full_b.shape)} "
            f"expected={(expected_rows, rank)}"
        )
    if not torch.equal(full_a[: original_a.shape[0]].to(original_a.dtype), original_a):
        max_diff = float(
            (full_a[: original_a.shape[0]].float() - original_a.float()).abs().max()
        )
        raise RuntimeError(f"rank0 lm_head LoRA-A shard mismatch; max_diff={max_diff}")
    if not torch.equal(
        full_b[: original_b.shape[0]].to(original_b.dtype), original_b[:, :rank]
    ):
        max_diff = float(
            (full_b[: original_b.shape[0]].float() - original_b[:, :rank].float())
            .abs()
            .max()
        )
        raise RuntimeError(f"rank0 lm_head LoRA-B shard mismatch; max_diff={max_diff}")
    return full_a.to(dtype=original_a.dtype), full_b.to(dtype=original_b.dtype)


def maybe_reconstruct_lm_head(
    *,
    lora_state: dict[str, torch.Tensor],
    adapter_repo: str,
    base_repo: str,
    weight_map: dict[str, str],
    cache_dir: str | None,
    rank: int,
) -> dict:
    stats = {
        "present": False,
        "needed": False,
        "reconstructed": False,
        "reconstructed_keys": [],
        "reason": "lm_head_lora_not_present",
    }
    if LM_HEAD_A_KEY not in lora_state or LM_HEAD_B_KEY not in lora_state:
        return stats
    if "lm_head.weight" not in weight_map:
        stats.update({"present": True, "reason": "base_lm_head_not_present"})
        return stats

    original_a = lora_state[LM_HEAD_A_KEY]
    original_b = lora_state[LM_HEAD_B_KEY]
    base_rows = tensor_shape(base_repo, weight_map, "lm_head.weight", cache_dir)[0]
    stats.update(
        {
            "present": True,
            "reason": "already_full_size",
            "rank": rank,
            "base_rows": base_rows,
            "original_a_shape": list(original_a.shape),
            "original_b_shape": list(original_b.shape),
        }
    )

    malformed_a = original_a.shape[0] != rank
    malformed_b = original_b.shape[0] != base_rows
    if not malformed_a and not malformed_b:
        return stats

    stats.update({"needed": True, "reason": "needs_mp_rank_reconstruction"})
    adapter_path = local_source_path(adapter_repo)
    if adapter_path is None:
        stats["reason"] = "adapter_repo_is_not_local"
        return stats

    pair = reconstruct_lm_head_pair(
        adapter_path=adapter_path,
        rank=rank,
        expected_rows=base_rows,
        original_a=original_a,
        original_b=original_b,
    )
    if pair is not None:
        full_a, full_b = pair
        lora_state[LM_HEAD_A_KEY] = full_a
        lora_state[LM_HEAD_B_KEY] = full_b
        stats.update(
            {
                "reconstructed": True,
                "reconstructed_keys": [LM_HEAD_A_KEY, LM_HEAD_B_KEY],
                "reason": "reconstructed_lm_head_pair_from_mp_rank_adapters",
                "reconstructed_a_shape": list(full_a.shape),
                "reconstructed_b_shape": list(full_b.shape),
            }
        )
        return stats

    if not malformed_a and malformed_b:
        full_b = reconstruct_lm_head_b(
            adapter_path=adapter_path,
            rank=rank,
            expected_rows=base_rows,
            original_b=original_b,
        )
        if full_b is not None:
            lora_state[LM_HEAD_B_KEY] = full_b
            stats.update(
                {
                    "reconstructed": True,
                    "reconstructed_keys": [LM_HEAD_B_KEY],
                    "reason": "reconstructed_lm_head_b_from_mp_rank_adapters",
                    "reconstructed_b_shape": list(full_b.shape),
                }
            )
            return stats

    stats["reason"] = "missing_mp_rank_lm_head_tensors"
    return stats


def routed_expert_parts(base_key: str) -> tuple[str, int, str] | None:
    match = ROUTED_EXPERT_RE.match(base_key)
    if match is None:
        return None
    return match.group("prefix"), int(match.group("expert")), match.group("suffix")


def format_expert_ids(ids: set[int], limit: int = 24) -> str:
    ordered = sorted(ids)
    rendered = ",".join(str(item) for item in ordered[:limit])
    if len(ordered) > limit:
        rendered += f",...,+{len(ordered) - limit} more"
    return rendered


def collect_base_expert_ids(
    weight_map: dict[str, str],
) -> dict[tuple[str, str], set[int]]:
    groups: dict[tuple[str, str], set[int]] = defaultdict(set)
    for base_key in weight_map:
        parts = routed_expert_parts(base_key)
        if parts is None:
            continue
        prefix, expert_id, suffix = parts
        groups[(prefix, suffix)].add(expert_id)
    return groups


def expand_sparse_expert_targets(
    raw_pairs: list[tuple[str, str, str]],
    weight_map: dict[str, str],
) -> tuple[list[tuple[str, str, str]], dict]:
    """Expand Tinker sparse routed-expert LoRA keys to all experts in each group.

    Tinker sparse expert exports can store only the representative expert in
    each EP group: 0, 8, 16, ..., 248 for a 256-expert GLM-5.1 layer. Those
    LoRA deltas must be applied to every base expert in the corresponding
    group. Fully expanded adapters with 0..255 expert keys are left unchanged.
    Any partial or mixed routed-expert coverage fails validation instead of
    silently merging an incomplete model.
    """

    base_expert_ids = collect_base_expert_ids(weight_map)
    lora_by_group: dict[tuple[str, str], dict[int, tuple[str, str, str]]] = defaultdict(
        dict
    )
    non_expert_pairs: list[tuple[str, str, str]] = []

    for base_key, a_key, b_key in raw_pairs:
        parts = routed_expert_parts(base_key)
        if parts is None:
            non_expert_pairs.append((base_key, a_key, b_key))
            continue
        prefix, expert_id, suffix = parts
        group_key = (prefix, suffix)
        if expert_id in lora_by_group[group_key]:
            raise RuntimeError(f"duplicate routed expert LoRA target for {base_key}")
        lora_by_group[group_key][expert_id] = (base_key, a_key, b_key)

    expanded_pairs = list(non_expert_pairs)
    expanded_groups = 0
    full_groups = 0
    expanded_targets = 0
    errors: list[str] = []

    for group_key, lora_items in sorted(lora_by_group.items()):
        prefix, suffix = group_key
        adapter_ids = set(lora_items)
        available_base_ids = base_expert_ids.get(group_key, set())
        if not available_base_ids:
            errors.append(
                f"{prefix}*{suffix}: no matching routed experts in base index"
            )
            continue

        full_ids = set(range(min(available_base_ids), max(available_base_ids) + 1))
        if available_base_ids != full_ids:
            errors.append(
                f"{prefix}*{suffix}: base routed expert ids are not contiguous: "
                f"{format_expert_ids(available_base_ids)}"
            )
            continue

        representative_ids = {
            expert_id
            for expert_id in available_base_ids
            if (expert_id - min(available_base_ids)) % SPARSE_EXPERT_GROUP_SIZE == 0
        }
        if adapter_ids == available_base_ids:
            full_groups += 1
            expanded_pairs.extend(
                lora_items[expert_id] for expert_id in sorted(adapter_ids)
            )
            continue

        if adapter_ids != representative_ids:
            errors.append(
                f"{prefix}*{suffix}: routed expert LoRA coverage must be either full "
                f"({len(available_base_ids)} experts) or sparse representatives "
                f"({len(representative_ids)} experts). got {len(adapter_ids)} ids: "
                f"{format_expert_ids(adapter_ids)}"
            )
            continue

        expanded_groups += 1
        for representative_id in sorted(adapter_ids):
            _, a_key, b_key = lora_items[representative_id]
            for offset in range(SPARSE_EXPERT_GROUP_SIZE):
                expert_id = representative_id + offset
                expanded_base_key = f"{prefix}{expert_id}{suffix}"
                if (
                    expert_id not in available_base_ids
                    or expanded_base_key not in weight_map
                ):
                    errors.append(
                        f"{prefix}{representative_id}{suffix}: cannot expand to missing "
                        f"base expert {expanded_base_key}"
                    )
                    continue
                expanded_pairs.append((expanded_base_key, a_key, b_key))
                expanded_targets += 1

    if errors:
        raise RuntimeError(
            "routed expert LoRA expansion validation failed:\n"
            + "\n".join(f"- {item}" for item in errors[:50])
        )

    stats = {
        "non_expert_pairs": len(non_expert_pairs),
        "routed_expert_groups_sparse_expanded": expanded_groups,
        "routed_expert_groups_full": full_groups,
        "routed_expert_expanded_targets": expanded_targets,
        "raw_pairs": len(raw_pairs),
        "planned_pairs": len(expanded_pairs),
        "sparse_expert_group_size": SPARSE_EXPERT_GROUP_SIZE,
    }
    return expanded_pairs, stats


def build_plan(
    base_repo: str, adapter_repo: str, cache_dir: str | None
) -> tuple[dict, dict, float]:
    base_index = load_json(base_repo, "model.safetensors.index.json", cache_dir)
    adapter_config = load_json(adapter_repo, "adapter_config.json", cache_dir)
    rank = int(adapter_config["r"])
    scale = float(adapter_config["lora_alpha"]) / float(rank)
    weight_map = base_index["weight_map"]

    adapter_path = resolve_file(adapter_repo, "adapter_model.safetensors", cache_dir)
    lora_state = load_file(adapter_path, device="cpu")
    lora_keys = set(lora_state)
    lm_head_stats = maybe_reconstruct_lm_head(
        lora_state=lora_state,
        adapter_repo=adapter_repo,
        base_repo=base_repo,
        weight_map=weight_map,
        cache_dir=cache_dir,
        rank=rank,
    )
    lora_keys = set(lora_state)

    raw_pairs: list[tuple[str, str, str]] = []
    missing: list[tuple[str, str]] = []
    for a_key in sorted(k for k in lora_keys if k.endswith(".lora_A.weight")):
        b_key = a_key.replace(".lora_A.weight", ".lora_B.weight")
        if b_key not in lora_keys:
            missing.append((a_key, "missing lora_B"))
            continue
        base_key = adapter_base_key(a_key)
        shard = weight_map.get(base_key)
        if shard is None:
            missing.append((a_key, f"missing base tensor {base_key}"))
            continue
        a_shape = tuple(lora_state[a_key].shape)
        b_shape = tuple(lora_state[b_key].shape)
        if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[0] != b_shape[1]:
            missing.append((a_key, f"bad LoRA shapes A={a_shape} B={b_shape}"))
            continue
        raw_pairs.append((base_key, a_key, b_key))

    if missing:
        msg = "\n".join(f"{k}: {reason}" for k, reason in missing[:50])
        raise RuntimeError(
            f"{len(missing)} adapter tensors could not be planned:\n{msg}"
        )

    expanded_pairs, expand_stats = expand_sparse_expert_targets(raw_pairs, weight_map)

    per_shard: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for base_key, a_key, b_key in expanded_pairs:
        per_shard[weight_map[base_key]].append((base_key, a_key, b_key))

    print(f"adapter raw pairs: {expand_stats['raw_pairs']}", flush=True)
    print(f"adapter planned pairs: {expand_stats['planned_pairs']}", flush=True)
    if lm_head_stats["present"]:
        print(
            "lm_head reconstruction: "
            f"needed={lm_head_stats['needed']} "
            f"reconstructed={lm_head_stats['reconstructed']} "
            f"reason={lm_head_stats['reason']}",
            flush=True,
        )
    print(
        "routed expert expand: "
        f"group_size={expand_stats['sparse_expert_group_size']} "
        f"sparse_groups={expand_stats['routed_expert_groups_sparse_expanded']} "
        f"full_groups={expand_stats['routed_expert_groups_full']} "
        f"expanded_targets={expand_stats['routed_expert_expanded_targets']}",
        flush=True,
    )
    print(
        f"touched shards: {len(per_shard)} / {len(set(weight_map.values()))}",
        flush=True,
    )
    print(f"lora scale: {scale}", flush=True)
    return (
        base_index,
        {
            "state": lora_state,
            "per_shard": per_shard,
            "expand_stats": expand_stats,
            "lm_head_reconstruction": lm_head_stats,
        },
        scale,
    )


def copy_repo_side_files(
    base_repo: str, adapter_repo: str, out_dir: Path, cache_dir: str | None
) -> None:
    base_path = local_source_path(base_repo)
    if base_path is not None:
        skip_names = {"model.safetensors.index.json"}
        for src in base_path.rglob("*"):
            rel = src.relative_to(base_path)
            if src.is_dir():
                continue
            if rel.name.endswith(".safetensors") or str(rel) in skip_names:
                continue
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    else:
        api = HfApi()
        info = api.model_info(base_repo)
        skip_suffixes = (".safetensors",)
        skip_names = {"model.safetensors.index.json"}
        for sibling in info.siblings:
            name = sibling.rfilename
            if name in skip_names or name.endswith(skip_suffixes):
                continue
            dest = out_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            src = hf_hub_download(base_repo, name, cache_dir=cache_dir)
            shutil.copy2(src, dest)

    adapter_files = ["adapter_config.json", "metadata.json", "training_meta.json"]
    for name in adapter_files:
        try:
            src = resolve_file(adapter_repo, name, cache_dir)
        except Exception:
            continue
        dest = out_dir / "merged_lora_info" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)


def copy_untouched_shard(
    base_path: str, out_path: Path, tmp_path: Path, mode: str
) -> bool:
    if mode == "none":
        return False
    if tmp_path.exists() or tmp_path.is_symlink():
        tmp_path.unlink()
    if mode == "symlink":
        if out_path.exists() or out_path.is_symlink():
            out_path.unlink()
        out_path.symlink_to(base_path)
        return True
    if mode == "hardlink":
        try:
            os.link(base_path, tmp_path)
        except OSError:
            shutil.copy2(base_path, tmp_path)
    else:
        shutil.copy2(base_path, tmp_path)
    os.replace(tmp_path, out_path)
    return True


def merge_shard(
    base_repo: str,
    shard_name: str,
    targets: list[tuple[str, str, str]],
    lora_state: dict[str, torch.Tensor],
    scale: float,
    out_dir: Path,
    cache_dir: str | None,
    compute_dtype: torch.dtype,
    device: str,
    copy_untouched: str,
) -> int:
    out_path = out_dir / shard_name
    tmp_path = out_dir / f".{shard_name}.tmp"
    base_path = resolve_file(base_repo, shard_name, cache_dir)
    if not targets and copy_untouched_shard(
        base_path, out_path, tmp_path, copy_untouched
    ):
        return 0
    tensors = load_file(base_path, device="cpu")
    changed = 0

    for base_key, a_key, b_key in targets:
        if base_key not in tensors:
            raise KeyError(f"{base_key} not present in {shard_name}")
        base = tensors[base_key]
        a = lora_state[a_key].to(device=device, dtype=compute_dtype)
        b = lora_state[b_key].to(device=device, dtype=compute_dtype)
        expected = (b.shape[0], a.shape[1])
        if tuple(base.shape) != expected:
            raise ValueError(
                f"shape mismatch for {base_key}: base={tuple(base.shape)} "
                f"A={tuple(a.shape)} B={tuple(b.shape)} expected={expected}"
            )
        merged = base.to(device=device, dtype=compute_dtype)
        merged.addmm_(b, a, beta=1.0, alpha=scale)
        tensors[base_key] = merged.to(device="cpu", dtype=base.dtype)
        changed += 1

    save_file(tensors, str(tmp_path))
    os.replace(tmp_path, out_path)
    return changed


def merge_shard_task(
    args: tuple[
        str, str, list[tuple[str, str, str]], str, str | None, float, str, str, str
    ],
) -> tuple[str, int]:
    (
        base_repo,
        shard_name,
        targets,
        out_dir,
        cache_dir,
        scale,
        dtype_name,
        device,
        copy_untouched,
    ) = args
    compute_dtype = torch.float32 if dtype_name == "float32" else torch.bfloat16
    changed = merge_shard(
        base_repo,
        shard_name,
        targets,
        LORA_STATE,
        scale,
        Path(out_dir),
        cache_dir,
        compute_dtype,
        device,
        copy_untouched,
    )
    return shard_name, changed


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    if not args.validate_only and out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"output directory already exists and is non-empty: {out_dir}. "
            "Remove it before running a fresh merge; resume mode is intentionally disabled."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    compute_dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    devices = parse_devices(args.device, args.devices)
    jobs = effective_jobs(args.jobs, devices)
    base_index, adapter_plan, scale = build_plan(
        args.base_repo, args.adapter_repo, args.cache_dir
    )
    if args.validate_only:
        print("validation ok", flush=True)
        return

    copy_repo_side_files(args.base_repo, args.adapter_repo, out_dir, args.cache_dir)
    with open(out_dir / "model.safetensors.index.json", "w", encoding="utf-8") as f:
        json.dump(base_index, f, indent=2, sort_keys=False)
        f.write("\n")

    manifest = {
        "base_repo": args.base_repo,
        "adapter_repo": args.adapter_repo,
        "lora_scale": scale,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "compute_dtype": args.dtype,
        "device": args.device,
        "devices": devices,
        "jobs": jobs,
        "copy_untouched": args.copy_untouched,
        "routed_expert_expand": adapter_plan["expand_stats"],
        "lm_head_reconstruction": adapter_plan["lm_head_reconstruction"],
        "output": str(out_dir),
    }
    with open(out_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    all_shards = sorted(set(base_index["weight_map"].values()))
    if args.max_shards is not None:
        all_shards = all_shards[: args.max_shards]

    per_shard = adapter_plan["per_shard"]
    global LORA_STATE
    LORA_STATE = adapter_plan["state"]

    pending: list[tuple[int, str, list[tuple[str, str, str]]]] = []
    for i, shard_name in enumerate(all_shards, 1):
        targets = per_shard.get(shard_name, [])
        pending.append((i, shard_name, targets))

    shard_summaries: list[dict] = []
    if jobs <= 1:
        for i, shard_name, targets in pending:
            device = devices[0]
            print(
                f"[{i}/{len(all_shards)}] merge {shard_name} targets={len(targets)} device={device}",
                flush=True,
            )
            changed = merge_shard(
                args.base_repo,
                shard_name,
                targets,
                LORA_STATE,
                scale,
                out_dir,
                args.cache_dir,
                compute_dtype,
                device,
                args.copy_untouched,
            )
            shard_summaries.append({"shard": shard_name, "changed_params": changed})
    else:
        print(f"parallel jobs: {jobs} devices={','.join(devices)}", flush=True)
        task_args = [
            (
                args.base_repo,
                shard_name,
                targets,
                str(out_dir),
                args.cache_dir,
                scale,
                args.dtype,
                devices[n % len(devices)],
                args.copy_untouched,
            )
            for n, (_, shard_name, targets) in enumerate(pending)
        ]
        started = {shard_name: (i, targets) for i, shard_name, targets in pending}
        with ProcessPoolExecutor(
            max_workers=jobs, mp_context=get_context("fork")
        ) as pool:
            futures = {}
            for item in task_args:
                shard_name = item[1]
                i, targets = started[shard_name]
                device = item[7]
                print(
                    f"[{i}/{len(all_shards)}] submit {shard_name} targets={len(targets)} device={device}",
                    flush=True,
                )
                futures[pool.submit(merge_shard_task, item)] = shard_name
            done = 0
            for future in as_completed(futures):
                shard_name = futures[future]
                _, changed = future.result()
                shard_summaries.append({"shard": shard_name, "changed_params": changed})
                done += 1
                print(f"done {done}/{len(futures)} {shard_name}", flush=True)

    manifest["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(out_dir / "merge_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    changed_shards = sorted(
        item["shard"] for item in shard_summaries if item["changed_params"] > 0
    )
    unchanged_shards = sorted(set(all_shards) - set(changed_shards))
    summary = {
        "base_model_path": args.base_repo,
        "lora_path": args.adapter_repo,
        "output_path": str(out_dir),
        "scaling": scale,
        "mapped_param_count": adapter_plan["expand_stats"]["planned_pairs"],
        "changed_shard_count": len(changed_shards),
        "unchanged_shard_count": len(unchanged_shards),
        "changed_shards": changed_shards,
        "unchanged_shards": unchanged_shards,
        "shards": sorted(shard_summaries, key=lambda item: item["shard"]),
        "plan_stats": {
            "routed_expert_expand": adapter_plan["expand_stats"],
            "lm_head_reconstruction": adapter_plan["lm_head_reconstruction"],
        },
    }
    with open(out_dir / "merge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("merge complete", flush=True)


if __name__ == "__main__":
    main()
