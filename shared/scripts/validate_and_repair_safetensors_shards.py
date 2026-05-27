#!/usr/bin/env python3
"""Validate sharded safetensors and repair dangling HF-cache shard links."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Merged model directory")
    parser.add_argument("--cache-dir", required=True, help="Hugging Face cache root")
    parser.add_argument("--expected-shards", type=int, default=282)
    parser.add_argument("--repair-dangling-links", action="store_true")
    parser.add_argument("--remove-temp-artifacts", action="store_true")
    return parser.parse_args()


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


def shard_is_valid(path: Path) -> tuple[bool, str | None]:
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            _ = list(handle.keys())[:1]
        return True, None
    except Exception as exc:
        return False, str(exc)


def blob_names_from_link(path: Path) -> list[str]:
    if not path.is_symlink():
        return []
    raw = os.readlink(path).rstrip("/")
    names = []
    basename = os.path.basename(raw)
    if basename:
        names.append(basename)
    parts = [part for part in raw.replace("\\", "/").split("/") if part]
    for i, part in enumerate(parts[:-1]):
        if part == "blobs" and i + 1 < len(parts):
            names.append(parts[i + 1])
    return list(dict.fromkeys(names))


def find_blob(cache_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        for pattern in (f"models--*/blobs/{name}", f"**/blobs/{name}"):
            for candidate in cache_dir.glob(pattern):
                if candidate.is_file():
                    return candidate
    return None


def replace_with_blob(link_path: Path, blob_path: Path) -> str:
    link_path.unlink()
    try:
        os.link(blob_path, link_path)
        return "hardlink"
    except OSError:
        shutil.copy2(blob_path, link_path)
        return "copy"


def main() -> int:
    args = parse_args()
    model = Path(args.model)
    cache_dir = Path(args.cache_dir)
    removed_temp = (
        cleanup_temp_model_artifacts(model) if args.remove_temp_artifacts else []
    )
    shards = sorted(model.glob("model-*.safetensors"))
    repaired: list[dict[str, str]] = []
    bad: list[dict[str, str]] = []

    if not (model / "model.safetensors.index.json").is_file():
        bad.append({"shard": "model.safetensors.index.json", "error": "missing index"})
    if not (model / "merge_manifest.json").is_file():
        bad.append({"shard": "merge_manifest.json", "error": "missing merge manifest"})
    if args.expected_shards and len(shards) != args.expected_shards:
        bad.append(
            {
                "shard": "model-*.safetensors",
                "error": f"expected {args.expected_shards} shards, found {len(shards)}",
            }
        )

    for shard in shards:
        ok, error = shard_is_valid(shard)
        if ok:
            continue

        if args.repair_dangling_links and shard.is_symlink() and not shard.exists():
            blob = find_blob(cache_dir, blob_names_from_link(shard))
            if blob is not None:
                mode = replace_with_blob(shard, blob)
                ok, error = shard_is_valid(shard)
                if ok:
                    repaired.append(
                        {
                            "shard": shard.name,
                            "source": str(blob),
                            "mode": mode,
                        }
                    )
                    continue

        bad.append({"shard": shard.name, "error": error or "invalid safetensors shard"})

    result = {
        "model": str(model),
        "checked": len(shards),
        "expected_shards": args.expected_shards,
        "repaired": repaired,
        "repaired_count": len(repaired),
        "removed_temp_artifacts": removed_temp,
        "removed_temp_artifacts_count": len(removed_temp),
        "bad": bad[:20],
        "bad_count": len(bad),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
