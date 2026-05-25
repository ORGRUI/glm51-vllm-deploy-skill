#!/usr/bin/env python3
"""Prefetch GLM-5.1 base-model files into a Hugging Face cache."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-repo", default="zai-org/GLM-5.1")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--include-side-files", action="store_true")
    return parser.parse_args()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def download_one(repo: str, filename: str, cache_dir: str) -> str:
    return hf_hub_download(repo, filename, cache_dir=cache_dir)


def main() -> None:
    args = parse_args()
    workers = max(1, args.workers)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    index_path = download_one(
        args.base_repo, "model.safetensors.index.json", args.cache_dir
    )
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    filenames = set(index["weight_map"].values())
    if args.include_side_files:
        api = HfApi()
        info = api.model_info(args.base_repo)
        for sibling in info.siblings:
            name = sibling.rfilename
            if not name.endswith(".safetensors"):
                filenames.add(name)

    filenames = sorted(filenames)
    log(
        f"prefetching {len(filenames)} files from {args.base_repo} with {workers} workers"
    )

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one, args.base_repo, filename, args.cache_dir
            ): filename
            for filename in filenames
        }
        for future in as_completed(futures):
            filename = futures[future]
            try:
                path = future.result()
            except Exception as exc:
                raise RuntimeError(f"failed to download {filename}: {exc}") from exc
            completed += 1
            log(f"[{completed}/{len(filenames)}] {filename} -> {path}")

    print(os.path.abspath(args.cache_dir))


if __name__ == "__main__":
    main()
