#!/usr/bin/env python3
import argparse
import subprocess
from pathlib import Path


SITE = "/opt/python/lib/python3.13/site-packages"

FILE_PATCHES = [
    ("rocm_aiter_mla_sparse.py", "vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py", "VLLM_ROCM_AITER_MLA_SPARSE_REFERENCE_FALLBACK"),
    ("ops_rocm_aiter_mla_sparse.py", "vllm/v1/attention/ops/rocm_aiter_mla_sparse.py", "rocm_aiter"),
    ("_aiter_ops.py", "vllm/_aiter_ops.py", "sparse"),
    ("sparse_attn_indexer.py", "vllm/model_executor/layers/sparse_attn_indexer.py", "topk"),
    ("mla.py", "vllm/model_executor/layers/mla.py", "sparse"),
    ("mla_attention.py", "vllm/model_executor/layers/attention/mla_attention.py", "mla"),
    ("custom_all_reduce.py", "vllm/distributed/device_communicators/custom_all_reduce.py", "return self.all_reduce(input, registered=False)"),
    ("pa_mqa_logits.py", "aiter/ops/triton/attention/pa_mqa_logits.py", "TileQCount = max(1"),
    ("aiter_mla.py", "aiter/mla.py", "nhead in (8, 16)"),
    ("topk.py", "aiter/ops/topk.py", "experts_per_group > 32"),
    ("asm_mla.cu", "aiter_meta/csrc/py_itfs_cu/asm_mla.cu", "asm"),
    ("responses_utils.py", "vllm/entrypoints/openai/responses/utils.py", "construct_tool_parser_tools"),
    ("responses_serving.py", "vllm/entrypoints/openai/responses/serving.py", "tool_calling_response"),
    ("responses_parser.py", "vllm/entrypoints/openai/parser/responses_parser.py", "construct_tool_parser_tools"),
    ("abstract_parser.py", "vllm/parser/abstract_parser.py", "tool_parse_content"),
    ("glm4_moe_tool_parser.py", "vllm/tool_parsers/glm4_moe_tool_parser.py", "_normalize_arg_key"),
]

DIR_PATCHES = [
    ("aiter_meta/hsa/gfx942/mla/", "aiter_meta/hsa/gfx942/mla/"),
]


def dockerfile_snippet(site: str) -> int:
    for src, dst, _ in FILE_PATCHES:
        print(f"COPY patches/{src} {site}/{dst}")
    for src, dst in DIR_PATCHES:
        print(f"COPY patches/{src} {site}/{dst}")
    py_files = " \\\n    ".join(f"{site}/{dst}" for _, dst, _ in FILE_PATCHES if dst.endswith(".py"))
    print(f"RUN /opt/python/bin/python -m py_compile \\\n    {py_files}")
    return 0


def verify_tree(site: Path) -> int:
    failed = 0
    for _, dst, sentinel in FILE_PATCHES:
        path = site / dst
        if not path.exists():
            print(f"MISSING {path}")
            failed += 1
            continue
        text = path.read_text(errors="ignore")
        if sentinel not in text:
            print(f"MISSING_SENTINEL {sentinel!r} in {path}")
            failed += 1
        else:
            print(f"OK {path}")
    for _, dst in DIR_PATCHES:
        path = site / dst
        if not path.exists():
            print(f"MISSING {path}")
            failed += 1
        else:
            print(f"OK {path}")
    return 1 if failed else 0


def verify_image(image: str, site: str) -> int:
    script = f"""
from pathlib import Path
site = Path({site!r})
checks = {[(dst, sentinel) for _, dst, sentinel in FILE_PATCHES]!r}
dirs = {[dst for _, dst in DIR_PATCHES]!r}
failed = 0
for dst, sentinel in checks:
    path = site / dst
    if not path.exists():
        print(f"MISSING {{path}}")
        failed += 1
        continue
    text = path.read_text(errors='ignore')
    if sentinel not in text:
        print(f"MISSING_SENTINEL {{sentinel!r}} in {{path}}")
        failed += 1
    else:
        print(f"OK {{path}}")
for dst in dirs:
    path = site / dst
    if not path.exists():
        print(f"MISSING {{path}}")
        failed += 1
    else:
        print(f"OK {{path}}")
raise SystemExit(1 if failed else 0)
"""
    return subprocess.call(
        [
            "docker",
            "run",
            "--rm",
            "-i",
            "--entrypoint",
            "/opt/python/bin/python",
            image,
            "-c",
            script,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dockerfile-snippet")
    p.add_argument("--site-packages", default=SITE)

    p = sub.add_parser("verify-tree")
    p.add_argument("--site-packages", default=SITE)

    p = sub.add_parser("verify-image")
    p.add_argument("--image", required=True)
    p.add_argument("--site-packages", default=SITE)

    args = parser.parse_args()
    if args.cmd == "dockerfile-snippet":
        return dockerfile_snippet(args.site_packages)
    if args.cmd == "verify-tree":
        return verify_tree(Path(args.site_packages))
    if args.cmd == "verify-image":
        return verify_image(args.image, args.site_packages)
    raise AssertionError(args.cmd)


if __name__ == "__main__":
    raise SystemExit(main())
