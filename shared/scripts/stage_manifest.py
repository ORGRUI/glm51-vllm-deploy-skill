#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

STAGE_FILES = {
    "merge": [
        "merge/SKILL.md",
        "merge/scripts/run_merge.sh",
        "shared/scripts/stage_manifest.py",
        "shared/scripts/resolve_model_source.py",
        "shared/scripts/prepare_oss_lora_source.py",
        "shared/scripts/prefetch_glm51_base.py",
        "shared/scripts/merge_glm51_lora_sharded.py",
        "shared/scripts/validate_and_repair_safetensors_shards.py",
    ],
    "quant": [
        "quant/SKILL.md",
        "quant/scripts/run_quant.sh",
        "shared/scripts/stage_manifest.py",
        "shared/scripts/quantize_glm51_fp8_block128.py",
    ],
    "serve": [
        "serve/SKILL.md",
        "serve/scripts/run_serve.sh",
        "shared/scripts/stage_manifest.py",
        "shared/scripts/serve_vllm_glm51.sh",
        "shared/scripts/serve_capture_proxy.sh",
        "shared/scripts/serve_observability.sh",
        "shared/scripts/serve_caddy_proxy.sh",
        "shared/scripts/capture_proxy.py",
        "shared/scripts/benchmark_vllm_glm51.sh",
        "shared/observability/grafana/provisioning/datasources/prometheus.yml",
        "shared/observability/grafana/provisioning/dashboards/dashboards.yml",
        "shared/observability/grafana/dashboards/vllm-overview.json",
    ],
}

RUN_STAGE_ARMS = {
    "merge": [
        "resolve-source",
        "sync-scripts",
        "preflight",
        "prepare-env",
        "fetch-source",
        "prefetch-base",
        "merge",
        "validate-bf16",
        "merge-all",
    ],
    "quant": [
        "sync-scripts",
        "preflight",
        "prepare-env",
        "quantize",
        "stage-model",
        "quant-all",
    ],
    "serve": [
        "sync-scripts",
        "write-serve-env",
        "serve-backend",
        "serve-proxy",
        "serve-observability",
        "serve-caddy",
        "smoke",
        "benchmark",
        "serve-all",
    ],
}

CASE_LABEL_RE = re.compile(r"^  ([A-Za-z0-9_-]+(?:\|[A-Za-z0-9_-]+)*)\)$")


def read_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    manifest_path = Path(path)
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_case_arm(script: str, label: str) -> str:
    lines = script.splitlines(keepends=True)
    start = None
    for index, line in enumerate(lines):
        match = CASE_LABEL_RE.match(line)
        if match and label in match.group(1).split("|"):
            start = index
            break
    if start is None:
        raise SystemExit(f"run_stage.sh case arm missing: {label}")

    end = len(lines)
    for index in range(start + 1, len(lines)):
        if CASE_LABEL_RE.match(lines[index]):
            end = index
            break
    return "".join(lines[start:end])


def stage_hash(repo_root: Path, stage: str) -> str:
    digest = hashlib.sha256()
    for rel in STAGE_FILES[stage]:
        path = repo_root / rel
        if not path.is_file():
            raise SystemExit(f"stage hash input missing: {rel}")
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(file_sha256(path).encode())
        digest.update(b"\0")
    run_stage_path = repo_root / "shared/scripts/run_stage.sh"
    if not run_stage_path.is_file():
        raise SystemExit("stage hash input missing: shared/scripts/run_stage.sh")
    run_stage = run_stage_path.read_text(encoding="utf-8")
    for label in RUN_STAGE_ARMS[stage]:
        digest.update(f"shared/scripts/run_stage.sh#{label}".encode())
        digest.update(b"\0")
        digest.update(
            hashlib.sha256(extract_case_arm(run_stage, label).encode())
            .hexdigest()
            .encode()
        )
        digest.update(b"\0")
    return digest.hexdigest()


def repo_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def parse_pairs(pairs: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"expected KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        result[key] = value
    return result


def manifest_hash(path: str | None) -> str:
    manifest = read_json(path)
    if manifest is None:
        return ""
    return stable_hash(manifest)


def write_manifest(args: argparse.Namespace) -> None:
    path = Path(args.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    inputs = parse_pairs(args.input)
    params = parse_pairs(args.param)
    manifest = {
        "schema_version": 1,
        "stage": args.stage,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_commit": args.repo_commit,
        "stage_hash": args.stage_hash,
        "input_manifest_hash": manifest_hash(args.input_manifest),
        "inputs": inputs,
        "params": params,
        "artifact_path": args.artifact_path,
    }
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(path)


def stage_mismatch(
    manifest: dict[str, Any] | None,
    stage: str,
    expected_hash: str,
    expected_input_hash: str | None,
    expected_params: dict[str, str],
) -> str:
    if manifest is None:
        return "missing_manifest"
    if manifest.get("stage") != stage:
        return "stage_mismatch"
    if manifest.get("stage_hash") != expected_hash:
        return "stage_hash_changed"
    if (
        expected_input_hash is not None
        and manifest.get("input_manifest_hash", "") != expected_input_hash
    ):
        return "input_manifest_changed"
    params = manifest.get("params", {})
    for key, value in expected_params.items():
        if str(params.get(key, "")) != value:
            return f"param_changed:{key}"
    return ""


def plan(args: argparse.Namespace) -> None:
    merge_params = parse_pairs(args.merge_param)
    quant_params = parse_pairs(args.quant_param)
    serve_params = parse_pairs(args.serve_param)
    merge_manifest = read_json(args.merge_manifest)
    quant_manifest = read_json(args.quant_manifest)
    serve_manifest = read_json(args.serve_manifest)
    merge_manifest_hash = manifest_hash(args.merge_manifest)
    quant_manifest_hash = manifest_hash(args.quant_manifest)

    checks = [
        (
            "merge",
            stage_mismatch(
                merge_manifest, "merge", args.merge_stage_hash, None, merge_params
            ),
        ),
        (
            "quant",
            stage_mismatch(
                quant_manifest,
                "quant",
                args.quant_stage_hash,
                merge_manifest_hash,
                quant_params,
            ),
        ),
        (
            "serve",
            stage_mismatch(
                serve_manifest,
                "serve",
                args.serve_stage_hash,
                quant_manifest_hash,
                serve_params,
            ),
        ),
    ]
    rerun_from = ""
    reason = ""
    for stage, mismatch in checks:
        if mismatch:
            rerun_from = stage
            reason = mismatch
            break

    result = {
        "decision": "reuse_all" if not rerun_from else f"rerun_from_{rerun_from}",
        "rerun_from": rerun_from,
        "reason": reason,
        "stage_hashes": {
            "merge": args.merge_stage_hash,
            "quant": args.quant_stage_hash,
            "serve": args.serve_stage_hash,
        },
        "manifest_hashes": {
            "merge": merge_manifest_hash,
            "quant": quant_manifest_hash,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    hash_parser = subparsers.add_parser("hash-stage")
    hash_parser.add_argument("--repo-root", required=True)
    hash_parser.add_argument("--stage", required=True, choices=sorted(STAGE_FILES))

    commit_parser = subparsers.add_parser("repo-commit")
    commit_parser.add_argument("--repo-root", required=True)

    manifest_hash_parser = subparsers.add_parser("manifest-hash")
    manifest_hash_parser.add_argument("--path", required=True)

    write_parser = subparsers.add_parser("write")
    write_parser.add_argument("--path", required=True)
    write_parser.add_argument("--stage", required=True, choices=sorted(STAGE_FILES))
    write_parser.add_argument("--stage-hash", required=True)
    write_parser.add_argument("--repo-commit", default="")
    write_parser.add_argument("--input-manifest", default="")
    write_parser.add_argument("--artifact-path", required=True)
    write_parser.add_argument("--input", action="append", default=[])
    write_parser.add_argument("--param", action="append", default=[])

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--merge-manifest", required=True)
    plan_parser.add_argument("--quant-manifest", required=True)
    plan_parser.add_argument("--serve-manifest", required=True)
    plan_parser.add_argument("--merge-stage-hash", required=True)
    plan_parser.add_argument("--quant-stage-hash", required=True)
    plan_parser.add_argument("--serve-stage-hash", required=True)
    plan_parser.add_argument("--merge-param", action="append", default=[])
    plan_parser.add_argument("--quant-param", action="append", default=[])
    plan_parser.add_argument("--serve-param", action="append", default=[])

    args = parser.parse_args()
    if args.command == "hash-stage":
        print(stage_hash(Path(args.repo_root), args.stage))
    elif args.command == "repo-commit":
        print(repo_commit(Path(args.repo_root)))
    elif args.command == "manifest-hash":
        print(manifest_hash(args.path))
    elif args.command == "write":
        write_manifest(args)
    elif args.command == "plan":
        plan(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
