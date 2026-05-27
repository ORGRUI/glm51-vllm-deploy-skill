from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAGE_MANIFEST = ROOT / "merge-quant-serve" / "scripts" / "stage_manifest.py"


def run_manifest(*args: str) -> str:
    result = subprocess.run(
        ["python3", str(STAGE_MANIFEST), *args],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def write_manifest(path: Path, stage: str, stage_hash: str, *extra: str) -> None:
    run_manifest(
        "write",
        "--path",
        str(path),
        "--stage",
        stage,
        "--stage-hash",
        stage_hash,
        "--repo-commit",
        "test-commit",
        "--artifact-path",
        str(path.parent),
        *extra,
    )


def test_stage_hash_changes_are_stage_scoped():
    merge_hash = run_manifest(
        "hash-stage", "--repo-root", str(ROOT), "--stage", "merge"
    )
    quant_hash = run_manifest(
        "hash-stage", "--repo-root", str(ROOT), "--stage", "quant"
    )

    assert len(merge_hash) == 64
    assert len(quant_hash) == 64
    assert merge_hash != quant_hash


def test_plan_reuses_all_when_manifest_hashes_and_params_match(tmp_path: Path):
    merge_manifest = tmp_path / "bf16" / "stage_manifest.json"
    quant_manifest = tmp_path / "fp8" / "stage_manifest.json"
    serve_manifest = tmp_path / "configs" / "serve.stage_manifest.json"

    write_manifest(
        merge_manifest,
        "merge",
        "merge-hash",
        "--param",
        "merge_jobs=8",
    )
    write_manifest(
        quant_manifest,
        "quant",
        "quant-hash",
        "--input-manifest",
        str(merge_manifest),
        "--param",
        "quant_workers=8",
    )
    write_manifest(
        serve_manifest,
        "serve",
        "serve-hash",
        "--input-manifest",
        str(quant_manifest),
        "--param",
        "docker_image=rocm/atom-dev:vllm-latest",
    )

    plan = json.loads(
        run_manifest(
            "plan",
            "--merge-manifest",
            str(merge_manifest),
            "--quant-manifest",
            str(quant_manifest),
            "--serve-manifest",
            str(serve_manifest),
            "--merge-stage-hash",
            "merge-hash",
            "--quant-stage-hash",
            "quant-hash",
            "--serve-stage-hash",
            "serve-hash",
            "--merge-param",
            "merge_jobs=8",
            "--quant-param",
            "quant_workers=8",
            "--serve-param",
            "docker_image=rocm/atom-dev:vllm-latest",
        )
    )

    assert plan["decision"] == "reuse_all"
    assert plan["rerun_from"] == ""


def test_plan_reruns_from_changed_stage_hash(tmp_path: Path):
    merge_manifest = tmp_path / "bf16" / "stage_manifest.json"
    quant_manifest = tmp_path / "fp8" / "stage_manifest.json"
    serve_manifest = tmp_path / "configs" / "serve.stage_manifest.json"

    write_manifest(merge_manifest, "merge", "merge-hash")
    write_manifest(
        quant_manifest,
        "quant",
        "old-quant-hash",
        "--input-manifest",
        str(merge_manifest),
    )
    write_manifest(
        serve_manifest,
        "serve",
        "serve-hash",
        "--input-manifest",
        str(quant_manifest),
    )

    plan = json.loads(
        run_manifest(
            "plan",
            "--merge-manifest",
            str(merge_manifest),
            "--quant-manifest",
            str(quant_manifest),
            "--serve-manifest",
            str(serve_manifest),
            "--merge-stage-hash",
            "merge-hash",
            "--quant-stage-hash",
            "new-quant-hash",
            "--serve-stage-hash",
            "serve-hash",
        )
    )

    assert plan["decision"] == "rerun_from_quant"
    assert plan["reason"] == "stage_hash_changed"
