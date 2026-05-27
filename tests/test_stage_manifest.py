from __future__ import annotations

import json
import shutil
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


def copy_stage_hash_inputs(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    for rel in (
        "merge/SKILL.md",
        "merge/scripts/run_merge.sh",
        "quant/SKILL.md",
        "quant/scripts/run_quant.sh",
        "serve/SKILL.md",
        "serve/scripts/run_serve.sh",
        "merge-quant-serve/SKILL.md",
        "merge-quant-serve/scripts/run_stage.sh",
        "merge-quant-serve/scripts/stage_manifest.py",
        "merge-quant-serve/scripts/resolve_model_source.py",
        "merge-quant-serve/scripts/prepare_oss_lora_source.py",
        "merge-quant-serve/scripts/prefetch_glm51_base.py",
        "merge-quant-serve/scripts/merge_glm51_lora_sharded.py",
        "merge-quant-serve/scripts/validate_and_repair_safetensors_shards.py",
        "merge-quant-serve/scripts/quantize_glm51_fp8_block128.py",
        "merge-quant-serve/scripts/serve_vllm_glm51.sh",
        "merge-quant-serve/scripts/serve_capture_proxy.sh",
        "merge-quant-serve/scripts/serve_observability.sh",
        "merge-quant-serve/scripts/serve_caddy_proxy.sh",
        "merge-quant-serve/scripts/capture_proxy.py",
        "merge-quant-serve/scripts/benchmark_vllm_glm51.sh",
        "merge-quant-serve/observability/grafana/provisioning/datasources/prometheus.yml",
        "merge-quant-serve/observability/grafana/provisioning/dashboards/dashboards.yml",
        "merge-quant-serve/observability/grafana/dashboards/vllm-overview.json",
    ):
        target = repo_root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    return repo_root


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


def test_serve_only_orchestrator_change_only_reruns_from_serve(tmp_path: Path):
    repo_root = copy_stage_hash_inputs(tmp_path)
    merge_hash = run_manifest(
        "hash-stage", "--repo-root", str(repo_root), "--stage", "merge"
    )
    quant_hash = run_manifest(
        "hash-stage", "--repo-root", str(repo_root), "--stage", "quant"
    )
    serve_hash = run_manifest(
        "hash-stage", "--repo-root", str(repo_root), "--stage", "serve"
    )

    run_stage = repo_root / "merge-quant-serve" / "scripts" / "run_stage.sh"
    original = run_stage.read_text(encoding="utf-8")
    updated = original.replace(
        'ATOM_ENV_FILE="$ENV_FILE" VLLM_ENV_FILE="$ENV_FILE" "$REMOTE_ROOT/scripts/serve_vllm_glm51.sh"',
        'echo "serve-only orchestration change"\n'
        'ATOM_ENV_FILE="$ENV_FILE" VLLM_ENV_FILE="$ENV_FILE" "$REMOTE_ROOT/scripts/serve_vllm_glm51.sh"',
    )
    assert updated != original
    run_stage.write_text(updated, encoding="utf-8")

    assert (
        run_manifest("hash-stage", "--repo-root", str(repo_root), "--stage", "merge")
        == merge_hash
    )
    assert (
        run_manifest("hash-stage", "--repo-root", str(repo_root), "--stage", "quant")
        == quant_hash
    )
    changed_serve_hash = run_manifest(
        "hash-stage", "--repo-root", str(repo_root), "--stage", "serve"
    )
    assert changed_serve_hash != serve_hash

    merge_manifest = tmp_path / "bf16" / "stage_manifest.json"
    quant_manifest = tmp_path / "fp8" / "stage_manifest.json"
    serve_manifest = tmp_path / "configs" / "serve.stage_manifest.json"
    write_manifest(merge_manifest, "merge", merge_hash)
    write_manifest(
        quant_manifest,
        "quant",
        quant_hash,
        "--input-manifest",
        str(merge_manifest),
    )
    write_manifest(
        serve_manifest,
        "serve",
        serve_hash,
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
            merge_hash,
            "--quant-stage-hash",
            quant_hash,
            "--serve-stage-hash",
            changed_serve_hash,
        )
    )

    assert plan["decision"] == "rerun_from_serve"
    assert plan["rerun_from"] == "serve"


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
