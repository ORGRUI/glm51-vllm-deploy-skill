from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_STAGE = ROOT / "merge-quant-serve" / "scripts" / "run_stage.sh"
SERVE_VLLM = ROOT / "merge-quant-serve" / "scripts" / "serve_vllm_glm51.sh"
DEFAULT_SPEC_CONFIG = '{"method":"mtp","num_speculative_tokens":3}'
EXPECTED_VLLM_VERSION = "0.19.1rc1.dev90+g5af684c31"
TOOL_PARSER_PATCH_PR = "https://github.com/vllm-project/vllm/pull/39253"
TOOL_PARSER_PATCH_REF = "refs/pull/39253/head"
TOOL_PARSER_PATCH_COMMIT = "920af3c7a1b29847fb237fa9a9aaedacf48e8bbd"
ATOM_BRANCH = "fix/mtp-arange-buffer-token-capacity"
ATOM_COMMIT = "d5f9a49bb2b6f3e82fda35e411d3cd962c19bf15"


def run_stage_derive(**env_overrides: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "REMOTE_ROOT": "/tmp/amd_profiling",
            "OSS_URL": "http://example.com/adapter.tar.gz",
        }
    )
    env.update(env_overrides)
    result = subprocess.run(
        [str(RUN_STAGE), "derive"],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_run_stage_defaults_mtp_off():
    derived = run_stage_derive()

    assert derived["VLLM_ENABLE_MTP"] == "0"
    assert derived["VLLM_SPECULATIVE_CONFIG"] == DEFAULT_SPEC_CONFIG
    assert derived["VLLM_ROCM_USE_AITER"] == "1"
    assert derived["VLLM_ROCM_QUICK_REDUCE_QUANTIZATION"] == "INT4"
    assert derived["VLLM_ROCM_USE_AITER_RMSNORM"] == "0"
    assert "--block-size=1" in derived["VLLM_EXTRA_ARGS"]
    assert "--enable-prefix-caching" in derived["VLLM_EXTRA_ARGS"]
    assert "--speculative-config=" not in derived["VLLM_EXTRA_ARGS"]


def test_run_stage_can_enable_mtp_canary():
    derived = run_stage_derive(VLLM_ENABLE_MTP="1")

    assert derived["VLLM_ENABLE_MTP"] == "1"
    assert derived["VLLM_SPECULATIVE_CONFIG"] == DEFAULT_SPEC_CONFIG
    assert "--block-size=1" in derived["VLLM_EXTRA_ARGS"]
    assert f"--speculative-config={DEFAULT_SPEC_CONFIG}" in derived["VLLM_EXTRA_ARGS"]


def test_run_stage_can_disable_default_mtp():
    derived = run_stage_derive(VLLM_ENABLE_MTP="0")

    assert derived["VLLM_ENABLE_MTP"] == "0"
    assert "--enable-prefix-caching" in derived["VLLM_EXTRA_ARGS"]
    assert "--speculative-config=" not in derived["VLLM_EXTRA_ARGS"]


def test_run_stage_explicit_extra_args_override_default_mtp():
    derived = run_stage_derive(VLLM_EXTRA_ARGS="--async-scheduling")

    assert derived["VLLM_EXTRA_ARGS"] == "--async-scheduling"


def test_run_stage_defaults_pinned_runtime_versions():
    derived = run_stage_derive()

    assert derived["VLLM_EXPECTED_VERSION"] == EXPECTED_VLLM_VERSION
    assert derived["VLLM_TOOL_PARSER_PATCH_PR"] == TOOL_PARSER_PATCH_PR
    assert derived["VLLM_TOOL_PARSER_PATCH_REF"] == TOOL_PARSER_PATCH_REF
    assert derived["VLLM_TOOL_PARSER_PATCH_COMMIT"] == TOOL_PARSER_PATCH_COMMIT
    assert derived["ATOM_BRANCH"] == ATOM_BRANCH
    assert derived["ATOM_PROD_COMMIT"] == ATOM_COMMIT


def test_run_stage_can_resume_with_run_slug_without_source_url():
    derived = run_stage_derive(OSS_URL="", RUN_SLUG="existing-model")

    assert derived["RUN_SLUG"] == "existing-model"
    assert (
        derived["BF16_OUT"]
        == "/local_nvme/amd_profiling/existing-model/models/existing-model-merged"
    )
    assert (
        derived["FP8_OUT"]
        == "/local_nvme/amd_profiling/existing-model/models/existing-model-merged-fp8-finegrained-block128"
    )
    assert derived["MODEL_PATH"] == derived["LOCAL_MODEL_PATH"]


def test_run_stage_preserves_explicit_artifact_paths():
    derived = run_stage_derive(
        OSS_URL="",
        BF16_OUT="/artifacts/bf16",
        FP8_OUT="/artifacts/fp8",
        LOCAL_MODEL_PATH="/artifacts/local-fp8",
        DURABLE_MODEL_PATH="/artifacts/durable-fp8",
        MODEL_PATH="/artifacts/served-fp8",
    )

    assert derived["BF16_OUT"] == "/artifacts/bf16"
    assert derived["FP8_OUT"] == "/artifacts/fp8"
    assert derived["LOCAL_MODEL_PATH"] == "/artifacts/local-fp8"
    assert derived["DURABLE_MODEL_PATH"] == "/artifacts/durable-fp8"
    assert derived["MODEL_PATH"] == "/artifacts/served-fp8"


def test_run_stage_can_resume_from_each_explicit_artifact_path_without_source_url():
    for name in (
        "BF16_OUT",
        "FP8_OUT",
        "LOCAL_MODEL_PATH",
        "DURABLE_MODEL_PATH",
        "MODEL_PATH",
    ):
        derived = run_stage_derive(OSS_URL="", **{name: f"/artifacts/{name.lower()}"})

        assert derived[name] == f"/artifacts/{name.lower()}"
        assert derived["RUN_SLUG"] == name.lower()


def test_run_stage_does_not_force_temperature_by_default():
    derived = run_stage_derive()

    assert derived["FORCE_TEMPERATURE"] == ""
    assert derived["DEFAULT_MAX_TOKENS"] == "8192"


def test_run_stage_can_force_temperature_when_explicit():
    derived = run_stage_derive(FORCE_TEMPERATURE="0")

    assert derived["FORCE_TEMPERATURE"] == "0"


def test_serve_dry_run_records_default_no_mtp_argv(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "AMD_PROFILING_ROOT": str(tmp_path),
            "VLLM_MODEL": "/tmp",
            "VLLM_DRY_RUN": "1",
            "VLLM_EXPECTED_VERSION": EXPECTED_VLLM_VERSION,
            "VLLM_TOOL_PARSER_PATCH_PR": TOOL_PARSER_PATCH_PR,
            "VLLM_TOOL_PARSER_PATCH_REF": TOOL_PARSER_PATCH_REF,
            "VLLM_TOOL_PARSER_PATCH_COMMIT": TOOL_PARSER_PATCH_COMMIT,
        }
    )
    subprocess.run(
        [str(SERVE_VLLM)],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    argv_files = sorted((tmp_path / "configs").glob("*.server_argv.json"))

    assert len(argv_files) == 1
    data = json.loads(argv_files[0].read_text())
    assert data["host"] == "127.0.0.1"
    assert data["enable_mtp"] == "0"
    assert data["speculative_config"] == DEFAULT_SPEC_CONFIG
    assert data["rocm_use_aiter"] == "1"
    assert data["rocm_quick_reduce_quantization"] == "INT4"
    assert data["rocm_use_aiter_rmsnorm"] == "0"
    assert data["expected_vllm_version"] == EXPECTED_VLLM_VERSION
    assert data["tool_parser_patch_pr"] == TOOL_PARSER_PATCH_PR
    assert data["tool_parser_patch_ref"] == TOOL_PARSER_PATCH_REF
    assert data["tool_parser_patch_commit"] == TOOL_PARSER_PATCH_COMMIT
    assert "--block-size=1" in data["server_argv"]
    assert all("--speculative-config" not in arg for arg in data["server_argv"])


def test_serve_dry_run_records_explicit_mtp_argv(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "AMD_PROFILING_ROOT": str(tmp_path),
            "VLLM_MODEL": "/tmp",
            "VLLM_DRY_RUN": "1",
            "VLLM_ENABLE_MTP": "1",
        }
    )
    subprocess.run(
        [str(SERVE_VLLM)],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    argv_files = sorted((tmp_path / "configs").glob("*.server_argv.json"))

    assert len(argv_files) == 1
    data = json.loads(argv_files[0].read_text())
    assert data["enable_mtp"] == "1"
    assert data["speculative_config"] == DEFAULT_SPEC_CONFIG
    assert f"--speculative-config={DEFAULT_SPEC_CONFIG}" in data["server_argv"]
