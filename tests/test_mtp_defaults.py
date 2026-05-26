from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN_STAGE = ROOT / "merge-quant-serve" / "scripts" / "run_stage.sh"
SERVE_VLLM = ROOT / "merge-quant-serve" / "scripts" / "serve_vllm_glm51.sh"
DEFAULT_SPEC_CONFIG = '{"method":"mtp","num_speculative_tokens":1}'


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


def test_run_stage_defaults_mtp_on():
    derived = run_stage_derive()

    assert derived["VLLM_ENABLE_MTP"] == "1"
    assert derived["VLLM_SPECULATIVE_CONFIG"] == DEFAULT_SPEC_CONFIG
    assert f"--speculative-config={DEFAULT_SPEC_CONFIG}" in derived["VLLM_EXTRA_ARGS"]


def test_run_stage_can_disable_default_mtp():
    derived = run_stage_derive(VLLM_ENABLE_MTP="0")

    assert derived["VLLM_ENABLE_MTP"] == "0"
    assert "--enable-prefix-caching" in derived["VLLM_EXTRA_ARGS"]
    assert "--speculative-config=" not in derived["VLLM_EXTRA_ARGS"]


def test_run_stage_explicit_extra_args_override_default_mtp():
    derived = run_stage_derive(VLLM_EXTRA_ARGS="--async-scheduling")

    assert derived["VLLM_EXTRA_ARGS"] == "--async-scheduling"


def test_serve_dry_run_records_default_mtp_argv(tmp_path: Path):
    env = os.environ.copy()
    env.update(
        {
            "AMD_PROFILING_ROOT": str(tmp_path),
            "VLLM_MODEL": "/tmp",
            "VLLM_DRY_RUN": "1",
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
