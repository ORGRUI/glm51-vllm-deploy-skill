#!/usr/bin/env bash
set -euo pipefail

ROOT="${AMD_PROFILING_ROOT:-/data/amd_profiling}"
ENV_FILE="${VLLM_ENV_FILE:-${ROOT}/configs/vllm_glm51_amd2.env}"

if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

CONTAINER_NAME="${VLLM_CONTAINER_NAME:-vllm-glm51-local-64k-seq2}"
MODEL="${VLLM_MODEL:-/data/sft_aug_v1_from_0429_retry10_state_r16_no_unembed_32k_lr1e5_batch32_20260501_075124_final_fp8}"
SERVER_PORT="${VLLM_PORT:-8000}"
SERVER_NAME="${VLLM_SERVED_MODEL_NAME:-glm51-local-fp8}"
INPUT_LENS="${VLLM_BENCH_INPUT_LENS:-10000 20000}"
CONCURRENCIES="${VLLM_BENCH_CONCURRENCIES:-1 2 4 8}"
OUTPUT_LEN="${VLLM_BENCH_OUTPUT_LEN:-128}"
PROMPTS_MULTIPLIER="${VLLM_BENCH_PROMPTS_MULTIPLIER:-2}"
MIN_PROMPTS="${VLLM_BENCH_MIN_PROMPTS:-8}"
FIXED_NUM_PROMPTS="${VLLM_BENCH_NUM_PROMPTS:-}"
REQUEST_RATE="${VLLM_BENCH_REQUEST_RATE:-inf}"
ENDPOINT="${VLLM_BENCH_ENDPOINT:-/v1/completions}"
BACKEND_BASE_URL="${VLLM_BENCH_BASE_URL:-http://127.0.0.1:${SERVER_PORT}}"
RESULT_ROOT="${VLLM_BENCH_RESULT_ROOT:-${ROOT}/results}"
SERVER_ARGV_JSON="${VLLM_SERVER_ARGV_JSON:-}"
SUDO_PASSWORD="${SUDO_PASSWORD:-}"
IMAGE="${VLLM_IMAGE:-vllm/vllm-openai-rocm:latest}"
PROMPT_PYTHON="${VLLM_BENCH_PROMPT_PYTHON:-${ROOT}/venv-merge/bin/python}"
if [[ ! -x "${PROMPT_PYTHON}" ]]; then
  PROMPT_PYTHON="python3"
fi

if [[ -z "${SERVER_ARGV_JSON}" ]]; then
  SERVER_ARGV_JSON="$(ls -t "${ROOT}"/configs/vllm_glm51_*.server_argv.json 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "${SERVER_ARGV_JSON}" || ! -f "${SERVER_ARGV_JSON}" ]]; then
  echo "ERROR: could not locate vLLM server argv JSON. Set VLLM_SERVER_ARGV_JSON." >&2
  exit 1
fi

while IFS='=' read -r key value; do
  case "${key}" in
    model) MODEL="${value}" ;;
    port) SERVER_PORT="${value}" ;;
    container_name) CONTAINER_NAME="${value}" ;;
    served_model_name) SERVER_NAME="${value}" ;;
    env_file) ENV_FILE="${value}" ;;
  esac
done < <(
  python3 - "${SERVER_ARGV_JSON}" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    text = f.read()
data, _ = json.JSONDecoder().raw_decode(text.lstrip())
for key in ("model", "port", "container_name", "served_model_name", "env_file"):
    value = data.get(key)
    if value:
        print(f"{key}={value}")
PY
)

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
out_dir="${VLLM_BENCH_OUT_DIR:-${RESULT_ROOT}/vllm_glm51_64k_seq2_${timestamp}}"
mkdir -p "${out_dir}"
echo "${out_dir}" >"${RESULT_ROOT}/latest_vllm_glm51_benchmark_dir.txt"

copy_if_exists() {
  local src="$1"
  local dst="$2"
  if [[ -n "${src}" && -e "${src}" ]]; then
    cp -a "${src}" "${dst}"
  fi
}

launch_timestamp="$(python3 - "${SERVER_ARGV_JSON}" <<'PY'
import json
import sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    text = f.read()
data, _ = json.JSONDecoder().raw_decode(text.lstrip())
print(data.get("timestamp_utc", ""))
PY
)"
wrapper="${ROOT}/configs/vllm_glm51_${launch_timestamp}.sh"

copy_if_exists "${SERVER_ARGV_JSON}" "${out_dir}/deployment.server_argv.json"
copy_if_exists "${ENV_FILE}" "${out_dir}/deployment.env"
copy_if_exists "${wrapper}" "${out_dir}/deployment.wrapper.sh"

python3 - "${out_dir}/benchmark_context.json" <<PY
import json
import os
from datetime import datetime, timezone

data = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "root": "${ROOT}",
    "env_file": "${ENV_FILE}",
    "server_argv_json": "${SERVER_ARGV_JSON}",
    "wrapper": "${wrapper}",
    "container_name": "${CONTAINER_NAME}",
    "model": "${MODEL}",
    "server_name": "${SERVER_NAME}",
    "base_url": "${BACKEND_BASE_URL}",
    "endpoint": "${ENDPOINT}",
    "input_lens": "${INPUT_LENS}".split(),
    "output_len": int("${OUTPUT_LEN}"),
    "concurrencies": "${CONCURRENCIES}".split(),
    "request_rate": "${REQUEST_RATE}",
    "prompts_multiplier": int("${PROMPTS_MULTIPLIER}"),
    "min_prompts": int("${MIN_PROMPTS}"),
    "fixed_num_prompts": "${FIXED_NUM_PROMPTS}" or None,
    "image": "${IMAGE}",
}
with open(os.sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

docker_cmd=(docker)
if ! docker ps >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    docker_cmd=(sudo -S docker)
  fi
fi

run_docker() {
  if [[ "${docker_cmd[0]}" == "sudo" && -n "${SUDO_PASSWORD}" ]]; then
    printf '%s\n' "${SUDO_PASSWORD}" | "${docker_cmd[@]}" "$@"
  else
    "${docker_cmd[@]}" "$@"
  fi
}

if [[ "${INPUT_LENS}" == "" ]]; then
  echo "ERROR: no input lengths configured." >&2
  exit 1
fi

prompt_dir="${out_dir}/prompts"
mkdir -p "${prompt_dir}"

generate_prompt() {
  local input_len="$1"
  local prompt_json="${prompt_dir}/prompt_${input_len}.json"
  if [[ -f "${prompt_json}" ]]; then
    return
  fi

  HF_HOME="${ROOT}/hf-cache" \
  HF_HUB_CACHE="${ROOT}/hf-cache/hub" \
  TRANSFORMERS_CACHE="${ROOT}/hf-cache/transformers" \
  "${PROMPT_PYTHON}" - "${MODEL}" "${input_len}" "${prompt_json}" <<'PY'
import json
import os
import sys

from transformers import AutoTokenizer

model = sys.argv[1]
target = int(sys.argv[2])
out = sys.argv[3]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)

base = "Throughput test line: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.\n"
lo, hi = 1, max(10, target)
best = None
for _ in range(28):
    mid = (lo + hi) // 2
    prompt = base * mid + "\nReturn a short answer."
    n = len(tok(prompt, add_special_tokens=False).input_ids)
    if n <= target:
      best = (mid, n, prompt)
      lo = mid + 1
    else:
      hi = mid - 1
if best is None:
    raise SystemExit("failed to generate prompt")
repeat, n, prompt = best
with open(out, "w", encoding="utf-8") as f:
    json.dump({"repeat": repeat, "prompt_tokens": n, "prompt": prompt}, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps({"repeat": repeat, "prompt_tokens": n, "out": out}, ensure_ascii=False))
PY
}

python3 - "${out_dir}" <<'PY'
import json
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
print(out)
PY

cat >"${out_dir}/benchmark_runner.py" <<'PY'
import argparse
import concurrent.futures as cf
import json
import pathlib
import statistics
import time
import urllib.error
import urllib.request


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * (pct / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    frac = rank - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def send_one(url, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            raw = resp.read()
            elapsed = time.perf_counter() - start
            return {
                "ok": True,
                "status": resp.status,
                "elapsed_s": elapsed,
                "body": raw.decode("utf-8", errors="replace"),
            }
    except urllib.error.HTTPError as e:
        elapsed = time.perf_counter() - start
        body = e.read().decode("utf-8", errors="replace")
        return {"ok": False, "status": e.code, "elapsed_s": elapsed, "body": body}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"ok": False, "status": None, "elapsed_s": elapsed, "error": repr(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--output-file", required=True)
    ap.add_argument("--num-prompts", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--input-len", type=int, required=True)
    args = ap.parse_args()

    url = args.base_url.rstrip("/") + args.endpoint
    prompt_info = json.loads(pathlib.Path(args.prompt_file).read_text(encoding="utf-8"))
    prompt = prompt_info["prompt"]
    prompt_tokens = prompt_info.get("prompt_tokens")
    payload = {
        "model": args.model_name,
        "prompt": prompt,
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }
    start = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(send_one, url, payload) for _ in range(args.num_prompts)]
        results = [f.result() for f in futures]
    wall = time.perf_counter() - start

    completed = [r for r in results if r.get("ok")]
    prompt_tok_total = 0
    completion_tok_total = 0
    for r in completed:
        try:
            data = json.loads(r["body"])
            usage = data.get("usage") or {}
            prompt_tok_total += int(usage.get("prompt_tokens") or 0)
            completion_tok_total += int(usage.get("completion_tokens") or 0)
        except Exception:
            pass

    summary = {
        "input_len": args.input_len,
        "num_prompts": args.num_prompts,
        "max_concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "prompt_tokens_estimate": prompt_tokens,
        "completed": len(completed),
        "wall_s": wall,
        "request_throughput": len(completed) / wall if wall else None,
        "prompt_token_throughput": prompt_tok_total / wall if wall else None,
        "completion_token_throughput": completion_tok_total / wall if wall else None,
        "total_token_throughput": (prompt_tok_total + completion_tok_total) / wall if wall else None,
        "p50_latency_s": percentile([r["elapsed_s"] for r in results], 50),
        "p95_latency_s": percentile([r["elapsed_s"] for r in results], 95),
        "results": results,
    }
    pathlib.Path(args.output_file).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
PY

echo "Post-deploy vLLM benchmark"
echo "Results: ${out_dir}"
echo "Model: ${MODEL}"
echo "Base URL: ${BACKEND_BASE_URL}${ENDPOINT}"
echo "Input lens: ${INPUT_LENS}; output len: ${OUTPUT_LEN}; concurrencies: ${CONCURRENCIES}"

curl -sS --max-time 10 "${BACKEND_BASE_URL}/v1/models" >"${out_dir}/models.json"

for input_len in ${INPUT_LENS}; do
  generate_prompt "${input_len}"
  prompt_json="${prompt_dir}/prompt_${input_len}.json"
  for concurrency in ${CONCURRENCIES}; do
    if [[ -n "${FIXED_NUM_PROMPTS}" ]]; then
      num_prompts="${FIXED_NUM_PROMPTS}"
    else
      num_prompts=$((concurrency * PROMPTS_MULTIPLIER))
      if (( num_prompts < MIN_PROMPTS )); then
        num_prompts="${MIN_PROMPTS}"
      fi
    fi

    run_id="in${input_len}_out${OUTPUT_LEN}_c${concurrency}_n${num_prompts}_${timestamp}"
    echo "=== ${run_id} ==="
    python3 "${out_dir}/benchmark_runner.py" \
      --base-url "${BACKEND_BASE_URL}" \
      --endpoint "${ENDPOINT}" \
      --model-name "${SERVER_NAME}" \
      --prompt-file "${prompt_json}" \
      --output-file "${out_dir}/${run_id}.json" \
      --num-prompts "${num_prompts}" \
      --concurrency "${concurrency}" \
      --max-tokens "${OUTPUT_LEN}" \
      --input-len "${input_len}" \
      2>&1 | tee "${out_dir}/${run_id}.log"
  done
done

python3 - "${out_dir}" <<'PY'
import json
import pathlib
import re
import sys

out = pathlib.Path(sys.argv[1])
rows = []
pattern = re.compile(r"in(?P<input>\d+)_out(?P<output>\d+)_c(?P<concurrency>\d+)_n(?P<num>\d+)_")
for path in sorted(out.glob("in*_out*_c*_n*_*.json")):
    if path.name == "benchmark_context.json":
        continue
    match = pattern.search(path.name)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    row = {
        "file": path.name,
        "input_len": int(match.group("input")) if match else data.get("input_len"),
        "output_len": int(match.group("output")) if match else data.get("max_tokens"),
        "max_concurrency": data.get("max_concurrency"),
        "num_prompts": data.get("num_prompts"),
        "completed": data.get("completed"),
        "wall_s": data.get("wall_s"),
        "request_throughput": data.get("request_throughput"),
        "prompt_token_throughput": data.get("prompt_token_throughput"),
        "completion_token_throughput": data.get("completion_token_throughput"),
        "total_token_throughput": data.get("total_token_throughput"),
        "p50_latency_s": data.get("p50_latency_s"),
        "p95_latency_s": data.get("p95_latency_s"),
        "prompt_tokens_estimate": data.get("prompt_tokens_estimate"),
    }
    rows.append(row)

rows.sort(key=lambda r: (r.get("input_len") or 0, r.get("max_concurrency") or 0))
best_by_input = {}
for row in rows:
    key = str(row["input_len"])
    prev = best_by_input.get(key)
    if prev is None or (row.get("completion_token_throughput") or 0) > (prev.get("completion_token_throughput") or 0):
        best_by_input[key] = row

summary = {
    "result_dir": str(out),
    "best_by_input_len": best_by_input,
    "rows": rows,
}
(out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

lines = [
    "# Post-deploy vLLM Benchmark",
    "",
    f"Result dir: `{out}`",
    "",
    "## Best By Input Length",
    "",
    "| input tokens | best concurrency | completion tok/s | request/s | p50 latency s | p95 latency s |",
    "|---:|---:|---:|---:|---:|---:|",
]
for input_len, row in sorted(best_by_input.items(), key=lambda item: int(item[0])):
    lines.append(
        f"| {input_len} | {row.get('max_concurrency')} | "
        f"{(row.get('completion_token_throughput') or 0):.2f} | "
        f"{(row.get('request_throughput') or 0):.2f} | "
        f"{(row.get('p50_latency_s') or 0):.2f} | "
        f"{(row.get('p95_latency_s') or 0):.2f} |"
    )
lines.extend([
    "",
    "## All Runs",
    "",
    "| input | concurrency | prompts | completion tok/s | total tok/s | p50 latency s | p95 latency s |",
    "|---:|---:|---:|---:|---:|---:|---:|",
])
for row in rows:
    lines.append(
        f"| {row.get('input_len')} | {row.get('max_concurrency')} | {row.get('num_prompts')} | "
        f"{(row.get('completion_token_throughput') or 0):.2f} | "
        f"{(row.get('total_token_throughput') or 0):.2f} | "
        f"{(row.get('p50_latency_s') or 0):.2f} | "
        f"{(row.get('p95_latency_s') or 0):.2f} |"
    )
(out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Summary: ${out_dir}/summary.json"
echo "Markdown: ${out_dir}/summary.md"
