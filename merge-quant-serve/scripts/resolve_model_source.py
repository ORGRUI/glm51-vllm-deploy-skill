#!/usr/bin/env python3
"""Resolve a deploy model source to an HTTP(S) OSS archive URL.

The deployment workflow accepts a direct OSS_URL, or a TINKER_URL that is
converted through either GPU Lease Manager's /api/transfer/jobs API or a
compatible transfer service /transfer/jobs API.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

TERMINAL_SUCCESS = {"succeeded", "completed"}
TERMINAL_FAILURE = {"failed", "error", "cancelled", "canceled"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oss-url", default=os.getenv("OSS_URL", ""))
    parser.add_argument("--tinker-url", default=os.getenv("TINKER_URL", ""))
    parser.add_argument(
        "--gpu-lease-base-url", default=os.getenv("GPU_LEASE_BASE_URL", "")
    )
    parser.add_argument(
        "--gpu-lease-api-key", default=os.getenv("GPU_LEASE_API_KEY", "")
    )
    parser.add_argument(
        "--transfer-jobs-endpoint", default=os.getenv("TRANSFER_JOBS_ENDPOINT", "")
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=float(os.getenv("TRANSFER_POLL_INTERVAL", "30")),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.getenv("TRANSFER_TIMEOUT_SECONDS", "7200")),
    )
    parser.add_argument(
        "--output-json", default=os.getenv("MODEL_SOURCE_RESOLUTION_JSON", "")
    )
    return parser.parse_args()


def http_request_json(
    url: str,
    method: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    body = None
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    if payload is not None:
        body = json.dumps(payload).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        url, data=body, headers=request_headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip() or exc.reason
        raise RuntimeError(
            f"{method} {url} returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{method} {url} returned non-JSON response: {raw[:500]!r}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{method} {url} returned a non-object JSON response")
    return parsed


def find_url_with_scheme(value: Any, schemes: tuple[str, ...]) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(schemes):
            return stripped
        return None
    if isinstance(value, dict):
        preferred = (
            "oss_http_url",
            "download_url",
            "http_url",
            "httpUrl",
            "signed_url",
            "signedUrl",
            "oss_url",
            "ossUrl",
            "url",
            "result",
            "data",
        )
        for key in preferred:
            if key in value:
                found = find_url_with_scheme(value[key], schemes)
                if found:
                    return found
        for nested in value.values():
            found = find_url_with_scheme(nested, schemes)
            if found:
                return found
    if isinstance(value, list):
        for nested in value:
            found = find_url_with_scheme(nested, schemes)
            if found:
                return found
    return None


def require_http_oss_url(value: str, name: str) -> str:
    url = value.strip()
    if not url:
        raise ValueError(f"{name} is empty")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{name} must be an HTTP(S) URL")
    return url


def api_headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key} if api_key else {}


def endpoint_join(base: str, suffix: str) -> str:
    return f"{base.rstrip('/')}/{suffix.lstrip('/')}"


def create_transfer_job(
    args: argparse.Namespace, tinker_url: str
) -> tuple[dict[str, Any], str, dict[str, str]]:
    if args.gpu_lease_base_url:
        if not args.gpu_lease_api_key:
            raise ValueError(
                "GPU_LEASE_API_KEY is required when GPU_LEASE_BASE_URL is used"
            )
        url = endpoint_join(args.gpu_lease_base_url, "/api/transfer/jobs")
        headers = api_headers(args.gpu_lease_api_key)
        return (
            http_request_json(url, "POST", {"model_url": tinker_url}, headers),
            url,
            headers,
        )

    if args.transfer_jobs_endpoint:
        url = args.transfer_jobs_endpoint.rstrip("/")
        return http_request_json(url, "POST", {"model_url": tinker_url}), url, {}

    raise ValueError(
        "TINKER_URL requires GPU_LEASE_BASE_URL + GPU_LEASE_API_KEY, "
        "or TRANSFER_JOBS_ENDPOINT"
    )


def job_status_url(
    create_payload: dict[str, Any], create_url: str, headers: dict[str, str]
) -> tuple[str, dict[str, str]]:
    job_id = str(create_payload.get("job_id") or "").strip()
    if not job_id:
        raise RuntimeError(
            f"transfer job create response did not include job_id: {create_payload}"
        )

    parsed_create = urllib.parse.urlparse(create_url)
    if parsed_create.path.rstrip("/").endswith("/api/transfer/jobs"):
        return (
            f"{create_url.rstrip('/')}/{urllib.parse.quote(job_id, safe='')}",
            headers,
        )

    status_url = str(create_payload.get("status_url") or "").strip()
    if status_url:
        parsed = urllib.parse.urlparse(status_url)
        if parsed.scheme in {"http", "https"}:
            return status_url, headers
        if status_url.startswith("/api/transfer/jobs/"):
            root = urllib.parse.urlparse(create_url)
            return f"{root.scheme}://{root.netloc}{status_url}", headers
        if status_url.startswith("/transfer/jobs/"):
            root = urllib.parse.urlparse(create_url)
            return f"{root.scheme}://{root.netloc}{status_url}", {}

    return f"{create_url.rstrip('/')}/{urllib.parse.quote(job_id, safe='')}", headers


def poll_transfer_job(
    status_url: str, headers: dict[str, str], interval: float, timeout: float
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    while True:
        last = http_request_json(status_url, "GET", headers=headers, timeout=60)
        status = str(last.get("status") or "").lower()
        stage = last.get("stage")
        error = last.get("error")
        print(
            f"transfer status={status or '-'} stage={stage or '-'} error={error or '-'}",
            file=sys.stderr,
            flush=True,
        )
        if status in TERMINAL_SUCCESS:
            return last
        if status in TERMINAL_FAILURE:
            raise RuntimeError(
                f"transfer job failed: {json.dumps(last, ensure_ascii=False)[:2000]}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"timed out waiting for transfer job: {json.dumps(last, ensure_ascii=False)[:2000]}"
            )
        time.sleep(interval)


def write_output(path: str, payload: dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> None:
    args = parse_args()
    oss_url = args.oss_url.strip()
    tinker_url = args.tinker_url.strip()

    if oss_url:
        resolved = require_http_oss_url(oss_url, "OSS_URL")
        result = {"source_type": "oss", "oss_url": resolved}
        write_output(args.output_json, result)
        print(resolved)
        return

    if not tinker_url:
        raise SystemExit("provide OSS_URL or TINKER_URL")
    if not tinker_url.startswith("tinker://"):
        raise SystemExit("TINKER_URL must start with tinker://")

    create_payload, create_url, headers = create_transfer_job(args, tinker_url)
    status_url, status_headers = job_status_url(create_payload, create_url, headers)
    final_payload = poll_transfer_job(
        status_url, status_headers, args.poll_interval, args.timeout
    )
    resolved = find_url_with_scheme(
        final_payload.get("result", final_payload), ("http://", "https://")
    )
    if not resolved:
        raise SystemExit(
            "transfer job succeeded but did not return an HTTP(S) OSS archive URL: "
            + json.dumps(final_payload, ensure_ascii=False)[:2000]
        )
    resolved = require_http_oss_url(resolved, "resolved OSS_URL")
    result = {
        "source_type": "tinker",
        "tinker_url": tinker_url,
        "oss_url": resolved,
        "create_response": create_payload,
        "final_status": final_payload,
        "status_url": status_url,
    }
    write_output(args.output_json, result)
    print(resolved)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, TimeoutError, ValueError) as exc:
        raise SystemExit(str(exc)) from None
