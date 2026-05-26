#!/usr/bin/env python3
"""Capture raw OpenAI-compatible requests and proxy them to an upstream server."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from aiohttp import ClientSession, ClientTimeout, web

HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def filtered_headers(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {
        key: value
        for key, value in headers
        if key.lower() not in HOP_BY_HOP_HEADERS
        and key.lower() not in {"host", "content-length"}
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def mask_replacement_chars_in_json(value: Any) -> int:
    if isinstance(value, dict):
        count = 0
        for key, nested in list(value.items()):
            if isinstance(nested, str):
                nested_count = nested.count("\ufffd")
                if nested_count:
                    value[key] = nested.replace("\ufffd", "")
                    count += nested_count
            else:
                count += mask_replacement_chars_in_json(nested)
        return count
    if isinstance(value, list):
        count = 0
        for index, nested in enumerate(value):
            if isinstance(nested, str):
                nested_count = nested.count("\ufffd")
                if nested_count:
                    value[index] = nested.replace("\ufffd", "")
                    count += nested_count
            else:
                count += mask_replacement_chars_in_json(nested)
        return count
    return 0


def as_json_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_tool_call_arguments_in_messages(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0

    count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict) or "arguments" not in function:
                continue
            arguments = function["arguments"]
            if isinstance(arguments, str):
                continue
            function["arguments"] = as_json_string(arguments)
            count += 1
    return count


def normalize_tool_call_arguments_in_json(value: Any) -> int:
    if not isinstance(value, dict):
        return 0

    count = 0
    for field in ("messages", "prompt", "history"):
        count += normalize_tool_call_arguments_in_messages(value.get(field))
    return count


def disable_thinking_in_chat_request(value: Any) -> bool:
    if not isinstance(value, dict):
        return False

    chat_template_kwargs = value.get("chat_template_kwargs")
    if chat_template_kwargs is None:
        value["chat_template_kwargs"] = {"enable_thinking": False}
        return True
    if not isinstance(chat_template_kwargs, dict):
        return False
    if "enable_thinking" in chat_template_kwargs:
        return False
    chat_template_kwargs["enable_thinking"] = False
    return True


def thinking_disabled_in_chat_request(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if value.get("enable_thinking") is False:
        return True
    chat_template_kwargs = value.get("chat_template_kwargs")
    return (
        isinstance(chat_template_kwargs, dict)
        and chat_template_kwargs.get("enable_thinking") is False
    )


def should_sanitize_thinking_markers(request: web.Request, body: bytes) -> bool:
    if request.path != "/v1/chat/completions":
        return False
    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False
    return thinking_disabled_in_chat_request(parsed)


class ReplacementCharMasker:
    """Drop literal UTF-8 U+FFFD bytes while preserving chunk-boundary matches."""

    NEEDLE = "\ufffd".encode("utf-8")
    HOLD_BACK = len(NEEDLE) - 1

    def __init__(self) -> None:
        self._pending = b""
        self.count = 0

    def feed(self, chunk: bytes) -> bytes:
        combined = self._pending + chunk
        if not combined:
            return b""

        hold_back = 0
        for size in range(min(self.HOLD_BACK, len(combined)), 0, -1):
            if combined.endswith(self.NEEDLE[:size]):
                hold_back = size
                break

        if len(combined) == hold_back:
            self._pending = combined
            return b""

        if hold_back:
            head = combined[:-hold_back]
            self._pending = combined[-hold_back:]
        else:
            head = combined
            self._pending = b""
        self.count += head.count(self.NEEDLE)
        return head.replace(self.NEEDLE, b"")

    def finish(self) -> bytes:
        tail = self._pending
        self._pending = b""
        self.count += tail.count(self.NEEDLE)
        return tail.replace(self.NEEDLE, b"")


class LiteralBytesMasker:
    """Drop configured byte sequences while preserving chunk-boundary matches."""

    def __init__(self, needles: Iterable[bytes]) -> None:
        self.needles = tuple(sorted(set(needles), key=len, reverse=True))
        self._pending = b""
        self.count = 0

    def feed(self, chunk: bytes) -> bytes:
        combined = self._pending + chunk
        if not combined:
            return b""

        hold_back = self._suffix_prefix_len(combined)
        if len(combined) == hold_back:
            self._pending = combined
            return b""

        if hold_back:
            head = combined[:-hold_back]
            self._pending = combined[-hold_back:]
        else:
            head = combined
            self._pending = b""
        return self._mask(head)

    def finish(self) -> bytes:
        tail = self._pending
        self._pending = b""
        return self._mask(tail)

    def _suffix_prefix_len(self, chunk: bytes) -> int:
        max_len = max((len(needle) for needle in self.needles), default=1) - 1
        for size in range(min(max_len, len(chunk)), 0, -1):
            suffix = chunk[-size:]
            if any(needle.startswith(suffix) for needle in self.needles):
                return size
        return 0

    def _mask(self, chunk: bytes) -> bytes:
        for needle in self.needles:
            self.count += chunk.count(needle)
            chunk = chunk.replace(needle, b"")
        return chunk


async def capture_request(
    request: web.Request,
    body: bytes,
    capture_dir: Path,
    upstream: str,
) -> dict[str, object]:
    now = datetime.now(timezone.utc)
    day_dir = capture_dir / now.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(body).hexdigest()
    base = f"{now.strftime('%H%M%S_%f')}_{safe_name(request.method)}_{safe_name(request.path)}_{digest[:12]}"
    body_path = day_dir / f"{base}.json"
    meta_path = day_dir / f"{base}.meta.json"
    replay_path = day_dir / f"{base}.replay.sh"
    response_body_path = day_dir / f"{base}.response.bin"
    response_headers_path = day_dir / f"{base}.response.headers.json"
    response_sse_path = day_dir / f"{base}.response.sse.jsonl"
    response_summary_path = day_dir / f"{base}.response.summary.json"

    body_path.write_bytes(body)

    meta = {
        "timestamp": now.isoformat(),
        "method": request.method,
        "path": request.path,
        "query_string": request.query_string,
        "content_length": len(body),
        "sha256": digest,
        "client": request.remote,
        "body_path": str(body_path),
        "meta_path": str(meta_path),
        "replay_path": str(replay_path),
        "response_body_path": str(response_body_path),
        "response_headers_path": str(response_headers_path),
        "response_sse_path": str(response_sse_path),
        "response_summary_path": str(response_summary_path),
        "upstream": upstream,
        "safe_headers": {
            key: request.headers[key]
            for key in ("content-type", "user-agent", "content-length", "host")
            if key in request.headers
        },
    }

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    replay_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"curl -sS http://127.0.0.1:7777{request.rel_url} \\\n"
        "  -H 'Content-Type: application/json' \\\n"
        f"  --data-binary @{body_path}\n",
        encoding="utf-8",
    )
    replay_path.chmod(0o755)

    return meta


def rewrite_request_body(
    request: web.Request,
    body: bytes,
    force_temperature: Optional[float],
    default_max_tokens: Optional[int],
    mask_replacement_char: bool,
    normalize_tool_call_arguments: bool,
    disable_thinking: bool,
) -> tuple[bytes, dict[str, Any]]:
    """Return the body sent upstream plus a short transform summary."""
    summary: dict[str, Any] = {
        "force_temperature": force_temperature,
        "default_max_tokens": default_max_tokens,
        "mask_replacement_char": mask_replacement_char,
        "normalize_tool_call_arguments": normalize_tool_call_arguments,
        "disable_thinking": disable_thinking,
        "changed": False,
    }
    if (
        force_temperature is None
        and default_max_tokens is None
        and not mask_replacement_char
        and not normalize_tool_call_arguments
        and not disable_thinking
    ):
        return body, summary

    content_type = request.headers.get("content-type", "")
    if request.method.upper() != "POST" or "json" not in content_type.lower():
        summary["skip_reason"] = "not_json_post"
        return body, summary
    if request.path not in {"/v1/chat/completions", "/v1/completions"}:
        summary["skip_reason"] = "unsupported_path"
        return body, summary

    try:
        parsed = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        summary["skip_reason"] = "invalid_json"
        return body, summary
    if not isinstance(parsed, dict):
        summary["skip_reason"] = "json_not_object"
        return body, summary

    changed = False
    if mask_replacement_char:
        masked_replacement_chars = mask_replacement_chars_in_json(parsed)
        if masked_replacement_chars:
            changed = True
            summary["masked_replacement_chars_in_request"] = masked_replacement_chars
    if normalize_tool_call_arguments:
        normalized_tool_call_arguments = normalize_tool_call_arguments_in_json(parsed)
        if normalized_tool_call_arguments:
            changed = True
            summary["normalized_tool_call_arguments"] = normalized_tool_call_arguments
    if disable_thinking and request.path == "/v1/chat/completions":
        previous_chat_template_kwargs = parsed.get("chat_template_kwargs")
        disabled_thinking = disable_thinking_in_chat_request(parsed)
        if disabled_thinking:
            changed = True
            summary.update(
                {
                    "previous_chat_template_kwargs": previous_chat_template_kwargs,
                    "forwarded_enable_thinking": False,
                }
            )
    if force_temperature is not None:
        previous = parsed.get("temperature")
        parsed["temperature"] = force_temperature
        changed = changed or previous != force_temperature
        summary.update(
            {
                "previous_temperature": previous,
                "forwarded_temperature": force_temperature,
            }
        )
    if default_max_tokens is not None and parsed.get("max_tokens") is None:
        previous_max_tokens = parsed.get("max_tokens")
        parsed["max_tokens"] = default_max_tokens
        changed = True
        summary.update(
            {
                "previous_max_tokens": previous_max_tokens,
                "forwarded_max_tokens": default_max_tokens,
            }
        )
    rewritten = json.dumps(parsed, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    summary["changed"] = changed
    return rewritten, summary


class SseCapture:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._buffer = b""
        self._event_lines: list[str] = []
        self.events = 0
        self.done = False
        self.content_delta_chars = 0
        self.content_preview_parts: list[str] = []
        self.tool_call_names: list[str] = []
        self.finish_reasons: list[str] = []

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk
        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            self._feed_line(line.rstrip(b"\r").decode("utf-8", "replace"))

    def finish(self) -> None:
        if self._buffer:
            self._feed_line(self._buffer.rstrip(b"\r").decode("utf-8", "replace"))
            self._buffer = b""
        self._flush_event()

    def summary(self) -> dict[str, Any]:
        return {
            "sse_events": self.events,
            "sse_done": self.done,
            "content_delta_chars": self.content_delta_chars,
            "content_preview": "".join(self.content_preview_parts)[:2000],
            "tool_call_names": self.tool_call_names,
            "finish_reasons": self.finish_reasons,
            "sse_path": str(self.path),
        }

    def _feed_line(self, line: str) -> None:
        if line == "":
            self._flush_event()
            return
        self._event_lines.append(line)

    def _flush_event(self) -> None:
        if not self._event_lines:
            return

        event_type = "message"
        data_parts: list[str] = []
        raw_lines = self._event_lines
        self._event_lines = []

        for line in raw_lines:
            if line.startswith(":"):
                continue
            field, sep, value = line.partition(":")
            if sep and value.startswith(" "):
                value = value[1:]
            if field == "event":
                event_type = value
            elif field == "data":
                data_parts.append(value)

        data = "\n".join(data_parts)
        row: dict[str, Any] = {
            "event_index": self.events,
            "event": event_type,
            "data": data,
        }

        if data == "[DONE]":
            self.done = True
            row["done"] = True
        elif data:
            try:
                parsed = json.loads(data)
            except json.JSONDecodeError:
                parsed = None
            if parsed is not None:
                row["json"] = parsed
                self._summarize_openai_event(parsed)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.events += 1

    def _summarize_openai_event(self, obj: Any) -> None:
        if not isinstance(obj, dict):
            return
        choices = obj.get("choices")
        if not isinstance(choices, list):
            return

        for choice in choices:
            if not isinstance(choice, dict):
                continue
            finish_reason = choice.get("finish_reason")
            if finish_reason is not None:
                self.finish_reasons.append(str(finish_reason))
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content")
            if isinstance(content, str) and content:
                self.content_delta_chars += len(content)
                if sum(len(part) for part in self.content_preview_parts) < 2000:
                    self.content_preview_parts.append(content)
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                if isinstance(name, str) and name:
                    self.tool_call_names.append(name)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


async def make_app(
    upstream: str,
    capture_dir: Path,
    force_temperature: Optional[float],
    default_max_tokens: Optional[int],
    mask_replacement_char: bool,
    normalize_tool_call_arguments: bool,
    disable_thinking: bool,
    sanitize_thinking_markers: bool,
) -> web.Application:
    timeout = ClientTimeout(total=None, sock_connect=30, sock_read=None)
    session = ClientSession(timeout=timeout)

    async def close_session(app: web.Application) -> None:
        await session.close()

    async def handler(request: web.Request) -> web.StreamResponse:
        started = time.perf_counter()
        body = await request.read()
        meta = await capture_request(request, body, capture_dir, upstream)
        upstream_body, request_transform = rewrite_request_body(
            request,
            body,
            force_temperature,
            default_max_tokens,
            mask_replacement_char,
            normalize_tool_call_arguments,
            disable_thinking,
        )
        meta["request_transform"] = request_transform
        meta["upstream_request_bytes"] = len(upstream_body)
        sanitize_response_thinking_markers = (
            sanitize_thinking_markers
            and should_sanitize_thinking_markers(request, upstream_body)
        )
        meta["sanitize_response_thinking_markers"] = sanitize_response_thinking_markers
        if upstream_body != body:
            forwarded_body_path = Path(str(meta["body_path"])).with_suffix(
                ".forwarded.json"
            )
            forwarded_body_path.write_bytes(upstream_body)
            meta["forwarded_body_path"] = str(forwarded_body_path)
        else:
            meta["forwarded_body_path"] = meta["body_path"]
        write_json(Path(str(meta["meta_path"])), meta)

        target = f"{upstream.rstrip('/')}{request.rel_url}"
        headers = filtered_headers(request.headers.items())

        async with session.request(
            request.method,
            target,
            data=upstream_body,
            headers=headers,
            allow_redirects=False,
        ) as upstream_resp:
            response_headers = filtered_headers(upstream_resp.headers.items())
            response_body_path = Path(str(meta["response_body_path"]))
            response_headers_path = Path(str(meta["response_headers_path"]))
            response_sse_path = Path(str(meta["response_sse_path"]))
            response_summary_path = Path(str(meta["response_summary_path"]))
            sse_capture = SseCapture(response_sse_path)

            write_json(
                response_headers_path,
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "status": upstream_resp.status,
                    "reason": upstream_resp.reason,
                    "headers": dict(upstream_resp.headers.items()),
                    "body_path": meta["body_path"],
                    "response_body_path": str(response_body_path),
                    "response_sse_path": str(response_sse_path),
                },
            )

            response = web.StreamResponse(
                status=upstream_resp.status,
                reason=upstream_resp.reason,
                headers=response_headers,
            )
            await response.prepare(request)

            bytes_out = 0
            response_digest = hashlib.sha256()
            response_masker = ReplacementCharMasker() if mask_replacement_char else None
            thinking_marker_masker = (
                LiteralBytesMasker((b"<|assistant|>", b"</think>"))
                if sanitize_response_thinking_markers
                else None
            )

            async def write_out_chunk(out_chunk: bytes) -> None:
                nonlocal bytes_out
                if not out_chunk:
                    return
                bytes_out += len(out_chunk)
                response_digest.update(out_chunk)
                response_file.write(out_chunk)
                sse_capture.feed(out_chunk)
                await response.write(out_chunk)

            with response_body_path.open("wb") as response_file:
                async for chunk in upstream_resp.content.iter_chunked(65536):
                    out_chunk = (
                        response_masker.feed(chunk) if response_masker else chunk
                    )
                    if thinking_marker_masker:
                        out_chunk = thinking_marker_masker.feed(out_chunk)
                    await write_out_chunk(out_chunk)
                if response_masker:
                    out_chunk = response_masker.finish()
                    if thinking_marker_masker:
                        out_chunk = thinking_marker_masker.feed(out_chunk)
                    await write_out_chunk(out_chunk)
                if thinking_marker_masker:
                    await write_out_chunk(thinking_marker_masker.finish())
            sse_capture.finish()
            await response.write_eof()

        elapsed = time.perf_counter() - started
        response_summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "path": request.path,
            "status": upstream_resp.status,
            "request_bytes": len(body),
            "upstream_request_bytes": len(upstream_body),
            "request_transform": request_transform,
            "forwarded_body_path": meta["forwarded_body_path"],
            "response_bytes": bytes_out,
            "response_sha256": response_digest.hexdigest(),
            "elapsed_s": round(elapsed, 3),
            "body_path": meta["body_path"],
            "response_body_path": meta["response_body_path"],
            "response_headers_path": meta["response_headers_path"],
            "response_sse_path": meta["response_sse_path"],
            "response_summary_path": meta["response_summary_path"],
            **sse_capture.summary(),
        }
        if mask_replacement_char and response_masker:
            response_summary["masked_replacement_chars_in_response"] = (
                response_masker.count
            )
        if thinking_marker_masker:
            response_summary["sanitized_thinking_markers_in_response"] = (
                thinking_marker_masker.count
            )
        meta.update(response_summary)
        write_json(Path(str(meta["meta_path"])), meta)
        write_json(Path(str(meta["response_summary_path"])), response_summary)

        with (capture_dir / "index.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

        print(
            json.dumps(response_summary, ensure_ascii=False),
            flush=True,
        )
        return response

    app = web.Application(client_max_size=512 * 1024 * 1024)
    app.router.add_route("*", "/{tail:.*}", handler)
    app.on_cleanup.append(close_session)
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("CAPTURE_PROXY_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("CAPTURE_PROXY_PORT", "18080"))
    )
    parser.add_argument(
        "--upstream",
        default=os.getenv("CAPTURE_PROXY_UPSTREAM", "http://127.0.0.1:7778"),
    )
    parser.add_argument(
        "--capture-dir",
        default=os.getenv(
            "CAPTURE_PROXY_DIR",
            "/data2/amd_profiling/request_captures",
        ),
    )
    parser.add_argument(
        "--force-temperature",
        type=float,
        default=(
            float(os.environ["CAPTURE_PROXY_FORCE_TEMPERATURE"])
            if os.getenv("CAPTURE_PROXY_FORCE_TEMPERATURE")
            else None
        ),
        help="Force JSON OpenAI completion requests to this temperature before forwarding.",
    )
    parser.add_argument(
        "--default-max-tokens",
        type=int,
        default=(
            int(os.environ["CAPTURE_PROXY_DEFAULT_MAX_TOKENS"])
            if os.getenv("CAPTURE_PROXY_DEFAULT_MAX_TOKENS")
            else None
        ),
        help="Fill in max_tokens for OpenAI completion requests when the client omits it.",
    )
    mask_group = parser.add_mutually_exclusive_group()
    mask_group.add_argument(
        "--mask-replacement-char",
        dest="mask_replacement_char",
        action="store_true",
        help="Remove literal U+FFFD replacement characters from forwarded request JSON and downstream responses.",
    )
    mask_group.add_argument(
        "--no-mask-replacement-char",
        dest="mask_replacement_char",
        action="store_false",
        help="Do not remove literal U+FFFD replacement characters.",
    )
    parser.set_defaults(
        mask_replacement_char=env_flag("CAPTURE_PROXY_MASK_REPLACEMENT_CHAR", True)
    )
    normalize_tool_call_group = parser.add_mutually_exclusive_group()
    normalize_tool_call_group.add_argument(
        "--normalize-tool-call-arguments",
        dest="normalize_tool_call_arguments",
        action="store_true",
        help=(
            "Convert historical messages/prompt/history "
            "tool_calls[*].function.arguments values to JSON strings before "
            "forwarding to vLLM."
        ),
    )
    normalize_tool_call_group.add_argument(
        "--no-normalize-tool-call-arguments",
        dest="normalize_tool_call_arguments",
        action="store_false",
        help="Forward historical tool-call arguments without normalizing their JSON type.",
    )
    parser.set_defaults(
        normalize_tool_call_arguments=env_flag(
            "CAPTURE_PROXY_NORMALIZE_TOOL_CALL_ARGUMENTS",
            True,
        )
    )
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        help=(
            "Fill chat_template_kwargs.enable_thinking=false for chat requests "
            "that omit an explicit value."
        ),
    )
    thinking_group.add_argument(
        "--no-disable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Forward chat requests without adding a default enable_thinking value.",
    )
    parser.set_defaults(
        disable_thinking=env_flag("CAPTURE_PROXY_DISABLE_THINKING", True)
    )
    sanitize_thinking_markers_group = parser.add_mutually_exclusive_group()
    sanitize_thinking_markers_group.add_argument(
        "--sanitize-thinking-markers",
        dest="sanitize_thinking_markers",
        action="store_true",
        help=(
            "Remove generated GLM thinking-template markers from downstream "
            "responses when enable_thinking=false."
        ),
    )
    sanitize_thinking_markers_group.add_argument(
        "--no-sanitize-thinking-markers",
        dest="sanitize_thinking_markers",
        action="store_false",
        help="Forward downstream thinking-template markers unchanged.",
    )
    parser.set_defaults(
        sanitize_thinking_markers=env_flag(
            "CAPTURE_PROXY_SANITIZE_THINKING_MARKERS",
            True,
        )
    )
    args = parser.parse_args()

    capture_dir = Path(args.capture_dir)
    capture_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    app = loop.run_until_complete(
        make_app(
            args.upstream,
            capture_dir,
            args.force_temperature,
            args.default_max_tokens,
            args.mask_replacement_char,
            args.normalize_tool_call_arguments,
            args.disable_thinking,
            args.sanitize_thinking_markers,
        )
    )

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, loop.stop)

    print(
        f"capture proxy listening on {args.host}:{args.port}, upstream={args.upstream}, "
        f"capture_dir={capture_dir}, force_temperature={args.force_temperature}, "
        f"default_max_tokens={args.default_max_tokens}, "
        f"mask_replacement_char={args.mask_replacement_char}, "
        f"normalize_tool_call_arguments={args.normalize_tool_call_arguments}, "
        f"disable_thinking={args.disable_thinking}, "
        f"sanitize_thinking_markers={args.sanitize_thinking_markers}",
        flush=True,
    )
    web.run_app(app, host=args.host, port=args.port, loop=loop, handle_signals=False)


if __name__ == "__main__":
    main()
