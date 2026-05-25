#!/usr/bin/env python3
"""Download an OSS LoRA archive and resolve it to a local PEFT adapter.

The accepted source is an HTTP(S) OSS archive. The archive may contain either
a complete PEFT adapter directory or a raw Tinker checkpoint directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

IGNORED_UNWRAP_NAMES = {".DS_Store", "__MACOSX"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url", required=True, help="Signed or public HTTP(S) OSS archive URL"
    )
    parser.add_argument(
        "--work-dir", required=True, help="Work directory for the downloaded archive"
    )
    parser.add_argument(
        "--base-repo",
        required=True,
        help="Base model repo/name used to build a Tinker adapter",
    )
    parser.add_argument("--out", required=True, help="Output PEFT adapter directory")
    parser.add_argument("--sha256", default="", help="Optional expected archive SHA256")
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload even if the archive exists",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract even if the extraction directory exists",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and verify the archive, then print its path",
    )
    parser.add_argument(
        "--extract-workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU workers for external decompression tools such as pigz",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def validate_url(url: str) -> urllib.parse.ParseResult:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("OSS_URL must be an HTTP(S) download link")
    if not parsed.netloc:
        raise ValueError("OSS_URL is missing a host")
    return parsed


def archive_name_from_url(parsed: urllib.parse.ParseResult) -> str:
    name = Path(urllib.parse.unquote(parsed.path.rstrip("/"))).name
    if not name:
        digest = hashlib.sha256(parsed.geturl().encode("utf-8")).hexdigest()[:16]
        name = f"oss-source-{digest}.tar.gz"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)


def download(url: str, archive_path: Path, force: bool) -> None:
    if archive_path.exists() and archive_path.stat().st_size > 0 and not force:
        log(f"using existing archive: {archive_path}")
        return

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = archive_path.with_suffix(archive_path.suffix + ".tmp")
    request = urllib.request.Request(
        url, headers={"User-Agent": "glm51-oss-lora-prepare/1.0"}
    )
    log(f"downloading OSS archive to {archive_path}")
    with urllib.request.urlopen(request) as response, open(tmp_path, "wb") as out:
        shutil.copyfileobj(response, out, length=16 * 1024 * 1024)
    os.replace(tmp_path, archive_path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path: Path, expected: str) -> None:
    if not expected:
        return
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ValueError(
            f"SHA256 mismatch for {path}: expected {expected}, got {actual}"
        )
    log(f"sha256 ok: {actual}")


def ensure_within_directory(root: Path, target: Path) -> None:
    root_resolved = root.resolve()
    target_resolved = target.resolve()
    if (
        root_resolved != target_resolved
        and root_resolved not in target_resolved.parents
    ):
        raise ValueError(f"archive member escapes extraction directory: {target}")


def safe_extract_tar(archive_path: Path, extract_dir: Path) -> None:
    with tarfile.open(archive_path, "r:*") as tf:
        for member in tf.getmembers():
            ensure_within_directory(extract_dir, extract_dir / member.name)
        tf.extractall(extract_dir)


def is_gzip_tar(archive_path: Path) -> bool:
    name = archive_path.name.lower()
    return name.endswith(".tar.gz") or name.endswith(".tgz")


def try_system_tar_extract(archive_path: Path, extract_dir: Path, workers: int) -> bool:
    if shutil.which("tar") is None:
        return False

    cmd = ["tar", "-xf", str(archive_path), "-C", str(extract_dir)]
    list_cmd = ["tar", "-tf", str(archive_path)]
    if is_gzip_tar(archive_path) and shutil.which("pigz") is not None:
        compress_program = f"pigz -dc -p {max(1, workers)}"
        cmd = [
            "tar",
            f"--use-compress-program={compress_program}",
            "-xf",
            str(archive_path),
            "-C",
            str(extract_dir),
        ]
        list_cmd = [
            "tar",
            f"--use-compress-program={compress_program}",
            "-tf",
            str(archive_path),
        ]

    log("extract command: " + " ".join(cmd))
    try:
        listing = subprocess.check_output(list_cmd, text=True)
        for name in listing.splitlines():
            ensure_within_directory(extract_dir, extract_dir / name)
        subprocess.run(cmd, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        log(f"external tar extraction failed, falling back to Python tarfile: {exc}")
        return False
    return True


def safe_extract_zip(archive_path: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        for member in zf.infolist():
            ensure_within_directory(extract_dir, extract_dir / member.filename)
        zf.extractall(extract_dir)


def extract_archive(
    archive_path: Path, extract_dir: Path, force: bool, workers: int
) -> None:
    if extract_dir.exists() and any(extract_dir.iterdir()) and not force:
        log(f"using existing extraction: {extract_dir}")
        return
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    log(f"extracting {archive_path} to {extract_dir}")
    try:
        if try_system_tar_extract(archive_path, extract_dir, workers):
            return
        safe_extract_tar(archive_path, extract_dir)
        return
    except tarfile.TarError as tar_exc:
        if zipfile.is_zipfile(archive_path):
            safe_extract_zip(archive_path, extract_dir)
            return
        raise ValueError(f"unsupported or corrupt archive: {archive_path}") from tar_exc


def validate_peft_dir(path: Path) -> None:
    config = path / "adapter_config.json"
    weights = path / "adapter_model.safetensors"
    if not config.exists() or not weights.exists():
        raise FileNotFoundError(
            f"{path} must contain adapter_config.json and adapter_model.safetensors"
        )
    with open(config, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("r", "lora_alpha"):
        if key not in cfg:
            raise KeyError(f"{config} is missing required PEFT key {key!r}")


def is_peft_dir(path: Path) -> bool:
    return (path / "adapter_config.json").is_file() and (
        path / "adapter_model.safetensors"
    ).is_file()


def find_peft_dir(root: Path) -> Path | None:
    if is_peft_dir(root):
        return root
    for config in root.rglob("adapter_config.json"):
        candidate = config.parent
        if is_peft_dir(candidate):
            return candidate
    return None


def clean_children(path: Path) -> list[Path]:
    return [child for child in path.iterdir() if child.name not in IGNORED_UNWRAP_NAMES]


def unwrap_single_directory(path: Path) -> Path:
    current = path
    while current.is_dir():
        children = clean_children(current)
        if len(children) != 1 or not children[0].is_dir():
            return current
        current = children[0]
    return current


def prepare_output_dir(out: Path) -> bool:
    if out.exists() and is_peft_dir(out):
        validate_peft_dir(out)
        return True
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            f"{out} exists but is not a complete PEFT adapter directory"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    return False


def copy_peft_dir(src: Path, out: Path) -> Path:
    validate_peft_dir(src)
    if src.resolve() == out.resolve():
        return out
    if prepare_output_dir(out):
        return out
    shutil.copytree(src, out, dirs_exist_ok=True)
    validate_peft_dir(out)
    return out


def build_from_tinker_raw(raw_path: Path, base_repo: str, out: Path) -> Path:
    try:
        from tinker_cookbook import weights
    except ImportError as exc:
        raise SystemExit(
            "Raw Tinker checkpoint archives require tinker-cookbook in the merge venv. "
            "Install it, then rerun this command."
        ) from exc

    if prepare_output_dir(out):
        return out
    weights.build_lora_adapter(
        base_model=base_repo, adapter_path=str(raw_path), output_path=str(out)
    )
    validate_peft_dir(out)
    return out


def main() -> None:
    args = parse_args()
    parsed = validate_url(args.url)
    work_dir = Path(args.work_dir)
    archive_path = work_dir / "archive" / archive_name_from_url(parsed)
    extract_dir = work_dir / "extracted"
    out = Path(args.out)

    if out.exists() and is_peft_dir(out):
        validate_peft_dir(out)
        print(out)
        return

    download(args.url, archive_path, args.force_download)
    verify_sha256(archive_path, args.sha256)
    if args.download_only:
        print(archive_path)
        return
    extract_archive(archive_path, extract_dir, args.force_extract, args.extract_workers)

    peft_dir = find_peft_dir(extract_dir)
    if peft_dir is not None:
        log(f"found PEFT adapter in archive: {peft_dir}")
        print(copy_peft_dir(peft_dir, out))
        return

    raw_dir = unwrap_single_directory(extract_dir)
    log(f"treating extracted archive as raw Tinker checkpoint: {raw_dir}")
    print(build_from_tinker_raw(raw_dir, args.base_repo, out))


if __name__ == "__main__":
    main()
