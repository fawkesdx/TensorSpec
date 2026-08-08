"""Remote scratch sidecar parse and best-effort wipe (no live SSH in tests)."""
from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

SIDECAR_NAME = ".tensorspec_remote_scratch"


def parse_remote_scratch_sidecar(text: str) -> tuple[str, str] | None:
    line = (text or "").strip().splitlines()[0] if text else ""
    if "\t" not in line:
        return None
    host, path = line.split("\t", 1)
    host, path = host.strip(), path.strip()
    if not host or not path.startswith("/") or ".." in path:
        return None
    return host, path


def wipe_remote_scratch_argv(host: str, path: str) -> list[str]:
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        host,
        "--",
        "rm",
        "-rf",
        "--",
        path,
    ]


def best_effort_wipe_remote_scratch(
    run_dir: Path,
    *,
    log: Callable[[str], None] | None = None,
    runner: Callable[[list[str]], int] | None = None,
) -> bool:
    sidecar = run_dir / SIDECAR_NAME
    if not sidecar.is_file():
        return False

    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError as exc:
        if log:
            log(f"remote scratch wipe: failed to read sidecar: {exc}")
        return False

    parsed = parse_remote_scratch_sidecar(text)
    if parsed is None:
        if log:
            log("remote scratch wipe: invalid sidecar contents")
        return False

    host, path = parsed
    argv = wipe_remote_scratch_argv(host, path)
    if log:
        log(f"remote scratch wipe: attempting ssh rm on {host}:{path}")

    if runner is None:
        try:
            result = subprocess.run(
                argv, check=False, capture_output=True, timeout=60
            )
            rc = result.returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            if log:
                log(f"remote scratch wipe: command failed: {exc}")
            return False
    else:
        try:
            rc = runner(argv)
        except Exception as exc:
            if log:
                log(f"remote scratch wipe: runner failed: {exc}")
            return False

    if rc == 0:
        if log:
            log("remote scratch wipe: success")
        return True

    if log:
        log(f"remote scratch wipe: ssh rm exited {rc}")
    return False
