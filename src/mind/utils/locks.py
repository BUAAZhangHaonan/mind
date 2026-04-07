"""Filesystem lock helpers for long-running pipeline writers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Iterator


LOCK_FILE_NAME = ".mind.lock"


def _read_lock_metadata(lock_path: Path) -> dict[str, object]:
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stale_lock_message(output_root: Path, metadata: dict[str, object]) -> str:
    pid = int(metadata.get("pid", -1))
    command = str(metadata.get("command", "unknown"))
    return (
        f"Removing stale lock for {output_root}: pid={pid} "
        f"command={command}"
    )


@contextmanager
def output_root_lock(output_root: Path, *, command: str) -> Iterator[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / LOCK_FILE_NAME
    payload = {
        "pid": os.getpid(),
        "command": command,
        "root": str(output_root),
    }

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            metadata = _read_lock_metadata(lock_path)
            pid = int(metadata.get("pid", -1)) if metadata else -1
            if _pid_is_alive(pid):
                raise RuntimeError(
                    f"Output root {output_root} is already locked by pid={pid} "
                    f"command={metadata.get('command', 'unknown')}"
                )
            print(_stale_lock_message(output_root, metadata))
            lock_path.unlink(missing_ok=True)
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        break

    try:
        yield lock_path
    finally:
        if lock_path.exists():
            metadata = _read_lock_metadata(lock_path)
            pid = int(metadata.get("pid", -1)) if metadata else -1
            if pid == os.getpid():
                lock_path.unlink(missing_ok=True)
