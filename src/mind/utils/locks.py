"""Lock helpers for long-running writers."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from typing import Iterator


LOCK_FILENAME = ".mind.lock"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


@contextmanager
def output_root_lock(root: Path, *, command: str) -> Iterator[Path]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / LOCK_FILENAME

    if lock_path.exists():
        try:
            payload = json.loads(lock_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        pid = int(payload.get("pid", -1)) if str(payload.get("pid", "")).strip() else -1
        if _pid_is_alive(pid):
            existing_command = str(payload.get("command", ""))
            raise RuntimeError(
                f"Output root {root} is already locked by pid {pid}"
                + (f" ({existing_command})" if existing_command else "")
            )
        lock_path.unlink()

    payload = {
        "pid": os.getpid(),
        "command": command,
        "root": str(root),
    }
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    try:
        yield lock_path
    finally:
        if lock_path.exists():
            try:
                current_payload = json.loads(lock_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                current_payload = {}
            if int(current_payload.get("pid", -1)) == os.getpid():
                lock_path.unlink()
