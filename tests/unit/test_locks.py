from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from mind.utils.locks import output_root_lock


def test_output_root_lock_cleans_up_stale_lock(tmp_path: Path) -> None:
    root = tmp_path / "output-root"
    root.mkdir()
    lock_path = root / ".mind.lock"
    lock_path.write_text(
        json.dumps({"pid": 999999, "command": "stale", "root": str(root)}),
        encoding="utf-8",
    )

    with output_root_lock(root, command="fresh-command") as acquired_path:
        assert acquired_path == lock_path
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
        assert payload["pid"] == os.getpid()
        assert payload["command"] == "fresh-command"

    assert not lock_path.exists()


def test_output_root_lock_rejects_live_lock(tmp_path: Path) -> None:
    root = tmp_path / "output-root"
    root.mkdir()
    lock_path = root / ".mind.lock"
    lock_path.write_text(
        json.dumps({"pid": os.getpid(), "command": "other-command", "root": str(root)}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="already locked"):
        with output_root_lock(root, command="fresh-command"):
            pass
