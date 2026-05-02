from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_cache_reference_states_import_does_not_load_model_wrappers_or_factory() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib.util
import json
import sys
from pathlib import Path

script_path = Path("scripts/cache_reference_states.py").resolve()
spec = importlib.util.spec_from_file_location("cache_reference_states", script_path)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)
print(json.dumps(sorted(name for name in sys.modules if name.startswith("mind.models"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo_root,
        check=True,
        text=True,
        capture_output=True,
    )

    imported_modules = set(json.loads(result.stdout))
    assert "mind.models.factory" not in imported_modules
    assert "mind.models.wrappers" not in imported_modules
