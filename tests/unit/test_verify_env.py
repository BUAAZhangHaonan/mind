from __future__ import annotations

import importlib.util
import types
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "verify_env.py"
SPEC = importlib.util.spec_from_file_location("verify_env", SCRIPT_PATH)
verify_env = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
SPEC.loader.exec_module(verify_env)


def test_parse_args_uses_empty_model_id_by_default(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["verify_env.py"])

    args = verify_env.parse_args()

    assert args.model_id == ""


def test_check_imports_collects_module_versions(monkeypatch) -> None:
    fake_modules = {
        "alpha": types.SimpleNamespace(__version__="1.0"),
        "beta": types.SimpleNamespace(__version__="2.0"),
    }

    def fake_import(name: str):
        return fake_modules[name]

    monkeypatch.setattr(verify_env.importlib, "import_module", fake_import)

    versions = verify_env.check_imports(("alpha", "beta"))

    assert versions == {"alpha": "1.0", "beta": "2.0"}
