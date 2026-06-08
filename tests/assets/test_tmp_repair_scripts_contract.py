from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


REPAIR_SCRIPTS = (
    "repair_gemma4_download.py",
    "repair_phi4_peft.py",
    "repair_molmo_asset.py",
    "repair_llava_v15_asset.py",
    "run_remaining_asset_repairs.py",
)

FORBIDDEN_PRODUCTION_FILES = (
    Path("src/mind/models/wrappers.py"),
    Path("src/mind/models/factory.py"),
    Path("src/mind/models/asset_validation.py"),
    Path("src/mind/models/registry.py"),
)


def _load_script(filename: str):
    path = Path("tmp/asset_repair") / filename
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repair_scripts_default_to_dry_run() -> None:
    for filename in REPAIR_SCRIPTS:
        module = _load_script(filename)
        args = module.build_parser().parse_args([])
        assert args.execute is False
        assert args.dry_run is True


def test_execute_mode_is_explicit() -> None:
    for filename in REPAIR_SCRIPTS:
        module = _load_script(filename)
        args = module.build_parser().parse_args(["--execute"])
        assert args.execute is True
        assert args.dry_run is False


def test_dry_run_writes_reports_without_modifying_production_wrappers(tmp_path: Path, monkeypatch) -> None:
    before = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FORBIDDEN_PRODUCTION_FILES}
    module = _load_script("run_remaining_asset_repairs.py")

    monkeypatch.setattr(module, "GEMMA4_LOCAL_PATH", tmp_path / "missing-gemma4")
    monkeypatch.setattr(module, "PHI4_LOCAL_PATH", tmp_path / "missing-phi4")
    monkeypatch.setattr(module, "MOLMO_LOCAL_PATH", tmp_path / "missing-molmo")
    monkeypatch.setattr(module, "LLAVA_V15_LOCAL_PATH", tmp_path / "missing-llava")

    result = module.run_repairs(execute=False, output_root=tmp_path / "repair")

    after = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in FORBIDDEN_PRODUCTION_FILES}
    assert before == after
    assert result["mode"] == "dry_run"
    assert (tmp_path / "repair" / "remaining_asset_repair_summary.json").is_file()
    assert (tmp_path / "repair" / "REMAINING_ASSET_REPAIR_SUMMARY.md").is_file()
