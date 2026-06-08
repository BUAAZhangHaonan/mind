from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module():
    path = Path("tmp/asset_repair/run_final_tmp_asset_smokes.py")
    spec = importlib.util.spec_from_file_location("run_final_tmp_asset_smokes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_summary_defaults_to_dry_run() -> None:
    module = _load_module()

    args = module.build_parser().parse_args([])

    assert args.execute is False
    assert args.dry_run is True
    assert args.run_smokes is False


def test_summary_classifies_four_tmp_models(tmp_path: Path) -> None:
    module = _load_module()
    output_root = tmp_path / "repair"
    output_root.mkdir(parents=True)
    (output_root / "gemma4_unified_tmp_smoke_report.json").write_text(
        json.dumps({"status": "blocked_missing_transformers_gemma4_unified_support", "reason": "missing runtime"}),
        encoding="utf-8",
    )
    (output_root / "phi4_tmp_smoke_report.json").write_text(
        json.dumps({"status": "verified_tmp", "reason": "phi ok"}),
        encoding="utf-8",
    )
    (output_root / "molmo_separate_env_acceptance.json").write_text(
        json.dumps({"status": "verified_separate_env", "reason": "molmo ok"}),
        encoding="utf-8",
    )
    (output_root / "llava_v15_tmp_smoke_report.json").write_text(
        json.dumps({"status": "verified", "reason": "llava ok"}),
        encoding="utf-8",
    )

    report = module.build_summary(
        output_root=output_root,
        execute=True,
        run_smokes=False,
        stage0_root=tmp_path / "stage0",
        device="cuda:0",
        smoke_limit=2,
    )

    assert report["models"]["gemma-4-12b-it"]["classification"] == "blocked_needs_separate_runtime"
    assert report["models"]["phi-4-multimodal-instruct"]["classification"] == "verified_tmp"
    assert report["models"]["molmo-7b-d-0924"]["classification"] == "verified_separate_env"
    assert report["models"]["llava-v1.5-7b"]["classification"] == "verified_tmp"
    assert report["status"] == "blocked"
    assert report["blocked"] == ["gemma-4-12b-it"]
    assert (output_root / "final_tmp_asset_smoke_summary.json").is_file()
    assert (output_root / "FINAL_TMP_ASSET_SMOKE_SUMMARY.md").is_file()


def test_summary_can_invoke_smokes_when_explicit(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    called = {}

    def fake_run_tmp_smokes(**kwargs):
        called.update(kwargs)

    monkeypatch.setattr(module, "_run_tmp_smokes", fake_run_tmp_smokes)

    module.build_summary(
        output_root=tmp_path / "repair",
        execute=True,
        run_smokes=True,
        stage0_root=tmp_path / "stage0",
        device="cuda:0",
        smoke_limit=2,
    )

    assert called["device"] == "cuda:0"
    assert called["smoke_limit"] == 2
