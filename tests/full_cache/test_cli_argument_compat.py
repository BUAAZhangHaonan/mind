from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

accept_existing = importlib.import_module("full_cache_accept_existing")
accept_separate_env = importlib.import_module("full_cache_accept_separate_env")


def _status_row(*, model_alias: str, route: str, status: str, source_cache_root: Path) -> dict[str, Any]:
    return {
        "model_alias": model_alias,
        "route": route,
        "status": status,
        "source_cache_root": str(source_cache_root),
        "copied_tensors": False,
        "total_entries": 1,
        "num_shards": 1,
        "validation_status": "passed",
        "failed_reason": "",
        "manifest_path": str(source_cache_root / "manifest.json"),
    }


def test_stage0_root_cli_resolves_to_stage0_cache(monkeypatch: Any, tmp_path: Path) -> None:
    stage0_root = tmp_path / "stage0"
    output_root = tmp_path / "full_cache"
    captured: list[tuple[str, Path, Path]] = []

    monkeypatch.setattr(accept_existing, "load_panel_config", lambda _config_path: {})
    monkeypatch.setattr(accept_existing, "resolve_output_root", lambda _config, _output_root: output_root)
    monkeypatch.setattr(accept_existing, "route_source_root", lambda _config, _route: None)
    monkeypatch.setattr(accept_existing, "route_models", lambda _config, _route: ["qwen3-vl-8b", "internvl3.5-8b"])
    monkeypatch.setattr(accept_existing, "write_csv", lambda _path, _rows: None)

    def fake_accept_model(*, model_alias: str, source_root: Path, output_root: Path) -> dict[str, Any]:
        captured.append((model_alias, source_root, output_root))
        return _status_row(
            model_alias=model_alias,
            route="accept_existing_stage0",
            status="accepted_existing_stage0",
            source_cache_root=source_root / model_alias,
        )

    monkeypatch.setattr(accept_existing, "accept_model", fake_accept_model)

    rc = accept_existing.main(
        [
            "--config",
            str(tmp_path / "panel.yaml"),
            "--stage0-root",
            str(stage0_root),
            "--output-root",
            str(output_root),
        ]
    )

    assert rc == 0
    assert captured == [
        ("qwen3-vl-8b", stage0_root / "cache", output_root),
        ("internvl3.5-8b", stage0_root / "cache", output_root),
    ]


def test_separate_env_model_specific_roots_resolve_to_full_cache_models(monkeypatch: Any, tmp_path: Path) -> None:
    output_root = tmp_path / "full_cache"
    gemma4_root = tmp_path / "assets_gemma4_tf5102"
    molmo_root = tmp_path / "assets_molmo_tf457"
    qwen35_root = tmp_path / "assets_qwen35_tf5102" / "full_cache"
    molmo_cache_root = molmo_root / "full_cache" / "molmo-7b-d-0924"
    molmo_cache_root.mkdir(parents=True)
    qwen35_cache_root = qwen35_root / "qwen3.5-4b"
    qwen35_cache_root.mkdir(parents=True)
    captured: dict[str, Path] = {}
    config = {
        "models": [
            {
                "alias": "qwen3.5-4b",
                "route": "extract_separate_env",
                "extraction_env_name": "mind-gemma4-py311",
                "cache_output_root": str(qwen35_root),
            }
        ]
    }

    monkeypatch.setattr(accept_separate_env, "load_panel_config", lambda _config_path: config)
    monkeypatch.setattr(accept_separate_env, "resolve_output_root", lambda _config, _output_root: output_root)
    monkeypatch.setattr(accept_separate_env, "route_source_root", lambda _config, _route: None)
    monkeypatch.setattr(
        accept_separate_env,
        "route_models",
        lambda _config, route: (
            ["gemma-4-12b-it"]
            if route == "accept_existing_separate_env"
            else ["molmo-7b-d-0924", "qwen3.5-4b"]
        ),
    )
    monkeypatch.setattr(
        accept_separate_env,
        "route_extraction_env_name",
        lambda _config, route: "mind-gemma4-py311" if route == "accept_existing_separate_env" else "mind-molmo-py311",
    )
    monkeypatch.setattr(accept_separate_env, "write_csv", lambda _path, _rows: None)

    def fake_accept_existing_separate_env_model(
        *,
        model_alias: str,
        source_root: Path,
        output_root: Path,
        extraction_env_name: str,
    ) -> dict[str, Any]:
        assert extraction_env_name == "mind-gemma4-py311"
        captured[model_alias] = accept_separate_env.resolve_model_cache_root(source_root, model_alias)
        return _status_row(
            model_alias=model_alias,
            route="accept_existing_separate_env",
            status="accepted_existing_separate_env",
            source_cache_root=captured[model_alias],
        )

    def fake_accept_extracted_separate_env_model(
        *,
        model_alias: str,
        source_cache_root: Path,
        output_root: Path,
        extraction_env_name: str,
    ) -> dict[str, Any]:
        expected_env = "mind-gemma4-py311" if model_alias == "qwen3.5-4b" else "mind-molmo-py311"
        assert extraction_env_name == expected_env
        captured[model_alias] = source_cache_root
        return _status_row(
            model_alias=model_alias,
            route="extract_separate_env",
            status="extracted_separate_env",
            source_cache_root=source_cache_root,
        )

    monkeypatch.setattr(accept_separate_env, "accept_existing_separate_env_model", fake_accept_existing_separate_env_model)
    monkeypatch.setattr(accept_separate_env, "accept_extracted_separate_env_model", fake_accept_extracted_separate_env_model)

    rc = accept_separate_env.main(
        [
            "--config",
            str(tmp_path / "panel.yaml"),
            "--output-root",
            str(output_root),
            "--gemma4-root",
            str(gemma4_root),
            "--molmo-root",
            str(molmo_root),
        ]
    )

    assert rc == 0
    assert captured == {
        "gemma-4-12b-it": gemma4_root / "full_cache" / "gemma-4-12b-it",
        "molmo-7b-d-0924": molmo_root / "full_cache" / "molmo-7b-d-0924",
        "qwen3.5-4b": qwen35_root / "qwen3.5-4b",
    }
