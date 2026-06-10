from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

full_cache_run = importlib.import_module("full_cache_run")


def test_extraction_commands_do_not_add_image_root_for_normalized_records(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    records_path = tmp_path / "outputs" / "stage0" / "normalized" / "pope" / "popular.jsonl"
    records_path.parent.mkdir(parents=True)
    records_path.write_text(
        '{"image_path":"data/coco/val2014/COCO_val2014_000000000001.jpg"}\n',
        encoding="utf-8",
    )
    (tmp_path / "data" / "coco" / "val2014").mkdir(parents=True)
    model_config_path = tmp_path / "configs" / "models" / "molmo_7b_d_0924_asset.yaml"
    model_config_path.parent.mkdir(parents=True)
    model_config_path.write_text("name: molmo-7b-d-0924\ndtype: float16\n", encoding="utf-8")

    monkeypatch.setattr(full_cache_run, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(full_cache_run.full_cache, "DATASET_MATRIX", (("pope", "popular"),))

    commands = full_cache_run.extraction_commands(
        model_alias="molmo-7b-d-0924",
        model_config_path=Path("configs/models/molmo_7b_d_0924_asset.yaml"),
        cache_output_root=Path("outputs/full_cache/full_cache"),
        device="cuda:0",
        python="python",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--records") + 1] == str(
        Path("outputs/stage0/normalized/pope/popular.jsonl")
    )
    assert "--image-root" not in command


def test_execute_separate_env_uses_molmo_asset_config_path(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config = full_cache_run.load_panel_config(Path("configs/full_cache/model_panel.yaml"))
    captured: dict[str, Path] = {}

    def fake_run_model_extraction_job(**kwargs: Any) -> dict[str, Any]:
        captured["model_config_path"] = kwargs["model_config_path"]
        return {
            "model_alias": kwargs["model_alias"],
            "route": kwargs["route"],
            "status": kwargs["status_on_pass"],
            "cache_root": str(kwargs["output_root"] / "full_cache" / kwargs["model_alias"]),
            "log_path": str(kwargs["log_path"]),
            "failed_reason": "",
        }

    monkeypatch.setattr(full_cache_run, "run_model_extraction_job", fake_run_model_extraction_job)

    rc = full_cache_run.execute_route(
        config=config,
        config_path=Path("configs/full_cache/model_panel.yaml"),
        output_root=tmp_path,
        requested_models=["molmo-7b-d-0924"],
        route_name="extract_separate_env",
        manifest_route="extract_separate_env",
        status_on_pass="extracted_separate_env",
        cache_origin="separate_env",
        extraction_env_name="mind-molmo-py311",
        gpus=["0"],
        python="python",
    )

    assert rc == 0
    assert captured["model_config_path"] == Path("configs/models/molmo_7b_d_0924_asset.yaml")

    route_map = full_cache_run.model_route_map(config)
    main_env_paths = {
        alias: path
        for alias, path in full_cache_run.model_config_paths(config).items()
        if route_map.get(alias) == "extract_main_env"
    }
    assert main_env_paths
    assert all(
        path != Path("configs/models/molmo_7b_d_0924_asset.yaml")
        for path in main_env_paths.values()
    )


def test_qwen35_4b_is_not_a_main_env_model() -> None:
    config = full_cache_run.load_panel_config(Path("configs/full_cache/model_panel.yaml"))

    route_map = full_cache_run.model_route_map(config)

    assert route_map["qwen3.5-4b"] == "extract_separate_env"
    assert "qwen3.5-4b" not in full_cache_run.route_models(config, "extract_main_env")


def test_execute_separate_env_uses_qwen35_4b_asset_config_path_and_gemma4_env(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    config = full_cache_run.load_panel_config(Path("configs/full_cache/model_panel.yaml"))
    captured: dict[str, Any] = {}

    def fake_run_model_extraction_job(**kwargs: Any) -> dict[str, Any]:
        captured["model_config_path"] = kwargs["model_config_path"]
        captured["cache_output_root"] = kwargs["cache_output_root"]
        captured["extraction_env_name"] = kwargs["extraction_env_name"]
        return {
            "model_alias": kwargs["model_alias"],
            "route": kwargs["route"],
            "status": kwargs["status_on_pass"],
            "cache_root": str(kwargs["cache_output_root"] / kwargs["model_alias"]),
            "log_path": str(kwargs["log_path"]),
            "failed_reason": "",
        }

    monkeypatch.setattr(full_cache_run, "run_model_extraction_job", fake_run_model_extraction_job)

    rc = full_cache_run.execute_route(
        config=config,
        config_path=Path("configs/full_cache/model_panel.yaml"),
        output_root=tmp_path,
        requested_models=["qwen3.5-4b"],
        route_name="extract_separate_env",
        manifest_route="extract_separate_env",
        status_on_pass="extracted_separate_env",
        cache_origin="separate_env",
        extraction_env_name=None,
        cache_output_root=None,
        gpus=["0"],
        python="python",
    )

    assert rc == 0
    assert captured["model_config_path"] == Path("configs/models/qwen3_5_4b_asset.yaml")
    assert captured["cache_output_root"] == Path("outputs/assets_qwen35_tf5102/full_cache")
    assert captured["extraction_env_name"] == "mind-gemma4-py311"


def test_plan_artifacts_use_panel_model_config_paths(tmp_path: Path) -> None:
    config_path = Path("configs/full_cache/model_panel.yaml")
    config = full_cache_run.load_panel_config(config_path)

    manifest = full_cache_run.write_plan_artifacts(
        config=config,
        config_path=config_path,
        output_root=tmp_path,
    )

    paths = {row["model_alias"]: row["config_path"] for row in manifest["models"]}
    routes = {row["model_alias"]: row["route"] for row in manifest["models"]}
    commands = {
        row["model_alias"]: row["execution_plan"]["command"]
        for row in manifest["models"]
    }
    assert paths["molmo-7b-d-0924"] == "configs/models/molmo_7b_d_0924_asset.yaml"
    assert paths["qwen3.5-4b"] == "configs/models/qwen3_5_4b_asset.yaml"
    assert routes["qwen3.5-4b"] == "extract_separate_env"
    assert "--execute-main-env" not in commands["qwen3.5-4b"]
    assert "--execute-separate-env" in commands["qwen3.5-4b"]
    assert "--cache-output-root outputs/assets_qwen35_tf5102/full_cache" in commands["qwen3.5-4b"]


@pytest.mark.parametrize(
    ("model_alias", "model_config_path"),
    [
        ("gemma-3-4b-it", Path("configs/models/gemma_3_4b_it.yaml")),
        ("gemma-3-12b-it", Path("configs/models/gemma_3_12b_it.yaml")),
    ],
)
def test_extraction_commands_use_model_config_dtype_for_gemma3(
    monkeypatch: Any,
    model_alias: str,
    model_config_path: Path,
) -> None:
    monkeypatch.setattr(
        full_cache_run,
        "resolve_records_path",
        lambda _dataset_name, _subset: Path("outputs/stage0/normalized/pope/popular.jsonl"),
    )
    monkeypatch.setattr(full_cache_run.full_cache, "DATASET_MATRIX", (("pope", "popular"),))

    commands = full_cache_run.extraction_commands(
        model_alias=model_alias,
        model_config_path=model_config_path,
        cache_output_root=Path("outputs/full_cache/main_env/cache"),
        device="cuda:0",
        python="python",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--dtype") + 1] == "bfloat16"


def test_extraction_commands_default_to_float16_when_model_config_omits_dtype(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    model_config_path = tmp_path / "model_without_dtype.yaml"
    model_config_path.write_text("name: no-dtype\n", encoding="utf-8")
    monkeypatch.setattr(
        full_cache_run,
        "resolve_records_path",
        lambda _dataset_name, _subset: Path("outputs/stage0/normalized/pope/popular.jsonl"),
    )
    monkeypatch.setattr(full_cache_run.full_cache, "DATASET_MATRIX", (("pope", "popular"),))

    commands = full_cache_run.extraction_commands(
        model_alias="no-dtype",
        model_config_path=model_config_path,
        cache_output_root=Path("outputs/full_cache/main_env/cache"),
        device="cuda:0",
        python="python",
    )

    assert len(commands) == 1
    command = commands[0]
    assert command[command.index("--dtype") + 1] == "float16"


def test_extraction_subprocess_disables_user_site_packages(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    validation_results = iter(({"status": "failed"}, {"status": "passed"}))
    captured_envs: list[dict[str, str] | None] = []

    monkeypatch.setattr(
        full_cache_run,
        "validate_existing_root",
        lambda **_kwargs: next(validation_results),
    )
    monkeypatch.setattr(
        full_cache_run,
        "extraction_commands",
        lambda **_kwargs: [["python", "extract.py"]],
    )
    monkeypatch.setattr(full_cache_run, "write_model_manifest", lambda *_args, **_kwargs: None)

    def fake_run(*_args: Any, **kwargs: Any) -> types.SimpleNamespace:
        captured_envs.append(kwargs.get("env"))
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr(full_cache_run.subprocess, "run", fake_run)

    result = full_cache_run.run_model_extraction_job(
        model_alias="phi-4-multimodal-instruct",
        model_config_path=Path("configs/models/phi_4_multimodal_instruct.yaml"),
        output_root=tmp_path,
        cache_output_root=tmp_path / "cache",
        route="extract_default_env",
        status_on_pass="extracted_default_env",
        cache_origin="default_env",
        extraction_env_name="mind-py311",
        device="cuda:0",
        python="python",
        log_path=tmp_path / "logs" / "phi4.log",
    )

    assert result["status"] == "extracted_default_env"
    assert captured_envs
    assert captured_envs[0] is not None
    assert captured_envs[0]["PYTHONNOUSERSITE"] == "1"
