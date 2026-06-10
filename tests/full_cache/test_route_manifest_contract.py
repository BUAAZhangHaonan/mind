from __future__ import annotations

from pathlib import Path

from mind.models.registry import REQUIRED_MODEL_ALIASES

from .conftest import SEPARATE_ENV_MODELS, STAGE0_ACCEPT_MODELS, VALID_ROUTES, full_cache_attr


def test_route_manifest_covers_all_16_final_panel_models_exactly_once(tmp_path: Path) -> None:
    build_route_manifest = full_cache_attr("build_route_manifest")

    manifest = build_route_manifest(output=tmp_path / "route_manifest.json")

    aliases = [model["model_alias"] for model in manifest["models"]]
    assert len(REQUIRED_MODEL_ALIASES) == 16
    assert aliases == list(REQUIRED_MODEL_ALIASES)
    assert len(aliases) == len(set(aliases))
    assert (tmp_path / "route_manifest.json").is_file()


def test_route_manifest_uses_only_valid_routes_and_keeps_acceptance_modes_distinct(tmp_path: Path) -> None:
    build_route_manifest = full_cache_attr("build_route_manifest")

    manifest = build_route_manifest(output=tmp_path / "route_manifest.json")

    routes = {model["model_alias"]: model["route"] for model in manifest["models"]}
    assert set(routes.values()) <= VALID_ROUTES
    assert {routes[alias] for alias in STAGE0_ACCEPT_MODELS} == {"accept_existing_stage0"}
    assert all(routes[alias] != "accept_existing_stage0" for alias in SEPARATE_ENV_MODELS)
    assert {routes[alias] for alias in SEPARATE_ENV_MODELS} <= {
        "accept_existing_separate_env",
        "extract_separate_env",
    }


def test_route_manifest_contains_execution_plan_before_any_extraction(tmp_path: Path) -> None:
    build_route_manifest = full_cache_attr("build_route_manifest")

    manifest = build_route_manifest(output=tmp_path / "route_manifest.json")

    assert manifest["extraction_started"] is False
    planned_aliases = [step["model_alias"] for step in manifest["execution_plan"]]
    assert planned_aliases == list(REQUIRED_MODEL_ALIASES)
    for step in manifest["execution_plan"]:
        assert step["route"] in VALID_ROUTES
        assert step["status"] in {"planned", "needs_extraction"}
        assert step["command"] or step["action"]
