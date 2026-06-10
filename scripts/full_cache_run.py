#!/usr/bin/env python3
"""Plan and run MIND Experiment 2 full-cache extraction routes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind import full_cache
from mind.models.registry import REQUIRED_MODEL_ALIASES


DEFAULT_CONFIG = Path("configs/full_cache/model_panel.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/full_cache")
DEFAULT_EXTRACTION_DTYPE = "float16"
EXTRACT_SCRIPT = Path("scripts/stage0_extract_full_layer_cache.py")
FULL_CACHE_DIRNAME = "full_cache"
MAIN_ENV_CACHE_DIR = Path("main_env") / "cache"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--execute-main-env", action="store_true")
    mode.add_argument("--execute-separate-env", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cache-output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--gpus", nargs="+", default=None)
    parser.add_argument("--python", default=sys.executable)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_panel_config(args.config)
    output_root = resolve_output_root(config, args.output_root)
    route_manifest = write_plan_artifacts(config=config, config_path=args.config, output_root=output_root)
    if args.plan_only:
        print_plan_summary(output_root=output_root, route_manifest=route_manifest)
        return 0
    if args.execute_main_env:
        return execute_route(
            config=config,
            config_path=args.config,
            output_root=output_root,
            requested_models=args.models,
            route_name="extract_main_env",
            manifest_route="extract_default_env",
            status_on_pass="extracted_default_env",
            cache_origin="default_env",
            extraction_env_name=route_extraction_env_name(config, "extract_main_env"),
            cache_output_root=args.cache_output_root,
            gpus=args.gpus,
            python=args.python,
        )
    return execute_route(
        config=config,
        config_path=args.config,
        output_root=output_root,
        requested_models=args.models,
        route_name="extract_separate_env",
        manifest_route="extract_separate_env",
        status_on_pass="extracted_separate_env",
        cache_origin="separate_env",
        extraction_env_name=route_extraction_env_name(config, "extract_separate_env"),
        cache_output_root=args.cache_output_root,
        gpus=args.gpus,
        python=args.python,
    )


def load_panel_config(config_path: Path) -> dict[str, Any]:
    config = full_cache.load_model_panel_config(config_path)
    aliases = configured_model_aliases(config)
    if aliases != list(REQUIRED_MODEL_ALIASES):
        raise ValueError(
            "full-cache model panel must match REQUIRED_MODEL_ALIASES exactly: "
            f"expected={list(REQUIRED_MODEL_ALIASES)!r} actual={aliases!r}"
        )
    return config


def resolve_output_root(config: Mapping[str, Any], output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root
    configured = config.get("output_root")
    return Path(str(configured)) if configured else DEFAULT_OUTPUT_ROOT


def configured_model_aliases(config: Mapping[str, Any]) -> list[str]:
    rows = config.get("models")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return []
    aliases: list[str] = []
    for row in rows:
        if isinstance(row, Mapping) and row.get("alias") is not None:
            aliases.append(str(row["alias"]))
    return aliases


def model_config_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    rows = config.get("models")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return paths
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        alias = row.get("alias")
        config_path = row.get("config_path")
        if alias is not None and config_path is not None:
            paths[str(alias)] = Path(str(config_path))
    return paths


def model_route_map(config: Mapping[str, Any]) -> dict[str, str]:
    routes: dict[str, str] = {}
    rows = config.get("models")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return routes
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        alias = row.get("alias")
        route = row.get("route")
        if alias is not None and route is not None:
            routes[str(alias)] = str(route)
    return routes


def model_settings(config: Mapping[str, Any], model_alias: str) -> dict[str, Any]:
    rows = config.get("models")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("alias", "")) == model_alias:
            return dict(row)
    return {}


def manifest_route_map(config: Mapping[str, Any]) -> dict[str, str]:
    routes: dict[str, str] = {}
    rows = config.get("models")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return routes
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        alias = row.get("alias")
        route = row.get("manifest_route") or row.get("route")
        if alias is not None and route is not None:
            value = str(route)
            routes[str(alias)] = "extract_default_env" if value == "extract_main_env" else value
    return routes


def route_settings(config: Mapping[str, Any], route_name: str) -> dict[str, Any]:
    routes = config.get("routes")
    if not isinstance(routes, Mapping):
        return {}
    value = routes.get(route_name)
    return dict(value) if isinstance(value, Mapping) else {}


def route_extraction_env_name(config: Mapping[str, Any], route_name: str) -> str | None:
    value = route_settings(config, route_name).get("extraction_env_name")
    return None if value is None else str(value)


def model_extraction_env_name(
    config: Mapping[str, Any],
    model_alias: str,
    route_name: str,
    *,
    route_default: str | None = None,
) -> str | None:
    value = model_settings(config, model_alias).get("extraction_env_name")
    if value is not None:
        return str(value)
    if route_default is not None:
        return str(route_default)
    return route_extraction_env_name(config, route_name)


def route_source_root(config: Mapping[str, Any], route_name: str) -> Path | None:
    value = route_settings(config, route_name).get("source_root")
    return None if value is None else Path(str(value))


def route_models(config: Mapping[str, Any], route_name: str) -> list[str]:
    settings = route_settings(config, route_name)
    value = settings.get("models")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    route_map = model_route_map(config)
    return [alias for alias in configured_model_aliases(config) if route_map.get(alias) == route_name]


def write_plan_artifacts(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    aliases = configured_model_aliases(config)
    manifest = full_cache.build_route_manifest(model_aliases=aliases)
    apply_panel_routes(manifest, config)
    apply_panel_config_paths(manifest, config)
    enrich_route_commands(manifest, config=config, config_path=config_path, output_root=output_root)
    manifests_dir = output_root / "manifests"
    route_json = manifests_dir / "model_extraction_routes.json"
    route_csv = manifests_dir / "model_extraction_routes.csv"
    plan_json = manifests_dir / "execution_plan.json"
    plan_csv = manifests_dir / "execution_plan.csv"
    full_cache.write_json_manifest(manifest, route_json)
    write_csv(route_csv, route_csv_rows(manifest))
    full_cache.write_json_manifest(
        {
            "schema_version": manifest.get("schema_version"),
            "extraction_started": manifest.get("extraction_started", False),
            "dataset_matrix": manifest.get("dataset_matrix", []),
            "execution_plan": manifest.get("execution_plan", []),
        },
        plan_json,
    )
    write_csv(plan_csv, execution_plan_csv_rows(manifest))
    return dict(manifest)


def apply_panel_config_paths(manifest: dict[str, Any], config: Mapping[str, Any]) -> None:
    paths_by_alias = {alias: str(path) for alias, path in model_config_paths(config).items()}
    for row in manifest.get("models", []):
        if not isinstance(row, dict):
            continue
        alias = str(row.get("model_alias", ""))
        if alias in paths_by_alias:
            row["config_path"] = paths_by_alias[alias]


def apply_panel_routes(manifest: dict[str, Any], config: Mapping[str, Any]) -> None:
    routes_by_alias = manifest_route_map(config)
    for row in manifest.get("models", []):
        if not isinstance(row, dict):
            continue
        alias = str(row.get("model_alias", ""))
        route = routes_by_alias.get(alias)
        if route is None:
            continue
        requires_extraction = route_requires_extraction(route)
        row["route"] = route
        row["status"] = "needs_extraction" if requires_extraction else "planned"
        plan = row.get("execution_plan")
        if isinstance(plan, dict):
            plan["requires_extraction"] = requires_extraction
            plan["action"] = route_action(route, alias)

    for step in manifest.get("execution_plan", []):
        if not isinstance(step, dict):
            continue
        alias = str(step.get("model_alias", ""))
        route = routes_by_alias.get(alias)
        if route is None:
            continue
        step["route"] = route
        step["status"] = "needs_extraction" if route_requires_extraction(route) else "planned"
        step["action"] = route_action(route, alias)


def route_requires_extraction(route: str) -> bool:
    return route in {"extract_default_env", "extract_separate_env"}


def route_action(route: str, model_alias: str) -> str:
    if route == "accept_existing_stage0":
        return f"accept existing Stage 0 cache for {model_alias}"
    if route == "accept_existing_separate_env":
        return f"accept existing separate-env cache for {model_alias}"
    if route == "extract_separate_env":
        return f"extract full cache in separate environment for {model_alias}"
    return f"extract full cache in main environment for {model_alias}"


def enrich_route_commands(
    manifest: dict[str, Any],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
) -> None:
    command_by_alias: dict[str, str] = {}
    for row in manifest.get("models", []):
        if not isinstance(row, dict):
            continue
        alias = str(row.get("model_alias", ""))
        route = str(row.get("route", ""))
        command = planned_command(alias, route, config=config, config_path=config_path, output_root=output_root)
        plan = row.get("execution_plan")
        if isinstance(plan, dict):
            plan["command"] = command
        command_by_alias[alias] = command
    for step in manifest.get("execution_plan", []):
        if isinstance(step, dict):
            alias = str(step.get("model_alias", ""))
            if alias in command_by_alias:
                step["command"] = command_by_alias[alias]


def planned_command(
    alias: str,
    route: str,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
) -> str:
    base = [sys.executable]
    if route == "accept_existing_stage0":
        return shlex.join(
            base
            + [
                "scripts/full_cache_accept_existing.py",
                "--config",
                str(config_path),
                "--output-root",
                str(output_root),
                "--models",
                alias,
            ]
        )
    if route == "accept_existing_separate_env":
        return shlex.join(
            base
            + [
                "scripts/full_cache_accept_separate_env.py",
                "--config",
                str(config_path),
                "--output-root",
                str(output_root),
                "--models",
                alias,
            ]
        )
    if route == "extract_separate_env":
        command = base + [
            "scripts/full_cache_run.py",
            "--execute-separate-env",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
        configured_cache_root = configured_model_cache_output_root(config, alias)
        if configured_cache_root is not None:
            command += ["--cache-output-root", str(configured_cache_root)]
        return shlex.join(command + ["--models", alias])
    command = base + [
        "scripts/full_cache_run.py",
        "--execute-main-env",
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ]
    configured_cache_root = configured_model_cache_output_root(config, alias)
    if configured_cache_root is not None:
        command += ["--cache-output-root", str(configured_cache_root)]
    return shlex.join(command + ["--models", alias])


def route_csv_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest.get("models", []):
        if not isinstance(row, Mapping):
            continue
        plan = row.get("execution_plan") if isinstance(row.get("execution_plan"), Mapping) else {}
        rows.append(
            {
                "model_alias": row.get("model_alias", ""),
                "route": row.get("route", ""),
                "status": row.get("status", ""),
                "config_path": row.get("config_path", ""),
                "requires_extraction": plan.get("requires_extraction", ""),
                "action": plan.get("action", ""),
                "command": plan.get("command", ""),
            }
        )
    return rows


def execution_plan_csv_rows(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in manifest.get("execution_plan", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {
                "model_alias": row.get("model_alias", ""),
                "route": row.get("route", ""),
                "status": row.get("status", ""),
                "action": row.get("action", ""),
                "command": row.get("command", ""),
            }
        )
    return rows


def execute_route(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_root: Path,
    requested_models: Sequence[str] | None,
    route_name: str,
    manifest_route: str,
    status_on_pass: str,
    cache_origin: str,
    extraction_env_name: str | None,
    gpus: Sequence[str] | None,
    python: str,
    cache_output_root: Path | None = None,
) -> int:
    del config_path
    allowed_models = route_models(config, route_name)
    models = filter_requested_models(allowed_models, requested_models=requested_models, route_name=route_name)
    devices = resolve_devices(gpus)
    max_workers = max(1, min(2, len(devices), len(models)))
    config_paths = model_config_paths(config)
    logs_dir = output_root / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for index, model_alias in enumerate(models):
            device = devices[index % len(devices)]
            model_cache_output_root = cache_output_root_for_model(
                config=config,
                output_root=output_root,
                model_alias=model_alias,
                route_name=route_name,
                cache_output_root=cache_output_root,
            )
            model_env_name = model_extraction_env_name(
                config,
                model_alias,
                route_name,
                route_default=extraction_env_name,
            )
            futures.append(
                executor.submit(
                    run_model_extraction_job,
                    model_alias=model_alias,
                    model_config_path=config_paths[model_alias],
                    output_root=output_root,
                    cache_output_root=model_cache_output_root,
                    route=manifest_route,
                    status_on_pass=status_on_pass,
                    cache_origin=cache_origin,
                    extraction_env_name=model_env_name,
                    device=device,
                    python=python,
                    log_path=logs_dir / f"{model_alias}_{route_name}.log",
                )
            )
        for future in as_completed(futures):
            results.append(future.result())
    write_csv(output_root / "reports" / f"{route_name}_execution_status.csv", results)
    failed = [row for row in results if str(row.get("status")) not in {status_on_pass, "skipped_valid"}]
    for row in sorted(results, key=lambda item: str(item.get("model_alias"))):
        print(
            "model={model_alias} route={route} status={status} log={log_path}".format(
                **{key: row.get(key, "") for key in ("model_alias", "route", "status", "log_path")}
            )
        )
    return 2 if failed else 0


def filter_requested_models(
    allowed_models: Sequence[str],
    *,
    requested_models: Sequence[str] | None,
    route_name: str,
) -> list[str]:
    allowed = list(allowed_models)
    if not requested_models:
        return allowed
    requested = [str(model) for model in requested_models]
    unknown = [model for model in requested if model not in allowed]
    if unknown:
        raise ValueError(f"models are not on route {route_name}: {unknown}")
    return requested


def resolve_devices(gpus: Sequence[str] | None) -> list[str]:
    values: list[str]
    if gpus:
        values = [str(item) for item in gpus]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        values = [item.strip() for item in visible.split(",") if item.strip()] if visible else ["0"]
    devices = [value if value.startswith("cuda") else f"cuda:{value}" for value in values]
    return devices[:2] or ["cuda:0"]


def extraction_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_model_extraction_job(
    *,
    model_alias: str,
    model_config_path: Path,
    output_root: Path,
    cache_output_root: Path,
    route: str,
    status_on_pass: str,
    cache_origin: str,
    extraction_env_name: str | None,
    device: str,
    python: str,
    log_path: Path,
) -> dict[str, Any]:
    model_cache_root = cache_output_root / model_alias
    validation = validate_existing_root(
        model_alias=model_alias,
        cache_root=model_cache_root,
        cache_origin=cache_origin,
        extraction_env_name=extraction_env_name,
    )
    if validation.get("status") == "passed":
        manifest = model_manifest_from_validation(
            model_alias=model_alias,
            route=route,
            status=status_on_pass,
            cache_origin=cache_origin,
            cache_root=model_cache_root,
            validation=validation,
            extraction_env_name=extraction_env_name,
            log_path=log_path,
            failed_reason="",
        )
        write_model_manifest(output_root, model_alias, manifest)
        return {
            "model_alias": model_alias,
            "route": route,
            "status": "skipped_valid",
            "cache_root": str(model_cache_root),
            "log_path": str(log_path),
        }

    commands = extraction_commands(
        model_alias=model_alias,
        model_config_path=model_config_path,
        cache_output_root=cache_output_root,
        device=device,
        python=python,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        for command in commands:
            log_handle.write("$ " + shlex.join(command) + "\n")
            log_handle.flush()
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                env=extraction_subprocess_env(),
            )
            log_handle.write(f"exit_code={result.returncode}\n")
            log_handle.flush()
            if result.returncode != 0:
                failed = failed_model_manifest(
                    model_alias=model_alias,
                    route=route,
                    status="failed_validation",
                    cache_origin=cache_origin,
                    cache_root=model_cache_root,
                    failed_reason=f"extraction command failed with exit_code={result.returncode}",
                    validation=validation,
                    extraction_env_name=extraction_env_name,
                    log_path=log_path,
                )
                write_model_manifest(output_root, model_alias, failed)
                return {
                    "model_alias": model_alias,
                    "route": route,
                    "status": "failed_validation",
                    "cache_root": str(model_cache_root),
                    "log_path": str(log_path),
                    "failed_reason": failed["failed_reason"],
                }

    validation = validate_existing_root(
        model_alias=model_alias,
        cache_root=model_cache_root,
        cache_origin=cache_origin,
        extraction_env_name=extraction_env_name,
    )
    status = status_on_pass if validation.get("status") == "passed" else "failed_validation"
    failed_reason = "" if status == status_on_pass else "; ".join(str(item) for item in validation.get("errors", [])[:3])
    manifest = model_manifest_from_validation(
        model_alias=model_alias,
        route=route,
        status=status,
        cache_origin=cache_origin,
        cache_root=model_cache_root,
        validation=validation,
        extraction_env_name=extraction_env_name,
        log_path=log_path,
        failed_reason=failed_reason,
    )
    write_model_manifest(output_root, model_alias, manifest)
    return {
        "model_alias": model_alias,
        "route": route,
        "status": status,
        "cache_root": str(model_cache_root),
        "log_path": str(log_path),
        "failed_reason": failed_reason,
    }


def extraction_commands(
    *,
    model_alias: str,
    model_config_path: Path,
    cache_output_root: Path,
    device: str,
    python: str,
) -> list[list[str]]:
    commands: list[list[str]] = []
    dtype = extraction_dtype_from_model_config(model_config_path)
    for dataset_name, subset in full_cache.DATASET_MATRIX:
        command = [
            python,
            str(EXTRACT_SCRIPT),
            "--records",
            str(resolve_records_path(dataset_name, subset)),
            "--model-config",
            str(model_config_path),
            "--output-root",
            str(cache_output_root),
            "--dataset-name",
            dataset_name,
            "--subset",
            subset,
            "--split",
            subset,
            "--device",
            device,
            "--dtype",
            dtype,
            "--max-new-tokens",
            "1",
            "--token-index",
            "-1",
            "--limit",
            "0",
            "--shard-size",
            "128",
            "--batch-size",
            "1",
        ]
        commands.append(command)
    del model_alias
    return commands


def extraction_dtype_from_model_config(model_config_path: Path) -> str:
    path = model_config_path if model_config_path.is_absolute() else REPO_ROOT / model_config_path
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"model config must be a YAML mapping: {model_config_path}")
    dtype = payload.get("dtype")
    return DEFAULT_EXTRACTION_DTYPE if dtype is None else str(dtype)


def resolve_records_path(dataset_name: str, subset: str) -> Path:
    candidates = [
        Path("outputs/stage0/normalized") / dataset_name / f"{subset}.jsonl",
        Path("outputs/round2_2026_04/normalized") / dataset_name / f"{subset}.jsonl",
        Path("data") / dataset_name / f"{subset}.jsonl",
    ]
    for candidate in candidates:
        if (REPO_ROOT / candidate).exists():
            return candidate
    return candidates[0]


def output_full_cache_root(output_root: Path) -> Path:
    return output_root / FULL_CACHE_DIRNAME


def output_cache_root_for_route(output_root: Path, route_name: str) -> Path:
    if route_name == "extract_main_env":
        return output_root / MAIN_ENV_CACHE_DIR
    return output_full_cache_root(output_root)


def configured_model_cache_output_root(config: Mapping[str, Any], model_alias: str) -> Path | None:
    value = model_settings(config, model_alias).get("cache_output_root")
    return None if value is None else Path(str(value))


def cache_output_root_for_model(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    model_alias: str,
    route_name: str,
    cache_output_root: Path | None = None,
) -> Path:
    if cache_output_root is not None:
        return cache_output_root
    configured = configured_model_cache_output_root(config, model_alias)
    if configured is not None:
        return configured
    return output_cache_root_for_route(output_root, route_name)


def validate_existing_root(
    *,
    model_alias: str,
    cache_root: Path,
    cache_origin: str,
    extraction_env_name: str | None,
) -> dict[str, Any]:
    return full_cache.validate_full_cache_root(
        cache_root=cache_root,
        expected_model_alias=model_alias,
        cache_origin=cache_origin,
        extraction_env_name=extraction_env_name,
        raise_on_error=False,
    )


def model_manifest_from_validation(
    *,
    model_alias: str,
    route: str,
    status: str,
    cache_origin: str,
    cache_root: Path,
    validation: Mapping[str, Any],
    extraction_env_name: str | None,
    log_path: Path | None,
    failed_reason: str,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema_version": full_cache.MODEL_MANIFEST_SCHEMA_VERSION,
        "model_alias": model_alias,
        "route": route,
        "status": status,
        "cache_origin": cache_origin,
        "cache_root": str(cache_root),
        "total_entries": int(validation.get("total_entries") or 0),
        "num_shards": int(validation.get("num_shards") or 0),
        "datasets": validation.get("datasets", {}),
        "validation_status": validation.get("status"),
        "validation_errors": validation.get("errors", []),
        "failed_reason": failed_reason,
    }
    if extraction_env_name is not None:
        manifest["extraction_env_name"] = extraction_env_name
    if log_path is not None:
        manifest["log_path"] = str(log_path)
    return manifest


def failed_model_manifest(
    *,
    model_alias: str,
    route: str,
    status: str,
    cache_origin: str,
    cache_root: Path,
    failed_reason: str,
    validation: Mapping[str, Any] | None = None,
    extraction_env_name: str | None = None,
    log_path: Path | None = None,
) -> dict[str, Any]:
    validation = validation or {}
    return model_manifest_from_validation(
        model_alias=model_alias,
        route=route,
        status=status,
        cache_origin=cache_origin,
        cache_root=cache_root,
        validation=validation,
        extraction_env_name=extraction_env_name,
        log_path=log_path,
        failed_reason=failed_reason,
    )


def write_model_manifest(output_root: Path, model_alias: str, manifest: Mapping[str, Any]) -> Path:
    manifest_path = output_root / model_alias / "full_cache_extraction_manifest.json"
    payload = dict(manifest)
    payload["manifest_path"] = str(manifest_path)
    full_cache.write_json_manifest(payload, manifest_path)
    return manifest_path


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in materialized for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            writer.writerow(row)


def print_plan_summary(*, output_root: Path, route_manifest: Mapping[str, Any]) -> None:
    manifests_dir = output_root / "manifests"
    print("full-cache plan written")
    print(f"routes_json={manifests_dir / 'model_extraction_routes.json'}")
    print(f"routes_csv={manifests_dir / 'model_extraction_routes.csv'}")
    print(f"execution_plan_json={manifests_dir / 'execution_plan.json'}")
    print(f"execution_plan_csv={manifests_dir / 'execution_plan.csv'}")
    print(f"models={len(route_manifest.get('models', []))}")


if __name__ == "__main__":
    raise SystemExit(main())
