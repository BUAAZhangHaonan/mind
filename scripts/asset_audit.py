#!/usr/bin/env python3
"""Audit local Experiment 1 model assets without loading model weights."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.models.asset_validation import AssetStatus, audit_asset_metadata, build_completion_summary
from mind.models.registry import load_asset_registry


BATCH1_TARGET_ALIASES = ("qwen2.5-vl-7b", "qwen3.5-4b", "qwen3.5-9b", "internvl3.5-8b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Reserved for explicit heavy checks. The default audit never loads weights.",
    )
    return parser


def run_audit(*, registry_path: Path, output_root: Path, load_model: bool = False) -> list[dict[str, object]]:
    if load_model:
        raise ValueError("--load-model is not implemented for the lightweight Experiment 1 audit")
    output_root.mkdir(parents=True, exist_ok=True)
    registry = load_asset_registry(registry_path)
    results = [audit_asset_metadata(model).as_dict() for model in registry.models]

    inventory_fields = [
        "alias",
        "local_path",
        "path_exists",
        "path_is_directory",
        "config_exists",
        "processor_tokenizer_assets",
        "model_family_detected",
        "architecture_detected",
        "status",
        "reason",
    ]
    capability_fields = [
        "alias",
        "model_family_detected",
        "architecture_detected",
        "moe_indicators",
        "thinking_detected",
        "thinking_disable_argument",
        "dtype",
        "trust_remote_code_required",
        "local_loading_class_candidate",
        "image_processor_candidate",
        "total_layers",
        "hidden_dim",
        "output_hidden_states_support",
        "generation_api_support",
        "hidden_state_index_offset",
        "prompt_template_id",
        "status",
        "reason",
    ]
    _write_csv(output_root / "asset_inventory.csv", results, inventory_fields)
    _write_csv(output_root / "model_capability_matrix.csv", results, capability_fields)

    unsupported = [
        row
        for row in results
        if row["status"] in {AssetStatus.UNSUPPORTED_BY_POLICY.value, AssetStatus.UNSUPPORTED_BY_WRAPPER.value}
    ]
    blocked = [row for row in results if row["status"] == AssetStatus.BLOCKED.value]
    failed = [row for row in results if row["status"] == AssetStatus.FAILED_VALIDATION.value]
    model_statuses = {str(row["alias"]): str(row["status"]) for row in results}
    model_reasons = {str(row["alias"]): str(row["reason"]) for row in results}
    summary = build_completion_summary(
        model_statuses=model_statuses,
        model_reasons=model_reasons,
        smoke_datasets=[],
        smoke_limit=0,
        tests_run=[],
        git_commit=get_git_commit(),
    )
    manifest = {
        "registry": str(registry_path),
        "output_root": str(output_root),
        "load_model": False,
        "git_commit": get_git_commit(),
        "models": results,
        "summary": summary,
    }
    _write_json(output_root / "model_asset_manifest.json", manifest)
    _write_json(output_root / "unsupported_models.json", unsupported)
    _write_json(output_root / "blocked_models.json", blocked)
    if failed:
        _write_json(output_root / "failed_models.json", failed)
    inspection_rows = build_wrapper_batch1_inspection(registry.models)
    inspection_fields = [
        "alias",
        "local_path",
        "config_json_path",
        "model_type",
        "architectures",
        "auto_map",
        "tokenizer_files",
        "processor_files",
        "chat_template_available",
        "image_processor_available",
        "generation_config_available",
        "hidden_size_candidates",
        "layer_count_candidates",
        "moe_indicators",
        "thinking_indicators",
        "required_trust_remote_code",
        "candidate_model_class",
        "candidate_processor_class",
        "status",
        "reason",
    ]
    _write_csv(output_root / "wrapper_batch1_asset_inspection.csv", inspection_rows, inspection_fields)
    _write_json(output_root / "wrapper_batch1_asset_inspection.json", inspection_rows)
    return results


def build_wrapper_batch1_inspection(models: list[object]) -> list[dict[str, object]]:
    return [_inspect_batch1_asset(model) for model in models if getattr(model, "alias", "") in BATCH1_TARGET_ALIASES]


def _inspect_batch1_asset(model: object) -> dict[str, object]:
    alias = str(getattr(model, "alias", ""))
    local_path = Path(str(getattr(model, "local_path", "")))
    row: dict[str, object] = {
        "alias": alias,
        "local_path": str(local_path),
        "config_json_path": str(local_path / "config.json"),
        "model_type": "",
        "architectures": [],
        "auto_map": {},
        "tokenizer_files": [],
        "processor_files": [],
        "chat_template_available": False,
        "image_processor_available": False,
        "generation_config_available": False,
        "hidden_size_candidates": {},
        "layer_count_candidates": {},
        "moe_indicators": [],
        "thinking_indicators": [],
        "required_trust_remote_code": bool(getattr(model, "trust_remote_code", False)),
        "candidate_model_class": "",
        "candidate_processor_class": "",
        "status": "blocked",
        "reason": "",
    }
    if not local_path.is_dir():
        row["reason"] = f"local path is missing or not a directory: {local_path}"
        return row
    config_path = local_path / "config.json"
    if not config_path.is_file():
        row["reason"] = f"config.json is missing: {config_path}"
        return row
    config = json.loads(config_path.read_text(encoding="utf-8"))
    row["model_type"] = config.get("model_type", "")
    row["architectures"] = config.get("architectures", [])
    row["auto_map"] = config.get("auto_map", {})
    row["tokenizer_files"] = sorted(
        path.name
        for path in local_path.iterdir()
        if path.name.startswith("tokenizer") or path.name in {"vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"}
    )
    row["processor_files"] = sorted(
        path.name
        for path in local_path.iterdir()
        if path.name.startswith("processor")
        or path.name.startswith("preprocessor")
        or path.name.startswith("image_processing")
        or path.name.startswith("video_preprocessor")
    )
    row["chat_template_available"] = any((local_path / filename).is_file() for filename in ("chat_template.json", "chat_template.jinja"))
    row["image_processor_available"] = any((local_path / filename).is_file() for filename in ("preprocessor_config.json", "processor_config.json"))
    row["generation_config_available"] = (local_path / "generation_config.json").is_file()
    row["hidden_size_candidates"] = _candidate_config_values(config, ("hidden_size",), ("text_config", "hidden_size"), ("llm_config", "hidden_size"))
    row["layer_count_candidates"] = _candidate_config_values(config, ("num_hidden_layers",), ("text_config", "num_hidden_layers"), ("llm_config", "num_hidden_layers"))
    row["moe_indicators"] = _find_keys(config, {"num_experts", "n_routed_experts", "moe", "experts_per_tok", "num_local_experts", "router_aux_loss_coef"})
    row["thinking_indicators"] = _thinking_indicator_files(local_path)
    row["required_trust_remote_code"] = bool(config.get("auto_map")) or bool(getattr(model, "trust_remote_code", False))
    row["candidate_model_class"] = _candidate_model_class(alias, config)
    row["candidate_processor_class"] = _candidate_processor_class(local_path, alias)
    row["status"] = "inspected"
    row["reason"] = "metadata inspection completed"
    return row


def _candidate_config_values(config: dict[str, object], *paths: tuple[str, ...]) -> dict[str, object]:
    values: dict[str, object] = {}
    for path in paths:
        current: object = config
        for key in path:
            if not isinstance(current, dict):
                current = None
                break
            current = current.get(key)
        if current is not None:
            values[".".join(path)] = current
    return values


def _find_keys(payload: object, keys: set[str], prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            label = key if not prefix else f"{prefix}.{key}"
            if key.lower() in keys:
                found.append(label)
            found.extend(_find_keys(value, keys, label))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_find_keys(value, keys, f"{prefix}[{index}]"))
    return found


def _thinking_indicator_files(path: Path) -> list[str]:
    markers = ("<think>", "</think>", "enable_thinking", "reasoning_content", "/nothink")
    files: list[str] = []
    for filename in ("chat_template.json", "chat_template.jinja", "tokenizer_config.json", "README.md"):
        candidate = path / filename
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore").lower()
        if any(marker in text for marker in markers):
            files.append(filename)
    return files


def _candidate_model_class(alias: str, config: dict[str, object]) -> str:
    if alias == "qwen2.5-vl-7b":
        return "Qwen2_5_VLForConditionalGeneration"
    if alias in {"qwen3.5-4b", "qwen3.5-9b"}:
        return "Qwen3_5ForConditionalGeneration"
    if alias == "internvl3.5-8b":
        return "InternVLChatModel"
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return ""


def _candidate_processor_class(path: Path, alias: str) -> str:
    if alias == "internvl3.5-8b":
        return "InternVLLocalProcessor"
    for filename in ("processor_config.json", "preprocessor_config.json"):
        candidate = path / filename
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        processor_class = payload.get("processor_class")
        if processor_class:
            return str(processor_class)
    return ""


def _write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _csv_value(value: object) -> object:
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True)
    return value


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        rows = run_audit(
            registry_path=args.registry,
            output_root=args.output_root,
            load_model=args.load_model,
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"asset_audit models={len(rows)} output_root={args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
