#!/usr/bin/env python3
"""Run Stage 0 audit, split, smoke extraction, and cache validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import shlex
import sys
from typing import Iterable, Mapping, Sequence

import yaml

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.models.types import resolve_torch_dtype
from mind.trajectory.audit import build_cache_label_balance_rows, run_audit, validate_required_datasets
from mind.trajectory.cache import validate_stage0_cache
from mind.trajectory.dataset import (
    DatasetSpec,
    discover_known_datasets,
    load_dataset_records,
    load_materializable_normalized_rows,
    materialize_missing_normalized_dataset_specs,
    normalized_dataset_path,
    planned_missing_normalized_dataset_materializations,
    raw_dataset_path,
    validate_extraction_ready_dataset_specs,
    validate_extraction_ready_rows,
)
from mind.trajectory.metadata import DATASET_SUBSETS
from mind.trajectory import splits as stage0_splits
from mind.trajectory.splits import build_split_manifest, write_split_manifest

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from stage0_extract_full_layer_cache import run_extraction


MODEL_ALIASES: dict[str, dict[str, str]] = {
    "qwen3-vl-8b": {
        "config": "configs/models/qwen3_vl_8b.yaml",
        "model_name": "qwen3-vl-8b",
    },
    "internvl3.5-8b": {
        "config": "configs/models/internvl3_5_8b.yaml",
        "model_name": "internvl3.5-8b",
    },
}
KNOWN_DATASETS = set(DATASET_SUBSETS)
KNOWN_SUBSETS = {subset for subsets in DATASET_SUBSETS.values() for subset in subsets}
DATASET_IMAGE_ROOTS = {
    "pope": Path("data/coco/val2014"),
    "repope": Path("data/coco/val2014"),
    "dash-b": Path("data/dash_b"),
}
PRIMARY_STAGE0_MODELS = ("qwen3-vl-8b", "internvl3.5-8b")
POPE_FULL_RUN_SUBSETS = ("popular", "random", "adversarial")
COMPLETE_STAGE0_CONFIG = Path("configs/stage0/stage0_complete.yaml")
FULL_CLOSURE_DATASET_SUBSETS = (
    ("pope", "popular"),
    ("pope", "random"),
    ("pope", "adversarial"),
    ("repope", "popular"),
    ("repope", "random"),
    ("repope", "adversarial"),
    ("dash-b", "all"),
)
SUMMARY_KEYS = (
    "stage",
    "status",
    "git_commit",
    "models_checked",
    "datasets_checked",
    "smoke_cache_paths",
    "audit_outputs",
    "split_manifest",
    "split_manifests",
    "cache_manifest",
    "required_cache_matrix",
    "completed_cache_matrix",
    "missing_cache_matrix",
    "cache_label_balance",
    "blocking_issues",
    "next_recommended_commands",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--smoke-limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--token-index", type=int, default=None)
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve Stage 0 commands and summary intent without writing artifacts or extracting cache.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    apply_config_defaults(args, parser=parser)
    return args


def apply_config_defaults(args: argparse.Namespace, *, parser: argparse.ArgumentParser) -> None:
    args.config_model_paths = {}
    args.config_dataset_specs = []
    args.split_seed = int(stage0_splits.DEFAULT_SEED)
    args.split_group_key = "image_id"
    args.split_ratios = list(stage0_splits.DEFAULT_RATIOS)
    if args.config is not None:
        payload = load_stage0_config(args.config)
        if args.output_root is None and payload.get("output_root") is not None:
            args.output_root = Path(str(payload["output_root"]))
        if args.smoke_limit is None and payload.get("smoke_limit") is not None:
            args.smoke_limit = int(payload["smoke_limit"])  # type: ignore[arg-type]
        if args.device is None and payload.get("device") is not None:
            args.device = str(payload["device"])
        if args.dtype is None and payload.get("dtype") is not None:
            args.dtype = str(payload["dtype"])
        if args.max_new_tokens is None and payload.get("max_new_tokens") is not None:
            args.max_new_tokens = int(payload["max_new_tokens"])  # type: ignore[arg-type]
        if args.token_index is None and payload.get("token_index") is not None:
            args.token_index = int(payload["token_index"])  # type: ignore[arg-type]
        if not args.full_run and str(payload.get("mode", "")).strip().lower() == "full":
            args.full_run = True

        split_config = parse_config_split(payload.get("split"), config_path=args.config)
        args.split_seed = split_config["seed"]
        args.split_group_key = split_config["group_key"]
        args.split_ratios = split_config["ratios"]

        models, model_paths = parse_config_models(payload.get("models", []), config_path=args.config)
        if args.models is None and models:
            args.models = models
        args.config_model_paths = model_paths

        dataset_specs = parse_config_dataset_specs(payload.get("datasets", []), config_path=args.config)
        if args.datasets is None and args.subsets is None and dataset_specs:
            args.config_dataset_specs = dataset_specs
            args.datasets = _unique([str(spec["dataset_name"]) for spec in dataset_specs])
            args.subsets = _unique([str(spec["subset"]) for spec in dataset_specs])

    if args.output_root is None:
        args.output_root = Path("outputs/stage0")
    if args.smoke_limit is None:
        args.smoke_limit = 8
    if args.device is None:
        args.device = "cuda:0"
    if args.dtype is None:
        args.dtype = "float16"
    if args.max_new_tokens is None:
        args.max_new_tokens = 1
    if args.token_index is None:
        args.token_index = -1
    if not args.models:
        parser.error("--models is required unless supplied by --config")
    if not args.config_dataset_specs and (not args.datasets or not args.subsets):
        parser.error("--datasets and --subsets are required unless supplied by --config")


def load_stage0_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"Stage 0 config must be a YAML mapping: {path}")
    return dict(payload)


def parse_config_models(value: object, *, config_path: Path) -> tuple[list[str], dict[str, Path]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return [], {}
    models: list[str] = []
    paths: dict[str, Path] = {}
    for index, item in enumerate(value, start=1):
        if isinstance(item, str):
            models.append(item)
            continue
        if not isinstance(item, Mapping):
            raise ValueError(f"{config_path}: models[{index}] must be a string or mapping")
        name = _required_config_text(item, ("name", "alias", "model_name"), config_path, f"models[{index}]")
        models.append(name)
        config_value = item.get("config_path") or item.get("config")
        if config_value is not None:
            paths[name] = Path(str(config_value))
    return models, paths


def parse_config_dataset_specs(value: object, *, config_path: Path) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    specs: list[dict[str, object]] = []
    for index, item in enumerate(value, start=1):
        if not isinstance(item, Mapping):
            raise ValueError(f"{config_path}: datasets[{index}] must be a mapping")
        dataset_name = _required_config_text(
            item,
            ("family", "dataset_name", "source_dataset", "dataset"),
            config_path,
            f"datasets[{index}]",
        )
        subset_values = item.get("subsets")
        subsets = (
            [str(subset) for subset in subset_values]
            if isinstance(subset_values, Sequence) and not isinstance(subset_values, (str, bytes))
            else [_required_config_text(item, ("subset", "split"), config_path, f"datasets[{index}]")]
        )
        record_path = item.get("record_path") or item.get("records_path") or item.get("path")
        for subset in subsets:
            specs.append(
                {
                    "dataset_name": dataset_name,
                    "subset": subset,
                    "path": None if record_path is None else Path(str(record_path)),
                }
            )
    return specs


def parse_config_split(value: object, *, config_path: Path) -> dict[str, object]:
    split = {
        "seed": int(stage0_splits.DEFAULT_SEED),
        "group_key": "image_id",
        "ratios": list(stage0_splits.DEFAULT_RATIOS),
    }
    if value is None:
        return split
    if not isinstance(value, Mapping):
        raise ValueError(f"{config_path}: split must be a mapping")

    if value.get("seed") is not None:
        split["seed"] = int(value["seed"])  # type: ignore[arg-type]
    if value.get("group_key") is not None:
        group_key = str(value["group_key"]).strip()
        if not group_key:
            raise ValueError(f"{config_path}: split.group_key must not be blank")
        split["group_key"] = group_key
    if value.get("ratios") is not None:
        ratios = value["ratios"]
        if not isinstance(ratios, Sequence) or isinstance(ratios, (str, bytes)):
            raise ValueError(f"{config_path}: split.ratios must be a sequence")
        split["ratios"] = list(stage0_splits._validate_ratios([float(ratio) for ratio in ratios]))
    return split


def _required_config_text(
    row: Mapping[str, object],
    keys: Sequence[str],
    config_path: Path,
    context: str,
) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise ValueError(f"{config_path}: {context} missing required field: {'/'.join(keys)}")


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_git_commit(repo_root: Path | str = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def run_environment_check(repo_root: Path) -> str:
    makefile = repo_root / "Makefile"
    verify_script = repo_root / "scripts" / "verify_env.py"
    if makefile.exists():
        command = ["make", "verify-env"]
    elif verify_script.exists():
        command = [sys.executable, str(verify_script)]
    else:
        return "environment check skipped: no Makefile or scripts/verify_env.py"
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return (result.stdout + result.stderr).strip()


def resolve_model_aliases(
    model_aliases: Sequence[str],
    repo_root: Path,
    *,
    config_model_paths: Mapping[str, Path] | None = None,
) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    configured = config_model_paths or {}
    for alias in model_aliases:
        if alias in configured:
            config_path = configured[alias]
            resolved.append(
                {
                    "alias": alias,
                    "model_name": alias,
                    "config_path": config_path if config_path.is_absolute() else repo_root / config_path,
                }
            )
            continue
        value = MODEL_ALIASES.get(alias)
        if value is None:
            raise ValueError(
                f"Unknown model alias: {alias}. Supported aliases: {', '.join(sorted(MODEL_ALIASES))}"
            )
        resolved.append(
            {
                "alias": alias,
                "model_name": value["model_name"],
                "config_path": repo_root / value["config"],
            }
        )
    return resolved


def resolve_dataset_specs(
    *,
    datasets: Sequence[str],
    subsets: Sequence[str],
    repo_root: Path,
    require_normalized: bool = False,
) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for dataset in datasets:
        normalized_dataset = dataset.strip().lower()
        if normalized_dataset not in KNOWN_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset}. Supported datasets: {', '.join(sorted(KNOWN_DATASETS))}"
            )
        for subset in subsets:
            normalized_subset = subset.strip().lower()
            supported_subsets = DATASET_SUBSETS[normalized_dataset]
            if normalized_subset not in supported_subsets:
                raise ValueError(
                    "Unknown subset for "
                    f"{normalized_dataset}: {subset}. Supported subsets: {', '.join(supported_subsets)}"
                )
            specs.append(
                DatasetSpec(
                    dataset_name=normalized_dataset,
                    subset=normalized_subset,
                    path=resolve_dataset_path(
                        repo_root=repo_root,
                        dataset_name=normalized_dataset,
                        subset=normalized_subset,
                        require_normalized=require_normalized,
                    ),
                )
            )
    return specs


def resolve_config_dataset_specs(
    config_specs: Sequence[Mapping[str, object]],
    *,
    repo_root: Path,
    require_normalized: bool = False,
) -> list[DatasetSpec]:
    specs: list[DatasetSpec] = []
    for item in config_specs:
        dataset_name = str(item["dataset_name"]).strip().lower()
        subset = str(item["subset"]).strip().lower()
        if dataset_name not in KNOWN_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. Supported datasets: {', '.join(sorted(KNOWN_DATASETS))}"
            )
        if subset not in DATASET_SUBSETS[dataset_name]:
            raise ValueError(
                "Unknown subset for "
                f"{dataset_name}: {subset}. Supported subsets: {', '.join(DATASET_SUBSETS[dataset_name])}"
            )
        configured_path = item.get("path")
        path = (
            Path(str(configured_path))
            if configured_path is not None
            else resolve_dataset_path(
                repo_root=repo_root,
                dataset_name=dataset_name,
                subset=subset,
                require_normalized=require_normalized,
            )
        )
        if not path.is_absolute():
            path = repo_root / path
        specs.append(DatasetSpec(dataset_name=dataset_name, subset=subset, path=path))
    return specs


def requested_dataset_specs(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    require_normalized: bool = False,
) -> list[DatasetSpec]:
    config_specs = getattr(args, "config_dataset_specs", [])
    if config_specs:
        return resolve_config_dataset_specs(
            config_specs,
            repo_root=repo_root,
            require_normalized=require_normalized,
        )
    return resolve_dataset_specs(
        datasets=args.datasets,
        subsets=args.subsets,
        repo_root=repo_root,
        require_normalized=require_normalized,
    )


def resolve_dataset_path(
    *,
    repo_root: Path,
    dataset_name: str,
    subset: str,
    require_normalized: bool = False,
) -> Path:
    normalized = normalized_dataset_path(
        repo_root=repo_root,
        dataset_name=dataset_name,
        subset=subset,
    )
    if require_normalized:
        return normalized
    candidates = [
        normalized,
        raw_dataset_path(repo_root=repo_root, dataset_name=dataset_name, subset=subset),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_dataset_image_root(*, repo_root: Path, dataset_name: str) -> Path | None:
    normalized_dataset = dataset_name.strip().lower()
    configured_root = DATASET_IMAGE_ROOTS.get(normalized_dataset)
    if configured_root is None:
        return None
    image_root = configured_root if configured_root.is_absolute() else repo_root / configured_root
    if not image_root.exists():
        raise FileNotFoundError(f"Image root for {normalized_dataset} is missing: {image_root}")
    if not image_root.is_dir():
        raise NotADirectoryError(f"Image root for {normalized_dataset} is not a directory: {image_root}")
    return image_root


def audit_output_paths(audit_dir: Path) -> list[str]:
    return [
        str(audit_dir / "dataset_audit.csv"),
        str(audit_dir / "label_balance.csv"),
        str(audit_dir / "cache_label_balance.csv"),
        str(audit_dir / "object_name_audit.csv"),
        str(audit_dir / "sample_overlap_audit.csv"),
    ]


def audit_dataset_specs_for_stage0(
    args: argparse.Namespace,
    *,
    requested_specs: Sequence[DatasetSpec],
    repo_root: Path,
) -> list[DatasetSpec]:
    if args.full_run:
        return list(requested_specs)
    return discover_known_datasets(repo_root)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def initial_summary(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
    return {
        "stage": "stage0",
        "status": "failed",
        "git_commit": get_git_commit(),
        "models_checked": list(args.models),
        "datasets_checked": datasets_checked_labels(args),
        "smoke_cache_paths": [],
        "audit_outputs": [],
        "split_manifest": None,
        "split_manifests": [],
        "cache_manifest": None,
        "required_cache_matrix": [],
        "completed_cache_matrix": [],
        "missing_cache_matrix": [],
        "cache_label_balance": [],
        "blocking_issues": [],
        "next_recommended_commands": [],
    }


def datasets_checked_labels(args: argparse.Namespace) -> list[str]:
    config_specs = getattr(args, "config_dataset_specs", [])
    if config_specs:
        return [f"{item['dataset_name']}/{item['subset']}" for item in config_specs]
    return [f"{dataset}/{subset}" for dataset in args.datasets for subset in args.subsets]


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    ordered = {key: summary.get(key) for key in SUMMARY_KEYS}
    write_json(path, ordered)


def required_cache_matrix(
    model_specs: Sequence[Mapping[str, object]],
    dataset_specs: Sequence[DatasetSpec],
    *,
    repo_root: Path | None = None,
) -> list[dict[str, object]]:
    return [
        {
            "model_name": str(model_spec["model_name"]),
            "dataset_name": spec.dataset_name,
            "source_dataset": spec.dataset_name,
            "subset": spec.subset,
            "expected_num_records": expected_record_count(spec, repo_root=repo_root),
        }
        for model_spec in model_specs
        for spec in dataset_specs
    ]


def completed_cache_matrix(
    cache_manifest: Mapping[str, object],
    *,
    required_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    observed_counts = cache_matrix_entry_counts(cache_manifest)
    rows: list[dict[str, object]] = []
    for required in required_rows:
        key = _matrix_key(required)
        expected_count = int(required.get("expected_num_records", 0))
        observed_count = observed_counts.get(key, 0)
        if observed_count == expected_count:
            row = dict(required)
            row["cache_num_entries"] = observed_count
            rows.append(row)
    return rows


def cache_matrix_entry_counts(cache_manifest: Mapping[str, object]) -> dict[tuple[str, str, str], int]:
    counts: dict[tuple[str, str, str], int] = {}
    shards = cache_manifest.get("shards", [])
    if isinstance(shards, Sequence) and not isinstance(shards, (str, bytes)):
        for shard in shards:
            if not isinstance(shard, Mapping):
                continue
            row = cache_matrix_row_from_shard(shard)
            if row is None:
                continue
            shard_status = str(shard.get("status", "passed"))
            shard_errors = shard.get("errors", [])
            if shard_status == "passed" and not shard_errors:
                key = _matrix_key(row)
                counts[key] = counts.get(key, 0) + int(row.get("cache_num_entries", 0))
    return counts


def cache_matrix_row_from_shard(shard: Mapping[str, object]) -> dict[str, object] | None:
    model_name = _optional_text(shard.get("model_name"))
    dataset_name = _optional_text(shard.get("dataset_name"))
    subset = _optional_text(shard.get("subset") or shard.get("split"))
    if model_name is None or dataset_name is None or subset is None:
        return None
    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "source_dataset": _optional_text(shard.get("source_dataset")) or dataset_name,
        "subset": subset,
        "cache_num_entries": _optional_int(shard.get("num_entries")) or 0,
    }


def missing_cache_matrix(
    required_rows: Sequence[Mapping[str, object]],
    completed_rows: Sequence[Mapping[str, object]],
    *,
    cache_manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    completed_keys = {_matrix_key(row) for row in completed_rows}
    observed_counts = cache_matrix_entry_counts(cache_manifest)
    rows: list[dict[str, object]] = []
    for row in required_rows:
        key = _matrix_key(row)
        if key in completed_keys:
            continue
        missing = dict(row)
        missing["cache_num_entries"] = observed_counts.get(key, 0)
        rows.append(missing)
    return rows


def matrix_labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    return ["/".join(_matrix_key(row)) for row in rows]


def cache_count_mismatch_labels(rows: Sequence[Mapping[str, object]]) -> list[str]:
    labels: list[str] = []
    for row in rows:
        observed = int(row.get("cache_num_entries", 0))
        if observed <= 0:
            continue
        labels.append(
            f"{'/'.join(_matrix_key(row))} expected {int(row.get('expected_num_records', 0))} "
            f"got {observed}"
        )
    return labels


def cache_label_balance_summary(
    dataset_specs: Sequence[DatasetSpec],
    *,
    cache_root: Path,
    model_specs: Sequence[Mapping[str, object]],
) -> list[dict[str, object | None]]:
    if not cache_root.exists():
        return []
    try:
        return build_cache_label_balance_rows(
            dataset_specs,
            cache_root=cache_root,
            model_names=[str(model_spec["model_name"]) for model_spec in model_specs],
        )
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return []


def _matrix_key(row: Mapping[str, object]) -> tuple[str, str, str]:
    return (str(row["model_name"]), str(row["dataset_name"]), str(row["subset"]))


def _optional_int(value: object | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _optional_text(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def expected_record_count(spec: DatasetSpec, *, repo_root: Path | None = None) -> int:
    if spec.path.exists():
        return len(load_dataset_records(spec))
    if repo_root is not None:
        rows = load_materializable_normalized_rows(spec, repo_root=repo_root)
        if rows is not None:
            return len(rows)
    return 0


def validate_full_run_closure_request(
    model_specs: Sequence[Mapping[str, object]],
    dataset_specs: Sequence[DatasetSpec],
) -> None:
    model_names = {str(model_spec["model_name"]) for model_spec in model_specs}
    dataset_keys = {(spec.dataset_name, spec.subset) for spec in dataset_specs}
    missing_models = [
        model_name
        for model_name in PRIMARY_STAGE0_MODELS
        if model_name not in model_names
    ]
    missing_datasets = [
        f"{dataset_name}/{subset}"
        for dataset_name, subset in FULL_CLOSURE_DATASET_SUBSETS
        if (dataset_name, subset) not in dataset_keys
    ]
    if not missing_models and not missing_datasets:
        return

    missing_parts = []
    if missing_models:
        missing_parts.append("models: " + ", ".join(missing_models))
    if missing_datasets:
        missing_parts.append("datasets: " + ", ".join(missing_datasets))
    raise ValueError(
        "Full-run requires the complete Stage 0 closure matrix "
        "(qwen3-vl-8b and internvl3.5-8b over POPE popular/random/adversarial, "
        "RePOPE popular/random/adversarial, and DASH-B all); missing "
        + "; ".join(missing_parts)
        + f". Use --config {COMPLETE_STAGE0_CONFIG}."
    )


def recommended_full_run_models(requested_models: Sequence[str]) -> list[str]:
    requested = list(requested_models)
    if all(model in requested for model in PRIMARY_STAGE0_MODELS):
        return requested
    return list(PRIMARY_STAGE0_MODELS)


def recommended_full_run_datasets_and_subsets(
    *,
    requested_datasets: Sequence[str],
    requested_subsets: Sequence[str],
) -> tuple[list[str], list[str]]:
    datasets = list(requested_datasets)
    normalized_datasets = [dataset.strip().lower() for dataset in datasets]
    if normalized_datasets == ["pope"]:
        return ["pope"], list(POPE_FULL_RUN_SUBSETS)
    return datasets, list(requested_subsets)


def full_run_command(args: argparse.Namespace) -> str:
    _ = args
    return shlex.join(
        [
            "conda",
            "run",
            "--no-capture-output",
            "-n",
            "mind-py311",
            "python",
            "scripts/stage0_run.py",
            "--config",
            str(COMPLETE_STAGE0_CONFIG),
        ]
    )


def audit_command(
    *,
    output_root: Path,
    dataset_specs: Sequence[DatasetSpec],
    required_specs: Sequence[DatasetSpec],
) -> str:
    parts = [
        "python",
        "scripts/stage0_audit_data.py",
        "--output-root",
        str(output_root),
        "--dry-run",
    ]
    for spec in dataset_specs:
        parts.extend(["--dataset", spec.dataset_name, spec.subset, str(spec.path)])
    for spec in required_specs:
        parts.extend(["--require", spec.dataset_name, spec.subset])
    return shlex.join(parts)


def split_command(
    *,
    spec: DatasetSpec,
    output: Path,
    seed: int,
    ratios: Sequence[float],
    group_key: str,
) -> str:
    parts = [
        "python",
        "scripts/stage0_build_splits.py",
        "--dataset-name",
        spec.dataset_name,
        "--subset",
        spec.subset,
        "--input-records",
        str(spec.path),
        "--output",
        str(output),
        "--seed",
        str(seed),
        "--ratios",
        *[str(ratio) for ratio in ratios],
        "--group-key",
        group_key,
        "--dry-run",
    ]
    return shlex.join(parts)


def split_manifest_index_row(manifest: Mapping[str, object], *, path: Path) -> dict[str, object]:
    counts = manifest.get("counts_per_split", {})
    total_records = sum(int(value) for value in counts.values()) if isinstance(counts, Mapping) else 0
    return {
        "dataset_name": manifest["dataset_name"],
        "subset": manifest["subset"],
        "path": str(path),
        "input_records": manifest["input_records"],
        "counts_per_split": counts,
        "label_counts_per_split": manifest.get("label_counts_per_split", {}),
        "object_counts_per_split": manifest.get("object_counts_per_split", {}),
        "image_id_overlap_validation": manifest.get("image_id_overlap_validation", {}),
        "sample_id_overlap_validation": manifest.get("sample_id_overlap_validation", {}),
        "total_records": total_records,
    }


def build_split_manifest_index(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    dataset_rows = [dict(row) for row in rows]
    return {
        "manifest_type": "stage0_split_manifest_index",
        "dataset_manifests": dataset_rows,
        "counts_per_split": aggregate_counts_per_split(dataset_rows, "counts_per_split"),
        "label_counts_per_split": aggregate_nested_counts_per_split(
            dataset_rows,
            "label_counts_per_split",
        ),
        "object_counts_per_split": aggregate_nested_counts_per_split(
            dataset_rows,
            "object_counts_per_split",
        ),
        "image_id_overlap_validation": aggregate_overlap_validation(
            dataset_rows,
            "image_id_overlap_validation",
        ),
        "sample_id_overlap_validation": aggregate_overlap_validation(
            dataset_rows,
            "sample_id_overlap_validation",
        ),
        "total_records": sum(int(row.get("total_records", 0)) for row in dataset_rows),
    }


def aggregate_counts_per_split(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, int]:
    totals = {split_name: 0 for split_name in stage0_splits.SPLIT_NAMES}
    for row in rows:
        counts = row.get(field, {})
        if not isinstance(counts, Mapping):
            continue
        for split_name, count in counts.items():
            split_text = str(split_name)
            if split_text in totals:
                totals[split_text] += int(count)
    return totals


def aggregate_nested_counts_per_split(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, dict[str, int]]:
    totals: dict[str, dict[str, int]] = {split_name: {} for split_name in stage0_splits.SPLIT_NAMES}
    for row in rows:
        split_counts = row.get(field, {})
        if not isinstance(split_counts, Mapping):
            continue
        for split_name, counts in split_counts.items():
            split_text = str(split_name)
            if split_text not in totals or not isinstance(counts, Mapping):
                continue
            split_total = totals[split_text]
            for value, count in counts.items():
                value_text = str(value)
                split_total[value_text] = split_total.get(value_text, 0) + int(count)
    return totals


def aggregate_overlap_validation(
    rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    per_dataset: dict[str, object] = {}
    overlaps: dict[str, object] = {}
    for row in rows:
        label = f"{row.get('dataset_name')}/{row.get('subset')}"
        validation = row.get(field, {})
        per_dataset[label] = validation
        if isinstance(validation, Mapping) and validation.get("valid") is not True:
            overlaps[label] = validation.get("overlaps", {})
    return {
        "valid": not overlaps,
        "overlaps": overlaps,
        "per_dataset": per_dataset,
    }


def extraction_dry_run_command(
    *,
    spec: DatasetSpec,
    model_spec: Mapping[str, object],
    cache_root: Path,
    image_root: Path | None,
    args: argparse.Namespace,
    limit: int,
) -> str:
    parts = [
        "python",
        "scripts/stage0_extract_full_layer_cache.py",
        "--records",
        str(spec.path),
        "--model-config",
        str(model_spec["config_path"]),
        "--output-root",
        str(cache_root),
        "--dataset-name",
        spec.dataset_name,
        "--subset",
        spec.subset,
        "--split",
        spec.subset,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--token-index",
        str(args.token_index),
        "--limit",
        str(limit),
        "--shard-size",
        "128",
        "--batch-size",
        "1",
        "--dry-run",
    ]
    if image_root is not None:
        parts.extend(["--image-root", str(image_root)])
    return shlex.join(parts)


def validate_dry_run_dataset_specs(
    specs: Sequence[DatasetSpec],
    *,
    repo_root: Path,
    require_normalized: bool = False,
) -> None:
    for spec in specs:
        if spec.path.exists():
            validate_extraction_ready_dataset_specs(
                [spec],
                repo_root=repo_root,
                require_normalized=require_normalized,
            )
            continue
        rows = load_materializable_normalized_rows(spec, repo_root=repo_root)
        if rows is None:
            validate_extraction_ready_dataset_specs(
                [spec],
                repo_root=repo_root,
                require_normalized=require_normalized,
            )
            continue
        validate_extraction_ready_rows(rows, path=spec.path)


def validate_required_datasets_for_dry_run(
    specs: Sequence[DatasetSpec],
    required: Iterable[tuple[str, str]],
    *,
    repo_root: Path,
) -> None:
    by_key = {(spec.dataset_name, spec.subset): spec for spec in specs}
    for dataset_name, subset in required:
        spec = by_key.get((dataset_name, subset))
        if spec is None:
            raise FileNotFoundError(
                f"Required dataset is missing: {dataset_name}/{subset} at <not configured>"
            )
        if spec.path.exists() or load_materializable_normalized_rows(spec, repo_root=repo_root) is not None:
            continue
        raise FileNotFoundError(
            f"Required dataset is missing: {dataset_name}/{subset} at {spec.path}"
        )


def build_dry_run_split_manifest(
    *,
    spec: DatasetSpec,
    repo_root: Path,
    seed: int,
    ratios: Sequence[float],
    group_key: str,
) -> dict[str, object]:
    if spec.path.exists():
        return build_split_manifest(
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            input_records=spec.path,
            seed=seed,
            ratios=ratios,
            group_key=group_key,
        )
    rows = load_materializable_normalized_rows(spec, repo_root=repo_root)
    if rows is None:
        return build_split_manifest(
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            input_records=spec.path,
            seed=seed,
            ratios=ratios,
            group_key=group_key,
        )
    return build_split_manifest_from_rows(
        dataset_name=spec.dataset_name,
        subset=spec.subset,
        input_records=spec.path,
        rows=rows,
        seed=seed,
        ratios=ratios,
        group_key=group_key,
    )


def build_split_manifest_from_rows(
    *,
    dataset_name: str,
    subset: str,
    input_records: Path,
    rows: Sequence[Mapping[str, object]],
    seed: int,
    ratios: Sequence[float],
    group_key: str,
) -> dict[str, object]:
    ratio_values = stage0_splits._validate_ratios(ratios)
    stage0_splits._validate_single_source(rows, dataset_name=dataset_name, subset=subset)
    grouped = stage0_splits._group_rows(rows, group_key=group_key)
    group_to_split = stage0_splits._assign_groups(
        grouped,
        ratios=ratio_values,
        seed=seed,
    )
    assignments = [
        stage0_splits._assignment_row(
            row,
            split=group_to_split[stage0_splits._required_text(row.get(group_key))],
            dataset_name=dataset_name,
            subset=subset,
        )
        for row in rows
    ]
    image_validation = stage0_splits._overlap_validation(assignments, key="image_id")
    sample_validation = stage0_splits._overlap_validation(assignments, key="sample_id")
    stage0_splits._raise_for_overlap("image_id", image_validation)
    stage0_splits._raise_for_overlap("sample_id", sample_validation)
    return {
        "seed": int(seed),
        "group_key": group_key,
        "split_names": list(stage0_splits.SPLIT_NAMES),
        "ratios": list(ratio_values),
        "dataset_name": dataset_name,
        "subset": subset,
        "input_records": str(input_records),
        "counts_per_split": stage0_splits._counts_per_split(assignments),
        "label_counts_per_split": stage0_splits._field_counts_per_split(assignments, "label"),
        "object_counts_per_split": stage0_splits._field_counts_per_split(assignments, "object_name"),
        "image_id_overlap_validation": image_validation,
        "sample_id_overlap_validation": sample_validation,
        "assignments": assignments,
    }


def build_dry_run_plan(
    args: argparse.Namespace,
    *,
    repo_root: Path,
    output_root: Path,
    manifests_dir: Path,
    cache_root: Path,
    cache_manifest_path: Path,
    split_manifest_path: Path,
) -> dict[str, object]:
    env_output = run_environment_check(repo_root)
    model_specs = resolve_model_aliases(
        args.models,
        repo_root=repo_root,
        config_model_paths=getattr(args, "config_model_paths", {}),
    )
    dataset_specs = requested_dataset_specs(
        args,
        repo_root=repo_root,
        require_normalized=args.full_run,
    )
    if args.full_run:
        validate_full_run_closure_request(model_specs, dataset_specs)
    materialization_plans = planned_missing_normalized_dataset_materializations(
        dataset_specs,
        repo_root=repo_root,
    )
    audit_dataset_specs = audit_dataset_specs_for_stage0(
        args,
        requested_specs=dataset_specs,
        repo_root=repo_root,
    )
    validate_dry_run_dataset_specs(
        dataset_specs,
        repo_root=repo_root,
        require_normalized=args.full_run,
    )
    validate_required_datasets_for_dry_run(
        dataset_specs,
        [(spec.dataset_name, spec.subset) for spec in dataset_specs],
        repo_root=repo_root,
    )
    image_roots = {
        spec.dataset_name: resolve_dataset_image_root(
            repo_root=repo_root,
            dataset_name=spec.dataset_name,
        )
        for spec in dataset_specs
    }

    split_manifest_paths: list[str] = []
    split_commands: list[str] = []
    split_summaries: list[dict[str, object]] = []
    split_index_rows: list[dict[str, object]] = []
    for index, spec in enumerate(dataset_specs):
        manifest = build_dry_run_split_manifest(
            spec=spec,
            repo_root=repo_root,
            seed=args.split_seed,
            ratios=args.split_ratios,
            group_key=args.split_group_key,
        )
        per_dataset_path = manifests_dir / f"split_manifest_{spec.dataset_name}_{spec.subset}.json"
        split_manifest_paths.append(str(per_dataset_path))
        split_commands.append(
            split_command(
                spec=spec,
                output=per_dataset_path,
                seed=args.split_seed,
                ratios=args.split_ratios,
                group_key=args.split_group_key,
            )
        )
        split_index_rows.append(split_manifest_index_row(manifest, path=per_dataset_path))
        split_summaries.append(
            {
                "dataset_name": spec.dataset_name,
                "subset": spec.subset,
                "input_records": str(spec.path),
                "output": str(per_dataset_path),
                "counts_per_split": manifest["counts_per_split"],
            }
        )
    if not split_index_rows:
        raise ValueError("No dataset specs were requested.")
    split_manifest_paths.append(str(split_manifest_path))

    limit = 0 if args.full_run else args.smoke_limit
    extraction_commands = [
        extraction_dry_run_command(
            spec=spec,
            model_spec=model_spec,
            cache_root=cache_root,
            image_root=image_roots[spec.dataset_name],
            args=args,
            limit=limit,
        )
        for model_spec in model_specs
        for spec in dataset_specs
    ]

    summary = initial_summary(args, repo_root=repo_root)
    summary["status"] = "dry_run"
    summary["audit_outputs"] = audit_output_paths(output_root / "audit")
    summary["split_manifest"] = str(split_manifest_path)
    summary["split_manifests"] = split_manifest_paths
    summary["cache_manifest"] = str(cache_manifest_path)
    summary["required_cache_matrix"] = required_cache_matrix(
        model_specs,
        dataset_specs,
        repo_root=repo_root,
    )
    summary["blocking_issues"] = []
    summary["next_recommended_commands"] = [] if args.full_run else [full_run_command(args)]

    return {
        "dry_run": True,
        "environment_check": env_output,
        "audit_command": audit_command(
            output_root=output_root,
            dataset_specs=audit_dataset_specs,
            required_specs=dataset_specs,
        ),
        "audit_outputs": summary["audit_outputs"],
        "materializations": materialization_plans,
        "split_commands": split_commands,
        "split_manifests": split_manifest_paths,
        "split_summaries": split_summaries,
        "extraction_commands": extraction_commands,
        "cache_manifest": str(cache_manifest_path),
        "summary": {key: summary.get(key) for key in SUMMARY_KEYS},
    }


def run_orchestration(args: argparse.Namespace, *, repo_root: Path | str = ".") -> int:
    repo_root = Path(repo_root)
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    args.output_root = output_root

    manifests_dir = output_root / "manifests"
    logs_dir = output_root / "logs"
    log_path = logs_dir / "stage0_run.log"
    summary_path = manifests_dir / "stage0_summary.json"
    cache_root = output_root / "cache"
    cache_manifest_path = manifests_dir / "cache_manifest.json"
    split_manifest_path = manifests_dir / "split_manifest.json"

    summary = initial_summary(args, repo_root=repo_root)

    def log(message: str) -> None:
        logs_dir.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{utc_now_iso()} {message}\n")

    try:
        if args.dry_run:
            plan = build_dry_run_plan(
                args,
                repo_root=repo_root,
                output_root=output_root,
                manifests_dir=manifests_dir,
                cache_root=cache_root,
                cache_manifest_path=cache_manifest_path,
                split_manifest_path=split_manifest_path,
            )
            print(json.dumps(plan, indent=2, sort_keys=False))
            return 0

        model_specs = resolve_model_aliases(
            args.models,
            repo_root=repo_root,
            config_model_paths=getattr(args, "config_model_paths", {}),
        )
        dataset_specs = requested_dataset_specs(
            args,
            repo_root=repo_root,
            require_normalized=args.full_run,
        )
        if args.full_run:
            validate_full_run_closure_request(model_specs, dataset_specs)
        dataset_specs = materialize_missing_normalized_dataset_specs(dataset_specs, repo_root=repo_root)
        required_matrix = required_cache_matrix(model_specs, dataset_specs, repo_root=repo_root)
        summary["required_cache_matrix"] = required_matrix
        audit_dataset_specs = audit_dataset_specs_for_stage0(
            args,
            requested_specs=dataset_specs,
            repo_root=repo_root,
        )
        try:
            validate_extraction_ready_dataset_specs(
                dataset_specs,
                repo_root=repo_root,
                require_normalized=args.full_run,
            )
        except Exception as error:
            if args.full_run:
                print(str(error), file=sys.stderr)
                return 2
            raise

        manifests_dir.mkdir(parents=True, exist_ok=True)
        log("stage0_run start")
        env_output = run_environment_check(repo_root)
        log(f"environment_check {env_output}")

        audit_result = run_audit(audit_dataset_specs, output_root=output_root, cache_root=None)
        summary["audit_outputs"] = audit_output_paths(audit_result.audit_dir)
        log(f"audit complete dir={audit_result.audit_dir}")

        validate_required_datasets(
            dataset_specs,
            [(spec.dataset_name, spec.subset) for spec in dataset_specs],
        )
        image_roots = {
            spec.dataset_name: resolve_dataset_image_root(
                repo_root=repo_root,
                dataset_name=spec.dataset_name,
            )
            for spec in dataset_specs
        }

        split_manifest_paths: list[str] = []
        split_index_rows: list[dict[str, object]] = []
        for spec in dataset_specs:
            manifest = build_split_manifest(
                dataset_name=spec.dataset_name,
                subset=spec.subset,
                input_records=spec.path,
                seed=args.split_seed,
                ratios=args.split_ratios,
                group_key=args.split_group_key,
            )
            per_dataset_path = manifests_dir / f"split_manifest_{spec.dataset_name}_{spec.subset}.json"
            write_split_manifest(manifest, per_dataset_path)
            split_manifest_paths.append(str(per_dataset_path))
            split_index_rows.append(split_manifest_index_row(manifest, path=per_dataset_path))
        if not split_index_rows:
            raise ValueError("No dataset specs were requested.")
        write_split_manifest(build_split_manifest_index(split_index_rows), split_manifest_path)
        split_manifest_paths.append(str(split_manifest_path))
        summary["split_manifest"] = str(split_manifest_path)
        summary["split_manifests"] = split_manifest_paths
        log(f"split manifest complete path={split_manifest_path}")

        limit = 0 if args.full_run else args.smoke_limit
        dtype = resolve_torch_dtype(args.dtype)
        smoke_paths: list[str] = []
        for model_spec in model_specs:
            for spec in dataset_specs:
                paths = run_extraction(
                    records_path=spec.path,
                    model_config_path=model_spec["config_path"],  # type: ignore[arg-type]
                    output_root=cache_root,
                    dataset_name=spec.dataset_name,
                    subset=spec.subset,
                    split=spec.subset,
                    image_root=image_roots[spec.dataset_name],
                    device=args.device,
                    dtype=dtype,
                    max_new_tokens=args.max_new_tokens,
                    token_index=args.token_index,
                    limit=limit,
                    shard_size=128,
                    batch_size=1,
                )
                smoke_paths.extend(str(path) for path in paths)
        summary["smoke_cache_paths"] = smoke_paths
        log(f"extraction complete shards={len(smoke_paths)} limit={limit}")

        cache_manifest = validate_stage0_cache(
            cache_root,
            output=cache_manifest_path,
            raise_on_error=False,
        )
        summary["cache_manifest"] = str(cache_manifest_path)
        completed_matrix = completed_cache_matrix(cache_manifest, required_rows=required_matrix)
        missing_matrix = missing_cache_matrix(
            required_matrix,
            completed_matrix,
            cache_manifest=cache_manifest,
        )
        summary["completed_cache_matrix"] = completed_matrix
        summary["missing_cache_matrix"] = missing_matrix
        summary["cache_label_balance"] = cache_label_balance_summary(
            dataset_specs,
            cache_root=cache_root,
            model_specs=model_specs,
        )
        if cache_manifest.get("status") != "passed":
            errors = cache_manifest.get("errors", [])
            if isinstance(errors, list) and errors:
                raise RuntimeError("; ".join(str(error) for error in errors[:3]))
            raise RuntimeError("cache validation failed")
        if missing_matrix:
            mismatch_labels = cache_count_mismatch_labels(missing_matrix)
            if mismatch_labels:
                raise RuntimeError(
                    "cache matrix count mismatch: " + ", ".join(mismatch_labels)
                )
            missing_text = ", ".join(matrix_labels(missing_matrix))
            raise RuntimeError(f"missing cache matrix entry: {missing_text}")
        log(f"cache validation passed path={cache_manifest_path}")

        summary["status"] = "passed"
        summary["blocking_issues"] = []
        if not args.full_run:
            command = full_run_command(args)
            summary["next_recommended_commands"] = [command]
            print(f"Smoke Stage 0 passed. Full-run command:")
            print(command)
        else:
            summary["next_recommended_commands"] = []
        write_summary(summary_path, summary)
        log(f"stage0_run complete summary={summary_path}")
        return 0
    except Exception as error:
        issue = str(error)
        if args.dry_run:
            print(issue, file=sys.stderr)
            return 2
        summary["status"] = "failed"
        summary["blocking_issues"] = [issue]
        manifests_dir.mkdir(parents=True, exist_ok=True)
        write_summary(summary_path, summary)
        log(f"stage0_run failed error={issue}")
        print(issue, file=sys.stderr)
        return 2


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_orchestration(args, repo_root=Path("."))


if __name__ == "__main__":
    raise SystemExit(main())
