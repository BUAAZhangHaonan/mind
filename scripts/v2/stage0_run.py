#!/usr/bin/env python3
"""Run v2 Stage 0 audit, split, smoke extraction, and cache validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import shlex
import sys
from typing import Mapping, Sequence

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.models.types import resolve_torch_dtype
from mind.trajectory.audit import run_audit, validate_required_datasets
from mind.trajectory.cache import validate_stage0_cache
from mind.trajectory.dataset import (
    DatasetSpec,
    discover_known_datasets,
    normalized_dataset_path,
    raw_dataset_path,
    validate_extraction_ready_dataset_specs,
)
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
KNOWN_DATASETS = {"pope", "repope"}
KNOWN_SUBSETS = {"popular", "random", "adversarial"}
DATASET_IMAGE_ROOTS = {
    "pope": Path("data/coco/val2014"),
    "repope": Path("data/coco/val2014"),
}
PRIMARY_STAGE0_MODELS = ("qwen3-vl-8b", "internvl3.5-8b")
POPE_FULL_RUN_SUBSETS = ("popular", "random", "adversarial")
SUMMARY_KEYS = (
    "stage",
    "status",
    "git_commit",
    "models_checked",
    "datasets_checked",
    "smoke_cache_paths",
    "audit_outputs",
    "split_manifest",
    "cache_manifest",
    "blocking_issues",
    "next_recommended_commands",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/v2_stage0"))
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--subsets", nargs="+", required=True)
    parser.add_argument("--smoke-limit", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve Stage 0 commands and summary intent without writing artifacts or extracting cache.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


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


def resolve_model_aliases(model_aliases: Sequence[str], repo_root: Path) -> list[dict[str, object]]:
    resolved: list[dict[str, object]] = []
    for alias in model_aliases:
        try:
            value = MODEL_ALIASES[alias]
        except KeyError as error:
            raise ValueError(
                f"Unknown model alias: {alias}. Supported aliases: {', '.join(sorted(MODEL_ALIASES))}"
            ) from error
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
            if normalized_subset not in KNOWN_SUBSETS:
                raise ValueError(
                    f"Unknown subset: {subset}. Supported subsets: {', '.join(sorted(KNOWN_SUBSETS))}"
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
        str(audit_dir / "object_name_audit.csv"),
        str(audit_dir / "sample_overlap_audit.csv"),
    ]


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def initial_summary(args: argparse.Namespace, *, repo_root: Path) -> dict[str, object]:
    return {
        "stage": "stage0",
        "status": "failed",
        "git_commit": get_git_commit(),
        "models_checked": list(args.models),
        "datasets_checked": [f"{dataset}/{subset}" for dataset in args.datasets for subset in args.subsets],
        "smoke_cache_paths": [],
        "audit_outputs": [],
        "split_manifest": None,
        "cache_manifest": None,
        "blocking_issues": [],
        "next_recommended_commands": [],
    }


def write_summary(path: Path, summary: Mapping[str, object]) -> None:
    ordered = {key: summary.get(key) for key in SUMMARY_KEYS}
    write_json(path, ordered)


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
    datasets, subsets = recommended_full_run_datasets_and_subsets(
        requested_datasets=args.datasets,
        requested_subsets=args.subsets,
    )
    parts = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        "mind-py311",
        "python",
        "scripts/v2/stage0_run.py",
        "--output-root",
        str(args.output_root),
        "--models",
        *recommended_full_run_models(args.models),
        "--datasets",
        *datasets,
        "--subsets",
        *subsets,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--full-run",
    ]
    return " ".join(parts)


def audit_command(
    *,
    output_root: Path,
    dataset_specs: Sequence[DatasetSpec],
    required_specs: Sequence[DatasetSpec],
) -> str:
    parts = [
        "python",
        "scripts/v2/stage0_audit_data.py",
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
) -> str:
    return shlex.join(
        [
            "python",
            "scripts/v2/stage0_build_splits.py",
            "--dataset-name",
            spec.dataset_name,
            "--subset",
            spec.subset,
            "--input-records",
            str(spec.path),
            "--output",
            str(output),
            "--dry-run",
        ]
    )


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
        "scripts/v2/stage0_extract_full_layer_cache.py",
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
        "1",
        "--token-index",
        "-1",
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
    model_specs = resolve_model_aliases(args.models, repo_root)
    dataset_specs = resolve_dataset_specs(
        datasets=args.datasets,
        subsets=args.subsets,
        repo_root=repo_root,
        require_normalized=args.full_run,
    )
    audit_dataset_specs = discover_known_datasets(repo_root)
    validate_extraction_ready_dataset_specs(
        dataset_specs,
        repo_root=repo_root,
        require_normalized=args.full_run,
    )
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
    split_commands: list[str] = []
    split_summaries: list[dict[str, object]] = []
    primary_manifest: dict[str, object] | None = None
    for index, spec in enumerate(dataset_specs):
        manifest = build_split_manifest(
            dataset_name=spec.dataset_name,
            subset=spec.subset,
            input_records=spec.path,
        )
        per_dataset_path = manifests_dir / f"split_manifest_{spec.dataset_name}_{spec.subset}.json"
        split_manifest_paths.append(str(per_dataset_path))
        split_commands.append(split_command(spec=spec, output=per_dataset_path))
        split_summaries.append(
            {
                "dataset_name": spec.dataset_name,
                "subset": spec.subset,
                "input_records": str(spec.path),
                "output": str(per_dataset_path),
                "counts_per_split": manifest["counts_per_split"],
            }
        )
        if index == 0:
            primary_manifest = manifest
    if primary_manifest is None:
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
    summary["cache_manifest"] = str(cache_manifest_path)
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

        model_specs = resolve_model_aliases(args.models, repo_root)
        dataset_specs = resolve_dataset_specs(
            datasets=args.datasets,
            subsets=args.subsets,
            repo_root=repo_root,
            require_normalized=args.full_run,
        )
        audit_dataset_specs = discover_known_datasets(repo_root)
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

        primary_manifest: dict[str, object] | None = None
        for index, spec in enumerate(dataset_specs):
            manifest = build_split_manifest(
                dataset_name=spec.dataset_name,
                subset=spec.subset,
                input_records=spec.path,
            )
            per_dataset_path = manifests_dir / f"split_manifest_{spec.dataset_name}_{spec.subset}.json"
            write_split_manifest(manifest, per_dataset_path)
            if index == 0:
                primary_manifest = manifest
        if primary_manifest is None:
            raise ValueError("No dataset specs were requested.")
        write_split_manifest(primary_manifest, split_manifest_path)
        summary["split_manifest"] = str(split_manifest_path)
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
                    max_new_tokens=1,
                    token_index=-1,
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
        if cache_manifest.get("status") != "passed":
            errors = cache_manifest.get("errors", [])
            if isinstance(errors, list) and errors:
                raise RuntimeError("; ".join(str(error) for error in errors[:3]))
            raise RuntimeError("cache validation failed")
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
