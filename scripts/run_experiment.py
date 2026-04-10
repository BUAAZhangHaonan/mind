#!/usr/bin/env python3
"""Plan or execute staged MIND experiment commands from a flat YAML preset."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.config import DatasetConfig, ModelConfig, load_yaml_config
from mind.evaluation import DEFAULT_FULL_VARIANT, resolve_highest_valid_num_folds


DEFAULT_STAGES = [
    "prepare",
    "build_reference",
    "cache_reference",
    "extract_eval",
    "build_manifolds",
    "compute_drift",
    "baselines",
    "train_detector",
    "evaluate",
    "plot",
]


def parse_stage_list(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(DEFAULT_STAGES)
    return [stage.strip() for stage in value.split(",") if stage.strip()]


def load_experiment_spec(config_path: Path) -> dict[str, object]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    required = [
        "name",
        "model_config",
        "dataset_config",
        "subset",
        "split",
        "selected_layers",
    ]
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(f"Experiment config is missing required fields: {', '.join(missing)}")
    payload.setdefault("limit", 0)
    payload.setdefault("detector", "logistic")
    payload.setdefault("device", "cuda")
    payload.setdefault("layer_range", "middle")
    payload.setdefault("reference_dataset_name", "pope-reference")
    payload.setdefault("reference_split", "train")
    payload.setdefault("reference_candidates", "outputs/reference_candidates/coco_train_candidates.json")
    payload.setdefault("reference_instances_json", "data/coco/annotations/instances_train2017.json")
    payload.setdefault("reference_image_root", "data/coco/train2017")
    payload.setdefault("reference_max_images_per_object", 64)
    payload.setdefault("bank_scope", "object")
    payload.setdefault("full_variant", DEFAULT_FULL_VARIANT)
    payload.setdefault("split_strategy", "row")
    payload.setdefault("num_folds", 5)
    payload.setdefault("test_size", 0.3)
    payload.setdefault("random_state", 13)
    payload.setdefault("bootstrap_resamples", 1000)
    payload.setdefault("split_seeds", "13,17,19,23,29")
    payload.setdefault("label_overrides", "")
    return payload


def resolve_reference_banks_root(output_root: Path, bank_scope: str) -> Path:
    if bank_scope == "object":
        return output_root / "reference_banks"
    if bank_scope == "shared":
        return output_root / "reference_banks_shared"
    if bank_scope == "shuffled_object":
        return output_root / "reference_banks_shuffled"
    raise ValueError(f"Unsupported bank scope: {bank_scope}")


def resolve_prepare_source(dataset_root: str, subset: str, *, dataset_name: str = "") -> str:
    root_path = Path(dataset_root)
    if root_path.is_file():
        return str(root_path)
    if dataset_name == "dash-b":
        return str(root_path)
    dash_b_neg = root_path / "images" / "dash_benchmark_neg.json"
    dash_b_pos = root_path / "images" / "dash_benchmark_pos.json"
    if dash_b_neg.exists() and dash_b_pos.exists():
        return str(root_path)
    for suffix in (".jsonl", ".json"):
        candidate = root_path / f"{subset}{suffix}"
        if candidate.exists():
            return str(candidate)
    return str(root_path / f"{subset}.jsonl")


def build_runtime_paths(
    *,
    experiment: dict[str, object],
    dataset: DatasetConfig,
    model: ModelConfig,
    output_root: Path,
) -> dict[str, str]:
    subset = str(experiment["subset"])
    dataset_name = dataset.name
    experiment_name = str(experiment["name"])
    reference_banks_root = resolve_reference_banks_root(output_root, str(experiment["bank_scope"]))
    return {
        "normalized": str(output_root / "normalized" / dataset_name / f"{subset}.jsonl"),
        "reference_cache": str(
            output_root / "cache" / model.name / str(experiment["reference_dataset_name"]) / str(experiment["reference_split"])
        ),
        "eval_cache": str(output_root / "cache" / model.name / dataset_name / subset),
        "reference_banks": str(reference_banks_root),
        "features": str(output_root / "features" / experiment_name / f"{subset}.parquet"),
        "reports": str(output_root / "reports" / experiment_name),
        "plots": str(output_root / "plots" / experiment_name),
    }


def build_stage_commands(
    *,
    config_path: Path,
    stages: list[str],
    python_bin: str | None = None,
    output_root: Path | None = None,
) -> dict[str, list[str]]:
    python_bin = python_bin or sys.executable
    output_root = output_root or Path("outputs")
    experiment = load_experiment_spec(config_path)
    dataset = load_yaml_config(Path(str(experiment["dataset_config"])), DatasetConfig)
    model = load_yaml_config(Path(str(experiment["model_config"])), ModelConfig)
    paths = build_runtime_paths(experiment=experiment, dataset=dataset, model=model, output_root=output_root)
    subset = str(experiment["subset"])
    split = str(experiment["split"])
    selected_layers = str(experiment["selected_layers"])
    layer_range = str(experiment["layer_range"])
    limit = str(experiment["limit"])
    bank_scope = str(experiment["bank_scope"])
    full_variant = str(experiment["full_variant"])
    dataset_source = dataset.source_dataset or ("repope" if dataset.name == "repope" else dataset.name)
    normalizer = dataset.normalizer.strip().lower()

    commands: dict[str, list[str]] = {}
    for stage in stages:
        if stage == "prepare":
            if normalizer not in {"object_yes_no", "object-yes-no", "pope"}:
                raise ValueError(f"Unsupported dataset normalizer: {dataset.normalizer}")
            command = [
                python_bin,
                "scripts/prepare_data.py",
                "normalize-object-yes-no",
                "--source",
                resolve_prepare_source(dataset.root, subset, dataset_name=dataset.name),
                "--output",
                paths["normalized"],
                "--subset",
                subset,
                "--split",
                split,
            ]
            if dataset_source != "pope":
                command.extend(["--source-dataset", dataset_source])
            if dataset.question_template:
                command.extend(["--question-template", dataset.question_template])
        elif stage == "cache_reference":
            command = [
                python_bin,
                "scripts/cache_reference_states.py",
                "--references",
                str(experiment["reference_candidates"]),
                "--image-root",
                str(experiment["reference_image_root"]),
                "--model-config",
                str(experiment["model_config"]),
                "--output-root",
                str(output_root / "cache"),
                "--dataset-name",
                str(experiment["reference_dataset_name"]),
                "--split",
                str(experiment["reference_split"]),
                "--device",
                str(experiment["device"]),
                "--selected-layers",
                selected_layers,
                "--layer-range",
                layer_range,
            ]
            if int(limit) > 0:
                command.extend(["--limit", limit])
        elif stage == "build_reference":
            command = [
                python_bin,
                "scripts/prepare_data.py",
                "build-reference",
                "--instances-json",
                str(experiment["reference_instances_json"]),
                "--output",
                str(experiment["reference_candidates"]),
                "--allowed-objects-from",
                paths["normalized"],
                "--max-images-per-object",
                str(experiment["reference_max_images_per_object"]),
            ]
        elif stage == "extract_eval":
            command = [
                python_bin,
                "scripts/extract_eval_states.py",
                "--records",
                paths["normalized"],
                "--model-config",
                str(experiment["model_config"]),
                "--output-root",
                str(output_root / "cache"),
                "--dataset-name",
                dataset.name,
                "--split",
                subset,
                "--device",
                str(experiment["device"]),
                "--selected-layers",
                selected_layers,
                "--layer-range",
                layer_range,
            ]
            if dataset.image_root:
                command.extend(["--image-root", dataset.image_root])
            if int(limit) > 0:
                command.extend(["--limit", limit])
        elif stage == "build_manifolds":
            command = [
                python_bin,
                "scripts/build_manifolds.py",
                "--reference-cache",
                paths["reference_cache"],
                "--output-root",
                paths["reference_banks"],
                "--model-name",
                model.name,
                "--bank-scope",
                bank_scope,
            ]
        elif stage == "compute_drift":
            command = [
                python_bin,
                "scripts/compute_drift.py",
                "--cache-path",
                paths["eval_cache"],
                "--reference-root",
                paths["reference_banks"],
                "--model-name",
                model.name,
                "--output-root",
                str(output_root / "features"),
                "--experiment-name",
                str(experiment["name"]),
                "--split",
                subset,
                "--bank-scope",
                bank_scope,
            ]
        elif stage == "baselines":
            command = [
                python_bin,
                "scripts/compute_baselines.py",
                "--features-path",
                paths["features"],
                "--cache-path",
                paths["eval_cache"],
                "--reference-root",
                paths["reference_banks"],
                "--model-name",
                model.name,
                "--output-root",
                str(output_root / "reports"),
                "--experiment-name",
                str(experiment["name"]),
                "--split-strategy",
                str(experiment["split_strategy"]),
                "--num-folds",
                str(experiment["num_folds"]),
                "--test-size",
                str(experiment["test_size"]),
                "--random-state",
                str(experiment["random_state"]),
                "--bootstrap-resamples",
                str(experiment["bootstrap_resamples"]),
                "--split-seeds",
                str(experiment["split_seeds"]),
                "--bank-scope",
                bank_scope,
                "--full-variant",
                full_variant,
            ]
            if str(experiment["label_overrides"]).strip():
                command.extend(["--label-overrides", str(experiment["label_overrides"])])
        elif stage == "train_detector":
            command = [
                python_bin,
                "scripts/train_detector.py",
                "--train-path",
                paths["features"],
                "--output-root",
                str(output_root / "reports"),
                "--experiment-name",
                str(experiment["name"]),
                "--feature-variant",
                full_variant,
                "--split-strategy",
                str(experiment["split_strategy"]),
                "--num-folds",
                str(experiment["num_folds"]),
                "--test-size",
                str(experiment["test_size"]),
                "--random-state",
                str(experiment["random_state"]),
            ]
        elif stage == "evaluate":
            command = [
                python_bin,
                "scripts/evaluate.py",
                "--input-path",
                f"{paths['reports']}/results.csv",
                "--output-root",
                str(output_root / "reports"),
                "--experiment-name",
                str(experiment["name"]),
            ]
            if str(experiment["label_overrides"]).strip():
                command.extend(["--label-overrides", str(experiment["label_overrides"])])
        elif stage == "plot":
            command = [
                python_bin,
                "scripts/plot_results.py",
                "--features-path",
                paths["features"],
                "--results-path",
                f"{paths['reports']}/results.csv",
                "--output-root",
                str(output_root / "plots"),
                "--experiment-name",
                str(experiment["name"]),
            ]
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        commands[stage] = command
    return commands


def resolve_highest_valid_object_heldout_folds(
    frame_paths: list[Path],
    *,
    candidate_folds: tuple[int, ...] = (5, 4, 3, 2),
    random_state: int = 13,
) -> int:
    frames = [
        pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        for path in frame_paths
    ]
    return resolve_highest_valid_num_folds(
        frames,
        split_strategy="object_heldout",
        candidate_folds=candidate_folds,
        random_state=random_state,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = parse_stage_list(args.stages)
    commands = build_stage_commands(config_path=args.config, stages=stages, output_root=args.output_root)
    for stage in stages:
        command = commands[stage]
        rendered = " ".join(shlex.quote(part) for part in command)
        print(f"{stage}: {rendered}")
        if args.execute:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
