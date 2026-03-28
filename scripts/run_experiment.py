#!/usr/bin/env python3
"""Plan or execute staged MIND experiment commands from a flat YAML preset."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml

from mind.config import DatasetConfig, ModelConfig, load_yaml_config


DEFAULT_STAGES = [
    "prepare",
    "build_reference",
    "cache_reference",
    "extract_eval",
    "build_manifolds",
    "compute_drift",
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
    payload.setdefault("reference_dataset_name", "pope-reference")
    payload.setdefault("reference_split", "train")
    payload.setdefault("reference_candidates", "outputs/reference_candidates/coco_train_candidates.json")
    payload.setdefault("reference_instances_json", "data/coco/annotations/instances_train2017.json")
    payload.setdefault("reference_image_root", "data/coco/train2017")
    payload.setdefault("reference_max_images_per_object", 64)
    return payload


def build_runtime_paths(
    *,
    experiment: dict[str, object],
    dataset: DatasetConfig,
    model: ModelConfig,
) -> dict[str, str]:
    subset = str(experiment["subset"])
    dataset_name = dataset.name
    experiment_name = str(experiment["name"])
    return {
        "normalized": f"outputs/normalized/{dataset_name}/{subset}.jsonl",
        "reference_cache": f"outputs/cache/{model.name}/{experiment['reference_dataset_name']}/{experiment['reference_split']}",
        "eval_cache": f"outputs/cache/{model.name}/{dataset_name}/{subset}",
        "reference_banks": "outputs/reference_banks",
        "features": f"outputs/features/{experiment_name}/{subset}.parquet",
        "reports": f"outputs/reports/{experiment_name}",
        "plots": f"outputs/plots/{experiment_name}",
    }


def build_stage_commands(
    *,
    config_path: Path,
    stages: list[str],
    python_bin: str | None = None,
) -> dict[str, list[str]]:
    python_bin = python_bin or sys.executable
    experiment = load_experiment_spec(config_path)
    dataset = load_yaml_config(Path(str(experiment["dataset_config"])), DatasetConfig)
    model = load_yaml_config(Path(str(experiment["model_config"])), ModelConfig)
    paths = build_runtime_paths(experiment=experiment, dataset=dataset, model=model)
    subset = str(experiment["subset"])
    split = str(experiment["split"])
    selected_layers = str(experiment["selected_layers"])
    limit = str(experiment["limit"])
    dataset_source = "repope" if dataset.name == "repope" else dataset.name

    commands: dict[str, list[str]] = {}
    for stage in stages:
        if stage == "prepare":
            command = [
                python_bin,
                "scripts/prepare_data.py",
                "normalize-pope",
                "--source",
                f"{dataset.root}/{subset}.jsonl",
                "--output",
                paths["normalized"],
                "--subset",
                subset,
                "--split",
                split,
            ]
            if dataset_source != "pope":
                command.extend(["--source-dataset", dataset_source])
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
                "outputs/cache",
                "--dataset-name",
                str(experiment["reference_dataset_name"]),
                "--split",
                str(experiment["reference_split"]),
                "--device",
                str(experiment["device"]),
                "--selected-layers",
                selected_layers,
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
                "outputs/cache",
                "--dataset-name",
                dataset.name,
                "--split",
                subset,
                "--device",
                str(experiment["device"]),
                "--selected-layers",
                selected_layers,
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
                "outputs/reference_banks",
                "--model-name",
                model.name,
            ]
        elif stage == "compute_drift":
            command = [
                python_bin,
                "scripts/compute_drift.py",
                "--cache-path",
                paths["eval_cache"],
                "--reference-root",
                "outputs/reference_banks",
                "--model-name",
                model.name,
                "--output-root",
                "outputs/features",
                "--experiment-name",
                str(experiment["name"]),
                "--split",
                subset,
            ]
        elif stage == "train_detector":
            command = [
                python_bin,
                "scripts/train_detector.py",
                "--train-path",
                paths["features"],
                "--output-root",
                "outputs/reports",
                "--experiment-name",
                str(experiment["name"]),
            ]
        elif stage == "evaluate":
            command = [
                python_bin,
                "scripts/evaluate.py",
                "--input-path",
                f"{paths['reports']}/results.csv",
                "--output-root",
                "outputs/reports",
                "--experiment-name",
                str(experiment["name"]),
            ]
        elif stage == "plot":
            command = [
                python_bin,
                "scripts/plot_results.py",
                "--features-path",
                paths["features"],
                "--results-path",
                f"{paths['reports']}/results.csv",
                "--output-root",
                "outputs/plots",
                "--experiment-name",
                str(experiment["name"]),
            ]
        else:
            raise ValueError(f"Unsupported stage: {stage}")
        commands[stage] = command
    return commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = parse_stage_list(args.stages)
    commands = build_stage_commands(config_path=args.config, stages=stages)
    for stage in stages:
        command = commands[stage]
        rendered = " ".join(shlex.quote(part) for part in command)
        print(f"{stage}: {rendered}")
        if args.execute:
            subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
