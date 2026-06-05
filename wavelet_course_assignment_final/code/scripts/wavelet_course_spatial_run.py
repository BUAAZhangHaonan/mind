#!/usr/bin/env python3
"""Run the spatial hidden-dimension wavelet supplement."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

import yaml

from mind.wavelet_course.cache_loading import load_repope_qwen_cache_entries
from mind.wavelet_course.common_sequence_models import parse_cuda_device_ordinals
from mind.wavelet_course.population import WaveletPopulation, build_wavelet_population
from mind.wavelet_course.spatial_wavelet_runner import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_PATIENCE,
    DEFAULT_QUICK_MAX_EPOCHS,
    DEFAULT_SEQUENCE_MODEL,
    SPATIAL_EXPERIMENT_NAME,
    run_spatial_hidden_wavelet_experiment,
)
from mind.wavelet_course.utils import DEFAULT_SPLIT_RATIOS, SPLIT_NAMES


DEFAULT_CONFIG = Path("configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/wavelet_course_v2")
DEFAULT_STAGE0_ROOT = Path("outputs/stage0")
OUTPUT_DIR_NAMES = ("audit", "reports")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = resolve_config(load_yaml_config(Path(args.config)), args)
        dirs = ensure_output_dirs(Path(str(config["output_root"])))
        device_audit = ensure_device_available(str(config["device"]), allow_cpu=bool(config.get("allow_cpu", False)))
        if device_audit.get("device_type") == "cpu":
            config["device"] = "cpu"
        write_json(config, dirs["audit"] / "spatial_hidden_wavelet_config_resolved.json")
        preflight = run_preflight(config, audit_dir=dirs["audit"])
        if args.preflight_only:
            print(json.dumps({"status": "success", "preflight_only": True}, sort_keys=True))
            return 0
        status = run_spatial_hidden_wavelet_experiment(
            config=config,
            preflight=preflight,
            output_root=Path(str(config["output_root"])),
            reports_dir=dirs["reports"],
        )
    except Exception as error:
        print(f"spatial_hidden_wavelet_failed={error}", file=sys.stderr)
        return 1
    print(str(status["summary_snippet"]))
    print(f"metrics_path={status['metrics_path']}")
    print(f"summary_path={status['summary_path']}")
    return 0


def load_yaml_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return dict(payload)


def resolve_config(config: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    resolved = json.loads(json.dumps(dict(config)))
    resolved.setdefault("experiment_name", "wavelet_course_repope_qwen3_vl_8b_spatial_hidden_wavelet")
    resolved.setdefault("runner_version", "spatial_hidden_wavelet_v1")
    resolved.setdefault("stage0_root", str(DEFAULT_STAGE0_ROOT))
    resolved.setdefault("output_root", str(DEFAULT_OUTPUT_ROOT))
    resolved.setdefault("seed", 20260506)
    resolved.setdefault("model_name", "qwen3-vl-8b")
    resolved.setdefault("dataset_name", "repope")
    resolved.setdefault("subsets", ["popular", "random", "adversarial"])
    resolved.setdefault("expected_num_layers", 36)
    resolved.setdefault("expected_hidden_dim", 4096)
    resolved.setdefault("split_ratios", {"train": 0.60, "validation": 0.20, "test": 0.20})
    resolved.setdefault("require_positive_in_each_split", True)
    resolved["device"] = args.device or str(resolved.get("device", "cuda:0"))
    resolved["allow_cpu"] = bool(args.allow_cpu or resolved.get("allow_cpu", False))
    resolved["quick_run"] = bool(args.quick or resolved.get("quick_run", False))

    sequence_section = _sequence_model_section(resolved, DEFAULT_SEQUENCE_MODEL)
    spatial = dict(resolved.get("spatial_hidden_wavelet", {}) or {})
    spatial.setdefault("name", SPATIAL_EXPERIMENT_NAME)
    spatial.setdefault("wavelet", "db2")
    spatial.setdefault("level", 2)
    spatial.setdefault("threshold", "universal_soft")
    spatial.setdefault("sequence_model", DEFAULT_SEQUENCE_MODEL)
    spatial.setdefault("learning_rate", DEFAULT_LEARNING_RATE)
    spatial.setdefault("patience", DEFAULT_PATIENCE)
    spatial.setdefault("batch_size", int(sequence_section.get("batch_size", DEFAULT_BATCH_SIZE)))
    spatial.setdefault("hidden_dim", int(sequence_section.get("hidden_dim", 128)))
    spatial.setdefault("dropout", float(sequence_section.get("dropout", 0.1)))
    spatial.setdefault("weight_decay", float(sequence_section.get("weight_decay", 1e-4)))
    spatial.setdefault("max_epochs", DEFAULT_MAX_EPOCHS)
    spatial.setdefault("quick_max_epochs", DEFAULT_QUICK_MAX_EPOCHS)
    if resolved["quick_run"]:
        spatial["max_epochs"] = min(int(spatial["max_epochs"]), int(spatial["quick_max_epochs"]))
    resolved["spatial_hidden_wavelet"] = spatial
    return resolved


def _sequence_model_section(config: Mapping[str, object], name: str) -> dict[str, object]:
    value = config.get("sequence_models")
    if not isinstance(value, Mapping):
        return {}
    section = value.get(name)
    if not isinstance(section, Mapping):
        return {}
    return dict(section)


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {name: output_root / name for name in OUTPUT_DIR_NAMES}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def ensure_device_available(device: str, *, allow_cpu: bool) -> dict[str, object]:
    normalized = device.strip().lower()
    if normalized == "cpu":
        if not allow_cpu:
            raise RuntimeError("requested device cpu, but allow_cpu=false")
        return {"device": "cpu", "device_type": "cpu"}
    if not normalized.startswith("cuda"):
        raise RuntimeError(f"unsupported device {device}; expected cuda device or cpu")
    try:
        import torch
    except ImportError as error:
        if allow_cpu:
            return {"device": "cpu", "device_type": "cpu", "fallback_reason": "torch is not installed"}
        raise RuntimeError(f"requested device {device}, but torch is not installed") from error
    if not torch.cuda.is_available():
        if allow_cpu:
            return {"device": "cpu", "device_type": "cpu", "fallback_reason": "cuda unavailable"}
        raise RuntimeError(f"requested device {device}, but torch.cuda.is_available() is false")
    ordinals = parse_cuda_device_ordinals(normalized, device_count=int(torch.cuda.device_count()))
    return {"device": device, "device_type": "cuda", "cuda_ordinals": list(ordinals)}


def run_preflight(config: Mapping[str, object], *, audit_dir: Path) -> dict[str, object]:
    stage0_root = Path(str(config.get("stage0_root", DEFAULT_STAGE0_ROOT)))
    entries = load_repope_qwen_cache_entries(
        stage0_root=stage0_root,
        manifest_path=stage0_root / "manifests" / "cache_manifest.json",
        model_name=str(config["model_name"]),
        dataset_name=str(config["dataset_name"]),
        subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
        expected_num_layers=int(config["expected_num_layers"]),
        expected_hidden_dim=int(config["expected_hidden_dim"]),
    )
    population = build_wavelet_population(
        entries,
        manifest_dir=stage0_root / "manifests",
        dataset_name=str(config["dataset_name"]),
        subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
        seed=int(config["seed"]),
        ratios=split_ratio_values(config),
    )
    split_validation = validate_population_splits(
        population,
        require_positive_in_each_split=bool(config.get("require_positive_in_each_split", False)),
    )
    audit = {
        "num_entries": len(entries),
        "num_primary_population": len(population.primary_entries),
        "split_validation": split_validation,
    }
    write_json(audit, audit_dir / "spatial_hidden_wavelet_preflight.json")
    return {"entries": entries, "population": population, "split_validation": split_validation}


def split_ratio_values(config: Mapping[str, object]) -> tuple[float, float, float]:
    ratios = config.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    if isinstance(ratios, Mapping):
        return tuple(float(ratios[name]) for name in SPLIT_NAMES)  # type: ignore[return-value]
    return tuple(float(value) for value in ratios)  # type: ignore[arg-type]


def validate_population_splits(
    population: WaveletPopulation,
    *,
    require_positive_in_each_split: bool,
) -> dict[str, object]:
    counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_NAMES}
    for entry, label in zip(population.primary_entries, population.labels, strict=True):
        split = str(entry.get("wavelet_split", ""))
        if split not in SPLIT_NAMES:
            raise RuntimeError(f"invalid wavelet split {split!r}")
        counts[split][int(label)] += 1
    for split in SPLIT_NAMES:
        if counts[split][0] + counts[split][1] == 0:
            raise RuntimeError(f"{split} split has no primary rows")
        if require_positive_in_each_split and counts[split][1] == 0:
            raise RuntimeError(f"{split} split has no positives")
    if counts["train"][0] == 0 or counts["train"][1] == 0:
        raise RuntimeError("train split must contain two classes")
    return {
        "valid": True,
        "counts": {
            split: {"neg": int(counts[split][0]), "pos": int(counts[split][1])}
            for split in SPLIT_NAMES
        },
    }


def write_json(payload: Mapping[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
