#!/usr/bin/env python3
"""Preflight and dispatch skeleton for the paired wavelet-course v2 run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import inspect
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
import shutil
import sys
import time
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

import numpy as np
import yaml

from mind.wavelet_course.cache_loading import load_repope_qwen_cache_entries
from mind.wavelet_course.common_sequence_models import parse_cuda_device_ordinals
from mind.wavelet_course.ours_wavelet_features import (
    NO_CANDIDATES,
    YES_CANDIDATES,
    resolve_yes_no_token_ids,
)
from mind.wavelet_course.paired_config import PAIR_BLOCKS, PAIR_SOURCES, PairedRunSpec
from mind.wavelet_course.paired_grid import PAIR_DEFINITIONS, assert_paired_grid_complete, build_paired_grid
from mind.wavelet_course.population import WaveletPopulation, build_wavelet_population, population_key
from mind.wavelet_course.reporting import write_json
from mind.wavelet_course.utils import DEFAULT_SPLIT_RATIOS, SPLIT_NAMES


DEFAULT_CONFIG = Path("configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml")
DEFAULT_OUTPUT_ROOT = Path("outputs/wavelet_course_v2")
DEFAULT_STAGE0_ROOT = Path("outputs/stage0")
DEFAULT_MODEL_CONFIG_PATH = Path("configs/models/qwen3_vl_8b.yaml")
PAIRED_RUNNER_MODULE = "mind.wavelet_course.paired_runner"
PAIRED_RUNNER_FUNCTIONS = (
    "run_paired_wavelet_experiment",
    "run_paired_experiment",
    "run_experiment",
    "run",
)
OUTPUT_DIR_NAMES = ("audit", "cache", "features", "logs", "reports")
SAMPLE_GRID_FIELDS = (
    "row_index",
    "population_key",
    "image_id",
    "subset",
    "split",
    "label",
    "row_order_hash",
)
TOKENIZER_AUDIT_FIELDS = (
    "yes_token_id",
    "no_token_id",
    "chosen_yes_token",
    "chosen_no_token",
    "yes_no_trace_source",
    "token_id_source",
    "tokenizer_candidate_table",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=None,
        help="Paired v2 blocks to run, as space-separated values or comma-separated groups.",
    )
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(load_yaml_config(Path(args.config)), args)
    output_root = Path(str(config.get("output_root", DEFAULT_OUTPUT_ROOT)))
    dirs = ensure_output_dirs(output_root)
    if not args.preflight_only:
        clean_stale_run_outputs(dirs)

    write_resolved_config(config, dirs)
    ensure_device_available(
        str(config["device"]),
        allow_cpu=bool(config.get("allow_cpu", False)),
        audit_dir=dirs["audit"],
    )

    try:
        preflight = run_preflight(config, audit_dir=dirs["audit"])
    except Exception as error:
        status = {
            "status": "failed",
            "training_started": False,
            "run_id": run_id_from_config(config),
            "failure_reason": str(error),
            "runner_module": PAIRED_RUNNER_MODULE,
        }
        status.update(write_failure_run_artifacts(status, dirs=dirs, config=config, preflight={}))
        write_json(status, dirs["reports"] / "full_run_status.json")
        print_final_summary(
            config=config,
            preflight={},
            status=status,
            output_root=output_root,
        )
        print(f"preflight_failed={error}", file=sys.stderr)
        return 2

    print_preflight_stats(preflight)
    if args.preflight_only:
        return 0

    try:
        status = run_full(config, preflight=preflight, output_root=output_root, dirs=dirs)
    except Exception as error:
        status = {
            "status": "failed",
            "training_started": False,
            "run_id": run_id_from_config(config),
            "failure_reason": str(error),
            "runner_module": PAIRED_RUNNER_MODULE,
        }
        status.update(write_failure_run_artifacts(status, dirs=dirs, config=config, preflight=preflight))
        write_json(status, dirs["reports"] / "full_run_status.json")
        print_final_summary(
            config=config,
            preflight=preflight,
            status=status,
            output_root=output_root,
        )
        print(f"full_run_failed={error}", file=sys.stderr)
        return 1
    if status.get("status") != "success":
        status.setdefault("run_id", run_id_from_config(config))
        status.update(write_failure_run_artifacts(status, dirs=dirs, config=config, preflight=preflight))
    write_json(status, dirs["reports"] / "full_run_status.json")
    print_final_summary(
        config=config,
        preflight=preflight,
        status=status,
        output_root=output_root,
    )
    return 0 if status.get("status") == "success" else 1


def load_yaml_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return dict(payload)


def resolve_config(config: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("experiment_name", "wavelet_course_repope_qwen3_vl_8b_v2_paired")
    resolved.setdefault("runner_version", "v2_paired_skeleton")
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
    resolved["allow_no_xgboost"] = bool(resolved.get("allow_no_xgboost", True))
    resolved["quick_run"] = bool(args.quick or resolved.get("quick_run", False))
    resolved["ours_signal"] = resolve_ours_signal_token_ids(resolved)

    paired = dict(resolved.get("paired_wavelet_v2", {}) or {})
    validate_yaml_pair_definitions(paired)
    paired.setdefault("enabled", True)
    paired.setdefault("run_id", "paired_wavelet_v2")
    paired.setdefault("expected_sources", list(PAIR_SOURCES))
    paired.setdefault("expected_blocks", list(PAIR_BLOCKS))
    if args.blocks:
        paired_blocks = parse_blocks(args.blocks)
        block_source = "cli"
    elif args.quick and isinstance(resolved.get("quick"), Mapping) and dict(resolved["quick"]).get("blocks"):
        paired_blocks = _coerce_blocks(dict(resolved["quick"])["blocks"])
        block_source = "quick_config"
    else:
        paired_blocks = _coerce_blocks(paired.get("blocks", PAIR_BLOCKS))
        block_source = "config"
    paired["blocks"] = list(paired_blocks)
    paired["expected_blocks"] = list(paired_blocks)
    paired["block_source"] = block_source

    pairs = assert_paired_grid_complete(
        build_paired_grid(blocks=paired_blocks),
        expected_blocks=paired_blocks,
        expected_sources=tuple(str(item) for item in paired.get("expected_sources", PAIR_SOURCES)),
    )
    run_spec = PairedRunSpec(
        run_id=str(paired["run_id"]),
        pairs=pairs,
        expected_blocks=tuple(paired_blocks),
        expected_sources=tuple(str(item) for item in paired.get("expected_sources", PAIR_SOURCES)),
        expected_num_layers=int(resolved["expected_num_layers"]),
        expected_hidden_dim=int(resolved["expected_hidden_dim"]),
        epsilon=float(paired.get("epsilon", resolved.get("epsilon", 1e-12))),
        description=str(paired.get("description", "")),
    )
    paired["resolved_run_spec"] = run_spec.as_dict()
    paired["pairs"] = [pair.as_dict() for pair in pairs]
    paired["num_pair_rows"] = len(pairs)
    paired["num_pair_ids"] = len({pair.pair_id for pair in pairs})
    resolved["paired_wavelet_v2"] = paired
    return resolved


def validate_yaml_pair_definitions(paired: Mapping[str, object]) -> None:
    if "pair_definitions" not in paired:
        return
    yaml_definitions = paired.get("pair_definitions")
    expected = [dict(definition) for definition in PAIR_DEFINITIONS]
    if yaml_definitions != expected:
        raise ValueError("paired_wavelet_v2.pair_definitions drift from code PAIR_DEFINITIONS")


def resolve_ours_signal_token_ids(config: Mapping[str, object]) -> dict[str, object]:
    ours_signal = dict(config.get("ours_signal", {}) or {})
    if _optional_int(ours_signal.get("yes_token_id")) is not None and _optional_int(ours_signal.get("no_token_id")) is not None:
        ours_signal["yes_token_id"] = int(ours_signal["yes_token_id"])
        ours_signal["no_token_id"] = int(ours_signal["no_token_id"])
        ours_signal.setdefault("yes_no_trace_source", "final_broadcast")
        ours_signal.setdefault("token_id_source", "explicit_config")
        ours_signal.setdefault("chosen_yes_token", str(ours_signal.get("yes_token", "")))
        ours_signal.setdefault("chosen_no_token", str(ours_signal.get("no_token", "")))
        ours_signal.setdefault("tokenizer_candidate_table", [])
        return ours_signal
    model_id = load_model_id(config)
    tokenizer = load_local_tokenizer(model_id)
    candidate_table = build_tokenizer_candidate_table(tokenizer=tokenizer)
    yes_id, no_id = resolve_yes_no_token_ids(tokenizer=tokenizer)
    _mark_selected_token_candidates(candidate_table, yes_token_id=yes_id, no_token_id=no_id)
    ours_signal["yes_token_id"] = int(yes_id)
    ours_signal["no_token_id"] = int(no_id)
    ours_signal["chosen_yes_token"] = _chosen_candidate_token(candidate_table, label="yes", token_id=yes_id)
    ours_signal["chosen_no_token"] = _chosen_candidate_token(candidate_table, label="no", token_id=no_id)
    ours_signal["yes_no_trace_source"] = "final_broadcast"
    ours_signal["token_id_source"] = str(getattr(tokenizer, "source", "local_auto_tokenizer"))
    ours_signal["tokenizer_candidate_table"] = candidate_table
    return ours_signal


def build_tokenizer_candidate_table(
    *,
    tokenizer: object | None = None,
    vocab: Mapping[str, int] | None = None,
    logits_size: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, candidates in (("yes", YES_CANDIDATES), ("no", NO_CANDIDATES)):
        for candidate in candidates:
            for source, token_id in _candidate_token_ids_with_source(
                candidate,
                tokenizer=tokenizer,
                vocab=vocab,
            ):
                valid = token_id >= 0 and (logits_size is None or token_id < int(logits_size))
                if not valid:
                    continue
                rows.append(
                    {
                        "label": label,
                        "candidate": candidate,
                        "token_id": int(token_id),
                        "selected": False,
                        "source": source,
                    }
                )
    return rows


def _candidate_token_ids_with_source(
    text: str,
    *,
    tokenizer: object | None,
    vocab: Mapping[str, int] | None,
) -> list[tuple[str, int]]:
    ids: list[tuple[str, int]] = []
    if vocab is not None and text in vocab:
        ids.append(("vocab", int(vocab[text])))
    if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
        token_id = tokenizer.convert_tokens_to_ids(text)  # type: ignore[attr-defined]
        if isinstance(token_id, int) and token_id >= 0:
            ids.append(("convert_tokens_to_ids", int(token_id)))
    if tokenizer is not None and hasattr(tokenizer, "encode"):
        encoded = tokenizer.encode(text, add_special_tokens=False)  # type: ignore[attr-defined]
        if isinstance(encoded, Sequence) and len(encoded) == 1:
            ids.append(("encode", int(encoded[0])))
    if tokenizer is not None and callable(tokenizer):
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, Mapping) else None
        if isinstance(input_ids, Sequence) and len(input_ids) == 1:
            ids.append(("callable", int(input_ids[0])))
    return ids


def _mark_selected_token_candidates(
    rows: list[dict[str, object]],
    *,
    yes_token_id: int,
    no_token_id: int,
) -> None:
    selected = {"yes": int(yes_token_id), "no": int(no_token_id)}
    for row in rows:
        row["selected"] = (
            int(row.get("token_id", -1)) == selected.get(str(row.get("label", "")))
        )


def _chosen_candidate_token(rows: Sequence[Mapping[str, object]], *, label: str, token_id: int) -> str:
    for row in rows:
        if (
            str(row.get("label", "")) == label
            and int(row.get("token_id", -1)) == int(token_id)
        ):
            return str(row.get("candidate", ""))
    return ""


def load_local_tokenizer(model_id: str) -> object:
    auto_error: BaseException | None = None
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        auto_error = error
    else:
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                local_files_only=True,
                trust_remote_code=True,
            )
            _set_tokenizer_source(tokenizer, "local_auto_tokenizer")
            return tokenizer
        except Exception as error:
            auto_error = error

    try:
        tokenizer_json = _resolve_local_tokenizer_json(model_id)
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_json))
        return _TokenizerJsonWrapper(tokenizer, tokenizer_json)
    except Exception as json_error:
        raise RuntimeError(
            "failed to load local tokenizer for "
            f"model_id={model_id!r}; "
            f"AutoTokenizer error: {_format_error(auto_error)}; "
            f"tokenizer.json error: {_format_error(json_error)}"
        ) from json_error


class _TokenizerJsonWrapper:
    source = "local_tokenizer_json"

    def __init__(self, tokenizer: object, tokenizer_json_path: Path) -> None:
        self._tokenizer = tokenizer
        self.tokenizer_json_path = str(tokenizer_json_path)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        encoded = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)  # type: ignore[attr-defined]
        ids = getattr(encoded, "ids", encoded)
        return [int(token_id) for token_id in ids]


def _set_tokenizer_source(tokenizer: object, source: str) -> None:
    try:
        setattr(tokenizer, "source", source)
    except Exception:
        pass


def _format_error(error: BaseException | None) -> str:
    if error is None:
        return "not attempted"
    return f"{type(error).__name__}: {error}"


def _resolve_local_tokenizer_json(model_id: str) -> Path:
    model_path = Path(model_id).expanduser()
    if model_path.is_dir():
        tokenizer_json = model_path / "tokenizer.json"
        if tokenizer_json.is_file():
            return tokenizer_json

    repo_dir = _hf_cache_repo_dir(model_id)
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"missing HF cache repo dir: {repo_dir}")

    refs_main = repo_dir / "refs" / "main"
    refs_error: FileNotFoundError | None = None
    if refs_main.is_file():
        snapshot_name = refs_main.read_text(encoding="utf-8").strip()
        if snapshot_name:
            tokenizer_json = repo_dir / "snapshots" / snapshot_name / "tokenizer.json"
            if tokenizer_json.is_file():
                return tokenizer_json
            refs_error = FileNotFoundError(f"refs/main points to missing tokenizer.json: {tokenizer_json}")

    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise FileNotFoundError(f"missing HF cache snapshots dir: {snapshots_dir}")
    candidates = [path / "tokenizer.json" for path in snapshots_dir.iterdir() if (path / "tokenizer.json").is_file()]
    if not candidates:
        if refs_error is not None:
            raise FileNotFoundError(
                f"{refs_error}; no tokenizer.json found under snapshots dir: {snapshots_dir}"
            ) from refs_error
        raise FileNotFoundError(f"no tokenizer.json found under snapshots dir: {snapshots_dir}")
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    return candidates[0]


def _hf_cache_repo_dir(model_id: str) -> Path:
    cache_root = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if cache_root:
        hub_dir = Path(cache_root).expanduser()
    else:
        hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
        hub_dir = hf_home / "hub"
    return hub_dir / f"models--{model_id.replace('/', '--')}"


def load_model_id(config: Mapping[str, object]) -> str:
    ours_signal = dict(config.get("ours_signal", {}) or {})
    model_id = str(ours_signal.get("model_id", "") or config.get("model_id", "") or "").strip()
    if model_id:
        return model_id
    model_config_path = Path(str(config.get("model_config_path", DEFAULT_MODEL_CONFIG_PATH)))
    payload = load_yaml_config(model_config_path)
    model_id = payload.get("model_id")
    if not model_id:
        raise ValueError(f"{model_config_path}: missing model_id")
    return str(model_id)


def _optional_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def parse_blocks(values: Sequence[str]) -> tuple[str, ...]:
    blocks: list[str] = []
    for value in values:
        for part in str(value).split(","):
            text = part.strip().upper()
            if text:
                blocks.append(text)
    return _coerce_blocks(blocks)


def _coerce_blocks(values: object) -> tuple[str, ...]:
    if isinstance(values, str):
        candidates = [part.strip().upper() for part in values.split(",") if part.strip()]
    elif isinstance(values, Sequence):
        candidates = [str(item).strip().upper() for item in values if str(item).strip()]
    else:
        raise ValueError("paired_wavelet_v2.blocks must be a sequence or comma-separated string")
    if not candidates:
        raise ValueError("at least one paired v2 block is required")
    unknown = sorted(set(candidates) - set(PAIR_BLOCKS))
    if unknown:
        raise ValueError(f"unknown paired v2 blocks: {unknown}")
    ordered = [block for block in PAIR_BLOCKS if block in set(candidates)]
    return tuple(ordered)


def ensure_output_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {name: output_root / name for name in OUTPUT_DIR_NAMES}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def clean_stale_run_outputs(dirs: Mapping[str, Path]) -> list[str]:
    reports_dir = dirs["reports"]
    features_dir = dirs["features"]
    stale_paths = [
        reports_dir / "metrics_long.csv",
        reports_dir / "metrics_wide_paired.csv",
        reports_dir / "best_by_block.csv",
        reports_dir / "pairwise_winrate.csv",
        reports_dir / "summary.md",
        reports_dir / "failure_report.csv",
        reports_dir / "metrics_ledger.csv",
        reports_dir / "full_run_status.json",
        features_dir / "feature_shape_manifest.csv",
    ]
    removed: list[str] = []
    for path in stale_paths:
        if path.exists():
            path.unlink()
            removed.append(str(path))
    curve_dir = reports_dir / "training_curves"
    if curve_dir.exists():
        shutil.rmtree(curve_dir)
        removed.append(str(curve_dir))
    return removed


def write_resolved_config(config: Mapping[str, object], dirs: Mapping[str, Path]) -> None:
    write_json(config, dirs["audit"] / "experiment_config_resolved.json")
    write_json(config, dirs["reports"] / "experiment_config_resolved.json")
    write_tokenizer_audit(config, audit_dir=dirs["audit"])


def write_tokenizer_audit(config: Mapping[str, object], *, audit_dir: Path) -> Path | None:
    ours_signal = _mapping(config.get("ours_signal"))
    if not ours_signal:
        return None
    payload = {
        field: ours_signal.get(field, "")
        for field in TOKENIZER_AUDIT_FIELDS
        if field in ours_signal
    }
    if not payload:
        return None
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / "tokenizer_audit.json"
    write_json(payload, path)
    return path


def ensure_device_available(
    device: str,
    *,
    allow_cpu: bool,
    audit_dir: Path | None = None,
) -> dict[str, object]:
    normalized_device = device.strip().lower()
    if normalized_device == "cpu":
        if allow_cpu:
            audit = {"device": device, "device_type": "cpu", "allow_cpu": True}
            _write_device_audit(audit, audit_dir=audit_dir)
            return audit
        raise RuntimeError(
            f"requested device {device}, but allow_cpu=false; "
            "pass --allow-cpu or set allow_cpu=true"
        )
    if not normalized_device.startswith("cuda"):
        raise RuntimeError(f"unsupported device {device}; expected cuda device or cpu")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(f"requested device {device}, but torch is not installed") from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"requested device {device}, but torch.cuda.is_available() is false"
        )
    device_count = int(torch.cuda.device_count())
    ordinals = parse_cuda_device_ordinals(normalized_device, device_count=device_count)
    primary_ordinal = int(ordinals[0])
    cuda_devices = []
    for ordinal in ordinals:
        free_bytes, total_bytes = _cuda_mem_get_info(torch.cuda, int(ordinal))
        cuda_devices.append(
            {
                "cuda_ordinal": int(ordinal),
                "gpu_name": str(torch.cuda.get_device_name(int(ordinal))),
                "memory_free_bytes": int(free_bytes),
                "memory_total_bytes": int(total_bytes),
            }
        )
    primary_device = cuda_devices[0]
    audit = {
        "device": device,
        "device_type": "cuda",
        "cuda_ordinal": primary_ordinal,
        "cuda_primary_ordinal": primary_ordinal,
        "cuda_ordinals": [int(ordinal) for ordinal in ordinals],
        "cuda_device_count": device_count,
        "gpu_name": primary_device["gpu_name"],
        "memory_free_bytes": primary_device["memory_free_bytes"],
        "memory_total_bytes": primary_device["memory_total_bytes"],
        "cuda_devices": cuda_devices,
        "data_parallel": len(ordinals) > 1,
    }
    _write_device_audit(audit, audit_dir=audit_dir)
    return audit


def _cuda_ordinal(normalized_device: str) -> int:
    return int(parse_cuda_device_ordinals(normalized_device)[0])


def _cuda_mem_get_info(cuda: object, ordinal: int) -> tuple[int, int]:
    mem_get_info = getattr(cuda, "mem_get_info", None)
    if not callable(mem_get_info):
        return 0, 0
    try:
        free_bytes, total_bytes = mem_get_info(ordinal)
    except TypeError:
        free_bytes, total_bytes = mem_get_info()
    return int(free_bytes), int(total_bytes)


def _write_device_audit(audit: Mapping[str, object], *, audit_dir: Path | None) -> None:
    if audit_dir is None:
        return
    audit_dir.mkdir(parents=True, exist_ok=True)
    write_json(dict(audit), audit_dir / "cuda_device_audit.json")


def run_preflight(config: Mapping[str, object], *, audit_dir: Path) -> dict[str, object]:
    cache_audit_path = audit_dir / "cache_acceptance.json"
    population_audit_path = audit_dir / "population_audit.csv"
    write_paired_grid_artifacts(config, audit_dir=audit_dir)

    stage0_root = Path(str(config.get("stage0_root", DEFAULT_STAGE0_ROOT)))
    try:
        entries = load_repope_qwen_cache_entries(
            stage0_root=stage0_root,
            manifest_path=stage0_root / "manifests" / "cache_manifest.json",
            model_name=str(config["model_name"]),
            dataset_name=str(config["dataset_name"]),
            subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
            expected_num_layers=int(config["expected_num_layers"]),
            expected_hidden_dim=int(config["expected_hidden_dim"]),
        )
    except Exception as error:
        cache_audit = {"accepted": False, "num_entries": 0, "failure_reason": str(error)}
        write_json(cache_audit, cache_audit_path)
        write_population_audit_csv([], population_audit_path)
        raise RuntimeError(str(error)) from error

    cache_audit = build_cache_acceptance(entries, config)
    try:
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
    except Exception as error:
        if "population" in locals():
            write_population_audit_csv(population.audit_rows, population_audit_path)
        else:
            write_population_audit_csv([], population_audit_path)
        cache_audit["split_validation"] = {"valid": False, "failure_reason": str(error)}
        write_json(cache_audit, cache_audit_path)
        raise RuntimeError(str(error)) from error

    population_audit = population_summary(population)
    cache_audit["split_validation"] = split_validation
    write_json(cache_audit, cache_audit_path)
    write_population_audit_csv(population.audit_rows, population_audit_path)
    sample_grid_audit = write_sample_grid_artifacts(population, audit_dir=audit_dir)
    paired_grid_audit = write_paired_grid_artifacts(
        config,
        audit_dir=audit_dir,
        population_summary=population_audit,
        split_validation=split_validation,
    )
    return {
        "entries": entries,
        "cache_audit": cache_audit,
        "population": population,
        "population_summary": population_audit,
        "split_validation": split_validation,
        "paired_grid_audit": paired_grid_audit,
        "sample_grid_audit": sample_grid_audit,
    }


def write_sample_grid_artifacts(population: WaveletPopulation, *, audit_dir: Path) -> dict[str, object]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    rows = build_sample_grid_rows(population)
    row_order_hash = sample_grid_row_order_hash(rows)
    rows_with_hash = [dict(row, row_order_hash=row_order_hash) for row in rows]
    sample_grid_path = audit_dir / "sample_grid.csv"
    configured_sample_grid_path = audit_dir / "configured_sample_grid.csv"
    for path in (sample_grid_path, configured_sample_grid_path):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SAMPLE_GRID_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows_with_hash:
                writer.writerow({field: row.get(field, "") for field in SAMPLE_GRID_FIELDS})
    audit = {
        "valid": True,
        "sample_grid_path": str(sample_grid_path),
        "configured_sample_grid_path": str(configured_sample_grid_path),
        "num_rows": len(rows),
        "configured_num_rows": len(rows),
        "row_order_hash": row_order_hash,
        "configured_row_order_hash": row_order_hash,
    }
    write_json(audit, audit_dir / "sample_grid_audit.json")
    return audit


def build_sample_grid_rows(population: WaveletPopulation) -> list[dict[str, object]]:
    labels = list(population.labels)
    if len(population.primary_entries) != len(labels):
        raise ValueError("sample grid population entries and labels must have matching lengths")
    rows: list[dict[str, object]] = []
    for index, (entry, label) in enumerate(zip(population.primary_entries, labels, strict=True)):
        key = str(entry.get("wavelet_population_key") or population_key(entry))
        row = {
            "row_index": index,
            "population_key": key,
            "image_id": _required_sample_text(entry, "image_id", index),
            "subset": _required_sample_text(entry, "subset", index),
            "split": _required_sample_text(entry, "wavelet_split", index),
            "label": int(label),
        }
        rows.append(row)
    return rows


def sample_grid_row_order_hash(rows: Sequence[Mapping[str, object]]) -> str:
    payload = [
        {
            "row_index": int(row["row_index"]),
            "population_key": str(row["population_key"]),
            "image_id": str(row["image_id"]),
            "subset": str(row["subset"]),
            "split": str(row["split"]),
            "label": int(row["label"]),
        }
        for row in rows
    ]
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _required_sample_text(entry: Mapping[str, object], field: str, index: int) -> str:
    value = entry.get(field)
    if value is None or str(value).strip() == "":
        raise ValueError(f"sample grid row {index} missing {field}")
    return str(value)


def write_paired_grid_artifacts(
    config: Mapping[str, object],
    *,
    audit_dir: Path,
    population_summary: Mapping[str, object] | None = None,
    split_validation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    paired_grid_path = audit_dir / "paired_grid.json"
    paired_grid_audit_path = audit_dir / "paired_grid_audit.json"
    paired_grid = build_paired_grid_audit(
        config,
        population_summary=population_summary,
        split_validation=split_validation,
    )
    paired_grid_audit = dict(paired_grid)
    paired_grid_audit["paired_grid_path"] = str(paired_grid_path)
    write_json(paired_grid, paired_grid_path)
    write_json(paired_grid_audit, paired_grid_audit_path)
    return paired_grid_audit


def build_paired_grid_audit(
    config: Mapping[str, object],
    *,
    population_summary: Mapping[str, object] | None = None,
    split_validation: Mapping[str, object] | None = None,
) -> dict[str, object]:
    paired = dict(config.get("paired_wavelet_v2", {}) or {})
    rows = [
        row
        for row in paired.get("pairs", [])
        if isinstance(row, Mapping)
    ]
    by_block: dict[str, list[str]] = defaultdict(list)
    by_pair: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        block = str(row.get("block", ""))
        pair_id = str(row.get("pair_id", ""))
        source = str(row.get("source", ""))
        if pair_id:
            by_block[block].append(pair_id)
            by_pair[pair_id].append(source)
    audit: dict[str, object] = {
        "valid": True,
        "audit_type": "paired_wavelet_v2_grid",
        "run_id": str(paired.get("run_id", "")),
        "requested_blocks": list(paired.get("blocks", [])),
        "expected_sources": list(paired.get("expected_sources", PAIR_SOURCES)),
        "num_pair_rows": len(rows),
        "num_pair_ids": len(by_pair),
        "pair_ids_by_block": {
            block: sorted(set(pair_ids))
            for block, pair_ids in sorted(by_block.items())
        },
        "sources_by_pair_id": {
            pair_id: sorted(sources)
            for pair_id, sources in sorted(by_pair.items())
        },
        "rows": rows,
    }
    if population_summary is not None:
        audit["population_summary"] = dict(population_summary)
    if split_validation is not None:
        audit["split_validation"] = dict(split_validation)
    return audit


def split_ratio_values(config: Mapping[str, object]) -> tuple[float, float, float]:
    ratios = config.get("split_ratios", DEFAULT_SPLIT_RATIOS)
    if isinstance(ratios, Mapping):
        return tuple(float(ratios[name]) for name in SPLIT_NAMES)  # type: ignore[return-value]
    return tuple(float(value) for value in ratios)  # type: ignore[arg-type]


def build_cache_acceptance(entries: Sequence[Mapping[str, object]], config: Mapping[str, object]) -> dict[str, object]:
    subset_counts = Counter(str(row.get("subset", "")) for row in entries)
    return {
        "accepted": True,
        "num_entries": len(entries),
        "model_name": str(config["model_name"]),
        "dataset_name": str(config["dataset_name"]),
        "expected_num_layers": int(config["expected_num_layers"]),
        "expected_hidden_dim": int(config["expected_hidden_dim"]),
        "subsets": {subset: int(count) for subset, count in sorted(subset_counts.items())},
    }


def validate_population_splits(
    population: WaveletPopulation,
    *,
    require_positive_in_each_split: bool,
) -> dict[str, object]:
    counts: dict[str, Counter[int]] = {split: Counter() for split in SPLIT_NAMES}
    for entry, label in zip(population.primary_entries, population.labels, strict=True):
        split = str(entry.get("wavelet_split", ""))
        if split not in SPLIT_NAMES:
            raise ValueError(f"invalid wavelet split {split!r}")
        counts[split][int(label)] += 1
    for split in SPLIT_NAMES:
        if counts[split][0] + counts[split][1] == 0:
            raise RuntimeError(f"{split} split has no primary rows")
        if require_positive_in_each_split and counts[split][1] == 0:
            raise RuntimeError(f"{split} split has no positives")
    if counts["train"][0] == 0 or counts["train"][1] == 0:
        raise RuntimeError("train split lacks two classes")
    return {
        "valid": True,
        "split_source": population.split_source,
        "counts": {
            split: {"neg": int(counts[split][0]), "pos": int(counts[split][1])}
            for split in SPLIT_NAMES
        },
    }


def population_summary(population: WaveletPopulation) -> dict[str, object]:
    labels = np.asarray(population.labels, dtype=np.int64)
    return {
        "num_primary_population": int(labels.shape[0]),
        "num_hard_hallucination": int(np.sum(labels == 1)),
        "num_correct": int(np.sum(labels == 0)),
        "split_source": population.split_source,
    }


def write_population_audit_csv(rows: Sequence[Mapping[str, object]], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "subset",
        "total",
        "gt_yes",
        "gt_no",
        "parsed_yes",
        "parsed_no",
        "correct",
        "hard_hallucination",
        "false_negative",
        "parsed_none",
        "invalid_label",
        "primary_pos",
        "primary_neg",
        "train_pos",
        "train_neg",
        "validation_pos",
        "validation_neg",
        "test_pos",
        "test_neg",
    ]
    extra_fields = sorted({key for row in rows for key in row if key not in base_fields})
    fields = base_fields + extra_fields
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return output


def run_full(
    config: Mapping[str, object],
    *,
    preflight: Mapping[str, object],
    output_root: Path,
    dirs: Mapping[str, Path],
) -> dict[str, object]:
    started_at = time.time()
    hook = find_paired_runner_hook()
    if hook is None:
        return {
            "status": "blocked",
            "training_started": False,
            "failure_reason": f"{PAIRED_RUNNER_MODULE} hook not available",
            "expected_hook_functions": list(PAIRED_RUNNER_FUNCTIONS),
            "output_root": str(output_root),
        }
    result = call_paired_runner_hook(
        hook,
        config=config,
        preflight=preflight,
        output_root=output_root,
        audit_dir=dirs["audit"],
        cache_dir=dirs["cache"],
        features_dir=dirs["features"],
        reports_dir=dirs["reports"],
    )
    if isinstance(result, Mapping):
        status = dict(result)
    else:
        status = {"status": "success", "result": result}
    status.setdefault("status", "success")
    status.setdefault("training_started", True)
    status.setdefault("runner_module", PAIRED_RUNNER_MODULE)
    status.setdefault("runner_function", getattr(hook, "__name__", ""))
    status.setdefault("elapsed_seconds", time.time() - started_at)
    return status


def write_failure_run_artifacts(
    status: Mapping[str, object],
    *,
    dirs: Mapping[str, Path],
    config: Mapping[str, object],
    preflight: Mapping[str, object],
) -> dict[str, object]:
    reports_dir = dirs["reports"]
    features_dir = dirs["features"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(status.get("run_id") or run_id_from_config(config))
    failure_reason = str(status.get("failure_reason", "") or status.get("status", "failed"))
    sample_grid = _mapping(preflight.get("sample_grid_audit"))
    sample_grid_path = str(
        status.get("sample_grid_path", "")
        or sample_grid.get("sample_grid_path", "")
        or sample_grid.get("selected_sample_grid_path", "")
    )
    sample_grid_rows = int(
        status.get("sample_grid_rows", 0)
        or sample_grid.get("num_rows", 0)
        or sample_grid.get("selected_num_rows", 0)
        or 0
    )
    sample_grid_hash = str(
        status.get("sample_grid_row_order_hash", "")
        or sample_grid.get("row_order_hash", "")
        or sample_grid.get("selected_row_order_hash", "")
    )
    metrics_ledger = reports_dir / "metrics_ledger.csv"
    row = {
        "run_id": run_id,
        "status": str(status.get("status", "failed") or "failed"),
        "failure_reason": failure_reason,
        "training_started": str(bool(status.get("training_started", False))).lower(),
        "runner_module": str(status.get("runner_module", PAIRED_RUNNER_MODULE)),
        "runner_function": str(status.get("runner_function", "")),
        "output_root": str(config.get("output_root", DEFAULT_OUTPUT_ROOT)),
        "sample_grid_path": sample_grid_path,
        "sample_grid_rows": sample_grid_rows,
        "sample_grid_row_order_hash": sample_grid_hash,
        "metrics_ledger": str(metrics_ledger),
    }
    metric_fields = (
        "run_id",
        "status",
        "failure_reason",
        "training_started",
        "runner_module",
        "runner_function",
        "output_root",
        "sample_grid_path",
        "sample_grid_rows",
        "sample_grid_row_order_hash",
        "metrics_ledger",
    )
    _write_failure_csv(metrics_ledger, [row], metric_fields)
    report_paths = {
        "metrics_long": reports_dir / "metrics_long.csv",
        "metrics_wide_paired": reports_dir / "metrics_wide_paired.csv",
        "best_by_block": reports_dir / "best_by_block.csv",
        "pairwise_winrate": reports_dir / "pairwise_winrate.csv",
        "failure_report": reports_dir / "failure_report.csv",
        "summary": reports_dir / "summary.md",
    }
    _write_failure_csv(report_paths["metrics_long"], [row], metric_fields)
    _write_failure_csv(
        report_paths["metrics_wide_paired"],
        [{**row, "paired_status": "failure", "paired_failure_reason": failure_reason}],
        (*metric_fields, "paired_status", "paired_failure_reason"),
    )
    _write_failure_csv(report_paths["best_by_block"], [], ("block", "metric", "status", "failure_reason", "run_id"))
    _write_failure_csv(report_paths["pairwise_winrate"], [], ("block", "metric", "status", "failure_reason", "run_id"))
    _write_failure_csv(report_paths["failure_report"], [row], metric_fields)
    _write_failure_csv(
        features_dir / "feature_shape_manifest.csv",
        [row],
        metric_fields,
    )
    report_paths["summary"].write_text(
        "\n".join(
            [
                "# Paired Wavelet V2 Summary",
                "",
                "## Run Status",
                "",
                "- final_status: failed",
                f"- run_id: {run_id}",
                f"- failure_reason: {failure_reason}",
                f"- sample_grid_path: {sample_grid_path}",
                f"- sample_grid_rows: {sample_grid_rows}",
                f"- sample_grid_row_order_hash: {sample_grid_hash}",
                f"- metrics_ledger.csv: {metrics_ledger}",
                "",
                "No success report from a previous run is current for this output root.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "metrics_ledger": str(metrics_ledger),
        "report_paths": {key: str(value) for key, value in report_paths.items()},
        "metrics_long_rows": 1,
        "success_rows": 0,
        "failure_rows": 1,
        "sample_grid_path": sample_grid_path,
        "sample_grid_rows": sample_grid_rows,
        "sample_grid_row_order_hash": sample_grid_hash,
    }


def _write_failure_csv(
    output: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str],
) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(str(field) for field in fields))
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return output


def run_id_from_config(config: Mapping[str, object]) -> str:
    paired = _mapping(config.get("paired_wavelet_v2"))
    return str(paired.get("run_id") or config.get("run_id") or "paired_wavelet_v2")


def find_paired_runner_hook() -> Any | None:
    try:
        module = importlib.import_module(PAIRED_RUNNER_MODULE)
    except ImportError:
        return None
    for name in PAIRED_RUNNER_FUNCTIONS:
        hook = getattr(module, name, None)
        if callable(hook):
            return hook
    return None


def call_paired_runner_hook(hook: Any, **available: object) -> object:
    signature = inspect.signature(hook)
    kwargs: dict[str, object] = {}
    accepts_var_kwargs = False
    for name, parameter in signature.parameters.items():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            accepts_var_kwargs = True
            continue
        if parameter.kind in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.VAR_POSITIONAL,
        }:
            continue
        if name in available:
            kwargs[name] = available[name]
    if accepts_var_kwargs:
        kwargs.update(available)
    if kwargs:
        return hook(**kwargs)
    return hook(available["config"], available["preflight"], available["output_root"])


def print_preflight_stats(preflight: Mapping[str, object]) -> None:
    summary = dict(preflight.get("population_summary", {}) or {})
    cache = dict(preflight.get("cache_audit", {}) or {})
    split_validation = dict(preflight.get("split_validation", {}) or {})
    print(f"cache_accepted={str(bool(cache.get('accepted', False))).lower()}")
    print(f"cache_entries={int(cache.get('num_entries', 0) or 0)}")
    print(f"primary_population={int(summary.get('num_primary_population', 0) or 0)}")
    print(f"hard_hallucinations={int(summary.get('num_hard_hallucination', 0) or 0)}")
    print(f"split_valid={str(bool(split_validation.get('valid', False))).lower()}")


def print_final_summary(
    *,
    config: Mapping[str, object],
    preflight: Mapping[str, object],
    status: Mapping[str, object],
    output_root: Path | str,
) -> None:
    cache = _mapping(preflight.get("cache_audit"))
    population = _mapping(preflight.get("population_summary"))
    split_validation = _mapping(preflight.get("split_validation"))
    paired_grid = _mapping(preflight.get("paired_grid_audit"))
    report_paths = _mapping(status.get("report_paths"))
    print(f"final_status={status.get('status', '')}")
    print("v2_paired_extension=true")
    print(f"v1_preservation={_v1_preservation_source(config, population)}")
    print(f"output_root={output_root}")
    print(f"cache_entries={int(cache.get('num_entries', 0) or 0)}")
    print(f"primary_population={int(population.get('num_primary_population', 0) or 0)}")
    print(f"hard_hallucinations={int(population.get('num_hard_hallucination', 0) or 0)}")
    print(f"correct={int(population.get('num_correct', 0) or 0)}")
    for line in _final_split_count_lines(split_validation):
        print(line)
    print(f"paired_grid_path={paired_grid.get('paired_grid_path', '')}")
    print(f"paired_grid_rows={int(paired_grid.get('num_pair_rows', status.get('pair_rows', 0)) or 0)}")
    print(f"paired_grid_pair_ids={int(paired_grid.get('num_pair_ids', status.get('pair_ids', 0)) or 0)}")
    print(f"configured_grid_rows={int(status.get('configured_grid_rows', paired_grid.get('num_pair_rows', 0)) or 0)}")
    print(f"configured_grid_pair_ids={int(status.get('configured_grid_pair_ids', paired_grid.get('num_pair_ids', 0)) or 0)}")
    print(f"selected_run_grid_rows={int(status.get('selected_run_grid_rows', status.get('pair_rows', 0)) or 0)}")
    print(f"selected_run_grid_pair_ids={int(status.get('selected_run_grid_pair_ids', status.get('pair_ids', 0)) or 0)}")
    print(f"metrics_long_rows={int(status.get('metrics_long_rows', 0) or 0)}")
    print(f"success_rows={int(status.get('success_rows', 0) or 0)}")
    print(f"failure_rows={int(status.get('failure_rows', 0) or 0)}")
    print(f"sample_grid_path={_sample_grid_status_value(status, preflight, 'sample_grid_path')}")
    print(f"sample_grid_rows={int(_sample_grid_status_value(status, preflight, 'sample_grid_rows') or 0)}")
    print(f"sample_grid_row_order_hash={_sample_grid_status_value(status, preflight, 'sample_grid_row_order_hash')}")
    print(f"metrics_ledger={status.get('metrics_ledger', '')}")
    print(f"metrics_long={report_paths.get('metrics_long', '')}")
    print(f"metrics_wide_paired={report_paths.get('metrics_wide_paired', '')}")
    print(f"failure_report={report_paths.get('failure_report', '')}")
    print(f"summary_md={report_paths.get('summary', '')}")
    print("limitations=failed configs remain in reports; paired comparison is limited to comparable successes")
    print("conclusion=see summary.md for paired results and wavelet rationale")


def _final_split_count_lines(split_validation: Mapping[str, object]) -> list[str]:
    counts = _mapping(split_validation.get("counts"))
    lines: list[str] = []
    for split in SPLIT_NAMES:
        split_counts = _mapping(counts.get(split))
        lines.append(f"{split}_pos={int(split_counts.get('pos', 0) or 0)}")
        lines.append(f"{split}_neg={int(split_counts.get('neg', 0) or 0)}")
    return lines


def _v1_preservation_source(
    config: Mapping[str, object],
    population: Mapping[str, object],
) -> str:
    population_grid = _mapping(config.get("population_grid"))
    return str(population_grid.get("source", "") or population.get("split_source", "") or "not_provided")


def _sample_grid_status_value(
    status: Mapping[str, object],
    preflight: Mapping[str, object],
    field: str,
) -> object:
    if field in status and status.get(field) not in {None, ""}:
        return status[field]
    sample_grid = _mapping(preflight.get("sample_grid_audit"))
    audit_field = {
        "sample_grid_rows": "num_rows",
        "sample_grid_row_order_hash": "row_order_hash",
    }.get(field, field)
    return sample_grid.get(audit_field, "")


def _mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
