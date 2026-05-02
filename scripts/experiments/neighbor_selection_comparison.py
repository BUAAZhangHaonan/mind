#!/usr/bin/env python3
"""Compare CUDA neighbor selection methods for Phase C."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from typing import Sequence

import pandas as pd
import torch


REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.geometry.neighbor_selection import (  # noqa: E402
    METHOD_NAMES,
    compute_neighbor_feature_row_gpu,
    tune_radius_for_target_count_gpu,
)
from mind.utils import output_root_lock  # noqa: E402


OUTPUT_MARKDOWN = Path("docs/tables/subtask1_neighbor_comparison.md")
OUTPUT_CSV = Path("docs/tables/subtask1_neighbor_comparison.csv")
OUTPUT_ANALYSIS = Path("docs/review/subtask1_neighbor_selection_analysis.md")
METRIC_COLUMNS = (
    "roc_auc",
    "roc_auc_ci_lower",
    "roc_auc_ci_upper",
    "pr_auc",
    "pr_auc_ci_lower",
    "pr_auc_ci_upper",
)
CALIBRATED_SUMMARY_COLUMNS = (
    "cal_mean_drift",
    "cal_max_drift",
    "cal_final_drift",
    "cal_drift_slope",
    "cal_drift_variance",
)


def validate_cuda_visible_devices() -> None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None and visible_devices != "0":
        raise ValueError(f"CUDA_VISIBLE_DEVICES must be '0' when set, found {visible_devices!r}.")


def resolve_cuda_device(device_name: str | torch.device = "cuda:0") -> torch.device:
    device = torch.device(device_name)
    if device.type != "cuda":
        raise ValueError("neighbor selection comparison requires CUDA.")
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available.")
    if device.index not in (None, 0):
        raise ValueError("Use GPU 0 only.")
    return torch.device("cuda:0")


def parse_methods(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return METHOD_NAMES
    methods = tuple(item.strip() for item in value.split(",") if item.strip())
    invalid = [method for method in methods if method not in METHOD_NAMES]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}")
    if not methods:
        raise ValueError("Provide at least one method or 'all'.")
    return methods


def _method_rank(method: str) -> int:
    return {name: index for index, name in enumerate(METHOD_NAMES)}.get(method, len(METHOD_NAMES))


def _sort_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return sorted(
        [dict(row) for row in rows],
        key=lambda row: (
            str(row.get("model", "")),
            str(row.get("benchmark", "")),
            _method_rank(str(row.get("method", ""))),
        ),
    )


def _format_metric_cell(row: dict[str, object] | None) -> str:
    if row is None:
        return "missing"
    if str(row.get("status", "ok")) != "ok":
        reason = str(row.get("reason", "")).strip()
        return f"{row.get('status')}: {reason}".rstrip()
    return (
        f"ROC-AUC {float(row['roc_auc']):.4f} "
        f"[{float(row['roc_auc_ci_lower']):.4f}, {float(row['roc_auc_ci_upper']):.4f}]; "
        f"PR-AUC {float(row['pr_auc']):.4f} "
        f"[{float(row['pr_auc_ci_lower']):.4f}, {float(row['pr_auc_ci_upper']):.4f}]"
    )


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _row_context(index: int, row: dict[str, object]) -> str:
    return (
        f"row {index + 1} "
        f"({row.get('model', '<missing>')}/{row.get('benchmark', '<missing>')}/{row.get('method', '<missing>')})"
    )


def validate_metric_rows(rows: Sequence[dict[str, object]]) -> None:
    for index, row in enumerate(rows):
        row_dict = dict(row)
        method = str(row_dict.get("method", ""))
        if method not in METHOD_NAMES:
            raise ValueError(f"Invalid metric row {_row_context(index, row_dict)}: unsupported method {method!r}.")
        if str(row_dict.get("status", "ok")) != "ok":
            continue
        for column in METRIC_COLUMNS:
            value = row_dict.get(column)
            if _is_missing(value):
                raise ValueError(f"Invalid metric row {_row_context(index, row_dict)}: missing {column}.")
            try:
                number = float(value)
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid metric row {_row_context(index, row_dict)}: {column} must be numeric, found {value!r}."
                ) from None
            if not math.isfinite(number):
                raise ValueError(
                    f"Invalid metric row {_row_context(index, row_dict)}: {column} must be finite, found {value!r}."
                )


def _write_markdown_table(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    sorted_rows = _sort_rows(rows)
    groups = sorted({(str(row["model"]), str(row["benchmark"])) for row in sorted_rows})
    row_map = {
        (str(row["model"]), str(row["benchmark"]), str(row["method"])): row
        for row in sorted_rows
    }
    columns = ["model", "benchmark", *METHOD_NAMES]
    lines = [
        "# Subtask 1 Neighbor Selection Comparison",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for model, benchmark in groups:
        values = [
            model,
            benchmark,
            *[_format_metric_cell(row_map.get((model, benchmark, method))) for method in METHOD_NAMES],
        ]
        lines.append("| " + " | ".join(values) + " |")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rank_lines(rows: Sequence[dict[str, object]]) -> list[str]:
    lines: list[str] = []
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if str(row.get("status", "ok")) != "ok":
            continue
        groups.setdefault((str(row["model"]), str(row["benchmark"])), []).append(dict(row))
    for (model, benchmark), group_rows in sorted(groups.items()):
        ranked = sorted(group_rows, key=lambda row: (float(row["pr_auc"]), float(row["roc_auc"])), reverse=True)
        if not ranked:
            continue
        best = ranked[0]
        order = " > ".join(str(row["method"]) for row in ranked)
        lines.append(
            f"- {model} / {benchmark}: Best method by PR-AUC is {best['method']} "
            f"(PR-AUC {float(best['pr_auc']):.4f}, ROC-AUC {float(best['roc_auc']):.4f})."
        )
        lines.append(f"- {model} / {benchmark}: {order}")
    return lines


def _write_analysis(rows: Sequence[dict[str, object]], output_path: Path) -> None:
    lines = [
        "# Subtask 1 Neighbor Selection Analysis",
        "",
        "## Rank Summary",
        "",
        *_rank_lines(rows),
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_neighbor_comparison_outputs(
    rows: Sequence[dict[str, object]],
    *,
    csv_path: Path = OUTPUT_CSV,
    markdown_path: Path = OUTPUT_MARKDOWN,
    analysis_path: Path = OUTPUT_ANALYSIS,
) -> None:
    sorted_rows = _sort_rows(rows)
    validate_metric_rows(sorted_rows)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sorted_rows).to_csv(csv_path, index=False)
    _write_markdown_table(sorted_rows, markdown_path)
    _write_analysis(sorted_rows, analysis_path)


def metadata_row_from_entry(entry: dict[str, object]) -> dict[str, object]:
    answer_label = None if entry.get("parsed_answer") is None else int(entry["parsed_answer"])
    ground_truth_label = int(entry["label"])
    return {
        "sample_id": str(entry["sample_id"]),
        "image_id": int(entry.get("image_id", -1)),
        "ground_truth_label": ground_truth_label,
        "answer_label": -1 if answer_label is None else int(answer_label),
        "label": int(answer_label == 1 and ground_truth_label == 0),
        "subset": str(entry.get("subset", "")),
        "object_name": str(entry.get("object_name", "")),
    }


def is_grounded_reference_entry(entry: dict[str, object]) -> bool:
    if "ground_truth_label" in entry:
        return int(entry["ground_truth_label"]) == 1
    return int(entry.get("label", 0)) == 1


def build_reference_layers_from_raw_entries(
    entries: Sequence[dict[str, object]],
    *,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    layers: dict[int, list[torch.Tensor]] = {}
    for entry in entries:
        if not is_grounded_reference_entry(dict(entry)):
            continue
        selected_layers = [int(layer) for layer in entry["selected_layers"]]
        vectors = torch.as_tensor(entry["layer_vectors"], dtype=torch.float32)
        for offset, layer_index in enumerate(selected_layers):
            layers.setdefault(layer_index, []).append(vectors[offset].unsqueeze(0))
    if not layers:
        raise ValueError("No grounded reference entries were found.")
    return {
        layer: torch.cat(vectors, dim=0).to(device=device, dtype=torch.float32)
        for layer, vectors in layers.items()
    }


def load_pooled_reference_layers(
    pooled_bank_root: Path,
    model_name: str,
    selected_layers: Sequence[int],
    *,
    device: torch.device,
) -> dict[int, torch.Tensor]:
    reference_layers: dict[int, torch.Tensor] = {}
    for layer in sorted({int(layer) for layer in selected_layers}):
        path = pooled_bank_root / model_name / f"layer-{layer:02d}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing pooled reference bank layer: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
            raise ValueError(f"{path} must contain a rank-2 tensor.")
        reference_layers[layer] = tensor.to(device=device, dtype=torch.float32)
    return reference_layers


def feature_columns(frame: pd.DataFrame) -> list[str]:
    raw_columns = sorted(
        [column for column in frame.columns if str(column).startswith("raw_drift_")],
        key=lambda column: int(str(column).rsplit("_", 1)[1]),
    )
    return [*raw_columns, *[column for column in CALIBRATED_SUMMARY_COLUMNS if column in frame.columns]]


def build_method_feature_frame(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_layers: dict[int, torch.Tensor],
    method: str,
    device: torch.device,
    k: int = 30,
    target_count: int = 30,
    reference_chunk_size: int = 16_384,
    limit_rows: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    entries = list(cache_entries[:limit_rows] if limit_rows is not None else cache_entries)
    layer_radii = None
    with torch.no_grad():
        if method == "radius_ball":
            layer_radii = tune_radius_ball_layer_radii_gpu(
                cache_entries=entries,
                reference_layers=reference_layers,
                device=device,
                target_count=target_count,
                reference_chunk_size=reference_chunk_size,
            )
        for entry in entries:
            selected_layers = [int(layer) for layer in entry["selected_layers"]]
            layer_vectors = torch.as_tensor(entry["layer_vectors"], dtype=torch.float32, device=device)
            row = metadata_row_from_entry(dict(entry))
            row.update(
                compute_neighbor_feature_row_gpu(
                    layer_vectors=layer_vectors,
                    selected_layers=selected_layers,
                    reference_layers=reference_layers,
                    method=method,
                    k=k,
                    target_count=target_count,
                    layer_radii=layer_radii,
                    reference_chunk_size=reference_chunk_size,
                )
            )
            rows.append(row)
    return pd.DataFrame(rows)


def tune_radius_ball_layer_radii_gpu(
    *,
    cache_entries: Sequence[dict[str, object]],
    reference_layers: dict[int, torch.Tensor],
    device: torch.device,
    target_count: int,
    reference_chunk_size: int,
) -> dict[int, torch.Tensor]:
    queries_by_layer: dict[int, list[torch.Tensor]] = {}
    for entry in cache_entries:
        selected_layers = [int(layer) for layer in entry["selected_layers"]]
        layer_vectors = torch.as_tensor(entry["layer_vectors"], dtype=torch.float32, device=device)
        if len(selected_layers) != int(layer_vectors.shape[0]):
            raise ValueError("selected_layers must align with layer_vectors rows")
        for offset, layer in enumerate(selected_layers):
            queries_by_layer.setdefault(layer, []).append(layer_vectors[offset : offset + 1])
    radii: dict[int, torch.Tensor] = {}
    for layer, queries in queries_by_layer.items():
        if layer not in reference_layers:
            raise KeyError(f"Missing reference layer: {layer}")
        radii[layer] = tune_radius_for_target_count_gpu(
            torch.cat(queries, dim=0),
            reference_layers[layer].to(device=device, dtype=torch.float32),
            target_count=target_count,
            reference_chunk_size=reference_chunk_size,
        ).detach()
    return radii


def _load_gpu_detector_module():
    path = Path(__file__).resolve().parent / "train_gpu_detector.py"
    spec = importlib.util.spec_from_file_location("train_gpu_detector", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluate_method_frame_gpu(
    frame: pd.DataFrame,
    *,
    device: torch.device,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
) -> dict[str, float]:
    detector = _load_gpu_detector_module()
    columns = feature_columns(frame)
    metrics, _predictions, _states = detector.evaluate_frame(
        frame,
        columns=columns,
        split_strategy="image_grouped",
        device=device,
        random_state=random_state,
        num_folds=num_folds,
        max_iter=max_iter,
        bootstrap_resamples=bootstrap_resamples,
    )
    return metrics


def run_comparison(
    *,
    cache_path: Path,
    pooled_bank_root: Path | None,
    raw_reference_cache: Path | None,
    model_name: str,
    benchmark: str,
    methods: Sequence[str],
    device: torch.device,
    k: int,
    target_count: int,
    reference_chunk_size: int,
    limit_rows: int | None,
    bootstrap_resamples: int,
    num_folds: int,
    random_state: int,
    max_iter: int,
) -> list[dict[str, object]]:
    from mind.evaluation.baselines import load_cache_entries

    cache_entries = load_cache_entries(
        cache_path,
        keep_fields={"sample_id", "image_id", "label", "parsed_answer", "subset", "object_name", "selected_layers", "layer_vectors"},
    )
    if not cache_entries:
        raise ValueError(f"No eval cache entries found: {cache_path}")
    selected_layers = sorted({int(layer) for entry in cache_entries for layer in entry["selected_layers"]})
    if raw_reference_cache is not None:
        reference_layers = build_reference_layers_from_raw_entries(
            load_cache_entries(raw_reference_cache, keep_fields={"label", "ground_truth_label", "selected_layers", "layer_vectors"}),
            device=device,
        )
    elif pooled_bank_root is not None:
        reference_layers = load_pooled_reference_layers(pooled_bank_root, model_name, selected_layers, device=device)
    else:
        raise ValueError("Provide --pooled-bank-root or --raw-reference-cache.")

    rows: list[dict[str, object]] = []
    for method in methods:
        frame = build_method_feature_frame(
            cache_entries=cache_entries,
            reference_layers=reference_layers,
            method=method,
            device=device,
            k=k,
            target_count=target_count,
            reference_chunk_size=reference_chunk_size,
            limit_rows=limit_rows,
        )
        metrics = evaluate_method_frame_gpu(
            frame,
            device=device,
            bootstrap_resamples=bootstrap_resamples,
            num_folds=num_folds,
            random_state=random_state,
            max_iter=max_iter,
        )
        row: dict[str, object] = {
            "model": model_name,
            "benchmark": benchmark,
            "method": method,
            "status": "ok",
            "n_rows": int(len(frame)),
            "split_strategy": "image_grouped",
        }
        row.update(metrics)
        rows.append(row)
    return rows


def load_metric_rows(path: Path) -> list[dict[str, object]]:
    frame = pd.read_csv(path)
    return frame.where(pd.notna(frame), None).to_dict(orient="records")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Production methods: {', '.join(METHOD_NAMES)}",
    )
    subparsers = parser.add_subparsers(dest="command")

    format_parser = subparsers.add_parser("format", help="Format existing metric rows without reading caches.")
    format_parser.add_argument("--metrics-csv", type=Path, required=True)
    format_parser.add_argument("--csv-path", type=Path, default=OUTPUT_CSV)
    format_parser.add_argument("--markdown-path", type=Path, default=OUTPUT_MARKDOWN)
    format_parser.add_argument("--analysis-path", type=Path, default=OUTPUT_ANALYSIS)

    run_parser = subparsers.add_parser("run", help="Run the CUDA neighbor-selection comparison.")
    run_parser.add_argument("--cache-path", type=Path, required=True)
    run_parser.add_argument("--pooled-bank-root", type=Path, default=None)
    run_parser.add_argument("--raw-reference-cache", type=Path, default=None)
    run_parser.add_argument("--model-name", required=True)
    run_parser.add_argument("--benchmark", required=True)
    run_parser.add_argument("--methods", default="all", help=f"Comma-separated methods or 'all': {', '.join(METHOD_NAMES)}")
    run_parser.add_argument("--device", default="cuda:0")
    run_parser.add_argument("--k", type=int, default=30)
    run_parser.add_argument("--target-count", type=int, default=30)
    run_parser.add_argument("--reference-chunk-size", type=int, default=16_384)
    run_parser.add_argument("--limit-rows", type=int, default=None)
    run_parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    run_parser.add_argument("--num-folds", type=int, default=5)
    run_parser.add_argument("--random-state", type=int, default=13)
    run_parser.add_argument("--max-iter", type=int, default=100)
    run_parser.add_argument("--csv-path", type=Path, default=OUTPUT_CSV)
    run_parser.add_argument("--markdown-path", type=Path, default=OUTPUT_MARKDOWN)
    run_parser.add_argument("--analysis-path", type=Path, default=OUTPUT_ANALYSIS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command is None:
        build_parser().print_help()
        return 0
    try:
        if args.command == "format":
            rows = load_metric_rows(args.metrics_csv)
        elif args.command == "run":
            validate_cuda_visible_devices()
            device = resolve_cuda_device(args.device)
            rows = run_comparison(
                cache_path=args.cache_path,
                pooled_bank_root=args.pooled_bank_root,
                raw_reference_cache=args.raw_reference_cache,
                model_name=args.model_name,
                benchmark=args.benchmark,
                methods=parse_methods(args.methods),
                device=device,
                k=args.k,
                target_count=args.target_count,
                reference_chunk_size=args.reference_chunk_size,
                limit_rows=args.limit_rows,
                bootstrap_resamples=args.bootstrap_resamples,
                num_folds=args.num_folds,
                random_state=args.random_state,
                max_iter=args.max_iter,
            )
        else:
            raise ValueError(f"Unsupported command: {args.command}")

        with output_root_lock(args.csv_path.parent, command="neighbor_selection_comparison"):
            write_neighbor_comparison_outputs(
                rows,
                csv_path=args.csv_path,
                markdown_path=args.markdown_path,
                analysis_path=args.analysis_path,
            )
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    print(json.dumps({"rows": len(rows), "csv_path": str(args.csv_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
