#!/usr/bin/env python3
"""Prepare decisive-round layer-count sensitivity table."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPORTS_ROOT = Path("/home/team/zhanghaonan/mind/outputs/decisive_round_2026_04/layer_scan/reports")
DEFAULT_OUTPUT_PATH = Path("docs/tables/decisive_layer_scan.md")
DEFAULT_MODELS = ("qwen3-vl-8b", "molmo-7b-d-0924")
DEFAULT_MODEL_COUNTS = {
    "qwen3-vl-8b": (8, 12, 16),
    "molmo-7b-d-0924": (8, 12),
}
DEFAULT_BANK_SCOPE = "object"
SETTINGS = ("popular-object-heldout", "dash-b")
METHODS = ("full", "linear_probe", "no_manifold")
METRIC_KEY_ALIASES = {
    "pr_auc": "pr_auc",
    "prauc": "pr_auc",
    "roc_auc": "roc_auc",
    "rocauc": "roc_auc",
}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_report_name(*, model: str, setting: str, layer_count: int, bank_scope: str = DEFAULT_BANK_SCOPE) -> str:
    if setting not in SETTINGS:
        raise ValueError(f"Unsupported setting: {setting}")
    return f"layer-scan-{model}-{setting}-lc{layer_count}-{bank_scope}"


def normalize_metric_key(key: object) -> str:
    return str(key).strip().lower().replace("-", "_").replace(" ", "_")


def metric_pair_from_mapping(payload: dict[object, object]) -> tuple[float, float] | None:
    metrics: dict[str, float] = {}
    for key, value in payload.items():
        metric_name = METRIC_KEY_ALIASES.get(normalize_metric_key(key))
        if metric_name is None:
            continue
        metrics[metric_name] = float(value)
    if "pr_auc" in metrics and "roc_auc" in metrics:
        return metrics["pr_auc"], metrics["roc_auc"]
    return None


def find_metric_pair(payload: Any) -> tuple[float, float] | None:
    if isinstance(payload, dict):
        direct = metric_pair_from_mapping(payload)
        if direct is not None:
            return direct
        for key in ("metrics", "summary", "results", "scores"):
            found = find_metric_pair(payload.get(key))
            if found is not None:
                return found
        for nested in payload.values():
            found = find_metric_pair(nested)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = find_metric_pair(item)
            if found is not None:
                return found
    return None


def collect_method_payloads(payload: Any, *, method: str) -> list[Any]:
    candidates: list[Any] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) == method:
                candidates.append(value)
        if str(payload.get("variant", "")).strip() == method or str(payload.get("method", "")).strip() == method:
            candidates.append(payload)
        for value in payload.values():
            candidates.extend(collect_method_payloads(value, method=method))
    elif isinstance(payload, list):
        for item in payload:
            candidates.extend(collect_method_payloads(item, method=method))
    return candidates


def extract_method_metrics(
    baselines: dict[str, object],
    *,
    method: str,
    source_path: Path,
) -> tuple[float, float] | None:
    for candidate in collect_method_payloads(baselines, method=method):
        metrics = find_metric_pair(candidate)
        if metrics is not None:
            return metrics
    if method == "full":
        raise ValueError(f"Missing full PR-AUC/ROC-AUC metrics in {source_path}")
    return None


def parse_model_counts(raw: str | None) -> dict[str, tuple[int, ...]]:
    if raw is None:
        return dict(DEFAULT_MODEL_COUNTS)
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--model-counts must be a JSON object of model names to layer-count lists")
    parsed: dict[str, tuple[int, ...]] = {}
    for model, counts in payload.items():
        if not isinstance(counts, list):
            raise ValueError(f"Layer counts for {model} must be a JSON list")
        parsed[str(model)] = tuple(int(count) for count in counts)
    return parsed


def format_float(value: float) -> str:
    return f"{value:.4f}"


def mark_best_layers(rows: list[dict[str, object]]) -> None:
    rows_by_group: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["setting"]), str(row["model"]), str(row["method"]))
        rows_by_group[key].append(row)

    for group_rows in rows_by_group.values():
        best_pr_auc = max(float(row["pr_auc"]) for row in group_rows)
        for row in group_rows:
            is_best = math.isclose(float(row["pr_auc"]), best_pr_auc, rel_tol=0.0, abs_tol=1e-12)
            row["best_layer_for_method"] = "yes" if is_best else "no"


def build_rows(
    *,
    reports_root: Path,
    models: tuple[str, ...],
    model_counts: dict[str, tuple[int, ...]],
    settings: tuple[str, ...],
    bank_scope: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        for setting in settings:
            for layer_count in model_counts[model]:
                report_name = expected_report_name(
                    model=model,
                    setting=setting,
                    layer_count=layer_count,
                    bank_scope=bank_scope,
                )
                baselines_path = reports_root / report_name / "baselines.json"
                if not baselines_path.exists():
                    raise FileNotFoundError(f"Missing expected layer-scan report: {baselines_path}")
                baselines = load_json(baselines_path)
                for method in METHODS:
                    metrics = extract_method_metrics(baselines, method=method, source_path=baselines_path)
                    if metrics is None:
                        continue
                    pr_auc, roc_auc = metrics
                    rows.append(
                        {
                            "setting": setting,
                            "model": model,
                            "layer_count": layer_count,
                            "bank_scope": bank_scope,
                            "method": method,
                            "pr_auc": pr_auc,
                            "roc_auc": roc_auc,
                        }
                    )
    mark_best_layers(rows)
    return rows


def interpretation_lines(
    rows: list[dict[str, object]],
    *,
    models: tuple[str, ...],
    settings: tuple[str, ...],
) -> list[str]:
    rows_by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["method"] == "full":
            rows_by_group[(str(row["model"]), str(row["setting"]))].append(row)

    lines: list[str] = []
    for model in models:
        for setting in settings:
            group_rows = rows_by_group.get((model, setting), [])
            if not group_rows:
                continue
            best_rows = [row for row in group_rows if row["best_layer_for_method"] == "yes"]
            best_layers = ", ".join(
                str(row["layer_count"])
                for row in sorted(best_rows, key=lambda row: int(row["layer_count"]))
            )
            evaluated_layers = {int(row["layer_count"]) for row in group_rows}
            best_layer_set = {int(row["layer_count"]) for row in best_rows}
            if 16 not in evaluated_layers:
                default_status = "16-layer default was not evaluated"
            elif 16 in best_layer_set:
                default_status = "16-layer default is best"
            else:
                default_status = "16-layer default is present but not best"
            lines.append(
                f"- {model} / {setting}: full MIND best at {best_layers} layers by PR-AUC; {default_status}."
            )
    if not lines:
        lines.append("- No complete full MIND layer-scan rows were available.")
    return lines


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    *,
    models: tuple[str, ...],
    settings: tuple[str, ...],
) -> None:
    columns = (
        "setting",
        "model",
        "layer_count",
        "bank_scope",
        "method",
        "pr_auc",
        "roc_auc",
        "best_layer_for_method",
    )
    lines = [
        "# Decisive Layer-Count Sensitivity",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["setting"]),
                    str(row["model"]),
                    str(row["layer_count"]),
                    str(row["bank_scope"]),
                    str(row["method"]),
                    format_float(float(row["pr_auc"])),
                    format_float(float(row["roc_auc"])),
                    str(row["best_layer_for_method"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Generated Interpretation", "", *interpretation_lines(rows, models=models, settings=settings)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_layer_scan_report(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    models: tuple[str, ...] = DEFAULT_MODELS,
    model_counts: dict[str, tuple[int, ...]] | None = None,
    settings: tuple[str, ...] = SETTINGS,
    bank_scope: str = DEFAULT_BANK_SCOPE,
) -> list[dict[str, object]]:
    counts = dict(DEFAULT_MODEL_COUNTS if model_counts is None else model_counts)
    missing_models = [model for model in models if model not in counts]
    if missing_models:
        raise ValueError(f"Missing layer counts for models: {', '.join(missing_models)}")
    rows = build_rows(
        reports_root=Path(reports_root),
        models=models,
        model_counts=counts,
        settings=settings,
        bank_scope=bank_scope,
    )
    write_markdown(Path(output_path), rows, models=models, settings=settings)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--bank-scope", default=DEFAULT_BANK_SCOPE)
    parser.add_argument("--model-counts", help="JSON object mapping model names to layer-count lists")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_layer_scan_report(
        reports_root=args.reports_root,
        output_path=args.output_path,
        bank_scope=args.bank_scope,
        model_counts=parse_model_counts(args.model_counts),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
