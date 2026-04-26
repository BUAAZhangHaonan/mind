#!/usr/bin/env python3
"""Prepare decisive-round bank-identity control table."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_REPORTS_ROOT = Path("/home/team/zhanghaonan/mind/outputs/decisive_round_2026_04/reports")
DEFAULT_OUTPUT_PATH = Path("docs/tables/decisive_bank_identity.md")
DEFAULT_MODELS = ("qwen3-vl-8b", "molmo-7b-d-0924")
DEFAULT_BANK_SCOPES = ("object", "shared", "shuffled_object")
SETTINGS = ("popular-object-heldout", "dash-b")
METRIC_KEY_ALIASES = {
    "pr_auc": "pr_auc",
    "prauc": "pr_auc",
    "roc_auc": "roc_auc",
    "rocauc": "roc_auc",
}


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def expected_report_name(*, model: str, setting: str, bank_scope: str) -> str:
    if setting == "popular-object-heldout":
        return f"bankid-{model}-popular-{bank_scope}-object-heldout"
    if setting == "dash-b":
        return f"bankid-{model}-dash-b-{bank_scope}"
    raise ValueError(f"Unsupported setting: {setting}")


def expected_report_names(
    *,
    models: tuple[str, ...] = DEFAULT_MODELS,
    bank_scopes: tuple[str, ...] = DEFAULT_BANK_SCOPES,
) -> list[str]:
    names: list[str] = []
    for model in models:
        for setting in SETTINGS:
            for bank_scope in bank_scopes:
                names.append(expected_report_name(model=model, setting=setting, bank_scope=bank_scope))
    return names


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
            nested = payload.get(key)
            found = find_metric_pair(nested)
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


def collect_full_payloads(payload: Any) -> list[Any]:
    candidates: list[Any] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if str(key) == "full":
                candidates.append(value)
        if str(payload.get("variant", "")).strip() == "full":
            candidates.append(payload)
        for value in payload.values():
            candidates.extend(collect_full_payloads(value))
    elif isinstance(payload, list):
        for item in payload:
            candidates.extend(collect_full_payloads(item))
    return candidates


def extract_full_metrics(baselines: dict[str, object], *, source_path: Path) -> tuple[float, float]:
    for candidate in collect_full_payloads(baselines):
        metrics = find_metric_pair(candidate)
        if metrics is not None:
            return metrics
    raise ValueError(f"Missing full PR-AUC/ROC-AUC metrics in {source_path}")


def format_float(value: float) -> str:
    return f"{value:.4f}"


def rank_rows_by_pr_auc(rows: list[dict[str, object]]) -> None:
    rows_by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_group[(str(row["model"]), str(row["setting"]))].append(row)

    for group_rows in rows_by_group.values():
        for row in group_rows:
            row_pr_auc = float(row["pr_auc"])
            greater_values: list[float] = []
            for other in group_rows:
                other_pr_auc = float(other["pr_auc"])
                if other_pr_auc <= row_pr_auc or math.isclose(other_pr_auc, row_pr_auc, rel_tol=0.0, abs_tol=1e-12):
                    continue
                if not any(math.isclose(other_pr_auc, value, rel_tol=0.0, abs_tol=1e-12) for value in greater_values):
                    greater_values.append(other_pr_auc)
            row["rank_by_pr_auc"] = len(greater_values) + 1


def build_rows(
    *,
    reports_root: Path,
    models: tuple[str, ...],
    bank_scopes: tuple[str, ...],
    allow_missing: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        for setting in SETTINGS:
            for bank_scope in bank_scopes:
                report_name = expected_report_name(model=model, setting=setting, bank_scope=bank_scope)
                baselines_path = reports_root / report_name / "baselines.json"
                if not baselines_path.exists():
                    if allow_missing:
                        continue
                    raise FileNotFoundError(f"Missing expected bank-identity report: {baselines_path}")
                try:
                    pr_auc, roc_auc = extract_full_metrics(load_json(baselines_path), source_path=baselines_path)
                except ValueError:
                    if allow_missing:
                        continue
                    raise
                rows.append(
                    {
                        "setting": setting,
                        "model": model,
                        "bank_scope": bank_scope,
                        "pr_auc": pr_auc,
                        "roc_auc": roc_auc,
                    }
                )
    rank_rows_by_pr_auc(rows)
    return rows


def interpretation_lines(rows: list[dict[str, object]], *, models: tuple[str, ...]) -> list[str]:
    rows_by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_group[(str(row["model"]), str(row["setting"]))].append(row)

    lines: list[str] = []
    for model in models:
        for setting in SETTINGS:
            group_rows = rows_by_group.get((model, setting), [])
            object_rows = [row for row in group_rows if row["bank_scope"] == "object"]
            if not object_rows:
                continue
            object_rank = int(object_rows[0]["rank_by_pr_auc"])
            first_rows = [row for row in group_rows if int(row["rank_by_pr_auc"]) == 1]
            if object_rank != 1:
                result = "did not rank first"
            elif len(first_rows) > 1:
                result = "tied for first"
            else:
                result = "ranked first"
            lines.append(f"- {model} / {setting}: object-conditioned {result} by PR-AUC.")
    if not lines:
        lines.append("- No complete object-conditioned bank-identity rows were available.")
    return lines


def write_markdown(path: Path, rows: list[dict[str, object]], *, models: tuple[str, ...]) -> None:
    columns = ("setting", "model", "bank_scope", "pr_auc", "roc_auc", "rank_by_pr_auc")
    lines = [
        "# Decisive Bank-Identity Controls",
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
                    str(row["bank_scope"]),
                    format_float(float(row["pr_auc"])),
                    format_float(float(row["roc_auc"])),
                    str(row["rank_by_pr_auc"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Generated Interpretation", "", *interpretation_lines(rows, models=models)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_bank_identity_report(
    *,
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    models: tuple[str, ...] = DEFAULT_MODELS,
    bank_scopes: tuple[str, ...] = DEFAULT_BANK_SCOPES,
    allow_missing: bool = False,
) -> list[dict[str, object]]:
    rows = build_rows(
        reports_root=Path(reports_root),
        models=models,
        bank_scopes=bank_scopes,
        allow_missing=allow_missing,
    )
    write_markdown(Path(output_path), rows, models=models)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", type=Path, default=DEFAULT_REPORTS_ROOT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--allow-missing", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare_bank_identity_report(
        reports_root=args.reports_root,
        output_path=args.output_path,
        allow_missing=args.allow_missing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
