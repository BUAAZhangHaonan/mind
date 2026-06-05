"""Reporting helpers for wavelet-course diagnostics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping, Sequence


METRIC_FIELDS = [
    "config_name",
    "method_family",
    "model_name",
    "dataset_name",
    "subset_scope",
    "train_samples",
    "val_samples",
    "test_samples",
    "train_pos",
    "val_pos",
    "test_pos",
    "feature_shape",
    "classifier",
    "wavelet",
    "wavelet_level",
    "transform",
    "feature_seconds",
    "train_eval_seconds",
    "total_seconds",
    "pr_auc",
    "average_precision",
    "roc_auc",
    "best_val_threshold",
    "test_f1",
    "test_precision",
    "test_recall",
    "balanced_accuracy",
    "tpr_at_1pct_fpr",
    "fpr_at_95pct_tpr",
    "status",
    "failure_reason",
]


BEST_FIELDS = [
    "rank_name",
    *METRIC_FIELDS,
]


def write_json(payload: Mapping[str, object], output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def write_metrics_csv(rows: Sequence[Mapping[str, object]], output: Path | str) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in METRIC_FIELDS})
    return path


def write_best_configs_csv(rows: Sequence[Mapping[str, object]], output: Path | str) -> Path:
    best_rows = best_config_rows(rows)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BEST_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in best_rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in BEST_FIELDS})
    return path


def best_config_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    successful = [dict(row) for row in rows if str(row.get("status", "")) == "success"]
    best_rows: list[dict[str, object]] = []
    for families, rank_name in (
        ({"teacher_bagua"}, "best_teacher_bagua"),
        ({"ours_wavelet"}, "best_ours_wavelet"),
        ({"halp_like_baseline", "mind_baseline"}, "best_baseline"),
    ):
        winner = _best_by_pr_auc(
            [row for row in successful if str(row.get("method_family", "")) in families]
        )
        if winner is not None:
            best_rows.append({"rank_name": rank_name, **winner})
    overall = _best_by_pr_auc(successful)
    if overall is not None:
        best_rows.append({"rank_name": "overall_best", **overall})
    return best_rows


def write_summary_md(
    *,
    output: Path | str,
    config: Mapping[str, object],
    cache_audit: Mapping[str, object],
    population_summary: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
    best_rows: Sequence[Mapping[str, object]],
    metrics_path: Path | str,
    best_configs_path: Path | str,
    quick: bool,
    failures: Sequence[Mapping[str, object]],
) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_name = str(config.get("model_name", ""))
    dataset_name = str(config.get("dataset_name", ""))
    lines = [
        "# Wavelet Course Summary",
        "",
        f"- model: {model_name}",
        f"- dataset: {dataset_name}",
        f"- quick_run: {str(bool(quick)).lower()}",
        f"- ours_token_id_source: {dict(config.get('ours_wavelet', {}) or {}).get('token_id_source', '')}",
        f"- cache_accepted: {str(bool(cache_audit.get('accepted', False))).lower()}",
        f"- primary_population: {population_summary.get('num_primary_population', 0)}",
        f"- hard_hallucinations: {population_summary.get('num_hard_hallucination', 0)}",
        f"- metrics_csv: {metrics_path}",
        f"- best_configs_csv: {best_configs_path}",
        "",
        "## Experiment Completion",
        "",
        *_experiment_completion_lines(
            quick=quick,
            cache_audit=cache_audit,
            metrics_rows=metrics_rows,
            failures=failures,
        ),
        "",
        "## Configuration Details",
        "",
        *_configuration_detail_lines(config, metrics_rows),
        "",
        "## Full Metrics",
        "",
        *_full_metrics_lines(metrics_rows),
        "",
        "## Course Narrative",
        "",
        "Traditional industrial fault detection templates assume two things. They assume the signal is a fixed sensor trace with stable axes, and they assume wavelet scales map cleanly onto local time-frequency events. VLM hidden dimensions do not satisfy those assumptions, because one hidden coordinate is not a stable physical channel and its order is not a sensor axis.",
        "",
        "The 36 transformer layers are the ordered computation depth here. Teacher-Bagua is intentionally strict transfer: it asks whether a standard temporal readout trained on layer trajectories can move into this hallucination setting without changing the problem to fit it.",
        "",
        "Ours puts the wavelet transform on layer-wise semantic traces. That keeps the ordered axis tied to computation depth instead of pretending hidden dimensions are physical sensors.",
        "",
        "## Results",
        "",
    ]
    if best_rows:
        for row in best_rows:
            lines.append(
                "- {rank}: {name} PR-AUC={pr_auc} F1={f1}".format(
                    rank=row.get("rank_name", ""),
                    name=row.get("config_name", ""),
                    pr_auc=row.get("pr_auc", ""),
                    f1=row.get("test_f1", ""),
                )
            )
    else:
        lines.append("- No successful configuration produced a scored test row.")
    lines.extend(["", "## Timing", ""])
    timing_lines = _timing_summary_lines(metrics_rows)
    if timing_lines:
        lines.extend(timing_lines)
    else:
        lines.append("- No timing rows recorded.")
    lines.extend(["", "## Failures", ""])
    if failures:
        for row in failures:
            lines.append(f"- {row.get('config_name', '')}: {row.get('failure_reason', '')}")
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            _conclusion(metrics_rows, failures),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _experiment_completion_lines(
    *,
    quick: bool,
    cache_audit: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> list[str]:
    success_count = sum(1 for row in metrics_rows if str(row.get("status", "")) == "success")
    failure_count = sum(1 for row in metrics_rows if str(row.get("status", "")) == "failure")
    if not failure_count and failures:
        failure_count = len(failures)
    return [
        f"- full_run: {str(not bool(quick)).lower()}",
        f"- quick_run: {str(bool(quick)).lower()}",
        f"- cache_accepted: {str(bool(cache_audit.get('accepted', False))).lower()}",
        f"- metrics_rows: {len(metrics_rows)}",
        f"- success_count: {success_count}",
        f"- failure_count: {failure_count}",
    ]


def _configuration_detail_lines(
    config: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    lines: list[str] = []
    lines.extend(_teacher_configuration_lines(config, metrics_rows))
    lines.append("")
    lines.extend(_ours_configuration_lines(config, metrics_rows))
    lines.append("")
    lines.extend(_baseline_configuration_lines(config, metrics_rows))
    return lines


def _teacher_configuration_lines(
    config: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    teacher = dict(config.get("teacher_bagua", {}) or {})
    classifier_cfg = dict(config.get("classifiers", {}) or {})
    lstm_cfg = dict(classifier_cfg.get("teacher_lstm", {}) or {})
    teacher_rows = [row for row in metrics_rows if str(row.get("method_family", "")) == "teacher_bagua"]
    sequence_len, input_dim = _teacher_input_shape(config, teacher, teacher_rows)
    hidden_dim = teacher.get("lstm_hidden_dim", lstm_cfg.get("hidden_dim", ""))
    epochs = lstm_cfg.get("epochs", teacher.get("epochs", ""))
    batch_size = lstm_cfg.get("batch_size", teacher.get("batch_size", ""))
    learning_rate = lstm_cfg.get("learning_rate", teacher.get("learning_rate", ""))
    patience = lstm_cfg.get("patience", teacher.get("patience", 3))
    lines = ["### Teacher-Bagua"]
    for item in _config_items(teacher.get("configs", [])):
        name = _config_name(item)
        lines.append(
            "- {name}: wavelet={wavelet} level={level} threshold={threshold}".format(
                name=name,
                wavelet=item.get("wavelet", ""),
                level=item.get("level", ""),
                threshold=item.get("threshold", ""),
            )
        )
    lines.append(
        "- LSTM hidden_dim={hidden_dim} epochs={epochs} batch_size={batch_size} "
        "lr={learning_rate} patience={patience} input_shape={sequence_len}x{input_dim}".format(
            hidden_dim=hidden_dim,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            patience=patience,
            sequence_len=sequence_len,
            input_dim=input_dim,
        )
    )
    return lines


def _ours_configuration_lines(
    config: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    ours = dict(config.get("ours_wavelet", {}) or {})
    ours_rows = [row for row in metrics_rows if str(row.get("method_family", "")) == "ours_wavelet"]
    classifier_variants = _classifier_variants(ours_rows, default=("logreg", "xgb"))
    trace_names = [str(item) for item in ours.get("trace_names", [])]
    final_broadcast = "yes" if {"yes_no_margin_trace", "yes_no_entropy_trace"} <= set(trace_names) else "no"
    token_source = ours.get("token_id_source", "")
    lines = ["### Ours-Wavelet"]
    for item in _config_items(ours.get("configs", [])):
        name = _config_name(item)
        transform = str(item.get("transform", ""))
        level_label = "SWT level" if transform.lower() == "swt" else "level"
        lines.append(
            "- {name}: transform={transform} wavelet={wavelet} {level_label}={level}".format(
                name=name,
                transform=transform,
                wavelet=item.get("wavelet", ""),
                level_label=level_label,
                level=item.get("level", ""),
            )
        )
    lines.append(f"- classifier_variants={','.join(classifier_variants)}")
    lines.append(f"- trace_list={', '.join(trace_names)}")
    lines.append(f"- final_broadcast={final_broadcast} token_source={token_source}")
    return lines


def _baseline_configuration_lines(
    config: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    classifier_cfg = dict(config.get("classifiers", {}) or {})
    logreg_cfg = dict(classifier_cfg.get("logreg", {}) or {})
    baseline_names = _baseline_names(config, metrics_rows)
    lines = ["### HALP-like/MIND Baselines"]
    for name in baseline_names:
        lines.append(
            "- {name}: feature={feature} classifier=logreg".format(
                name=name,
                feature=_baseline_feature_definition(name),
            )
        )
    lines.append(
        "- logreg=max_iter={max_iter} class_weight={class_weight}".format(
            max_iter=logreg_cfg.get("max_iter", ""),
            class_weight=logreg_cfg.get("class_weight", ""),
        )
    )
    return lines


def _full_metrics_lines(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        return ["- No metrics rows recorded."]
    lines = [
        "| family | config | status | PR-AUC | F1 | feature_seconds | train_eval_seconds | total_seconds | failure_reason |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in sorted(rows, key=_metrics_sort_key):
        lines.append(
            "| {family} | {config_name} | {status} | {pr_auc} | {f1} | {feature_seconds} | "
            "{train_eval_seconds} | {total_seconds} | {failure_reason} |".format(
                family=_md_cell(row.get("method_family", "")),
                config_name=_md_cell(row.get("config_name", "")),
                status=_md_cell(row.get("status", "")),
                pr_auc=_md_cell(row.get("pr_auc", "")),
                f1=_md_cell(row.get("test_f1", "")),
                feature_seconds=_md_cell(row.get("feature_seconds", "")),
                train_eval_seconds=_md_cell(row.get("train_eval_seconds", "")),
                total_seconds=_md_cell(row.get("total_seconds", "")),
                failure_reason=_md_cell(row.get("failure_reason", "")),
            )
        )
    return lines


def _teacher_input_shape(
    config: Mapping[str, object],
    teacher: Mapping[str, object],
    teacher_rows: Sequence[Mapping[str, object]],
) -> tuple[object, object]:
    for row in teacher_rows:
        shape = _parse_shape(row.get("feature_shape"))
        if len(shape) >= 3:
            return shape[1], shape[2]
    num_layers = int(config.get("expected_num_layers", 36) or 36)
    hidden_dim = int(config.get("expected_hidden_dim", 4096) or 4096)
    window_size = int(teacher.get("window_size", 4) or 4)
    stride = int(teacher.get("stride", 4) or 4)
    features_per_window = int(teacher.get("num_features_per_window", 28) or 28)
    sequence_len = 0 if num_layers < window_size else ((num_layers - window_size) // stride) + 1
    return sequence_len, hidden_dim * features_per_window


def _parse_shape(value: object) -> list[int]:
    parts = str(value or "").split("x")
    shape: list[int] = []
    for part in parts:
        if not part.isdigit():
            return []
        shape.append(int(part))
    return shape


def _config_items(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _config_name(item: Mapping[str, object]) -> str:
    return str(item.get("config_name") or item.get("name") or "")


def _classifier_variants(
    rows: Sequence[Mapping[str, object]],
    *,
    default: Sequence[str],
) -> list[str]:
    observed = {str(row.get("classifier", "")) for row in rows if str(row.get("classifier", ""))}
    order = [name for name in default if name in observed or not observed]
    extras = sorted(observed.difference(order))
    return order + extras


def _baseline_names(
    config: Mapping[str, object],
    metrics_rows: Sequence[Mapping[str, object]],
) -> list[str]:
    configured = config.get("baselines", [])
    names: list[str] = []
    if isinstance(configured, list):
        for item in configured:
            if isinstance(item, str):
                names.append(item)
            elif isinstance(item, Mapping):
                names.append(_config_name(item))
    elif isinstance(configured, Mapping):
        for item in _config_items(configured.get("configs", [])):
            names.append(_config_name(item))
    for row in metrics_rows:
        if str(row.get("method_family", "")) not in {"mind_baseline", "halp_like_baseline"}:
            continue
        name = str(row.get("config_name", ""))
        if name and name not in names:
            names.append(name)
    return names


def _baseline_feature_definition(name: str) -> str:
    definitions = {
        "final_hidden_logreg": "final-layer hidden vector",
        "mean_layer_hidden_logreg": "mean-pooled hidden vector across layers",
        "norm_traj_logreg": "36-point hidden-norm trajectory",
        "sphere_traj_meanpool_logreg": "mean-pooled unit-sphere layer trajectory",
    }
    return definitions.get(name, name)


def _metrics_sort_key(row: Mapping[str, object]) -> tuple[int, str, str]:
    family_order = {
        "teacher_bagua": 0,
        "ours_wavelet": 1,
        "halp_like_baseline": 2,
        "mind_baseline": 3,
    }
    family = str(row.get("method_family", ""))
    return family_order.get(family, 99), family, str(row.get("config_name", ""))


def _md_cell(value: object) -> str:
    if value is None:
        return ""
    return str(value).replace("|", "\\|")


def _best_by_pr_auc(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    scored = [(float(row.get("pr_auc", "nan")), dict(row)) for row in rows if _is_number(row.get("pr_auc"))]
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[0][1]


def _is_number(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return number == number


def _csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def _timing_summary_lines(rows: Sequence[Mapping[str, object]]) -> list[str]:
    if not rows:
        return []
    families = sorted({str(row.get("method_family", "")) for row in rows if str(row.get("method_family", ""))})
    lines: list[str] = []
    totals_by_family: dict[str, dict[str, float | int]] = {}
    for family in families:
        family_rows = [row for row in rows if str(row.get("method_family", "")) == family]
        feature_total = _sum_numeric(row.get("feature_seconds") for row in family_rows)
        train_total = _sum_numeric(row.get("train_eval_seconds") for row in family_rows)
        total_seconds = _sum_numeric(row.get("total_seconds") for row in family_rows)
        total_count = sum(1 for row in family_rows if _is_number(row.get("total_seconds")))
        totals_by_family[family] = {
            "total": total_seconds,
            "count": total_count,
        }
        lines.append(
            "- {family}: configs={configs} feature_seconds={feature:.6f} "
            "train_eval_seconds={train:.6f} total_seconds={total:.6f}".format(
                family=family,
                configs=len(family_rows),
                feature=feature_total,
                train=train_total,
                total=total_seconds,
            )
        )

    teacher = totals_by_family.get("teacher_bagua")
    ours = totals_by_family.get("ours_wavelet")
    if teacher is not None or ours is not None:
        lines.append("")
    if teacher is not None:
        lines.append(
            "- Teacher-Bagua total_seconds={total:.6f} avg_total_seconds={average:.6f}".format(
                total=float(teacher["total"]),
                average=_average(float(teacher["total"]), int(teacher["count"])),
            )
        )
    if ours is not None:
        lines.append(
            "- Ours-Wavelet total_seconds={total:.6f} avg_total_seconds={average:.6f}".format(
                total=float(ours["total"]),
                average=_average(float(ours["total"]), int(ours["count"])),
            )
        )
    return lines


def _sum_numeric(values: Sequence[object]) -> float:
    total = 0.0
    for value in values:
        if _is_number(value):
            total += float(value)  # type: ignore[arg-type]
    return total


def _average(total: float, count: int) -> float:
    if count <= 0:
        return 0.0
    return total / count


def _conclusion(
    metrics_rows: Sequence[Mapping[str, object]],
    failures: Sequence[Mapping[str, object]],
) -> str:
    best = _best_by_pr_auc(metrics_rows)
    if best is None:
        return "No honest positive conclusion is available yet, because every scored configuration failed or lacked valid metrics."
    failure_text = ""
    if failures:
        failure_text = f" {len(failures)} configurations failed and remain listed in metrics.csv."
    return (
        f"The best observed test PR-AUC is from {best.get('config_name', '')}. "
        "This result should be read as a diagnostic comparison, not as proof that hidden dimensions form sensor-like wavelet channels."
        + failure_text
    )
