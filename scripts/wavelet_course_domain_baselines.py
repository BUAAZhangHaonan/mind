#!/usr/bin/env python3
"""Run official HALP and linear probes on the wavelet-course v2 RePOPE split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import sys
from typing import Any

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

import yaml

from mind.wavelet_course.cache_loading import load_repope_qwen_cache_entries
from mind.wavelet_course.domain_baselines import (
    load_halp_readout_cache,
    merge_halp_readout_cache,
    run_domain_baselines,
)
from mind.wavelet_course.population import build_wavelet_population
from mind.wavelet_course.reporting import write_json
from mind.wavelet_course.utils import DEFAULT_SPLIT_RATIOS


DEFAULT_CONFIG = Path("configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = resolve_config(load_yaml_config(Path(args.config)), args)
    ensure_device_available(str(config["device"]), allow_cpu=bool(config["allow_cpu"]))

    stage0_root = Path(str(config["stage0_root"]))
    output_root = Path(str(config["output_root"]))
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

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
    validate_population(population)

    domain_cfg = dict(config.get("domain_baselines", {}) or {})
    primary_entries = maybe_merge_halp_readout_cache(population.primary_entries, domain_cfg)
    result = run_domain_baselines(
        primary_entries,
        population.labels,
        output_dir=reports_dir,
        model_name=str(config["model_name"]),
        dataset_name=str(config["dataset_name"]),
        subsets=[str(item) for item in config["subsets"]],  # type: ignore[index]
        seed=int(config["seed"]),
        device=str(config["device"]),
        logreg_max_iter=int(domain_cfg.get("logreg_max_iter", 20000 if not config["quick_run"] else 2000)),
        halp_hidden_dims=parse_int_tuple(str(domain_cfg.get("halp_hidden_dims", "512,256,128"))),
        halp_dropout=float(domain_cfg.get("halp_dropout", 0.3)),
        halp_learning_rate=float(domain_cfg.get("halp_learning_rate", 1e-3)),
        halp_batch_size=int(domain_cfg.get("halp_batch_size", 32)),
        halp_epochs=resolve_halp_epochs(domain_cfg, quick=bool(config["quick_run"])),
        quick=bool(config["quick_run"]),
    )

    status = build_status(config, population, result)
    status_path = reports_dir / "domain_baseline_status.json"
    write_json(status, status_path)
    append_domain_section(
        output_root / "reports" / "summary.md",
        domain_csv=Path(result["csv_path"]),
        domain_summary=Path(result["summary_path"]),
        rows=result["rows"],
        paired_metrics_long=reports_dir / "metrics_long.csv",
        halp_cache_path=resolve_halp_cache_path(domain_cfg),
    )
    print_final_summary(status, status_path=status_path)
    return 0


def load_yaml_config(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: config must be a YAML mapping")
    return dict(payload)


def resolve_config(config: Mapping[str, object], args: argparse.Namespace) -> dict[str, object]:
    resolved = json.loads(json.dumps(config))
    resolved.setdefault("seed", 20260506)
    resolved.setdefault("stage0_root", "outputs/stage0")
    resolved.setdefault("output_root", "outputs/wavelet_course_v2")
    resolved.setdefault("model_name", "qwen3-vl-8b")
    resolved.setdefault("dataset_name", "repope")
    resolved.setdefault("subsets", ["popular", "random", "adversarial"])
    resolved.setdefault("expected_num_layers", 36)
    resolved.setdefault("expected_hidden_dim", 4096)
    resolved.setdefault("split_ratios", {"train": 0.60, "validation": 0.20, "test": 0.20})
    resolved["device"] = args.device or str(resolved.get("device", "cuda:0"))
    resolved["allow_cpu"] = bool(args.allow_cpu or resolved.get("allow_cpu", False))
    resolved["quick_run"] = bool(args.quick or resolved.get("quick_run", False))
    return resolved


def split_ratio_values(config: Mapping[str, object]) -> tuple[float, float, float]:
    ratios = config.get("split_ratios", {})
    if isinstance(ratios, Mapping):
        return (
            float(ratios.get("train", DEFAULT_SPLIT_RATIOS[0])),
            float(ratios.get("validation", DEFAULT_SPLIT_RATIOS[1])),
            float(ratios.get("test", DEFAULT_SPLIT_RATIOS[2])),
        )
    values = list(ratios)  # type: ignore[arg-type]
    if len(values) != 3:
        raise ValueError("split_ratios must contain train/validation/test")
    return (float(values[0]), float(values[1]), float(values[2]))


def parse_int_tuple(value: str) -> tuple[int, ...]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if not parts:
        raise ValueError("HALP hidden dims must not be empty")
    return tuple(int(part) for part in parts)


def resolve_halp_epochs(domain_cfg: Mapping[str, object], *, quick: bool) -> int:
    full_epochs = int(domain_cfg.get("halp_epochs", 50))
    if not quick:
        return full_epochs
    return int(domain_cfg.get("halp_quick_epochs", min(5, full_epochs)))


def maybe_merge_halp_readout_cache(
    entries: Sequence[Mapping[str, object]],
    domain_cfg: Mapping[str, object],
) -> list[Mapping[str, object]]:
    cache_value = resolve_halp_cache_path(domain_cfg)
    if cache_value is None or str(cache_value).strip() == "":
        return list(entries)
    cache_path = Path(str(cache_value))
    if not cache_path.exists():
        return list(entries)
    readout_rows = load_halp_readout_cache(cache_path)
    return merge_halp_readout_cache(entries, readout_rows)


def resolve_halp_cache_path(domain_cfg: Mapping[str, object]) -> object | None:
    return (
        domain_cfg.get("halp_readout_cache_path")
        or domain_cfg.get("halp_cache_path")
        or domain_cfg.get("official_halp_readout_cache_path")
    )


def ensure_device_available(device: str, *, allow_cpu: bool) -> None:
    normalized = str(device).strip().lower()
    if normalized.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"requested device {device!r}, but CUDA is not available")
        return
    if normalized == "cpu" and allow_cpu:
        return
    raise RuntimeError(f"device {device!r} is not allowed; pass --allow-cpu for CPU")


def validate_population(population: Any) -> None:
    labels = list(population.labels)
    if not labels:
        raise ValueError("primary population is empty")
    if sorted(set(labels)) != [0, 1]:
        raise ValueError("primary population must contain both classes")
    for split in ("train", "validation", "test"):
        split_labels = [
            int(label)
            for entry, label in zip(population.primary_entries, labels, strict=True)
            if str(entry.get("wavelet_split")) == split
        ]
        if sorted(set(split_labels)) != [0, 1]:
            raise ValueError(f"{split} split must contain both classes")


def build_status(config: Mapping[str, object], population: Any, result: Mapping[str, object]) -> dict[str, object]:
    rows = [dict(row) for row in result["rows"]]  # type: ignore[index]
    successes = [row for row in rows if row.get("status") == "success"]
    failures = [row for row in rows if row.get("status") == "failure"]
    return {
        "status": "success",
        "quick_run": bool(config.get("quick_run", False)),
        "primary_population": len(population.primary_entries),
        "hard_hallucinations": int(sum(population.labels)),
        "success_rows": len(successes),
        "failure_rows": len(failures),
        "domain_baselines_csv": result["csv_path"],
        "domain_baseline_summary": result["summary_path"],
        "best_rows": result["best_rows"],
    }


def append_domain_section(
    summary_path: Path,
    *,
    domain_csv: Path,
    domain_summary: Path,
    rows: Sequence[Mapping[str, object]],
    paired_metrics_long: Path,
    halp_cache_path: object | None = None,
) -> None:
    marker_begin = "<!-- domain-baseline-comparison:start -->"
    marker_end = "<!-- domain-baseline-comparison:end -->"
    existing = summary_path.read_text(encoding="utf-8") if summary_path.exists() else "# Wavelet Course V2 Summary\n"
    before = existing
    if marker_begin in existing and marker_end in existing:
        start = existing.index(marker_begin)
        end = existing.index(marker_end) + len(marker_end)
        before = existing[:start].rstrip() + "\n"
        after = existing[end:].lstrip()
    else:
        after = ""
    section = build_domain_section(
        rows,
        domain_csv=domain_csv,
        domain_summary=domain_summary,
        paired_metrics_long=paired_metrics_long,
        halp_cache_path=halp_cache_path,
        marker_begin=marker_begin,
        marker_end=marker_end,
    )
    summary_path.write_text(before.rstrip() + "\n\n" + section + ("\n\n" + after if after else "\n"), encoding="utf-8")


def build_domain_section(
    rows: Sequence[Mapping[str, object]],
    *,
    domain_csv: Path,
    domain_summary: Path,
    paired_metrics_long: Path,
    halp_cache_path: object | None = None,
    marker_begin: str,
    marker_end: str,
) -> str:
    best = best_by_family(rows)
    paired_best = read_paired_best(paired_metrics_long)
    lines = [
        marker_begin,
        "## Domain Baselines on the Same RePOPE Split",
        "",
        "这些补充结果只写在小波课程 v2 输出目录中。",
        "",
        f"- domain_baselines_csv: {domain_csv}",
        f"- domain_baseline_summary: {domain_summary}",
        f"- official_halp_cache: {halp_cache_path or 'not_configured'}",
        "- official_halp_policy: train on train split, choose probe and threshold on validation split, report test metrics.",
        "- included_domain_methods: official HALP and linear probe only; MIND and HALP-like are not included.",
    ]
    for family in ("halp_official", "linear_probe"):
        row = best.get(family)
        if row is None:
            failures = [item for item in rows if item.get("method_family") == family and item.get("status") == "failure"]
            reason = failures[0].get("failure_reason", "") if failures else "no successful row"
            lines.append(f"- best_{family}: unavailable ({reason})")
        else:
            lines.append(
                "- best_{family}: {name}, PR-AUC={pr_auc}, F1={f1}".format(
                    family=family,
                    name=row.get("baseline_name", ""),
                    pr_auc=_fmt(row.get("test_pr_auc")),
                    f1=_fmt(row.get("test_f1")),
                )
            )
    if paired_best:
        lines.extend(["", "### Current Wavelet V2 Best Rows", ""])
        for family, row in paired_best.items():
            lines.append(
                "- best_{family}: {name}, PR-AUC={pr_auc}, F1={f1}".format(
                    family=family,
                    name=row.get("config_name", ""),
                    pr_auc=_fmt(row.get("test_pr_auc")),
                    f1=_fmt(row.get("test_f1")),
                )
            )
    lines.extend(["", marker_end])
    return "\n".join(lines)


def best_by_family(rows: Sequence[Mapping[str, object]]) -> dict[str, Mapping[str, object]]:
    result: dict[str, Mapping[str, object]] = {}
    for row in rows:
        if row.get("status") != "success":
            continue
        family = str(row.get("method_family", ""))
        current = result.get(family)
        if current is None or _metric(row.get("test_pr_auc")) > _metric(current.get("test_pr_auc")):
            result[family] = row
    return result


def read_paired_best(path: Path) -> dict[str, Mapping[str, object]]:
    if not path.exists():
        return {}
    result: dict[str, Mapping[str, object]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "success":
                continue
            family = str(row.get("method_family", ""))
            if family not in {"teacher_bagua", "ours_wavelet"}:
                continue
            current = result.get(family)
            if current is None or _metric(row.get("test_pr_auc")) > _metric(current.get("test_pr_auc")):
                result[family] = row
    return result


def _metric(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return -float("inf")
    if not math.isfinite(number):
        return -float("inf")
    return number


def _fmt(value: object) -> str:
    number = _metric(value)
    if number == -float("inf"):
        return "undefined"
    return f"{number:.6f}"


def print_final_summary(status: Mapping[str, object], *, status_path: Path) -> None:
    print(f"domain_baseline_status={status.get('status')}")
    print(f"primary_population={status.get('primary_population')}")
    print(f"hard_hallucinations={status.get('hard_hallucinations')}")
    for row in status.get("best_rows", []):  # type: ignore[assignment]
        rank = str(row.get("rank_name", "")).removeprefix("best_")
        print(
            "best_{rank}={name} pr_auc={pr_auc} f1={f1}".format(
                rank=rank,
                name=row.get("baseline_name", ""),
                pr_auc=_fmt(row.get("test_pr_auc")),
                f1=_fmt(row.get("test_f1")),
            )
        )
    print(f"domain_baselines_csv={status.get('domain_baselines_csv')}")
    print(f"domain_baseline_summary={status.get('domain_baseline_summary')}")
    print(f"status_json={status_path}")


if __name__ == "__main__":
    raise SystemExit(main())
