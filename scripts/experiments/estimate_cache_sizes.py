#!/usr/bin/env python3
"""Estimate hidden-state cache tensor payload sizes for extraction plans."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    hidden_dim: int


@dataclass(frozen=True)
class SplitSpec:
    dataset: str
    split: str
    samples: int


DEFAULT_MODELS = [
    ModelSpec("qwen3-vl-8b", 4096),
    ModelSpec("qwen3-vl-4b", 2560),
    ModelSpec("internvl3.5-8b", 4096),
    ModelSpec("llava-onevision-7b", 3584),
]

DEFAULT_SPLITS = [
    SplitSpec("pope", "popular", 3000),
    SplitSpec("dash-b", "minimal", 1000),
]

ADVERSARIAL_SPLITS = [
    SplitSpec("pope", "adversarial", 3000),
]

REPOPE_SPLITS = [
    SplitSpec("repope", "popular", 3000),
    SplitSpec("repope", "adversarial", 3000),
]


def estimate_bytes(*, samples: int, layers: int, hidden_dim: int) -> int:
    return int(samples) * int(layers) * int(hidden_dim) * 2


def parse_model_spec(value: str) -> ModelSpec:
    try:
        name, hidden_dim = value.split(":", maxsplit=1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("model specs must use name:hidden_dim") from exc
    return ModelSpec(name=name, hidden_dim=int(hidden_dim))


def parse_split_spec(value: str) -> SplitSpec:
    try:
        dataset, split, samples = value.split(":", maxsplit=2)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("split specs must use dataset:split:samples") from exc
    return SplitSpec(dataset=dataset, split=split, samples=int(samples))


def build_plan(
    *,
    models: list[ModelSpec],
    splits: list[SplitSpec],
    include_adversarial: bool,
    include_repope: bool,
) -> tuple[list[ModelSpec], list[SplitSpec]]:
    planned_splits = list(splits)
    if include_adversarial:
        planned_splits.extend(ADVERSARIAL_SPLITS)
    if include_repope:
        planned_splits.extend(REPOPE_SPLITS)
    return list(models), planned_splits


def build_rows(*, models: list[ModelSpec], splits: list[SplitSpec], layers: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model in models:
        for split in splits:
            rows.append(
                {
                    "model": model.name,
                    "dataset": split.dataset,
                    "split": split.split,
                    "samples": split.samples,
                    "layers": layers,
                    "hidden_dim": model.hidden_dim,
                    "bytes": estimate_bytes(
                        samples=split.samples,
                        layers=layers,
                        hidden_dim=model.hidden_dim,
                    ),
                }
            )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument(
        "--model",
        dest="models",
        action="append",
        type=parse_model_spec,
        default=None,
        help="Override/add model as name:hidden_dim. May be repeated.",
    )
    parser.add_argument(
        "--split",
        dest="splits",
        action="append",
        type=parse_split_spec,
        default=None,
        help="Override/add split as dataset:split:samples. May be repeated.",
    )
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--include-repope", action="store_true")
    parser.add_argument("--format", choices=["table", "json"], default="table")
    return parser


def _format_bytes(value: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024
    raise AssertionError("unreachable")


def print_table(rows: list[dict[str, object]]) -> None:
    print("model\tdataset\tsplit\tsamples\tlayers\thidden_dim\tbytes\thuman")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["model"]),
                    str(row["dataset"]),
                    str(row["split"]),
                    str(row["samples"]),
                    str(row["layers"]),
                    str(row["hidden_dim"]),
                    str(row["bytes"]),
                    _format_bytes(int(row["bytes"])),
                ]
            )
        )
    total = sum(int(row["bytes"]) for row in rows)
    print(f"TOTAL\t\t\t\t\t\t{total}\t{_format_bytes(total)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    models, splits = build_plan(
        models=args.models or DEFAULT_MODELS,
        splits=args.splits or DEFAULT_SPLITS,
        include_adversarial=args.include_adversarial,
        include_repope=args.include_repope,
    )
    rows = build_rows(models=models, splits=splits, layers=args.layers)
    total = sum(int(row["bytes"]) for row in rows)
    if args.format == "json":
        print(json.dumps({"rows": rows, "total_bytes": total}, indent=2, sort_keys=True))
    else:
        print_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
