#!/usr/bin/env python3
"""Minimal experiment stage orchestration for MIND."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_STAGES = [
    "prepare",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stages", default="all")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stages = parse_stage_list(args.stages)
    print(f"config={args.config}")
    print("stages=" + ",".join(stages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
