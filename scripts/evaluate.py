#!/usr/bin/env python3
"""Aggregate detector outputs into experiment reports."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_report_paths(*, output_root: Path, experiment_name: str) -> dict[str, Path]:
    report_root = output_root / experiment_name
    return {
        "metrics": report_root / "metrics.json",
        "results": report_root / "results.csv",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    paths = build_report_paths(
        output_root=args.output_root,
        experiment_name=args.experiment_name,
    )
    for path in paths.values():
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
