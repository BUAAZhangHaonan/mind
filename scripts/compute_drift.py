#!/usr/bin/env python3
"""Compute drift features from cached hidden-state shards."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_feature_output_path(
    *,
    output_root: Path,
    experiment_name: str,
    split: str,
) -> Path:
    return output_root / experiment_name / f"{split}.parquet"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--split", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        build_feature_output_path(
            output_root=args.output_root,
            experiment_name=args.experiment_name,
            split=args.split,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
