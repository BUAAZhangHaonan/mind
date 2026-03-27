#!/usr/bin/env python3
"""Cache reference hidden-state shards for grounded samples."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_reference_cache_output_path(
    *,
    output_root: Path,
    model_name: str,
    dataset_name: str,
    split: str,
    shard_index: int,
) -> Path:
    return output_root / model_name / dataset_name / split / f"shard-{shard_index:05d}.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(
        build_reference_cache_output_path(
            output_root=args.output_root,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            split=args.split,
            shard_index=args.shard_index,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
