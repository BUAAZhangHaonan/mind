#!/usr/bin/env python3
"""Build local manifold artifacts from cached reference states."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_output_path(
    *,
    output_root: Path,
    model_name: str,
    object_name: str,
    layer_index: int,
) -> Path:
    return output_root / model_name / object_name / f"layer-{layer_index:02d}.pt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--object-name", required=True)
    parser.add_argument("--layer-index", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    path = build_output_path(
        output_root=args.output_root,
        model_name=args.model_name,
        object_name=args.object_name,
        layer_index=args.layer_index,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
