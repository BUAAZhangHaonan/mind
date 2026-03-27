#!/usr/bin/env python3
"""Prepare local benchmark assets for MIND."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from mind.data import build_reference_candidates, load_pope_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_pope = subparsers.add_parser("normalize-pope")
    normalize_pope.add_argument("--source", type=Path, required=True)
    normalize_pope.add_argument("--output", type=Path, required=True)
    normalize_pope.add_argument("--subset", required=True)
    normalize_pope.add_argument("--split", required=True)

    reference = subparsers.add_parser("build-reference")
    reference.add_argument("--instances-json", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--allowed-object", action="append", dest="allowed_objects", default=[])
    reference.add_argument("--exclude-image-ids", type=Path, default=None)

    return parser


def _write_jsonl(output_path: Path, rows: Sequence[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _normalize_pope(args: argparse.Namespace) -> int:
    records = load_pope_records(args.source, subset=args.subset, split=args.split)
    _write_jsonl(args.output, [asdict(record) for record in records])
    return 0


def _load_excluded_image_ids(path: Path | None) -> set[int]:
    if path is None:
        return set()
    return {
        int(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _build_reference(args: argparse.Namespace) -> int:
    coco_instances = json.loads(args.instances_json.read_text(encoding="utf-8"))
    candidates = build_reference_candidates(
        coco_instances,
        allowed_objects=set(args.allowed_objects),
        exclude_image_ids=_load_excluded_image_ids(args.exclude_image_ids),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(candidates, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "normalize-pope":
        return _normalize_pope(args)
    if args.command == "build-reference":
        return _build_reference(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
