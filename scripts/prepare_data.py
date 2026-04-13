#!/usr/bin/env python3
"""Prepare local benchmark assets for MIND."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Sequence

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.data import build_reference_candidates, load_object_yes_no_records, load_pope_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    normalize_pope = subparsers.add_parser("normalize-pope")
    normalize_pope.add_argument("--source", type=Path, required=True)
    normalize_pope.add_argument("--output", type=Path, required=True)
    normalize_pope.add_argument("--subset", required=True)
    normalize_pope.add_argument("--split", required=True)
    normalize_pope.add_argument("--source-dataset", default="pope")

    normalize_object_yes_no = subparsers.add_parser("normalize-object-yes-no")
    normalize_object_yes_no.add_argument("--source", type=Path, required=True)
    normalize_object_yes_no.add_argument("--output", type=Path, required=True)
    normalize_object_yes_no.add_argument("--subset", required=True)
    normalize_object_yes_no.add_argument("--split", required=True)
    normalize_object_yes_no.add_argument("--source-dataset", default="pope")
    normalize_object_yes_no.add_argument("--question-template", default=None)

    reference = subparsers.add_parser("build-reference")
    reference.add_argument("--instances-json", type=Path, required=True)
    reference.add_argument("--output", type=Path, required=True)
    reference.add_argument("--allowed-object", action="append", dest="allowed_objects", default=[])
    reference.add_argument("--allowed-objects-from", action="append", dest="allowed_objects_from", default=[])
    reference.add_argument("--exclude-image-ids", type=Path, default=None)
    reference.add_argument("--max-images-per-object", type=int, default=0)

    return parser


def _write_jsonl(output_path: Path, rows: Sequence[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _normalize_pope(args: argparse.Namespace) -> int:
    records = load_pope_records(
        args.source,
        subset=args.subset,
        split=args.split,
        source_dataset=args.source_dataset,
    )
    _write_jsonl(args.output, [asdict(record) for record in records])
    return 0


def _normalize_object_yes_no(args: argparse.Namespace) -> int:
    records = load_object_yes_no_records(
        args.source,
        subset=args.subset,
        split=args.split,
        source_dataset=args.source_dataset,
        question_template=args.question_template,
    )
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


def _load_allowed_objects(paths: Sequence[str], explicit_objects: Sequence[str]) -> set[str]:
    allowed = {value.strip() for value in explicit_objects if value.strip()}
    for raw_path in paths:
        path = Path(raw_path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            object_name = str(row.get("object_name") or row.get("object") or "").strip()
            if object_name:
                allowed.add(object_name)
    return allowed


def _build_reference(args: argparse.Namespace) -> int:
    coco_instances = json.loads(args.instances_json.read_text(encoding="utf-8"))
    allowed_objects = _load_allowed_objects(args.allowed_objects_from, args.allowed_objects)
    candidates = build_reference_candidates(
        coco_instances,
        allowed_objects=allowed_objects,
        exclude_image_ids=_load_excluded_image_ids(args.exclude_image_ids),
        max_images_per_object=args.max_images_per_object,
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
    if args.command == "normalize-object-yes-no":
        return _normalize_object_yes_no(args)
    if args.command == "build-reference":
        return _build_reference(args)
    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
