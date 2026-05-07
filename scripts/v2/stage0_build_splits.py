#!/usr/bin/env python3
"""Build v2 Stage 0 grouped split manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.trajectory.splits import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SPLIT_NAMES,
    build_split_manifest,
    write_split_manifest,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--subset", required=True)
    parser.add_argument("--input-records", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/v2_stage0/manifests/split_manifest.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--ratios",
        nargs=4,
        type=float,
        default=list(DEFAULT_RATIOS),
        metavar=("ENCODER_TRAIN", "BANK", "CAL", "TEST"),
    )
    parser.add_argument("--group-key", default="image_id")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the manifest without writing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = build_split_manifest(
            dataset_name=args.dataset_name,
            subset=args.subset,
            input_records=args.input_records,
            seed=args.seed,
            ratios=args.ratios,
            group_key=args.group_key,
        )
        if not args.dry_run:
            write_split_manifest(manifest, args.output)
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    _print_summary(manifest, output=args.output, dry_run=args.dry_run)
    return 0


def _print_summary(
    manifest: dict[str, object],
    *,
    output: Path,
    dry_run: bool,
) -> None:
    counts = manifest["counts_per_split"]
    print("Stage 0 split manifest complete")
    print(f"output={output}")
    print(f"dataset_name={manifest['dataset_name']} subset={manifest['subset']}")
    print(f"seed={manifest['seed']} group_key={manifest['group_key']}")
    print(
        "counts="
        + ",".join(
            f"{split_name}:{counts[split_name]}"  # type: ignore[index]
            for split_name in SPLIT_NAMES
        )
    )
    print(f"dry_run={str(dry_run).lower()}")
    if dry_run:
        print(
            json.dumps(
                {
                    "dry_run": dry_run,
                    "output": str(output),
                    "manifest": manifest,
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    raise SystemExit(main())
