#!/usr/bin/env python3
"""Audit v2 Stage 0 datasets and optional cache shards."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.trajectory import DatasetSpec, discover_known_datasets, run_audit, validate_required_datasets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/v2_stage0"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        nargs=3,
        action="append",
        metavar=("DATASET_NAME", "SUBSET", "PATH"),
        default=[],
        help="Dataset tuple to audit. May be repeated.",
    )
    parser.add_argument(
        "--require",
        nargs=2,
        action="append",
        metavar=("DATASET_NAME", "SUBSET"),
        default=[],
        help="Fail if this dataset/subset tuple is missing. May be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    specs = _dataset_specs_from_args(args.dataset) if args.dataset else discover_known_datasets(Path("."))

    try:
        validate_required_datasets(specs, [(dataset, subset) for dataset, subset in args.require])
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 2

    result = run_audit(specs, output_root=args.output_root, cache_root=args.cache_root)
    _print_summary(result, dry_run=args.dry_run, cache_root=args.cache_root)
    return 0


def _dataset_specs_from_args(values: list[list[str]]) -> list[DatasetSpec]:
    return [
        DatasetSpec(dataset_name=dataset_name, subset=subset, path=Path(path))
        for dataset_name, subset, path in values
    ]


def _print_summary(result, *, dry_run: bool, cache_root: Path | None) -> None:
    present = sum(1 for row in result.dataset_audit_rows if row["status"] == "present")
    missing = sum(1 for row in result.dataset_audit_rows if row["status"] == "missing")
    records = sum(int(row["num_records"]) for row in result.dataset_audit_rows)
    parsed_statuses = sorted({str(row["parsed_answer_status"]) for row in result.label_balance_rows})
    print("Stage 0 audit complete")
    print(f"audit_dir={result.audit_dir}")
    print(f"datasets_present={present} datasets_missing={missing} records={records}")
    print(f"parsed_answer_status={','.join(parsed_statuses) if parsed_statuses else 'none'}")
    print(f"cache_root={cache_root if cache_root is not None else 'not_supplied'} dry_run={str(dry_run).lower()}")


if __name__ == "__main__":
    raise SystemExit(main())
