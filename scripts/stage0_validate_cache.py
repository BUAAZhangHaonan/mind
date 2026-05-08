#!/usr/bin/env python3
"""Validate Stage 0 cache shards and sidecar metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.trajectory.cache import validate_stage0_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "cache_root_positional",
        nargs="?",
        type=Path,
        metavar="CACHE_ROOT",
        help="Cache root to validate. May be supplied instead of --cache-root.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=None,
        help="Cache root to validate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/stage0/manifests/cache_manifest.json"),
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and write the manifest, then print a dry-run summary.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cache_root = _resolve_cache_root(parser, args.cache_root, args.cache_root_positional)

    manifest = validate_stage0_cache(
        cache_root,
        output=args.output,
        dataset_name=args.dataset_name,
        split=args.split,
        model_name=args.model_name,
        raise_on_error=False,
    )
    _print_summary(manifest, output=args.output, dry_run=args.dry_run)
    return 0 if manifest["status"] == "passed" else 2


def _resolve_cache_root(
    parser: argparse.ArgumentParser,
    option_value: Path | None,
    positional_value: Path | None,
) -> Path:
    if option_value is None and positional_value is None:
        parser.error("cache root is required as --cache-root or positional CACHE_ROOT")
    if option_value is not None and positional_value is not None and option_value != positional_value:
        parser.error("provide cache root either as --cache-root or positional CACHE_ROOT, not both")
    if option_value is not None:
        return option_value
    assert positional_value is not None
    return positional_value


def _print_summary(
    manifest: dict[str, object],
    *,
    output: Path,
    dry_run: bool,
) -> None:
    shards = manifest["shards"]
    print("Stage 0 cache validation complete")
    print(f"output={output}")
    print(f"cache_root={manifest['cache_root']}")
    print(f"status={manifest['status']}")
    print(f"shards={len(shards)} total_entries={manifest['total_entries']}")
    print(f"duplicate_keys={len(manifest['duplicate_keys'])}")
    print(f"errors={len(manifest['errors'])}")
    print(f"dry_run={str(dry_run).lower()}")


if __name__ == "__main__":
    raise SystemExit(main())
