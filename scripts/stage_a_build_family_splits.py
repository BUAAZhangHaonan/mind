#!/usr/bin/env python3
"""Build Stage A POPE-family image-id splits from Stage 0 cache rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.trajectory.stage_a_population import stream_stage0_cache_entries
from mind.trajectory.stage_a_splits import (
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    SPLIT_NAMES,
    build_pope_family_split,
    write_family_split_manifest,
)


POPE_FAMILY_DATASETS = ("pope",)
POPE_FAMILY_SUBSETS = ("popular", "random", "adversarial")
DEFAULT_OUTPUT = Path("outputs/stageA/manifests/pope_family_split_manifest.json")


def build_family_splits(
    *,
    stage0_root: Path | str = Path("outputs/stage0"),
    output_root: Path | str = Path("outputs/stageA"),
    output: Path | str | None = None,
    dataset_names: Sequence[str] = POPE_FAMILY_DATASETS,
    seed: int = DEFAULT_SEED,
    ratios: Sequence[float] = DEFAULT_RATIOS,
    write_output: bool = True,
) -> dict[str, object]:
    """Read Stage 0 cache rows and build the Stage A POPE-family split manifest."""

    stage0_root = Path(stage0_root)
    output_root = Path(output_root)
    output_path = Path(output) if output is not None else output_root / "manifests" / DEFAULT_OUTPUT.name
    dataset_names = _validate_dataset_names(dataset_names)
    entries = list(
        stream_stage0_cache_entries(
            stage0_root,
            dataset_names=dataset_names,
            include_tensors=False,
        )
    )
    _validate_pope_family_subsets(entries)
    manifest = build_pope_family_split(entries, seed=seed, ratios=ratios)
    if int(manifest["num_entries"]) == 0:
        raise ValueError("no POPE-family cache entries found in Stage 0 cache")
    manifest["stage0_root"] = str(stage0_root)
    manifest["output"] = str(output_path)
    manifest["dataset_names"] = list(dataset_names)
    manifest["allowed_subsets"] = list(POPE_FAMILY_SUBSETS)
    if write_output:
        write_family_split_manifest(manifest, output_path)
    return manifest


def _validate_dataset_names(dataset_names: Sequence[str]) -> tuple[str, ...]:
    values = tuple(str(name).strip() for name in dataset_names)
    if values != POPE_FAMILY_DATASETS:
        raise ValueError("Stage A split builder only accepts dataset_names=['pope']")
    return values


def _validate_pope_family_subsets(entries: Sequence[Mapping[str, object]]) -> None:
    subset_values = {str(row.get("subset", "")).strip() or "<blank>" for row in entries}
    unsupported = sorted(subset for subset in subset_values if subset not in POPE_FAMILY_SUBSETS)
    if unsupported:
        raise ValueError("unsupported Stage A POPE subsets: " + ", ".join(unsupported))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage0-root", type=Path, default=Path("outputs/stage0"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageA"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--dataset-name",
        action="append",
        dest="dataset_names",
        default=None,
        help="POPE-family dataset to include. May be supplied more than once.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--ratios",
        nargs=4,
        type=float,
        default=list(DEFAULT_RATIOS),
        metavar=("ENCODER_TRAIN", "BANK", "CAL", "TEST"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and print the manifest without writing it.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset_names = tuple(args.dataset_names or POPE_FAMILY_DATASETS)
    try:
        manifest = build_family_splits(
            stage0_root=args.stage0_root,
            output_root=args.output_root,
            output=args.output,
            dataset_names=dataset_names,
            seed=args.seed,
            ratios=args.ratios,
            write_output=not args.dry_run,
        )
    except (FileNotFoundError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2

    output = Path(str(manifest["output"]))
    _print_summary(manifest, output=output, dry_run=args.dry_run)
    return 0


def _print_summary(
    manifest: dict[str, object],
    *,
    output: Path,
    dry_run: bool,
) -> None:
    counts = manifest["counts_per_split"]
    print("Stage A POPE-family split manifest complete")
    print(f"output={output}")
    print(f"seed={manifest['seed']} group_key={manifest['group_key']}")
    print(f"num_entries={manifest['num_entries']} num_image_ids={manifest['num_image_ids']}")
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
