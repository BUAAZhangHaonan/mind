#!/usr/bin/env python3
"""Build Stage A closeout family-level splits from the unified full-cache manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_SRC = REPO_ROOT / "src"
if str(REPO_SRC) in sys.path:
    sys.path.remove(str(REPO_SRC))
sys.path.insert(0, str(REPO_SRC))

from mind.trajectory.stage_a_closeout import (  # noqa: E402
    FAMILY_SUBSETS,
    build_closeout_family_split,
    load_closeout_panel_manifest,
    stream_full_cache_entries,
    write_split_manifest,
)


OUTPUT_NAMES = {
    "pope": "pope_family_split_manifest.json",
    "repope": "repope_family_split_manifest.json",
    "dash-b": "dash_b_split_manifest.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-cache-root", type=Path, default=Path("outputs/full_cache"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/stageA_closeout"))
    parser.add_argument("--datasets", nargs="+", default=["pope", "repope", "dash-b"])
    parser.add_argument("--seed", type=int, default=20260506)
    parser.add_argument(
        "--ratios",
        nargs=4,
        type=float,
        default=[0.50, 0.20, 0.10, 0.20],
        metavar=("ENCODER_TRAIN", "BANK", "CAL", "TEST"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    panel = load_closeout_panel_manifest(args.full_cache_root)
    split_source_model = panel.models[0]
    for family in args.datasets:
        if family not in FAMILY_SUBSETS:
            raise SystemExit(f"unsupported dataset family: {family}")
        rows = _split_source_rows(
            split_source_model,
            full_cache_root=args.full_cache_root,
            family=family,
        )
        manifest = build_closeout_family_split(
            rows,
            family=family,
            seed=args.seed,
            ratios=args.ratios,
        )
        manifest["split_source_model"] = split_source_model["model_alias"]
        manifest["split_application"] = "image_id assignments are applied to every panel model"
        output_path = args.output_root / "manifests" / OUTPUT_NAMES[family]
        write_split_manifest(manifest, output_path)
        print(f"wrote {output_path} entries={manifest['num_entries']}")
    return 0


def _split_source_rows(
    model_row: dict[str, object],
    *,
    full_cache_root: Path,
    family: str,
) -> list[dict[str, object]]:
    return list(
        stream_full_cache_entries(
            model_row,
            full_cache_root,
            dataset_family=family,
            include_tensors=False,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
