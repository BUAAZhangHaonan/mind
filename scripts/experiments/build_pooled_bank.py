#!/usr/bin/env python3
"""Pool object-conditioned reference banks into model/layer banks."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import torch


REPO_SRC = Path(__file__).resolve().parents[2] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

SPECIAL_BANK_NAMES = {"__shared__", "__pooled__"}


def _layer_index(path: Path) -> int:
    try:
        return int(path.stem.split("-")[-1])
    except ValueError as exc:
        raise ValueError(f"Cannot parse layer index from {path}") from exc


def _object_dirs(model_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(model_root.iterdir(), key=lambda item: item.name)
        if path.is_dir() and path.name not in SPECIAL_BANK_NAMES and not path.name.startswith("__")
    ]


def _collect_layer_sources(reference_root: Path, model_name: str) -> dict[int, list[tuple[str, torch.Tensor]]]:
    model_root = reference_root / model_name
    if not model_root.exists():
        raise FileNotFoundError(f"Reference model root does not exist: {model_root}")

    layers: dict[int, list[tuple[str, torch.Tensor]]] = {}
    for object_dir in _object_dirs(model_root):
        for layer_path in sorted(object_dir.glob("layer-*.pt"), key=_layer_index):
            tensor = torch.load(layer_path, map_location="cpu", weights_only=True)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"{layer_path} did not contain a tensor")
            if tensor.ndim != 2:
                raise ValueError(f"{layer_path} must contain a rank-2 tensor")
            layers.setdefault(_layer_index(layer_path), []).append((object_dir.name, tensor.detach().cpu()))
    if not layers:
        raise ValueError(f"No object-conditioned layer tensors found under {model_root}")
    return layers


def _validate_layer_tensors(layer_index: int, sources: list[tuple[str, torch.Tensor]]) -> int:
    if not sources:
        raise ValueError(f"Layer {layer_index} has no source tensors")
    feature_dim = int(sources[0][1].shape[1])
    for object_name, tensor in sources:
        if tensor.shape[0] == 0:
            raise ValueError(f"Layer {layer_index} object {object_name} has no rows")
        if int(tensor.shape[1]) != feature_dim:
            raise ValueError(
                f"Layer {layer_index} dimension mismatch: object {object_name} has "
                f"{tensor.shape[1]} columns, expected {feature_dim}"
            )
    return feature_dim


def _write_metadata(path: Path, *, layer_index: int, sources: list[tuple[str, torch.Tensor]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    pooled_index = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer_index", "pooled_index", "object_name", "source_row_index"],
        )
        writer.writeheader()
        for object_name, tensor in sources:
            for source_row_index in range(int(tensor.shape[0])):
                writer.writerow(
                    {
                        "layer_index": int(layer_index),
                        "pooled_index": pooled_index,
                        "object_name": object_name,
                        "source_row_index": source_row_index,
                    }
                )
                pooled_index += 1
    return pooled_index


def _write_counts(path: Path, rows: list[dict[str, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["layer_index", "pooled_count", "source_count", "object_count"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_pooled_bank(
    *,
    reference_root: Path,
    output_root: Path,
    model_name: str,
) -> list[dict[str, int]]:
    """Pool saved object-conditioned bank tensors by layer.

    The output layout is ``output_root/model_name/layer-XX.pt`` plus one
    ``layer-XX.metadata.csv`` file mapping pooled rows back to source objects.
    """
    layer_sources = _collect_layer_sources(reference_root, model_name)
    output_model_root = output_root / model_name
    output_model_root.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, int]] = []

    for layer_index in sorted(layer_sources):
        sources = layer_sources[layer_index]
        _validate_layer_tensors(layer_index, sources)
        pooled = torch.cat([tensor.contiguous() for _, tensor in sources], dim=0).contiguous()
        source_count = sum(int(tensor.shape[0]) for _, tensor in sources)
        if int(pooled.shape[0]) != source_count:
            raise RuntimeError(
                f"Layer {layer_index} pooled count {pooled.shape[0]} did not match source count {source_count}"
            )
        tensor_path = output_model_root / f"layer-{layer_index:02d}.pt"
        torch.save(pooled, tensor_path)
        metadata_count = _write_metadata(
            output_model_root / f"layer-{layer_index:02d}.metadata.csv",
            layer_index=layer_index,
            sources=sources,
        )
        if metadata_count != source_count:
            raise RuntimeError(
                f"Layer {layer_index} metadata count {metadata_count} did not match source count {source_count}"
            )
        summary_rows.append(
            {
                "layer_index": int(layer_index),
                "pooled_count": int(pooled.shape[0]),
                "source_count": int(source_count),
                "object_count": len(sources),
            }
        )

    _write_counts(output_model_root / "pooled_counts.csv", summary_rows)
    return summary_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--device", default="cuda", help="Accepted for orchestration symmetry; pooling uses CPU file I/O.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = build_pooled_bank(
        reference_root=args.reference_root,
        output_root=args.output_root,
        model_name=args.model_name,
    )
    for row in rows:
        layer_path = args.output_root / args.model_name / f"layer-{row['layer_index']:02d}.pt"
        print(
            f"{layer_path} pooled_count={row['pooled_count']} "
            f"source_count={row['source_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
