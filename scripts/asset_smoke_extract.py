#!/usr/bin/env python3
"""Run tiny deterministic smoke extraction for Experiment 1 model assets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import torch

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
repo_src_path = str(REPO_SRC)
if repo_src_path in sys.path:
    sys.path.remove(repo_src_path)
sys.path.insert(0, repo_src_path)

from mind.config import ModelConfig, load_yaml_config
from mind.data import HallucinationRecord
from mind.extractors.prefill import estimate_prefill_cache_tensor_bytes, save_prefill_cache_shard
from mind.models.asset_validation import (
    AssetStatus,
    audit_asset_metadata,
    build_completion_summary,
    tensor_checksum,
    validate_determinism_pair,
    DeterminismPair,
)
from mind.models.factory import create_model_wrapper
from mind.models.registry import REQUIRED_MODEL_ALIASES, load_asset_registry
from mind.models.types import parse_yes_no_answer, resolve_torch_dtype
from mind.trajectory.dataset import validate_extraction_ready_row


DATASET_SPECS = {
    "pope": ("pope", "popular"),
    "repope": ("repope", "popular"),
    "dash-b": ("dash-b", "all"),
}
DATASET_IMAGE_ROOTS = {
    "pope": Path("data/coco/val2014"),
    "repope": Path("data/coco/val2014"),
    "dash-b": Path("data/dash_b"),
}
REQUIRED_RECORD_FIELDS = tuple(field.name for field in fields(HallucinationRecord))
REPORT_FIELDS = (
    "model_alias",
    "dataset",
    "subset",
    "status",
    "reason",
    "shard_path",
    "sidecar_path",
    "num_records",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--stage0-root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--smoke-limit", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    return parser


def run_smoke(
    *,
    registry_path: Path,
    output_root: Path,
    stage0_root: Path,
    datasets: Sequence[str],
    smoke_limit: int,
    device: str,
) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ.setdefault("MKL_NUM_THREADS", "4")
    torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
    if smoke_limit <= 0:
        raise ValueError("--smoke-limit must be positive for Experiment 1")
    unknown = sorted(set(datasets) - set(DATASET_SPECS))
    if unknown:
        raise ValueError(f"unsupported smoke datasets: {unknown}")

    output_root.mkdir(parents=True, exist_ok=True)
    registry = load_asset_registry(registry_path)
    missing_paths = required_dataset_paths(stage0_root=stage0_root, datasets=datasets)
    audit_results = {result.alias: result for result in (audit_asset_metadata(model) for model in registry.models)}
    rows: list[dict[str, object]] = []
    checksums: dict[str, object] = {
        "determinism": {},
        "image_sensitivity_canary": {},
        "created_at_utc": utc_now_iso(),
    }

    if missing_paths:
        reason = "required smoke dataset file missing before model loading: " + ", ".join(str(path) for path in missing_paths)
        for alias in REQUIRED_MODEL_ALIASES:
            audit = audit_results[alias]
            status = AssetStatus.BLOCKED.value if audit.status == AssetStatus.VERIFIED else audit.status.value
            row_reason = reason if audit.status == AssetStatus.VERIFIED else f"{audit.reason}; {reason}"
            for dataset in datasets:
                dataset_name, subset = DATASET_SPECS[dataset]
                rows.append(_report_row(alias, dataset_name, subset, status, row_reason))
        write_smoke_outputs(output_root, rows, checksums)
        write_summary_from_rows(output_root, rows, datasets=datasets, smoke_limit=smoke_limit)
        print(reason, file=sys.stderr)
        return 2

    records_by_dataset = {
        dataset: load_smoke_records(stage0_root=stage0_root, dataset_key=dataset, limit=smoke_limit)
        for dataset in datasets
    }
    for asset in registry.models:
        audit = audit_results[asset.alias]
        if audit.status != AssetStatus.VERIFIED:
            for dataset in datasets:
                dataset_name, subset = DATASET_SPECS[dataset]
                rows.append(_report_row(asset.alias, dataset_name, subset, audit.status.value, audit.reason))
            continue
        try:
            model_config = merge_asset_model_config(load_yaml_config(asset.model_config_path, ModelConfig), asset)
            wrapper = create_model_wrapper(model_config)
            processor = wrapper.load_processor()
            model = wrapper.load_model(device=device)
            total_layers = wrapper.resolve_total_layers(model)
            hidden_dim = wrapper.resolve_hidden_dim(model)
            offset = wrapper.resolve_hidden_state_index_offset()
            for dataset in datasets:
                dataset_name, subset = DATASET_SPECS[dataset]
                records = records_by_dataset[dataset]
                entries, hidden_state_count = extract_entries(
                    model=model,
                    processor=processor,
                    wrapper=wrapper,
                    records=records,
                    device=device,
                    total_layers=total_layers,
                    offset=offset,
                    model_config=model_config,
                    dataset_name=dataset_name,
                    subset=subset,
                )
                repeat_entries, _ = extract_entries(
                    model=model,
                    processor=processor,
                    wrapper=wrapper,
                    records=records[:2],
                    device=device,
                    total_layers=total_layers,
                    offset=offset,
                    model_config=model_config,
                    dataset_name=dataset_name,
                    subset=subset,
                )
                determinism_results = [
                    validate_determinism_pair(
                        DeterminismPair(first=entry, second=repeat),
                        layer_tolerance=1e-3,
                        logits_tolerance=1e-3,
                    )
                    for entry, repeat in zip(entries[:2], repeat_entries)
                ]
                failed_determinism = [result for result in determinism_results if result.status != "verified"]
                if failed_determinism:
                    raise ValueError(f"determinism validation failed: {failed_determinism[0].reason}")
                checksums["determinism"][f"{asset.alias}/{dataset_name}/{subset}"] = {
                    "status": "verified",
                    "pairs": [result.details for result in determinism_results],
                    "primary": [entry_checksums(entry) for entry in entries[:2]],
                    "repeat": [entry_checksums(entry) for entry in repeat_entries],
                }
                shard_path = output_root / "smoke_cache" / asset.alias / dataset_name / subset / "shard-00000.pt"
                metadata = {
                    "stage": "assets",
                    "cache_type": "asset_smoke_prefill",
                    "model_name": model_config.name,
                    "model_family": model_config.family,
                    "dataset_name": dataset_name,
                    "source_dataset": dataset_name,
                    "subset": subset,
                    "split": subset,
                    "total_layers": total_layers,
                    "hidden_dim": hidden_dim,
                    "hidden_state_index_offset": offset,
                    "hidden_state_count": hidden_state_count,
                    "hidden_state_index_offset_source": "configs/assets/model_assets.yaml",
                    "selected_layer_hidden_state_indices": [layer + offset for layer in range(total_layers)],
                    "selected_layers": list(range(total_layers)),
                    "token_index": int(entries[0]["token_index"]),
                    "max_new_tokens": 1,
                    "dtype": model_config.dtype,
                    "prompt_template_id": wrapper.prompt_template_id(),
                    "prompt_template_text": wrapper.prompt_template_text(),
                    "script": "scripts/asset_smoke_extract.py",
                    "git_commit": get_git_commit(),
                    "created_at_utc": utc_now_iso(),
                }
                sidecar = save_prefill_cache_shard(
                    entries,
                    shard_path,
                    dtype=resolve_torch_dtype(model_config.dtype),
                    cast_all_floating_tensors=False,
                    estimated_tensor_bytes=estimate_prefill_cache_tensor_bytes(entries, dtype=resolve_torch_dtype(model_config.dtype)),
                    metadata=metadata,
                )
                merge_top_level_sidecar_metadata(shard_path, metadata)
                rows.append(
                    _report_row(
                        asset.alias,
                        dataset_name,
                        subset,
                        AssetStatus.VERIFIED.value,
                        "smoke extraction completed",
                        shard_path=str(shard_path),
                        sidecar_path=str(shard_path) + ".json",
                        num_records=len(entries),
                    )
                )
            canary = run_image_sensitivity_canary(
                model=model,
                processor=processor,
                wrapper=wrapper,
                records_by_dataset=records_by_dataset,
                device=device,
                total_layers=total_layers,
                offset=offset,
                model_config=model_config,
            )
            checksums["image_sensitivity_canary"][asset.alias] = canary
            if canary.get("status") == AssetStatus.FAILED_VALIDATION.value:
                raise ValueError(str(canary.get("reason", "image sensitivity canary failed")))
        except ImportError as error:
            reason = f"missing dependency while loading model: {error}"
            rows.extend(_blocked_model_rows(asset.alias, datasets, reason, AssetStatus.BLOCKED.value))
        except Exception as error:
            reason = f"smoke extraction failed: {type(error).__name__}: {error}"
            rows.extend(_blocked_model_rows(asset.alias, datasets, reason, AssetStatus.FAILED_VALIDATION.value))
        finally:
            try:
                del model
                del processor
            except UnboundLocalError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_smoke_outputs(output_root, rows, checksums)
    write_summary_from_rows(output_root, rows, datasets=datasets, smoke_limit=smoke_limit)
    return 0


def required_dataset_paths(*, stage0_root: Path, datasets: Sequence[str]) -> list[Path]:
    missing: list[Path] = []
    for dataset in datasets:
        dataset_name, subset = DATASET_SPECS[dataset]
        path = stage0_root / "normalized" / dataset_name / f"{subset}.jsonl"
        if not path.is_file():
            missing.append(path)
    return missing


def merge_asset_model_config(model_config: ModelConfig, asset: object) -> ModelConfig:
    return model_config.model_copy(
        update={
            "local_path": asset.local_path,
            "deterministic_generation": asset.deterministic_generation.model_dump(),
            "thinking": asset.thinking.model_dump(),
            "policy": asset.policy.model_dump(),
            "prompt_template_id": asset.prompt_template_id,
            "prompt_template_text": asset.prompt_template_text,
            "hidden_state_index_offset": asset.hidden_state_index_offset,
        }
    )


def load_smoke_records(*, stage0_root: Path, dataset_key: str, limit: int) -> list[HallucinationRecord]:
    dataset_name, subset = DATASET_SPECS[dataset_key]
    path = stage0_root / "normalized" / dataset_name / f"{subset}.jsonl"
    rows: list[HallucinationRecord] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        validate_extraction_ready_row(
            row,
            path=path,
            record_number=line_number,
            required_fields=REQUIRED_RECORD_FIELDS,
        )
        payload = {field: row[field] for field in REQUIRED_RECORD_FIELDS}
        record = HallucinationRecord(**payload)
        rows.append(resolve_record_image_path(record, dataset_name=dataset_name))
        if len(rows) >= limit:
            break
    if len(rows) < limit:
        raise ValueError(f"{path} has fewer than {limit} smoke records")
    return rows


def resolve_record_image_path(record: HallucinationRecord, *, dataset_name: str) -> HallucinationRecord:
    image_path = Path(record.image_path)
    if image_path.is_absolute():
        return record
    root = DATASET_IMAGE_ROOTS[dataset_name]
    return replace(record, image_path=str(root / image_path))


def extract_entries(
    *,
    model: object,
    processor: object,
    wrapper: object,
    records: Sequence[HallucinationRecord],
    device: str,
    total_layers: int,
    offset: int,
    model_config: ModelConfig,
    dataset_name: str,
    subset: str,
) -> tuple[list[dict[str, object]], int]:
    entries: list[dict[str, object]] = []
    selected_layers = list(range(total_layers))
    hidden_state_count = -1
    with torch.inference_mode():
        for record in records:
            model_inputs = wrapper.prepare_asset_batch_inputs(
                processor,
                questions=[record.question],
                image_paths=[record.image_path],
                device=device,
            )
            generation_output = wrapper.generate(
                model,
                processor,
                model_inputs=model_inputs,
                max_new_tokens=1,
            )
            hidden_states = wrapper.resolve_prefill_hidden_states(
                model,
                processor,
                model_inputs=model_inputs,
                generation_output=generation_output,
            )
            hidden_state_count = len(hidden_states)
            if hidden_state_count != total_layers + offset:
                raise ValueError(
                    "hidden_state_index_offset mismatch: "
                    f"hidden_states={hidden_state_count} total_layers={total_layers} offset={offset}"
                )
            token_index = wrapper.resolve_query_token_index(
                processor,
                model_inputs=model_inputs,
                batch_index=0,
            )
            layer_vectors = torch.stack(
                [
                    hidden_states[layer + offset][0, token_index, :].detach().cpu()
                    for layer in selected_layers
                ],
                dim=0,
            )
            answer_text = wrapper.decode_generation(
                processor,
                generated_ids=generation_output.sequences[0:1],
                prompt_input_ids=model_inputs["input_ids"][0:1],
            )
            first_token_logits = generation_output.scores[0][0].detach().cpu()
            if not torch.isfinite(first_token_logits).all().item():
                raise ValueError(f"first_token_logits contain non-finite values for sample_id={record.sample_id}")
            entries.append(
                {
                    "sample_id": record.sample_id,
                    "image_id": record.image_id,
                    "image_path": record.image_path,
                    "question": record.question,
                    "label": record.label,
                    "object_name": record.object_name,
                    "source_dataset": dataset_name,
                    "subset": subset,
                    "answer_text": answer_text,
                    "parsed_answer": parse_yes_no_answer(answer_text),
                    "first_token_logits": first_token_logits,
                    "selected_layers": selected_layers,
                    "layer_vectors": layer_vectors,
                    "model_name": model_config.name,
                    "model_family": model_config.family,
                    "token_index": token_index,
                    "prompt_template_id": wrapper.prompt_template_id(),
                }
            )
    return entries, hidden_state_count


def run_image_sensitivity_canary(
    *,
    model: object,
    processor: object,
    wrapper: object,
    records_by_dataset: Mapping[str, Sequence[HallucinationRecord]],
    device: str,
    total_layers: int,
    offset: int,
    model_config: ModelConfig,
) -> dict[str, object]:
    for dataset_key, records in records_by_dataset.items():
        if len(records) < 2:
            continue
        first, second = records[0], records[1]
        if first.image_path == second.image_path:
            continue
        dataset_name, subset = DATASET_SPECS[dataset_key]
        canary_records = [first, replace(second, question=first.question)]
        entries, _ = extract_entries(
            model=model,
            processor=processor,
            wrapper=wrapper,
            records=canary_records,
            device=device,
            total_layers=total_layers,
            offset=offset,
            model_config=model_config,
            dataset_name=dataset_name,
            subset=subset,
        )
        first_checksum = tensor_checksum(torch.as_tensor(entries[0]["layer_vectors"]))
        second_checksum = tensor_checksum(torch.as_tensor(entries[1]["layer_vectors"]))
        return {
            "status": "verified" if first_checksum != second_checksum else AssetStatus.FAILED_VALIDATION.value,
            "dataset": dataset_name,
            "subset": subset,
            "first_checksum": first_checksum,
            "second_checksum": second_checksum,
            "reason": "" if first_checksum != second_checksum else "image sensitivity canary checksums are identical",
        }
    return {"status": "skipped_with_reason", "reason": "no two smoke records with different images were available"}


def entry_checksums(entry: Mapping[str, object]) -> dict[str, object]:
    return {
        "sample_id": entry["sample_id"],
        "layer_vectors_checksum": tensor_checksum(torch.as_tensor(entry["layer_vectors"])),
        "first_token_logits_checksum": tensor_checksum(torch.as_tensor(entry["first_token_logits"])),
    }


def _blocked_model_rows(alias: str, datasets: Sequence[str], reason: str, status: str) -> list[dict[str, object]]:
    rows = []
    for dataset in datasets:
        dataset_name, subset = DATASET_SPECS[dataset]
        rows.append(_report_row(alias, dataset_name, subset, status, reason))
    return rows


def _report_row(
    alias: str,
    dataset: str,
    subset: str,
    status: str,
    reason: str,
    *,
    shard_path: str = "",
    sidecar_path: str = "",
    num_records: int = 0,
) -> dict[str, object]:
    return {
        "model_alias": alias,
        "dataset": dataset,
        "subset": subset,
        "status": status,
        "reason": reason,
        "shard_path": shard_path,
        "sidecar_path": sidecar_path,
        "num_records": num_records,
    }


def write_smoke_outputs(output_root: Path, rows: list[dict[str, object]], checksums: dict[str, object]) -> None:
    _write_csv(output_root / "smoke_extraction_report.csv", rows, REPORT_FIELDS)
    _write_json(output_root / "validation_checksums.json", checksums)


def write_summary_from_rows(output_root: Path, rows: Sequence[Mapping[str, object]], *, datasets: Sequence[str], smoke_limit: int) -> None:
    severity = {
        AssetStatus.FAILED_VALIDATION.value: 5,
        AssetStatus.BLOCKED.value: 4,
        AssetStatus.UNSUPPORTED_BY_POLICY.value: 3,
        AssetStatus.UNSUPPORTED_BY_WRAPPER.value: 3,
        AssetStatus.VERIFIED.value: 1,
    }
    statuses: dict[str, str] = {}
    reasons: dict[str, str] = {}
    for alias in REQUIRED_MODEL_ALIASES:
        alias_rows = [row for row in rows if row.get("model_alias") == alias]
        status = (
            AssetStatus.VERIFIED.value
            if alias_rows and all(row.get("status") == AssetStatus.VERIFIED.value for row in alias_rows)
            else max(
                (str(row.get("status", AssetStatus.BLOCKED.value)) for row in alias_rows),
                key=lambda value: severity.get(value, 0),
                default=AssetStatus.BLOCKED.value,
            )
        )
        statuses[alias] = status
        reasons[alias] = "; ".join(sorted({str(row.get("reason", "")) for row in alias_rows if row.get("reason")}))
    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons=reasons,
        smoke_datasets=list(datasets),
        smoke_limit=smoke_limit,
        tests_run=[],
        git_commit=get_git_commit(),
    )
    _write_json(output_root / "asset_completion_summary.json", summary)


def merge_top_level_sidecar_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    sidecar_path = Path(str(path) + ".json")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    sidecar.update(metadata)
    sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run_smoke(
            registry_path=args.registry,
            output_root=args.output_root,
            stage0_root=args.stage0_root,
            datasets=args.datasets,
            smoke_limit=args.smoke_limit,
            device=args.device,
        )
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
