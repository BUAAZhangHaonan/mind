#!/usr/bin/env python3
"""Run a tmp-only LLaVA-v1.5 HF smoke hidden-state extraction.

This script is not part of the production asset wrapper path. It exists only to
test whether the complete local HF LLaVA-v1.5 7B asset can run deterministic
single-image smoke extraction and expose pre-generation hidden states.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args, normalize_mode, read_json


ALIAS = "llava-v1.5-7b"
FAMILY = "llava_v15_tmp"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/llava-1.5-7b-hf")
PROMPT_TEMPLATE_ID = "tmp_llava_v15_single_image_chat_v1"
PROMPT_TEMPLATE_TEXT = (
    "LLaVA-v1.5 chat template receives one image content item followed by the "
    "normalized question text unchanged, with add_generation_prompt=True."
)
DATASET_SPECS = {
    "pope": ("pope", "popular", Path("normalized/pope/popular.jsonl")),
    "repope": ("repope", "popular", Path("normalized/repope/popular.jsonl")),
    "dash-b": ("dash-b", "all", Path("normalized/dash-b/all.jsonl")),
}
YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no_answer(text: str) -> int | None:
    match = YES_NO_PATTERN.search(text)
    if match is None:
        return None
    return 1 if match.group(1).lower() == "yes" else 0


def get_git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        return "unknown"
    if completed.returncode != 0:
        return "unknown"
    return completed.stdout.strip() or "unknown"


def build_prompt(processor: Any, question: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    if not hasattr(processor, "apply_chat_template"):
        raise ValueError("LLaVA processor does not expose apply_chat_template")
    return str(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))


def resolve_total_layers_and_hidden_dim(config: Any) -> tuple[int, int]:
    text_config = getattr(config, "text_config", None)
    total_layers = getattr(text_config, "num_hidden_layers", None)
    hidden_dim = getattr(text_config, "hidden_size", None)
    if total_layers is None:
        total_layers = getattr(config, "num_hidden_layers", None)
    if hidden_dim is None:
        hidden_dim = getattr(config, "hidden_size", None)
    if not isinstance(total_layers, int) or total_layers <= 0:
        raise ValueError("total_layers cannot be resolved from LLaVA config")
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("hidden_dim cannot be resolved from LLaVA config")
    return total_layers, hidden_dim


def select_full_layer_vectors(
    hidden_states: Sequence[Any],
    *,
    token_index: int,
    total_layers: int,
    hidden_dim: int,
):
    import torch

    hidden_state_count = len(hidden_states)
    if hidden_state_count == total_layers + 1:
        offset = 1
    elif hidden_state_count == total_layers:
        offset = 0
    else:
        raise ValueError(
            "hidden_state_index_offset cannot be determined: "
            f"hidden_state_count={hidden_state_count}, total_layers={total_layers}"
        )
    selected_indices = [layer + offset for layer in range(total_layers)]
    vectors = []
    for index in selected_indices:
        tensor = hidden_states[index]
        if tensor.ndim != 3:
            raise ValueError(f"hidden state {index} must have ndim=3, got {tensor.ndim}")
        vector = tensor[0, token_index, :].detach().float().cpu()
        if int(vector.numel()) != hidden_dim:
            raise ValueError(f"hidden state {index} hidden_dim mismatch: expected {hidden_dim}, got {vector.numel()}")
        vectors.append(vector)
    stacked = torch.stack(vectors, dim=0)
    if not torch.isfinite(stacked).all():
        raise ValueError("layer_vectors contain non-finite values")
    return stacked, selected_indices


def hidden_state_index_offset(selected_indices: Sequence[int]) -> int:
    if not selected_indices:
        raise ValueError("selected layer hidden-state indices are empty")
    first = int(selected_indices[0])
    if first not in (0, 1):
        raise ValueError(f"hidden_state_index_offset must be 0 or 1, got {first}")
    return first


def deterministic_generation_kwargs() -> dict[str, object]:
    return {
        "max_new_tokens": 1,
        "do_sample": False,
        "temperature": 0,
        "return_dict_in_generate": True,
        "output_scores": True,
        "output_hidden_states": True,
    }


def build_sidecar_metadata(
    *,
    dataset_name: str,
    subset: str,
    total_layers: int,
    hidden_dim: int,
    hidden_state_count: int,
    selected_indices: Sequence[int],
    token_index: int,
    processor_class: str,
    model_class: str,
    local_path: str,
) -> dict[str, object]:
    offset = hidden_state_index_offset(selected_indices)
    return {
        "stage": "tmp_asset_repair",
        "cache_type": "tmp_llava_v15_prefill_smoke",
        "model_alias": ALIAS,
        "model_name": ALIAS,
        "model_family": FAMILY,
        "local_path": local_path,
        "wrapper_class": "TmpLlavaV15SmokeRunner",
        "processor_class": processor_class,
        "model_class": model_class,
        "dataset_name": dataset_name,
        "source_dataset": dataset_name,
        "subset": subset,
        "split": subset,
        "total_layers": int(total_layers),
        "hidden_dim": int(hidden_dim),
        "hidden_state_index_offset": int(offset),
        "hidden_state_count": int(hidden_state_count),
        "hidden_state_index_offset_source": "tmp forward output hidden_state_count",
        "selected_layer_hidden_state_indices": [int(value) for value in selected_indices],
        "selected_layers": list(range(int(total_layers))),
        "token_index": int(token_index),
        "max_new_tokens": 1,
        "dtype": "float16",
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "prompt_template_text": PROMPT_TEMPLATE_TEXT,
        "deterministic_generation_kwargs": deterministic_generation_kwargs(),
        "thinking_disabled": True,
        "trust_remote_code": False,
        "validation_commit": get_git_commit(),
        "script": "tmp/asset_repair/run_llava_v15_tmp_smoke.py",
        "git_commit": get_git_commit(),
    }


def load_records(stage0_root: Path, dataset_keys: Sequence[str], smoke_limit: int) -> dict[tuple[str, str], list[dict[str, object]]]:
    records: dict[tuple[str, str], list[dict[str, object]]] = {}
    for key in dataset_keys:
        dataset_name, subset, rel_path = DATASET_SPECS[key]
        path = stage0_root / rel_path
        if not path.is_file():
            raise FileNotFoundError(f"missing smoke source file: {path}")
        rows: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= smoke_limit:
                break
        if len(rows) < smoke_limit:
            raise ValueError(f"{path} contains {len(rows)} records, need {smoke_limit}")
        records[(dataset_name, subset)] = rows
    return records


def resolve_image_path(repo_root: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"missing image file: {candidate}")
    return candidate


def move_to_device(batch: Any, device: str) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    raise TypeError(f"unsupported processor batch type: {type(batch).__name__}")


def resolve_token_index(model_inputs: Mapping[str, Any]) -> int:
    import torch

    attention_mask = model_inputs.get("attention_mask")
    if attention_mask is not None:
        nonzero = torch.nonzero(attention_mask[0], as_tuple=False).flatten()
        if len(nonzero) > 0:
            return int(nonzero[-1].item())
    input_ids = model_inputs.get("input_ids")
    if input_ids is None:
        raise ValueError("model_inputs missing input_ids")
    return int(input_ids.shape[-1] - 1)


def decode_generation(processor: Any, sequences: Any, input_ids: Any) -> str:
    prompt_length = int(input_ids.shape[-1])
    continuation = sequences[:, prompt_length:]
    decoded = processor.batch_decode(
        continuation.tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return str(decoded[0]).strip()


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_markdown(path: Path, report: Mapping[str, object]) -> None:
    lines = [
        "# LLaVA-v1.5 Tmp Smoke Report",
        "",
        f"- status: {report.get('status')}",
        f"- reason: {report.get('reason')}",
        f"- local_path: {report.get('local_path')}",
        "",
        "```json",
        json.dumps(report, indent=2, sort_keys=True, default=str),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields = ("model_alias", "dataset", "subset", "status", "reason", "shard_path", "sidecar_path", "num_records")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def tensor_checksum(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def inspect_local_asset(local_path: Path) -> dict[str, object]:
    config = read_json(local_path / "config.json")
    processor_config = read_json(local_path / "processor_config.json")
    preprocessor_config = read_json(local_path / "preprocessor_config.json")
    index = read_json(local_path / "model.safetensors.index.json")
    weight_map = index.get("weight_map") if isinstance(index.get("weight_map"), dict) else {}
    shards = sorted({str(value) for value in weight_map.values()}) if isinstance(weight_map, dict) else []
    missing = []
    if not local_path.is_dir():
        missing.append("local_path")
    if not config:
        missing.append("config.json")
    if not processor_config:
        missing.append("processor_config.json")
    if not preprocessor_config:
        missing.append("preprocessor_config.json")
    if not any((local_path / name).is_file() for name in ("tokenizer.json", "tokenizer.model")):
        missing.append("tokenizer files")
    if not index:
        missing.append("model.safetensors.index.json")
    if shards and not all((local_path / shard).is_file() for shard in shards):
        missing.append("referenced model shards")
    if isinstance(weight_map, dict) and not any("vision_tower" in key or "vision_model" in key for key in weight_map):
        missing.append("vision tower weights in index")
    return {
        "local_path_exists": local_path.is_dir(),
        "missing": missing,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures"),
        "processor_class": processor_config.get("processor_class"),
        "image_processor_type": preprocessor_config.get("image_processor_type"),
        "num_shards": len(shards),
        "has_onevision_marker": any(
            "onevision" in (local_path / name).read_text(encoding="utf-8", errors="ignore").lower()
            for name in ("config.json", "processor_config.json", "preprocessor_config.json")
            if (local_path / name).is_file()
        ),
    }


def dry_run_report(*, local_path: Path, stage0_root: Path, datasets: Sequence[str], smoke_limit: int) -> dict[str, object]:
    inspection = inspect_local_asset(local_path)
    reason = "tmp smoke can be attempted"
    status = "ready_for_execute"
    if inspection["missing"]:
        status = "blocked"
        reason = "missing " + ", ".join(str(item) for item in inspection["missing"])
    try:
        load_records(stage0_root, datasets, smoke_limit)
    except Exception as error:
        status = "blocked"
        reason = str(error)
    return {
        "model_alias": ALIAS,
        "status": status,
        "reason": reason,
        "mode": "dry_run",
        "local_path": str(local_path),
        "datasets": list(datasets),
        "smoke_limit": smoke_limit,
        "inspection": inspection,
    }


def run_tmp_smoke(
    *,
    local_path: Path = LOCAL_PATH,
    stage0_root: Path = Path("outputs/stage0"),
    output_root: Path = Path("outputs/assets/repair"),
    datasets: Sequence[str] = ("pope", "repope", "dash-b"),
    smoke_limit: int = 2,
    device: str = "cuda:0",
    execute: bool = False,
    allow_cpu: bool = False,
) -> dict[str, object]:
    if smoke_limit < 1 or smoke_limit > 2:
        raise ValueError("tmp LLaVA smoke only permits smoke_limit in [1, 2]")
    unknown = [dataset for dataset in datasets if dataset not in DATASET_SPECS]
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")

    output_root.mkdir(parents=True, exist_ok=True)
    report_json = output_root / "llava_v15_tmp_smoke_report.json"
    report_md = output_root / "llava_v15_tmp_smoke_report.md"
    if not execute:
        report = dry_run_report(local_path=local_path, stage0_root=stage0_root, datasets=datasets, smoke_limit=smoke_limit)
        report["report_json"] = str(report_json)
        report["report_markdown"] = str(report_md)
        write_json(report_json, report)
        write_markdown(report_md, report)
        return report

    if device == "cpu" and not allow_cpu:
        raise ValueError("CPU execution requires explicit --allow-cpu")

    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError(f"requested {device}, but CUDA is not available")

    records_by_pair = load_records(stage0_root, datasets, smoke_limit)
    config = AutoConfig.from_pretrained(local_path, local_files_only=True)
    total_layers, hidden_dim = resolve_total_layers_and_hidden_dim(config)
    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)

    load_kwargs: dict[str, object] = {
        "local_files_only": True,
        "torch_dtype": torch.float16,
        "low_cpu_mem_usage": True,
    }
    if device.startswith("cuda"):
        load_kwargs["device_map"] = {"": device}
    model = LlavaForConditionalGeneration.from_pretrained(local_path, **load_kwargs)
    model.eval()
    if not device.startswith("cuda"):
        model.to(device)

    rows: list[dict[str, object]] = []
    checksums: dict[str, object] = {"entries": {}}
    repo_root = Path.cwd()
    with torch.inference_mode():
        for (dataset_name, subset), records in records_by_pair.items():
            entries: list[dict[str, object]] = []
            selected_indices: list[int] | None = None
            token_index_for_sidecar: int | None = None
            for record in records:
                image_path = resolve_image_path(repo_root, str(record["image_path"]))
                image = Image.open(image_path).convert("RGB")
                prompt = build_prompt(processor, str(record["question"]))
                batch = processor(text=[prompt], images=[image], return_tensors="pt")
                model_inputs = move_to_device(batch, device)
                token_index = resolve_token_index(model_inputs)
                generation = model.generate(**model_inputs, **deterministic_generation_kwargs())
                outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
                hidden_states = getattr(outputs, "hidden_states", None)
                if not hidden_states:
                    raise ValueError("forward output did not include hidden_states")
                layer_vectors, current_indices = select_full_layer_vectors(
                    hidden_states,
                    token_index=token_index,
                    total_layers=total_layers,
                    hidden_dim=hidden_dim,
                )
                logits = outputs.logits[0, token_index, :].detach().float().cpu()
                answer_text = decode_generation(processor, generation.sequences, model_inputs["input_ids"])
                selected_indices = current_indices
                token_index_for_sidecar = token_index
                entry = {
                    "sample_id": record.get("sample_id"),
                    "image_id": record.get("image_id"),
                    "image_path": record.get("image_path"),
                    "question": record.get("question"),
                    "label": record.get("label"),
                    "object_name": record.get("object_name"),
                    "source_dataset": record.get("source_dataset"),
                    "subset": subset,
                    "answer_text": answer_text,
                    "parsed_answer": parse_yes_no_answer(answer_text),
                    "first_token_logits": logits,
                    "selected_layers": list(range(total_layers)),
                    "layer_vectors": layer_vectors,
                    "model_name": ALIAS,
                    "model_family": FAMILY,
                    "token_index": int(token_index),
                    "prompt_template_id": PROMPT_TEMPLATE_ID,
                }
                entries.append(entry)
                checksums["entries"][f"{dataset_name}/{subset}/{record.get('sample_id')}"] = {
                    "layer_vectors_sha256": tensor_checksum(layer_vectors),
                    "first_token_logits_sha256": tensor_checksum(logits),
                    "answer_text": answer_text,
                    "parsed_answer": entry["parsed_answer"],
                }

            if selected_indices is None or token_index_for_sidecar is None:
                raise ValueError(f"no entries produced for {dataset_name}/{subset}")
            shard_path = output_root / "llava_v15_tmp_smoke_cache" / ALIAS / dataset_name / subset / "shard-00000.pt"
            sidecar_path = Path(str(shard_path) + ".json")
            shard_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(entries, shard_path)
            sidecar = build_sidecar_metadata(
                dataset_name=dataset_name,
                subset=subset,
                total_layers=total_layers,
                hidden_dim=hidden_dim,
                hidden_state_count=total_layers + hidden_state_index_offset(selected_indices),
                selected_indices=selected_indices,
                token_index=token_index_for_sidecar,
                processor_class=type(processor).__name__,
                model_class=type(model).__name__,
                local_path=str(local_path),
            )
            write_json(sidecar_path, sidecar)
            rows.append(
                {
                    "model_alias": ALIAS,
                    "dataset": dataset_name,
                    "subset": subset,
                    "status": "verified",
                    "reason": "tmp LLaVA-v1.5 smoke extraction completed",
                    "shard_path": str(shard_path),
                    "sidecar_path": str(sidecar_path),
                    "num_records": len(entries),
                }
            )

    write_csv(output_root / "llava_v15_tmp_smoke_report.csv", rows)
    report = {
        "model_alias": ALIAS,
        "status": "verified",
        "reason": "tmp LLaVA-v1.5 HF smoke extraction completed",
        "mode": "execute",
        "local_path": str(local_path),
        "datasets": list(datasets),
        "smoke_limit": smoke_limit,
        "device": device,
        "total_layers": total_layers,
        "hidden_dim": hidden_dim,
        "rows": rows,
        "checksums": checksums,
        "report_json": str(report_json),
        "report_markdown": str(report_md),
    }
    write_json(report_json, report)
    write_markdown(report_md, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--model-path", type=Path, default=LOCAL_PATH)
    parser.add_argument("--stage0-root", type=Path, default=Path("outputs/stage0"))
    parser.add_argument("--datasets", nargs="+", default=["pope", "repope", "dash-b"], choices=sorted(DATASET_SPECS))
    parser.add_argument("--smoke-limit", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = normalize_mode(build_parser().parse_args(argv))
    run_tmp_smoke(
        local_path=args.model_path,
        stage0_root=args.stage0_root,
        output_root=args.output_root,
        datasets=tuple(args.datasets),
        smoke_limit=args.smoke_limit,
        device=args.device,
        execute=bool(args.execute),
        allow_cpu=bool(args.allow_cpu),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
