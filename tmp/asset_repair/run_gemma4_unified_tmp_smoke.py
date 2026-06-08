#!/usr/bin/env python3
"""Tmp-only Gemma4 Unified smoke-path probe.

This script does not touch the production model wrapper path. It validates the
local Gemma4 12B Unified asset, verifies image input wiring with
enable_thinking=False, and only attempts model execution when the installed
Transformers runtime has a real gemma4_unified implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args, read_json


ALIAS = "gemma-4-12b-it"
FAMILY = "gemma4_unified"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/gemma-4-12B-it")
DEFAULT_STAGE0_ROOT = Path("outputs/stage0")
REPORT_JSON = "gemma4_unified_tmp_smoke_report.json"
REPORT_MD = "gemma4_unified_tmp_smoke_report.md"
PROMPT_TEMPLATE_ID = "gemma4_unified_chat_enable_thinking_false_v1"
EXPECTED_SHA256 = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"
IMAGE_TOKEN_ID = 258880


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_safetensors_keys(path: Path) -> list[str]:
    try:
        from safetensors import safe_open

        with safe_open(path, framework="pt", device="cpu") as handle:
            return list(handle.keys())
    except Exception:
        return []


def _active_moe_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null"}
    return True


def _active_moe_indicators(payload: object, prefix: str = "") -> list[str]:
    keys = {
        "enable_moe_block",
        "num_experts",
        "n_routed_experts",
        "experts_per_tok",
        "num_local_experts",
        "router_aux_loss_coef",
        "moe_intermediate_size",
        "top_k_experts",
    }
    found: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            label = key if not prefix else f"{prefix}.{key}"
            if (key.lower() in keys or key.lower() == "moe") and _active_moe_value(value):
                found.append(label)
            found.extend(_active_moe_indicators(value, label))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found.extend(_active_moe_indicators(value, f"{prefix}[{index}]"))
    return found


def _resolve_text_config(config: Mapping[str, object]) -> Mapping[str, object]:
    text_config = config.get("text_config")
    return text_config if isinstance(text_config, Mapping) else config


def inspect_local_asset(local_path: Path) -> dict[str, object]:
    config = read_json(local_path / "config.json")
    processor_config = read_json(local_path / "processor_config.json")
    tokenizer_config = read_json(local_path / "tokenizer_config.json")
    text_config = _resolve_text_config(config)
    weight_path = local_path / "model.safetensors"
    keys = inspect_safetensors_keys(weight_path)
    sha = sha256_file(weight_path) if weight_path.is_file() else ""
    prefix_counts = {
        "model.language_model": sum(key.startswith("model.language_model.") for key in keys),
        "model.vision_embedder": sum(key.startswith("model.vision_embedder.") for key in keys),
        "model.vision_tower": sum(key.startswith("model.vision_tower.") for key in keys),
        "model.audio_tower": sum(key.startswith("model.audio_tower.") for key in keys),
    }
    return {
        "local_path": str(local_path),
        "path_exists": local_path.exists(),
        "config_exists": (local_path / "config.json").is_file(),
        "processor_config_exists": (local_path / "processor_config.json").is_file(),
        "tokenizer_config_exists": (local_path / "tokenizer_config.json").is_file(),
        "chat_template_exists": (local_path / "chat_template.jinja").is_file(),
        "model_safetensors_exists": weight_path.is_file(),
        "model_safetensors_sha256": sha,
        "sha256_matches_expected_upload": sha == EXPECTED_SHA256,
        "tensor_key_count": len(keys),
        "sample_tensor_keys": keys[:10],
        "prefix_counts": prefix_counts,
        "checkpoint_looks_unified": prefix_counts["model.language_model"] > 0 and prefix_counts["model.vision_embedder"] > 0,
        "model_type": config.get("model_type"),
        "architectures": config.get("architectures", []),
        "processor_class": processor_config.get("processor_class") or tokenizer_config.get("processor_class"),
        "image_processor_type": (processor_config.get("image_processor") or {}).get("image_processor_type")
        if isinstance(processor_config.get("image_processor"), Mapping)
        else None,
        "total_layers": text_config.get("num_hidden_layers"),
        "hidden_size": text_config.get("hidden_size"),
        "active_moe_indicators": _active_moe_indicators(config),
    }


def inspect_transformers_runtime(local_path: Path) -> dict[str, object]:
    runtime: dict[str, object] = {"python_executable": sys.executable}
    try:
        import transformers

        runtime["transformers_version"] = getattr(transformers, "__version__", "unknown")
        runtime["transformers_file"] = getattr(transformers, "__file__", "")
        runtime["has_gemma4_unified_module"] = importlib.util.find_spec("transformers.models.gemma4_unified") is not None
        runtime["has_gemma4_unified_model_class"] = hasattr(transformers, "Gemma4UnifiedForConditionalGeneration")
        runtime["has_gemma4_unified_processor_class"] = hasattr(transformers, "Gemma4UnifiedProcessor")
        runtime["has_gemma4_model_class"] = hasattr(transformers, "Gemma4ForConditionalGeneration")
        runtime["has_gemma4_processor_class"] = hasattr(transformers, "Gemma4Processor")
        try:
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(local_path, local_files_only=True)
            runtime["auto_config_status"] = "loaded"
            runtime["auto_config_class"] = type(config).__name__
            runtime["auto_config_model_type"] = getattr(config, "model_type", "")
        except Exception as error:
            runtime["auto_config_status"] = "failed"
            runtime["auto_config_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        runtime["transformers_import_error"] = f"{type(error).__name__}: {error}"
        runtime["has_gemma4_unified_module"] = False
        runtime["has_gemma4_unified_model_class"] = False
        runtime["has_gemma4_unified_processor_class"] = False
        runtime["auto_config_status"] = "failed"
    return runtime


def _load_unified_processor(local_path: Path):
    from transformers import AutoProcessor, Gemma4UnifiedProcessor

    processor = AutoProcessor.from_pretrained(local_path, local_files_only=True)
    processor_class = type(processor).__name__
    image_processor_class = type(getattr(processor, "image_processor", None)).__name__
    if not isinstance(processor, Gemma4UnifiedProcessor) or image_processor_class != "Gemma4UnifiedImageProcessor":
        raise RuntimeError(
            "Gemma4 Unified tmp smoke requires Gemma4UnifiedProcessor and "
            f"Gemma4UnifiedImageProcessor, got {processor_class}/{image_processor_class}"
        )
    return processor


def _shape_list(value: object) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def validate_unified_processor_batch(batch: Mapping[str, object]) -> dict[str, object]:
    """Validate that the processor emitted Gemma4 Unified raw merged patches."""
    missing = [key for key in ("input_ids", "pixel_values", "image_position_ids", "mm_token_type_ids") if key not in batch]
    if missing:
        return {"status": "blocked", "reason": "processor batch missing keys: " + ", ".join(missing)}
    pixel_shape = _shape_list(batch["pixel_values"])
    if not pixel_shape or len(pixel_shape) != 3:
        return {"status": "blocked", "reason": f"pixel_values must be rank 3, got shape {pixel_shape}"}
    if pixel_shape[-1] != 6912:
        return {
            "status": "blocked",
            "reason": (
                "Gemma4 Unified model expects raw merged image patches with last dimension 6912; "
                f"processor emitted pixel_values shape {pixel_shape}"
            ),
        }
    image_position_shape = _shape_list(batch["image_position_ids"])
    if not image_position_shape or image_position_shape[:2] != pixel_shape[:2] or image_position_shape[-1] != 2:
        return {
            "status": "blocked",
            "reason": (
                "image_position_ids must align with pixel_values as [batch, num_patches, 2]; "
                f"got image_position_ids={image_position_shape}, pixel_values={pixel_shape}"
            ),
        }
    return {
        "status": "ok",
        "reason": "",
        "pixel_values_shape": pixel_shape,
        "image_position_ids_shape": image_position_shape,
        "input_ids_shape": _shape_list(batch["input_ids"]),
        "mm_token_type_ids_shape": _shape_list(batch["mm_token_type_ids"]),
    }


def _load_first_record(stage0_root: Path) -> dict[str, object]:
    path = stage0_root / "normalized" / "pope" / "popular.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        return json.loads(handle.readline())


def inspect_processor_wiring(local_path: Path, stage0_root: Path = DEFAULT_STAGE0_ROOT) -> dict[str, object]:
    try:
        from PIL import Image

        processor = _load_unified_processor(local_path)
        record = _load_first_record(stage0_root)
        image_path = Path(str(record["image_path"]))
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": str(record["question"])},
                ],
            }
        ]
        prompt = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        batch = processor(images=[image], text=[prompt], return_tensors="pt")
        image_token_count = int((batch["input_ids"] == IMAGE_TOKEN_ID).sum().item())
        batch_validation = validate_unified_processor_batch(batch)
        wired = image_token_count > 0 and batch_validation["status"] == "ok"
        return {
            "status": "processor_wired" if wired else "blocked",
            "reason": "" if wired else str(batch_validation.get("reason") or "image token missing"),
            "processor_class": type(processor).__name__,
            "tokenizer_class": type(processor.tokenizer).__name__,
            "image_processor_class": type(processor.image_processor).__name__,
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "prompt_template_text": "Gemma4 chat template with one image item and normalized question text; enable_thinking=False; add_generation_prompt=True.",
            "enable_thinking": False,
            "contains_image_token": "<|image|>" in prompt,
            "contains_think_token": "<|think|>" in prompt,
            "image_token_count": image_token_count,
            "input_ids_shape": _shape_list(batch["input_ids"]),
            "pixel_values_shape": _shape_list(batch["pixel_values"]) if "pixel_values" in batch else None,
            "image_position_ids_shape": _shape_list(batch["image_position_ids"]) if "image_position_ids" in batch else None,
            "mm_token_type_ids_shape": _shape_list(batch["mm_token_type_ids"]) if "mm_token_type_ids" in batch else None,
            "batch_validation": batch_validation,
            "sample_id": record.get("sample_id"),
            "image_path": str(image_path),
            "question": record.get("question"),
        }
    except Exception as error:
        return {"status": "blocked", "reason": f"{type(error).__name__}: {error}", "enable_thinking": False}


def _runtime_supports_unified(runtime: Mapping[str, object]) -> bool:
    return bool(
        runtime.get("has_gemma4_unified_module")
        and runtime.get("has_gemma4_unified_model_class")
        and runtime.get("has_gemma4_unified_processor_class")
    )


def non_unified_class_incompatibility(asset: Mapping[str, object], runtime: Mapping[str, object]) -> dict[str, object]:
    prefix_counts = asset.get("prefix_counts")
    if not isinstance(prefix_counts, Mapping):
        prefix_counts = {}
    checkpoint_uses_unified_prefixes = bool(
        prefix_counts.get("model.language_model", 0)
        and prefix_counts.get("model.vision_embedder", 0)
        and not prefix_counts.get("model.vision_tower", 0)
    )
    return {
        "has_non_unified_gemma4_class": bool(runtime.get("has_gemma4_model_class")),
        "has_unified_runtime_class": bool(runtime.get("has_gemma4_unified_model_class")),
        "checkpoint_uses_unified_prefixes": checkpoint_uses_unified_prefixes,
        "checkpoint_prefix_counts": dict(prefix_counts),
        "reason": (
            "The installed Gemma4ForConditionalGeneration maps to model_type gemma4, while this checkpoint declares "
            "gemma4_unified and stores weights under model.language_model plus model.vision_embedder. Loading it with "
            "the non-unified class initializes missing tower weights and cannot be treated as a valid hidden-state path."
        ),
    }


def _write_report(output_root: Path, report: Mapping[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / REPORT_JSON
    md_path = output_root / REPORT_MD
    payload = dict(report)
    payload["report_json"] = str(json_path)
    payload["report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Gemma4 Unified Tmp Smoke Report",
                "",
                f"- status: {payload.get('status')}",
                f"- runnable: {payload.get('runnable')}",
                f"- reason: {payload.get('reason')}",
                f"- family: {payload.get('family')}",
                f"- thinking_disabled: {payload.get('thinking_disabled')}",
                "",
                "```json",
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_supported_smoke(*, local_path: Path, output_root: Path, stage0_root: Path, device: str, smoke_limit: int) -> dict[str, object]:
    """Run a tiny hidden-state smoke only when gemma4_unified runtime exists."""
    import torch
    from PIL import Image
    from transformers import AutoModelForMultimodalLM

    processor = _load_unified_processor(local_path)
    records: list[dict[str, object]] = []
    source_path = stage0_root / "normalized" / "pope" / "popular.jsonl"
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if len(records) >= smoke_limit:
                break
            records.append(json.loads(line))
    model = AutoModelForMultimodalLM.from_pretrained(
        local_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    cache_dir = output_root / "gemma4_unified_tmp_smoke_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, object]] = []
    total_layers = 48
    hidden_dim = 3840
    for record in records:
        image_path = Path(str(record["image_path"]))
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        image = Image.open(image_path).convert("RGB")
        prompt = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": str(record["question"])}]}],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
        batch = processor(images=[image], text=[prompt], return_tensors="pt")
        batch_validation = validate_unified_processor_batch(batch)
        if batch_validation["status"] != "ok":
            raise RuntimeError(str(batch_validation["reason"]))
        inputs = {key: value.to(device) for key, value in batch.items()}
        with torch.inference_mode():
            output = model(**inputs, output_hidden_states=True, return_dict=True, use_cache=False, logits_to_keep=1)
        hidden_states = output.hidden_states
        if len(hidden_states) == total_layers + 1:
            hidden_state_index_offset = 1
        elif len(hidden_states) == total_layers:
            hidden_state_index_offset = 0
        else:
            raise RuntimeError(f"unexpected hidden_states length: {len(hidden_states)}")
        vectors = torch.stack(
            [hidden_states[index + hidden_state_index_offset][0, -1, :].detach().float().cpu() for index in range(total_layers)]
        )
        logits = output.logits[0, -1, :].detach().float().cpu()
        token_id = int(torch.argmax(logits).item())
        answer_text = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        entries.append(
            {
                "sample_id": record.get("sample_id"),
                "image_id": record.get("image_id"),
                "image_path": str(image_path),
                "question": record.get("question"),
                "label": record.get("label"),
                "object_name": record.get("object_name"),
                "source_dataset": record.get("source_dataset"),
                "subset": record.get("subset"),
                "answer_text": answer_text,
                "parsed_answer": None,
                "first_token_logits": logits,
                "selected_layers": list(range(total_layers)),
                "layer_vectors": vectors,
                "model_name": ALIAS,
                "model_family": FAMILY,
                "token_index": -1,
                "prompt_template_id": PROMPT_TEMPLATE_ID,
                "hidden_state_index_offset": hidden_state_index_offset,
            }
        )
    shard_path = cache_dir / "shard-00000.pt"
    torch.save(entries, shard_path)
    sidecar = {
        "model_alias": ALIAS,
        "model_family": FAMILY,
        "local_path": str(local_path),
        "wrapper_class": "tmp_gemma4_unified_smoke",
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "total_layers": total_layers,
        "hidden_dim": hidden_dim,
        "hidden_state_index_offset": entries[0]["hidden_state_index_offset"] if entries else None,
        "selected_layers": list(range(total_layers)),
        "token_index": -1,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "deterministic_generation_kwargs": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
        "thinking_disabled": True,
        "trust_remote_code": False,
        "validation_commit": "",
        "note": "tmp-only smoke cache; not production wrapper output",
    }
    (cache_dir / "shard-00000.pt.json").write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "verified_tmp_smoke", "smoke_cache": str(cache_dir), "num_entries": len(entries), "sidecar": sidecar}


def run_tmp_smoke(
    *,
    local_path: Path = LOCAL_PATH,
    output_root: Path = Path("outputs/assets/repair"),
    stage0_root: Path = DEFAULT_STAGE0_ROOT,
    execute: bool = False,
    device: str = "cuda:0",
    smoke_limit: int = 2,
) -> dict[str, object]:
    asset = inspect_local_asset(local_path)
    runtime = inspect_transformers_runtime(local_path)
    processor_wiring = inspect_processor_wiring(local_path, stage0_root)
    status = "ready_for_execute"
    reason = "Gemma4 Unified runtime support exists; execute mode can attempt tmp smoke."
    runnable = False

    if not asset.get("path_exists"):
        status = "blocked_missing_local_path"
        reason = f"local path does not exist: {local_path}"
    elif asset.get("model_type") != FAMILY:
        status = "blocked_wrong_family"
        reason = f"config model_type is not {FAMILY}: {asset.get('model_type')}"
    elif asset.get("active_moe_indicators"):
        status = "blocked_policy_moe"
        reason = "active MoE indicators detected: " + ", ".join(str(value) for value in asset["active_moe_indicators"])
    elif processor_wiring.get("status") != "processor_wired":
        status = "blocked_processor_wiring"
        reason = str(processor_wiring.get("reason") or "processor did not wire image input")
    elif not _runtime_supports_unified(runtime):
        status = "blocked_missing_transformers_gemma4_unified_support"
        reason = (
            "installed Transformers does not expose a gemma4_unified implementation. "
            f"AutoConfig status={runtime.get('auto_config_status')}; error={runtime.get('auto_config_error', '')}"
        )
    elif not execute:
        runnable = True
    else:
        try:
            smoke = run_supported_smoke(
                local_path=local_path,
                output_root=output_root,
                stage0_root=stage0_root,
                device=device,
                smoke_limit=smoke_limit,
            )
            status = str(smoke["status"])
            reason = "tmp Gemma4 Unified smoke completed"
            runnable = status == "verified_tmp_smoke"
        except Exception as error:
            status = "blocked_tmp_smoke_failed"
            reason = f"{type(error).__name__}: {error}"

    report: dict[str, object] = {
        "alias": ALIAS,
        "family": FAMILY,
        "status": status,
        "runnable": runnable,
        "mode": "execute" if execute else "dry_run",
        "reason": reason,
        "local_path": str(local_path),
        "asset": asset,
        "runtime": runtime,
        "non_unified_class_incompatibility": non_unified_class_incompatibility(asset, runtime),
        "processor_wiring": processor_wiring,
        "thinking_disabled": processor_wiring.get("enable_thinking") is False,
        "disable_thinking_kwargs": {"enable_thinking": False},
        "vision_tower_check_used": False,
        "required_validation_if_runnable": [
            "deterministic repeat check",
            "same question with different images must produce different prefill trajectory",
            "full-layer hidden-state extraction",
            "non-constant layer trajectory",
            "finite first-token logits",
        ],
    }
    _write_report(output_root, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--local-path", type=Path, default=LOCAL_PATH)
    parser.add_argument("--stage0-root", type=Path, default=DEFAULT_STAGE0_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke-limit", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_tmp_smoke(
        local_path=args.local_path,
        output_root=args.output_root,
        stage0_root=args.stage0_root,
        execute=bool(args.execute),
        device=args.device,
        smoke_limit=args.smoke_limit,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
