#!/usr/bin/env python3
"""Run a tmp-only Phi-4 multimodal smoke hidden-state extraction."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import site
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import add_common_args


ALIAS = "phi-4-multimodal-instruct"
LOCAL_PATH = Path("/home/team/lvshuyang/Models/Phi-4-multimodal-instruct")
REPORT_JSON = "phi4_tmp_smoke_report.json"
REPORT_MD = "phi4_tmp_smoke_report.md"
PROMPT_TEMPLATE_ID = "phi4mm_tmp_single_image_raw_question_v1"


def build_phi4_prompt(question: str) -> str:
    return f"<|user|><|image_1|>{question}<|end|><|assistant|>"


def tmp_loading_plan() -> dict[str, object]:
    return {
        "python_no_user_site_required": True,
        "attention_override": "eager",
        "torch_dtype": "bfloat16",
        "low_cpu_mem_usage": False,
        "device_map": None,
        "peft_prepare_inputs_patch": "Phi4MMModel.prepare_inputs_for_generation",
        "generation_kwargs": {"do_sample": False, "max_new_tokens": 1, "use_cache": False, "num_logits_to_keep": 1},
    }


def parse_yes_no(text: str) -> int | None:
    normalized = text.strip().lower()
    if normalized.startswith("yes"):
        return 1
    if normalized.startswith("no"):
        return 0
    return None


def _remove_user_site_paths() -> list[str]:
    user_site = site.getusersitepackages()
    candidates = [user_site] if isinstance(user_site, str) else list(user_site)
    removed: list[str] = []
    for candidate in candidates:
        for path in list(sys.path):
            if path == candidate or path.startswith(str(candidate) + "/"):
                sys.path.remove(path)
                removed.append(path)
    loaded = sys.modules.get("transformers")
    loaded_file = str(getattr(loaded, "__file__", "")) if loaded is not None else ""
    if loaded_file and any(loaded_file.startswith(str(candidate)) for candidate in candidates):
        raise RuntimeError(f"transformers was already imported from user-site path: {loaded_file}")
    return removed


def _sha256_tensor(tensor: Any) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _read_first_record(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                payload = json.loads(line)
                if isinstance(payload, dict):
                    return payload
    raise ValueError(f"no JSONL records found: {path}")


def _read_first_record_with_different_image(path: Path, image_path: str) -> dict[str, object] | None:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict) and str(payload.get("image_path")) != image_path:
                return payload
    return None


def _resolve_image_path(repo_root: Path, image_path: str) -> Path:
    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if not candidate.is_file():
        raise FileNotFoundError(f"image_path does not exist: {candidate}")
    return candidate


def _patch_phi4_for_peft(local_path: Path) -> bool:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    phi4_model = get_class_from_dynamic_module("modeling_phi4mm.Phi4MMModel", str(local_path), local_files_only=True)
    if hasattr(phi4_model, "prepare_inputs_for_generation"):
        return False

    def _tmp_prepare_inputs_for_generation(self: object, input_ids: Any, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": input_ids, **kwargs}

    phi4_model.prepare_inputs_for_generation = _tmp_prepare_inputs_for_generation
    return True


def _load_bundle(local_path: Path, device: str) -> tuple[Any, Any, dict[str, object]]:
    removed_user_site_paths = _remove_user_site_paths()
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
    import transformers
    import peft

    patch_applied = _patch_phi4_for_peft(local_path)
    processor = AutoProcessor.from_pretrained(str(local_path), trust_remote_code=True, local_files_only=True)
    config = AutoConfig.from_pretrained(str(local_path), trust_remote_code=True, local_files_only=True)
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_pretrained(
        str(local_path),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
        device_map=None,
        config=config,
        attn_implementation="eager",
    ).eval()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device is not available: {device}")
        model = model.to(device)
    else:
        model = model.to(device)
    meta_names = [name for name, parameter in model.named_parameters() if str(parameter.device) == "meta"]
    metadata = {
        "removed_user_site_paths": removed_user_site_paths,
        "transformers_version": transformers.__version__,
        "transformers_file": getattr(transformers, "__file__", ""),
        "peft_version": peft.__version__,
        "processor_class": type(processor).__name__,
        "model_class": type(model).__name__,
        "patch_applied": patch_applied,
        "num_meta_parameters": len(meta_names),
        "meta_parameter_names": meta_names[:20],
        "attn_implementation": getattr(model.config, "_attn_implementation", None),
        "total_layers": int(model.config.num_hidden_layers),
        "hidden_dim": int(model.config.hidden_size),
    }
    return processor, model, metadata


def _prepare_inputs(processor: Any, *, question: str, image_path: Path, device: str) -> tuple[dict[str, Any], dict[str, object]]:
    from PIL import Image
    import torch

    image = Image.open(image_path).convert("RGB")
    prompt = build_phi4_prompt(question)
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    metadata = {
        "prompt": prompt,
        "input_keys": sorted(inputs.keys()),
        "input_shapes": {key: list(value.shape) for key, value in inputs.items() if isinstance(value, torch.Tensor)},
        "input_mode": int(inputs["input_mode"][0].item()),
        "has_image_embeds": "input_image_embeds" in inputs and int(inputs["input_image_embeds"].numel()) > 0,
        "has_image_attention_mask": "image_attention_mask" in inputs and int(inputs["image_attention_mask"].numel()) > 0,
    }
    moved = {key: (value.to(device) if isinstance(value, torch.Tensor) else value) for key, value in inputs.items()}
    return moved, metadata


def _forward_extract(model: Any, processor: Any, inputs: dict[str, Any]) -> dict[str, Any]:
    import torch

    total_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    with torch.no_grad():
        output = model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
            num_logits_to_keep=1,
        )
    hidden_states = output.hidden_states
    if len(hidden_states) != total_layers + 1:
        raise RuntimeError(f"expected {total_layers + 1} hidden-state tensors, got {len(hidden_states)}")
    layer_vectors = torch.stack([hidden_states[index + 1][0, -1, :].detach().float().cpu() for index in range(total_layers)])
    logits = output.logits[0, -1, :].detach().float().cpu()
    next_id = int(torch.argmax(logits).item())
    answer_text = processor.batch_decode(
        torch.tensor([[next_id]]),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "answer_text": answer_text,
        "parsed_answer": parse_yes_no(answer_text),
        "generated_token_id": next_id,
        "selected_layers": list(range(total_layers)),
        "hidden_state_index_offset": 1,
        "layer_vectors": layer_vectors,
        "first_token_logits": logits,
        "layer_vectors_shape": list(layer_vectors.shape),
        "first_token_logits_shape": list(logits.shape),
        "layer_vectors_finite": bool(torch.isfinite(layer_vectors).all().item()),
        "first_token_logits_finite": bool(torch.isfinite(logits).all().item()),
        "hidden_dim": hidden_dim,
        "layer_vectors_checksum": _sha256_tensor(layer_vectors),
        "first_token_logits_checksum": _sha256_tensor(logits),
    }


def _generate_one(model: Any, processor: Any, inputs: dict[str, Any]) -> dict[str, object]:
    import torch

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=1,
            use_cache=False,
            num_logits_to_keep=1,
        )
    new_ids = generated[:, inputs["input_ids"].shape[1] :]
    text = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {"answer_text": text, "parsed_answer": parse_yes_no(text), "token_ids": new_ids.detach().cpu().tolist()}


def _entry_from_result(record: dict[str, object], result: dict[str, Any], metadata: dict[str, object]) -> dict[str, Any]:
    return {
        "sample_id": record.get("sample_id"),
        "image_id": record.get("image_id"),
        "image_path": record.get("image_path"),
        "question": record.get("question"),
        "label": record.get("label"),
        "object_name": record.get("object_name"),
        "source_dataset": record.get("source_dataset"),
        "subset": record.get("subset"),
        "answer_text": result["answer_text"],
        "parsed_answer": result["parsed_answer"],
        "first_token_logits": result["first_token_logits"],
        "selected_layers": result["selected_layers"],
        "layer_vectors": result["layer_vectors"],
        "model_name": ALIAS,
        "model_family": "phi4mm",
        "token_index": -1,
        "prompt_template_id": PROMPT_TEMPLATE_ID,
        "hidden_state_index_offset": result["hidden_state_index_offset"],
        "model_alias": ALIAS,
        "wrapper_class": "tmp_phi4mm_smoke",
        "processor_class": metadata["processor_class"],
        "model_class": metadata["model_class"],
    }


def run_phi4_smoke(
    *,
    local_path: Path = LOCAL_PATH,
    stage0_root: Path = Path("outputs/stage0"),
    output_root: Path = Path("outputs/assets/repair"),
    device: str = "cuda:0",
) -> dict[str, object]:
    import gc
    import torch

    repo_root = Path.cwd()
    source_path = stage0_root / "normalized" / "pope" / "popular.jsonl"
    canary_path = stage0_root / "normalized" / "dash-b" / "all.jsonl"
    record = _read_first_record(source_path)
    canary_record = _read_first_record_with_different_image(canary_path, str(record["image_path"]))
    image_path = _resolve_image_path(repo_root, str(record["image_path"]))
    processor, model, load_metadata = _load_bundle(local_path, device)
    if load_metadata["num_meta_parameters"] != 0:
        raise RuntimeError("meta parameters remain after tmp Phi4 load")
    try:
        inputs, input_metadata = _prepare_inputs(processor, question=str(record["question"]), image_path=image_path, device=device)
        first = _forward_extract(model, processor, inputs)
        repeat = _forward_extract(model, processor, inputs)
        generated = _generate_one(model, processor, inputs)

        repeat_layer_diff = float((first["layer_vectors"] - repeat["layer_vectors"]).abs().max().item())
        repeat_logits_diff = float((first["first_token_logits"] - repeat["first_token_logits"]).abs().max().item())
        canary_status = "skipped_with_reason"
        canary_reason = "no different-image smoke record found"
        canary_diff = None
        if canary_record is not None:
            other_image_path = _resolve_image_path(repo_root, str(canary_record["image_path"]))
            canary_inputs, _ = _prepare_inputs(
                processor,
                question=str(record["question"]),
                image_path=other_image_path,
                device=device,
            )
            canary = _forward_extract(model, processor, canary_inputs)
            canary_diff = float((first["layer_vectors"] - canary["layer_vectors"]).abs().max().item())
            canary_status = "passed" if canary_diff > 0 else "failed"
            canary_reason = "" if canary_diff > 0 else "same-question different-image trajectories are identical"

        cache_dir = output_root / "phi4_tmp_smoke_cache" / "pope" / "popular"
        cache_dir.mkdir(parents=True, exist_ok=True)
        shard_path = cache_dir / "shard-00000.pt"
        sidecar_path = cache_dir / "shard-00000.pt.json"
        entry = _entry_from_result(record, first, load_metadata)
        torch.save([entry], shard_path)
        sidecar = {
            "model_alias": ALIAS,
            "model_family": "phi4mm",
            "local_path": str(local_path),
            "wrapper_class": "tmp_phi4mm_smoke",
            "processor_class": load_metadata["processor_class"],
            "model_class": load_metadata["model_class"],
            "total_layers": load_metadata["total_layers"],
            "hidden_dim": load_metadata["hidden_dim"],
            "hidden_state_index_offset": 1,
            "selected_layers": first["selected_layers"],
            "token_index": -1,
            "prompt_template_id": PROMPT_TEMPLATE_ID,
            "deterministic_generation_kwargs": tmp_loading_plan()["generation_kwargs"],
            "thinking_disabled": None,
            "trust_remote_code": True,
            "validation_scope": "tmp_only_phi4_single_sample",
        }
        sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        status = "verified_tmp"
        blockers: list[str] = []
        if not input_metadata["has_image_embeds"] or int(input_metadata["input_mode"]) != 1:
            blockers.append("processor did not produce vision-mode image inputs")
        if not first["layer_vectors_finite"] or not first["first_token_logits_finite"]:
            blockers.append("non-finite tensors observed")
        if repeat_layer_diff > 1e-3 or repeat_logits_diff > 1e-3:
            blockers.append("determinism tolerance exceeded")
        if canary_status == "failed":
            blockers.append(canary_reason)
        if blockers:
            status = "failed_validation"
        return {
            "status": status,
            "reason": "tmp-only Phi4 smoke passed" if status == "verified_tmp" else "; ".join(blockers),
            "sample_id": record.get("sample_id"),
            "dataset": "pope",
            "subset": "popular",
            "answer_text": generated["answer_text"],
            "parsed_answer": generated["parsed_answer"],
            "forward_argmax_answer_text": first["answer_text"],
            "forward_argmax_parsed_answer": first["parsed_answer"],
            "generated_token_ids": generated["token_ids"],
            "hidden_states_len": int(load_metadata["total_layers"]) + 1,
            "total_layers": load_metadata["total_layers"],
            "hidden_dim": load_metadata["hidden_dim"],
            "hidden_state_index_offset": 1,
            "layer_vectors_shape": first["layer_vectors_shape"],
            "first_token_logits_shape": first["first_token_logits_shape"],
            "layer_vectors_checksum": first["layer_vectors_checksum"],
            "first_token_logits_checksum": first["first_token_logits_checksum"],
            "repeat_layer_vectors_max_abs_diff": repeat_layer_diff,
            "repeat_first_token_logits_max_abs_diff": repeat_logits_diff,
            "image_sensitivity_status": canary_status,
            "image_sensitivity_reason": canary_reason,
            "image_sensitivity_max_abs_diff": canary_diff,
            "input_metadata": input_metadata,
            "load_metadata": load_metadata,
            "shard_path": str(shard_path),
            "sidecar_path": str(sidecar_path),
        }
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()


def _write_report(output_root: Path, report: dict[str, object]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / REPORT_JSON
    md_path = output_root / REPORT_MD
    report["report_json"] = str(json_path)
    report["report_markdown"] = str(md_path)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    md_path.write_text(
        "\n".join(
            [
                "# Phi4 Tmp Smoke Report",
                "",
                f"- status: {report['status']}",
                f"- reason: {report['reason']}",
                "",
                "```json",
                json.dumps(report, indent=2, sort_keys=True, default=str),
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run(
    *,
    local_path: Path = LOCAL_PATH,
    stage0_root: Path = Path("outputs/stage0"),
    output_root: Path = Path("outputs/assets/repair"),
    device: str = "cuda:0",
    execute: bool = False,
) -> dict[str, object]:
    report: dict[str, object] = {
        "alias": ALIAS,
        "mode": "execute" if execute else "dry_run",
        "tmp_only": True,
        "local_path": str(local_path),
        "loading_plan": tmp_loading_plan(),
        "status": "planned",
        "reason": "dry-run only; no model loading was attempted",
    }
    if execute:
        try:
            smoke_result = run_phi4_smoke(
                local_path=local_path,
                stage0_root=stage0_root,
                output_root=output_root,
                device=device,
            )
            report["status"] = smoke_result["status"]
            report["reason"] = smoke_result.get("reason", str(smoke_result["status"]))
            report["smoke_result"] = smoke_result
        except Exception as error:
            report["status"] = "blocked"
            report["reason"] = f"{type(error).__name__}: {error}"
    _write_report(output_root, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--local-path", type=Path, default=LOCAL_PATH)
    parser.add_argument("--stage0-root", type=Path, default=Path("outputs/stage0"))
    parser.add_argument("--device", default="cuda:0")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        local_path=args.local_path,
        stage0_root=args.stage0_root,
        output_root=args.output_root,
        device=args.device,
        execute=bool(args.execute),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
