"""Asset audit and hidden-state validation helpers for Experiment 1."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F

from mind.models.registry import REQUIRED_MODEL_ALIASES, AssetModel
from mind.models.types import parse_yes_no_answer


class AssetStatus(str, Enum):
    VERIFIED = "verified"
    VERIFIED_SEPARATE_ENV = "verified_separate_env"
    BLOCKED = "blocked"
    UNSUPPORTED_BY_POLICY = "unsupported_by_policy"
    UNSUPPORTED_BY_WRAPPER = "unsupported_by_wrapper"
    FAILED_VALIDATION = "failed_validation"
    NOT_ATTEMPTED_DUE_TO_DEPENDENCY = "not_attempted_due_to_dependency"


@dataclass(frozen=True)
class ValidationResult:
    status: str
    reason: str = ""
    details: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class AssetAuditResult:
    alias: str
    status: AssetStatus
    reason: str
    local_path: str
    path_exists: bool
    path_is_directory: bool
    config_exists: bool
    processor_tokenizer_assets: bool
    model_family_detected: str
    architecture_detected: str
    moe_indicators: list[str]
    thinking_detected: bool
    thinking_disable_argument: str | None
    dtype: str
    trust_remote_code_required: bool
    local_loading_class_candidate: str
    image_processor_candidate: str
    total_layers: int | None
    hidden_dim: int | None
    output_hidden_states_support: str
    generation_api_support: str
    hidden_state_index_offset: int | str
    prompt_template_id: str
    moe_policy_decision: str = "not_evaluated"
    moe_indicators_seen: list[str] = field(default_factory=list)
    moe_indicators_ignored_with_reason: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "alias": self.alias,
            "status": self.status.value,
            "reason": self.reason,
            "local_path": self.local_path,
            "path_exists": self.path_exists,
            "path_is_directory": self.path_is_directory,
            "config_exists": self.config_exists,
            "processor_tokenizer_assets": self.processor_tokenizer_assets,
            "model_family_detected": self.model_family_detected,
            "architecture_detected": self.architecture_detected,
            "moe_indicators": self.moe_indicators,
            "thinking_detected": self.thinking_detected,
            "thinking_disable_argument": self.thinking_disable_argument,
            "dtype": self.dtype,
            "trust_remote_code_required": self.trust_remote_code_required,
            "local_loading_class_candidate": self.local_loading_class_candidate,
            "image_processor_candidate": self.image_processor_candidate,
            "total_layers": self.total_layers,
            "hidden_dim": self.hidden_dim,
            "output_hidden_states_support": self.output_hidden_states_support,
            "generation_api_support": self.generation_api_support,
            "hidden_state_index_offset": self.hidden_state_index_offset,
            "prompt_template_id": self.prompt_template_id,
            "moe_policy_decision": self.moe_policy_decision,
            "moe_indicators_seen": self.moe_indicators_seen,
            "moe_indicators_ignored_with_reason": self.moe_indicators_ignored_with_reason,
        }


@dataclass(frozen=True)
class DeterminismPair:
    first: Mapping[str, object]
    second: Mapping[str, object]


MOE_INDICATOR_KEYS = {
    "num_experts",
    "n_routed_experts",
    "experts_per_tok",
    "num_local_experts",
    "router_aux_loss_coef",
    "enable_moe_block",
    "moe_intermediate_size",
    "top_k_experts",
}

SUPPORTED_WRAPPER_FAMILIES = {
    "glm4v": "Glm4vForConditionalGeneration",
    "qwen_vl": "AutoModelForImageTextToText",
    "qwen2_5_vl": "Qwen2_5_VLForConditionalGeneration",
    "llava_onevision": "AutoModelForImageTextToText",
    "qwen3_vl": "AutoModelForImageTextToText",
    "qwen3_5": "Qwen3_5ForConditionalGeneration",
    "internvl": "AutoModel",
    "minicpmv": "MiniCPMV",
    "molmo": "AutoModelForCausalLM",
    "gemma3": "Gemma3ForConditionalGeneration",
    "gemma4": "Gemma4ForConditionalGeneration",
    "gemma4_unified": "AutoModelForMultimodalLM",
    "phi3_v": "Phi3VForCausalLM",
    "phi4mm": "Phi4MMForCausalLM",
    "llava_v15": "LlavaForConditionalGeneration",
}

SINGLE_IMAGE_VLM_FAMILIES = SUPPORTED_WRAPPER_FAMILIES.keys() | {
    "glm4v",
    "gemma3",
    "gemma4",
    "gemma4_unified",
    "qwen3_5",
    "qwen2_5_vl",
    "minicpmv",
    "phi4mm",
    "phi3_v",
    "llava_v15",
    "internvl",
}

UNSUPPORTED_LOCAL_WRAPPER_REASONS: dict[str, str] = {}


def detect_moe_indicators(payload: object, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            label = key if not prefix else f"{prefix}.{key}"
            normalized = key.lower()
            if (normalized in MOE_INDICATOR_KEYS or normalized == "moe") and _moe_indicator_is_active(value):
                found.append(label)
            found.extend(detect_moe_indicators(value, prefix=label))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            label = f"{prefix}[{index}]" if prefix else f"[{index}]"
            found.extend(detect_moe_indicators(value, prefix=label))
    return found


def _moe_indicator_is_active(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "none", "null"}
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def classify_moe_policy(asset: AssetModel, config: Mapping[str, object]) -> tuple[list[str], str, dict[str, str]]:
    active = detect_moe_indicators(config)
    ignored: dict[str, str] = {}
    if asset.family == "gemma4_unified":
        ignored = {
            label: "Gemma4 12B Unified local config carries inactive expert-capacity fields; only active routing fields or the 26B A4B MoE architecture block this family."
            for label in _inactive_moe_indicator_labels(config)
        }
        architectures = config.get("architectures")
        architecture_text = " ".join(str(item) for item in architectures) if isinstance(architectures, list) else str(architectures or "")
        if "26B" in architecture_text or "A4B" in architecture_text:
            active = active or ["architectures"]
    decision = "moe_disallowed" if active and not asset.policy.allow_moe else "non_moe"
    return active, decision, ignored


def _inactive_moe_indicator_labels(payload: object, *, prefix: str = "") -> list[str]:
    labels: list[str] = []
    if isinstance(payload, Mapping):
        for raw_key, value in payload.items():
            key = str(raw_key)
            label = key if not prefix else f"{prefix}.{key}"
            normalized = key.lower()
            if (normalized in MOE_INDICATOR_KEYS or normalized == "moe") and not _moe_indicator_is_active(value):
                labels.append(label)
            labels.extend(_inactive_moe_indicator_labels(value, prefix=label))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            label = f"{prefix}[{index}]" if prefix else f"[{index}]"
            labels.extend(_inactive_moe_indicator_labels(value, prefix=label))
    return labels


def audit_asset_metadata(asset: AssetModel) -> AssetAuditResult:
    local_path = Path(asset.local_path)
    path_exists = local_path.exists()
    path_is_directory = local_path.is_dir()
    config_path = local_path / "config.json"
    config_exists = config_path.is_file()
    empty = _base_audit_result(
        asset,
        path_exists=path_exists,
        path_is_directory=path_is_directory,
        config_exists=config_exists,
        processor_tokenizer_assets=False,
    )
    if not path_exists:
        return _replace_audit(empty, status=AssetStatus.BLOCKED, reason=f"local path does not exist: {local_path}")
    if not path_is_directory:
        return _replace_audit(empty, status=AssetStatus.BLOCKED, reason=f"local path is not a directory: {local_path}")
    if not config_exists:
        return _replace_audit(empty, status=AssetStatus.BLOCKED, reason=f"config.json is missing: {config_path}")

    config = _read_json(config_path)
    processor_tokenizer_assets = _has_processor_or_tokenizer_assets(local_path)
    model_family = str(config.get("model_type") or asset.family)
    architecture = _architecture_name(config)
    moe_indicators, moe_policy_decision, moe_ignored = classify_moe_policy(asset, config)
    thinking_detected = _detect_thinking_markers(local_path)
    total_layers = resolve_total_layers_from_config(config)
    hidden_dim = resolve_hidden_dim_from_config(config)
    if asset.family == "llava_v15":
        total_layers = total_layers or resolve_total_layers_from_weight_index(local_path)
        hidden_dim = hidden_dim or resolve_hidden_dim_from_safetensors_header(local_path)
    loading_class = SUPPORTED_WRAPPER_FAMILIES.get(asset.family, "unsupported")
    output_support = "resolved_by_wrapper" if asset.family in SUPPORTED_WRAPPER_FAMILIES else "unsupported_by_wrapper"
    generation_support = output_support
    image_processor_candidate = _image_processor_candidate(local_path)
    trust_remote_code_required = bool(config.get("auto_map")) or asset.trust_remote_code

    result = AssetAuditResult(
        alias=asset.alias,
        status=AssetStatus.VERIFIED,
        reason="metadata audit passed",
        local_path=str(local_path),
        path_exists=path_exists,
        path_is_directory=path_is_directory,
        config_exists=config_exists,
        processor_tokenizer_assets=processor_tokenizer_assets,
        model_family_detected=model_family,
        architecture_detected=architecture,
        moe_indicators=moe_indicators,
        thinking_detected=thinking_detected,
        thinking_disable_argument=asset.thinking.disable_argument,
        dtype=asset.dtype,
        trust_remote_code_required=trust_remote_code_required,
        local_loading_class_candidate=loading_class,
        image_processor_candidate=image_processor_candidate,
        total_layers=total_layers,
        hidden_dim=hidden_dim,
        output_hidden_states_support=output_support,
        generation_api_support=generation_support,
        hidden_state_index_offset=asset.hidden_state_index_offset,
        prompt_template_id=asset.prompt_template_id,
        moe_policy_decision=moe_policy_decision,
        moe_indicators_seen=moe_indicators,
        moe_indicators_ignored_with_reason=moe_ignored,
    )

    if moe_policy_decision == "moe_disallowed":
        return _replace_audit(
            result,
            status=AssetStatus.UNSUPPORTED_BY_POLICY,
            reason="MoE indicators are disallowed: " + ", ".join(moe_indicators),
        )
    thinking_requires_disable = thinking_detected or asset.thinking.supported is True
    if thinking_requires_disable and not asset.policy.allow_thinking:
        if asset.thinking.disable_argument is None or asset.thinking.disabled_by_default is not True:
            return _replace_audit(
                result,
                status=AssetStatus.UNSUPPORTED_BY_POLICY,
                reason="thinking mode was detected but is not explicitly disabled",
            )
    if asset.family not in SINGLE_IMAGE_VLM_FAMILIES:
        return _replace_audit(result, status=AssetStatus.UNSUPPORTED_BY_POLICY, reason="family is not registered as a single-image VLM")
    if not _deterministic_generation_is_valid(asset):
        return _replace_audit(result, status=AssetStatus.UNSUPPORTED_BY_POLICY, reason="deterministic yes/no generation contract is not satisfied")
    family_result = _audit_family_specific_constraints(asset, local_path, config)
    if family_result is not None:
        status, reason = family_result
        return _replace_audit(result, status=status, reason=reason)
    if not processor_tokenizer_assets:
        return _replace_audit(result, status=AssetStatus.BLOCKED, reason="processor/tokenizer metadata is missing")
    if asset.alias in UNSUPPORTED_LOCAL_WRAPPER_REASONS:
        return _replace_audit(
            result,
            status=AssetStatus.UNSUPPORTED_BY_WRAPPER,
            reason=UNSUPPORTED_LOCAL_WRAPPER_REASONS[asset.alias],
        )
    if asset.family not in SUPPORTED_WRAPPER_FAMILIES:
        return _replace_audit(result, status=AssetStatus.UNSUPPORTED_BY_WRAPPER, reason=f"no Experiment 1 wrapper is implemented for family={asset.family}")
    if total_layers is None:
        return _replace_audit(result, status=AssetStatus.BLOCKED, reason="total_layers could not be resolved from local config")
    if hidden_dim is None:
        return _replace_audit(result, status=AssetStatus.BLOCKED, reason="hidden_dim could not be resolved from local config")
    if asset.hidden_state_index_offset not in (0, 1):
        return _replace_audit(result, status=AssetStatus.FAILED_VALIDATION, reason="hidden_state_index_offset is unknown")
    return result


def resolve_total_layers_from_config(config: Mapping[str, object]) -> int | None:
    for path in (
        ("num_hidden_layers",),
        ("text_config", "num_hidden_layers"),
        ("llm_config", "num_hidden_layers"),
        ("language_config", "num_hidden_layers"),
    ):
        value = _lookup_path(config, path)
        integer = _positive_int_or_none(value)
        if integer is not None:
            return integer
    return None


def resolve_hidden_dim_from_config(config: Mapping[str, object]) -> int | None:
    for path in (
        ("hidden_size",),
        ("text_config", "hidden_size"),
        ("llm_config", "hidden_size"),
        ("language_config", "hidden_size"),
    ):
        value = _lookup_path(config, path)
        integer = _positive_int_or_none(value)
        if integer is not None:
            return integer
    return None


def resolve_total_layers_from_weight_index(local_path: Path) -> int | None:
    index_path = local_path / "model.safetensors.index.json"
    if not index_path.is_file():
        return None
    try:
        payload = _read_json(index_path)
    except Exception:
        return None
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        return None
    layer_indices: set[int] = set()
    for key_object in weight_map:
        key = str(key_object)
        match = re.search(r"(?:language_model\.)?model\.layers\.(\d+)\.", key)
        if match:
            layer_indices.add(int(match.group(1)))
    if not layer_indices:
        return None
    return max(layer_indices) + 1


def resolve_hidden_dim_from_safetensors_header(local_path: Path) -> int | None:
    try:
        from safetensors import safe_open
    except Exception:
        return None
    for shard_path in sorted(local_path.glob("*.safetensors")):
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in (
                    "language_model.model.embed_tokens.weight",
                    "model.embed_tokens.weight",
                    "language_model.model.layers.0.self_attn.q_proj.weight",
                    "model.layers.0.self_attn.q_proj.weight",
                ):
                    if key in handle.keys():
                        shape = handle.get_slice(key).get_shape()
                        if len(shape) >= 2:
                            return int(shape[-1])
        except Exception:
            continue
    return None


def validate_hidden_state_entries(
    entries: Sequence[Mapping[str, object]],
    sidecar: Mapping[str, object],
) -> ValidationResult:
    if not entries:
        return ValidationResult("failed_validation", "shard contains no entries")
    sidecar_result = _validate_required_sidecar_metadata(sidecar)
    if sidecar_result.status != "verified":
        return sidecar_result
    try:
        total_layers = int(sidecar["total_layers"])
        hidden_dim = int(sidecar["hidden_dim"])
        token_index = int(sidecar["token_index"])
        prompt_template_id = str(sidecar["prompt_template_id"])
        hidden_state_index_offset = int(sidecar["hidden_state_index_offset"])
        hidden_state_count = int(sidecar["hidden_state_count"])
        selected_layer_hidden_state_indices = list(sidecar["selected_layer_hidden_state_indices"])
    except (KeyError, TypeError, ValueError) as error:
        return ValidationResult("failed_validation", f"sidecar metadata missing or invalid: {error}")
    if hidden_state_index_offset not in (0, 1):
        return ValidationResult("failed_validation", "hidden_state_index_offset must be 0 or 1")
    if hidden_state_count != total_layers + hidden_state_index_offset:
        return ValidationResult(
            "failed_validation",
            "hidden_state_index_offset does not map selected layers to transformer blocks",
            {"hidden_state_count": hidden_state_count},
        )
    expected_layers = list(range(total_layers))
    expected_hidden_state_indices = [layer + hidden_state_index_offset for layer in expected_layers]
    if selected_layer_hidden_state_indices != expected_hidden_state_indices:
        return ValidationResult(
            "failed_validation",
            "selected layer to hidden-state index mapping is not explicit or correct",
            {"expected": expected_hidden_state_indices, "actual": selected_layer_hidden_state_indices},
        )
    for index, entry in enumerate(entries):
        result = _validate_entry(
            entry,
            index=index,
            expected_layers=expected_layers,
            total_layers=total_layers,
            hidden_dim=hidden_dim,
            prompt_template_id=prompt_template_id,
        )
        if result.status != "verified":
            return result
    return ValidationResult("verified")


def _validate_required_sidecar_metadata(sidecar: Mapping[str, object]) -> ValidationResult:
    required_nonblank = (
        "model_alias",
        "model_family",
        "local_path",
        "wrapper_class",
        "processor_class",
        "model_class",
        "prompt_template_id",
        "validation_commit",
    )
    for key in required_nonblank:
        value = str(sidecar.get(key) or "").strip()
        if not value:
            return ValidationResult("failed_validation", f"sidecar metadata missing or blank: {key}")
    if str(sidecar.get("model_family")) == "unknown":
        return ValidationResult("failed_validation", "sidecar metadata model_family must not be unknown")
    if str(sidecar.get("wrapper_class")) == "generic_unknown":
        return ValidationResult("failed_validation", "sidecar metadata wrapper_class must not be generic_unknown")
    if str(sidecar.get("processor_class")) == "unknown":
        return ValidationResult("failed_validation", "sidecar metadata processor_class must not be unknown")
    if str(sidecar.get("model_class")) == "unknown":
        return ValidationResult("failed_validation", "sidecar metadata model_class must not be unknown")
    deterministic = sidecar.get("deterministic_generation_kwargs")
    if not isinstance(deterministic, Mapping):
        return ValidationResult("failed_validation", "sidecar metadata deterministic_generation_kwargs missing or invalid")
    if deterministic.get("do_sample") is not False:
        return ValidationResult("failed_validation", "sidecar metadata deterministic_generation_kwargs.do_sample must be false")
    try:
        max_new_tokens = int(deterministic["max_new_tokens"])
        temperature = float(deterministic["temperature"])
    except (KeyError, TypeError, ValueError) as error:
        return ValidationResult("failed_validation", f"sidecar metadata deterministic generation invalid: {error}")
    if max_new_tokens != 1 or temperature != 0.0:
        return ValidationResult("failed_validation", "sidecar metadata deterministic generation must use max_new_tokens=1 and temperature=0")
    thinking_disabled = sidecar.get("thinking_disabled")
    if thinking_disabled not in (True, False):
        return ValidationResult("failed_validation", "sidecar metadata thinking_disabled must be boolean")
    trust_remote_code = sidecar.get("trust_remote_code")
    if trust_remote_code not in (True, False):
        return ValidationResult("failed_validation", "sidecar metadata trust_remote_code must be boolean")
    family = str(sidecar.get("model_family"))
    if family == "gemma4_unified":
        for key, expected in (
            ("unified_multimodal", True),
            ("has_separate_vision_encoder", False),
            ("image_sensitivity_canary_required", True),
            ("enable_thinking", False),
        ):
            if sidecar.get(key) is not expected:
                return ValidationResult("failed_validation", f"sidecar metadata {key} must be {expected!r}")
    if family == "phi4mm":
        required_phi4 = (
            "attn_implementation_effective",
            "disabled_flash_attention_2",
            "low_cpu_mem_usage",
            "device_map_policy",
            "no_meta_tensors_after_load",
            "peft_version",
        )
        for key in required_phi4:
            if key not in sidecar:
                return ValidationResult("failed_validation", f"sidecar metadata missing Phi4 loading field: {key}")
        if sidecar.get("attn_implementation_effective") not in {"eager", "sdpa"}:
            return ValidationResult("failed_validation", "Phi4 sidecar attention must be eager or sdpa")
        if sidecar.get("low_cpu_mem_usage") is not False:
            return ValidationResult("failed_validation", "Phi4 sidecar low_cpu_mem_usage must be false")
        if sidecar.get("no_meta_tensors_after_load") is not True:
            return ValidationResult("failed_validation", "Phi4 sidecar no_meta_tensors_after_load must be true")
    if family == "llava_v15":
        for key in ("hf_complete_asset_path", "vision_tower_status", "image_token_prompt_policy", "copied_metadata_from_onevision"):
            if key not in sidecar:
                return ValidationResult("failed_validation", f"sidecar metadata missing LLaVA-v1.5 field: {key}")
        if sidecar.get("copied_metadata_from_onevision") is not False:
            return ValidationResult("failed_validation", "LLaVA-v1.5 sidecar copied_metadata_from_onevision must be false")
    return ValidationResult("verified")


def validate_determinism_pair(
    pair: DeterminismPair,
    *,
    layer_tolerance: float,
    logits_tolerance: float,
) -> ValidationResult:
    for key in ("answer_text", "parsed_answer", "selected_layers"):
        if pair.first.get(key) != pair.second.get(key):
            return ValidationResult("failed_validation", f"{key} differs between deterministic runs")
    first_layers = _require_tensor(pair.first, "layer_vectors")
    second_layers = _require_tensor(pair.second, "layer_vectors")
    first_logits = _require_tensor(pair.first, "first_token_logits")
    second_logits = _require_tensor(pair.second, "first_token_logits")
    layer_diff = _max_abs_diff(first_layers, second_layers)
    logits_diff = _max_abs_diff(first_logits, second_logits)
    if layer_diff > layer_tolerance:
        return ValidationResult("failed_validation", f"layer_vectors max_abs_diff {layer_diff:.6g} exceeds tolerance")
    if logits_diff > logits_tolerance:
        return ValidationResult("failed_validation", f"first_token_logits max_abs_diff {logits_diff:.6g} exceeds tolerance")
    return ValidationResult(
        "verified",
        details={"layer_vectors_max_abs_diff": layer_diff, "first_token_logits_max_abs_diff": logits_diff},
    )


def validate_smoke_report_contract(
    rows: Sequence[Mapping[str, object]],
    *,
    datasets: Sequence[str],
) -> ValidationResult:
    expected = {(alias, dataset) for alias in REQUIRED_MODEL_ALIASES for dataset in datasets}
    actual: set[tuple[str, str]] = set()
    duplicates: list[tuple[str, str]] = []
    unknown_aliases: set[str] = set()
    unknown_statuses: set[str] = set()
    allowed_statuses = {status.value for status in AssetStatus}
    for row in rows:
        alias = str(row.get("model_alias"))
        dataset = str(row.get("dataset"))
        pair = (alias, dataset)
        if pair in actual:
            duplicates.append(pair)
        actual.add(pair)
        if alias not in REQUIRED_MODEL_ALIASES:
            unknown_aliases.add(alias)
        status = str(row.get("status"))
        if status not in allowed_statuses:
            unknown_statuses.add(status)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if duplicates:
        return ValidationResult("failed_validation", "smoke report contains duplicate model/dataset pairs", {"duplicates": duplicates})
    if unknown_aliases:
        return ValidationResult("failed_validation", "smoke report contains unknown model aliases", {"unknown_aliases": sorted(unknown_aliases)})
    if unknown_statuses:
        return ValidationResult("failed_validation", "smoke report contains unknown statuses", {"unknown_statuses": sorted(unknown_statuses)})
    if missing:
        return ValidationResult("failed_validation", "smoke report is missing model/dataset pairs", {"missing": missing})
    if extra:
        return ValidationResult("failed_validation", "smoke report contains unexpected model/dataset pairs", {"extra": extra})
    return ValidationResult("verified")


def build_completion_summary(
    *,
    model_statuses: Mapping[str, str],
    model_reasons: Mapping[str, str],
    smoke_datasets: Sequence[str],
    smoke_limit: int,
    tests_run: Sequence[str],
    git_commit: str,
) -> dict[str, object]:
    normalized_statuses = {
        alias: str(model_statuses.get(alias, AssetStatus.NOT_ATTEMPTED_DUE_TO_DEPENDENCY.value))
        for alias in REQUIRED_MODEL_ALIASES
    }
    verified = sorted(alias for alias, status in normalized_statuses.items() if status == AssetStatus.VERIFIED.value)
    verified_separate_env = sorted(alias for alias, status in normalized_statuses.items() if status == AssetStatus.VERIFIED_SEPARATE_ENV.value)
    blocked = sorted(alias for alias, status in normalized_statuses.items() if status == AssetStatus.BLOCKED.value)
    unsupported_policy = sorted(alias for alias, status in normalized_statuses.items() if status == AssetStatus.UNSUPPORTED_BY_POLICY.value)
    unsupported_wrapper = sorted(alias for alias, status in normalized_statuses.items() if status == AssetStatus.UNSUPPORTED_BY_WRAPPER.value)
    unsupported = sorted(unsupported_policy + unsupported_wrapper)
    failed = sorted(alias for alias, status in normalized_statuses.items() if status == AssetStatus.FAILED_VALIDATION.value)
    not_attempted = sorted(alias for alias, status in normalized_statuses.items() if status == AssetStatus.NOT_ATTEMPTED_DUE_TO_DEPENDENCY.value)
    passing_statuses = {AssetStatus.VERIFIED.value, AssetStatus.VERIFIED_SEPARATE_ENV.value}
    return {
        "final_status": "passed" if all(normalized_statuses[alias] in passing_statuses for alias in REQUIRED_MODEL_ALIASES) else "blocked",
        "total_models_requested": len(REQUIRED_MODEL_ALIASES),
        "num_verified": len(verified),
        "num_verified_separate_env": len(verified_separate_env),
        "num_blocked": len(blocked),
        "num_unsupported_by_policy": len(unsupported_policy),
        "num_unsupported_by_wrapper": len(unsupported_wrapper),
        "num_failed_validation": len(failed),
        "num_not_attempted_due_to_dependency": len(not_attempted),
        "model_statuses": normalized_statuses,
        "verified_models": verified,
        "verified_separate_env_models": verified_separate_env,
        "blocked_models": blocked,
        "unsupported_by_policy_models": unsupported_policy,
        "unsupported_by_wrapper_models": unsupported_wrapper,
        "unsupported_models": unsupported,
        "failed_models": failed,
        "not_attempted_due_to_dependency_models": not_attempted,
        "verified_separate_env_reasons": {alias: model_reasons.get(alias, "") for alias in verified_separate_env},
        "blocked_reasons": {alias: model_reasons.get(alias, "") for alias in blocked},
        "unsupported_reasons": {alias: model_reasons.get(alias, "") for alias in unsupported},
        "unsupported_by_policy_reasons": {alias: model_reasons.get(alias, "") for alias in unsupported_policy},
        "unsupported_by_wrapper_reasons": {alias: model_reasons.get(alias, "") for alias in unsupported_wrapper},
        "failed_reasons": {alias: model_reasons.get(alias, "") for alias in failed},
        "not_attempted_due_to_dependency_reasons": {alias: model_reasons.get(alias, "") for alias in not_attempted},
        "smoke_datasets_used": list(smoke_datasets),
        "smoke_limit": int(smoke_limit),
        "tests_run": list(tests_run),
        "git_commit": git_commit,
        "stageA_started": False,
        "full_cache_extraction_started": False,
        "training_started": False,
    }


def tensor_checksum(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(str(list(cpu.shape)).encode("utf-8"))
    if cpu.dtype == torch.bfloat16:
        digest.update(cpu.view(torch.uint16).numpy().tobytes())
    else:
        digest.update(cpu.numpy().tobytes())
    return digest.hexdigest()


def _validate_entry(
    entry: Mapping[str, object],
    *,
    index: int,
    expected_layers: list[int],
    total_layers: int,
    hidden_dim: int,
    prompt_template_id: str,
) -> ValidationResult:
    if "layer_vectors" not in entry:
        return ValidationResult("failed_validation", f"entry {index}: layer_vectors missing")
    layer_vectors = _require_tensor(entry, "layer_vectors")
    if layer_vectors.ndim != 2:
        return ValidationResult("failed_validation", f"entry {index}: layer_vectors.ndim must be 2")
    selected_layers = entry.get("selected_layers")
    if selected_layers != expected_layers:
        return ValidationResult("failed_validation", f"entry {index}: selected_layers must equal list(range(total_layers))")
    if int(layer_vectors.shape[0]) != total_layers:
        return ValidationResult("failed_validation", f"entry {index}: layer_vectors row count must equal total_layers")
    if int(layer_vectors.shape[1]) != hidden_dim:
        return ValidationResult("failed_validation", f"entry {index}: hidden_dim mismatch")
    if not torch.isfinite(layer_vectors).all().item():
        return ValidationResult("failed_validation", f"entry {index}: layer_vectors must be finite")
    try:
        entry_token_index = int(entry["token_index"])
    except (KeyError, TypeError, ValueError) as error:
        return ValidationResult("failed_validation", f"entry {index}: token_index missing or invalid: {error}")
    if entry_token_index < 0:
        return ValidationResult("failed_validation", f"entry {index}: token_index must be non-negative")
    if str(entry.get("prompt_template_id")) != prompt_template_id:
        return ValidationResult("failed_validation", f"entry {index}: prompt_template_id mismatch")
    first_token_logits = _require_tensor(entry, "first_token_logits")
    if not torch.isfinite(first_token_logits).all().item():
        return ValidationResult("failed_validation", f"entry {index}: first_token_logits must be finite")
    answer_text = str(entry.get("answer_text") or "")
    if not answer_text.strip():
        return ValidationResult("failed_validation", f"entry {index}: answer_text must be non-empty")
    parsed_answer = entry.get("parsed_answer")
    if parsed_answer not in (0, 1, None):
        return ValidationResult("failed_validation", f"entry {index}: parsed_answer must be 0, 1, or null")
    if parse_yes_no_answer(answer_text) != parsed_answer:
        return ValidationResult("failed_validation", f"entry {index}: parsed_answer parser result does not match answer_text")
    norms = torch.linalg.vector_norm(layer_vectors.float(), dim=1)
    if not torch.isfinite(norms).all().item() or bool((norms == 0).any().item()):
        return ValidationResult("failed_validation", f"entry {index}: layer norm is zero or non-finite")
    if total_layers > 1:
        cosine = F.cosine_similarity(layer_vectors[:-1].float(), layer_vectors[1:].float(), dim=1)
        if not torch.isfinite(cosine).all().item():
            return ValidationResult("failed_validation", f"entry {index}: adjacent layer cosine is non-finite")
        if bool((cosine > 0.999999).all().item()):
            return ValidationResult("failed_validation", f"entry {index}: hidden states appear constant across layers")
    return ValidationResult("verified")


def _base_audit_result(
    asset: AssetModel,
    *,
    path_exists: bool,
    path_is_directory: bool,
    config_exists: bool,
    processor_tokenizer_assets: bool,
) -> AssetAuditResult:
    return AssetAuditResult(
        alias=asset.alias,
        status=AssetStatus.BLOCKED,
        reason="not audited",
        local_path=asset.local_path,
        path_exists=path_exists,
        path_is_directory=path_is_directory,
        config_exists=config_exists,
        processor_tokenizer_assets=processor_tokenizer_assets,
        model_family_detected="unknown",
        architecture_detected="unknown",
        moe_indicators=[],
        thinking_detected=False,
        thinking_disable_argument=asset.thinking.disable_argument,
        dtype=asset.dtype,
        trust_remote_code_required=asset.trust_remote_code,
        local_loading_class_candidate="unknown",
        image_processor_candidate="unknown",
        total_layers=None,
        hidden_dim=None,
        output_hidden_states_support="unknown",
        generation_api_support="unknown",
        hidden_state_index_offset=asset.hidden_state_index_offset,
        prompt_template_id=asset.prompt_template_id,
        moe_policy_decision="not_evaluated",
        moe_indicators_seen=[],
        moe_indicators_ignored_with_reason={},
    )


def _replace_audit(result: AssetAuditResult, *, status: AssetStatus, reason: str) -> AssetAuditResult:
    return AssetAuditResult(**{**result.as_dict(), "status": status, "reason": reason})


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return payload


def _has_processor_or_tokenizer_assets(path: Path) -> bool:
    tokenizer = any(
        (path / filename).is_file()
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
    )
    processor = any(path.glob("processor_config.json")) or any(path.glob("preprocessor_config.json")) or any(path.glob("processing_*.py"))
    return tokenizer and bool(processor)


def _detect_thinking_markers(path: Path) -> bool:
    template_markers = ("<think>", "</think>", "enable_thinking", "/nothink", "reasoning_content")
    for filename in ("chat_template.json", "chat_template.jinja"):
        candidate = path / filename
        if not candidate.is_file():
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore").lower()
        if any(marker in text for marker in template_markers):
            return True
    tokenizer_config = path / "tokenizer_config.json"
    if tokenizer_config.is_file():
        try:
            payload = json.loads(tokenizer_config.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        chat_template = payload.get("chat_template") if isinstance(payload, Mapping) else None
        if isinstance(chat_template, str):
            lowered = chat_template.lower()
            return any(marker in lowered for marker in template_markers)
    return False


def _architecture_name(config: Mapping[str, object]) -> str:
    architectures = config.get("architectures")
    if isinstance(architectures, list) and architectures:
        return str(architectures[0])
    return "unknown"


def _image_processor_candidate(path: Path) -> str:
    for filename in ("processor_config.json", "preprocessor_config.json"):
        if (path / filename).is_file():
            return filename
    for candidate in sorted(path.glob("image_processing_*.py")):
        return candidate.name
    return "unknown"


def _audit_family_specific_constraints(
    asset: AssetModel,
    local_path: Path,
    config: Mapping[str, object],
) -> tuple[AssetStatus, str] | None:
    family = asset.family
    if family == "glm4v":
        if config.get("model_type") != "glm4v" or "vision_config" not in config or not _has_any_key(config, ("image_token_id", "image_token_index")):
            return AssetStatus.UNSUPPORTED_BY_POLICY, "GLM-4V image-text config is required"
        processor_class = (_processor_class_from_files(local_path) or "").lower()
        image_processor = (_image_processor_type(local_path) or "").lower()
        if not processor_class.startswith("glm") or "processor" not in processor_class or not image_processor.startswith("glm"):
            return AssetStatus.BLOCKED, "GLM image processor metadata is required"
        missing = _missing_transformers_classes(("Glm4vForConditionalGeneration", "Glm46VProcessor"))
        if missing:
            return AssetStatus.BLOCKED, "installed transformers is missing required GLM classes: " + ", ".join(missing)
    if family == "minicpmv":
        if config.get("model_type") != "minicpmv" or "vision_config" not in config:
            return AssetStatus.UNSUPPORTED_BY_POLICY, "MiniCPM-V image-text config is required"
        if _processor_class_from_files(local_path) != "MiniCPMVProcessor" or _image_processor_type(local_path) != "MiniCPMVImageProcessor":
            return AssetStatus.BLOCKED, "MiniCPMV image processor metadata is required"
        custom_chat = config.get("custom_chat_api")
        if isinstance(custom_chat, Mapping) and custom_chat.get("returns_hidden_states") is False:
            return AssetStatus.UNSUPPORTED_BY_WRAPPER, "MiniCPM custom chat API does not expose hidden-state access"
        auto_map = config.get("auto_map")
        if not isinstance(auto_map, Mapping) or not any("MiniCPMV" in str(value) for value in auto_map.values()):
            return AssetStatus.UNSUPPORTED_BY_WRAPPER, "MiniCPM local remote-code MiniCPMV mapping is required for hidden-state extraction"
    if family == "gemma3":
        if config.get("model_type") != "gemma3" or "vision_config" not in config or "image_token_index" not in config:
            return AssetStatus.UNSUPPORTED_BY_POLICY, "Gemma3 multimodal image-text config is required"
        if _processor_class_from_files(local_path) != "Gemma3Processor" or _image_processor_type(local_path) != "Gemma3ImageProcessor":
            return AssetStatus.BLOCKED, "Gemma3Processor and Gemma3ImageProcessor metadata are required"
        missing = _missing_transformers_classes(("Gemma3Processor", "Gemma3ForConditionalGeneration"))
        if missing:
            return AssetStatus.BLOCKED, "installed transformers is missing required Gemma3 classes: " + ", ".join(missing)
    if family == "gemma4":
        if config.get("model_type") != "gemma4" or not _gemma4_config_has_image_text_path(config):
            return AssetStatus.UNSUPPORTED_BY_POLICY, "Gemma4 Unified image-text config is required"
        if _processor_class_from_files(local_path) != "Gemma4Processor" or _image_processor_type(local_path) != "Gemma4ImageProcessor":
            return AssetStatus.BLOCKED, "Gemma4Processor and Gemma4ImageProcessor metadata are required"
        if not _has_safetensors_asset(local_path):
            return AssetStatus.BLOCKED, "Gemma4 safetensors shards are required in the local asset"
        missing = _missing_transformers_classes(("Gemma4Processor", "Gemma4ForConditionalGeneration", "AutoModelForMultimodalLM"))
        if missing:
            return AssetStatus.BLOCKED, "installed transformers is missing required Gemma4 classes: " + ", ".join(missing)
    if family == "gemma4_unified":
        if config.get("model_type") != "gemma4_unified" or "image_token_id" not in config:
            return AssetStatus.UNSUPPORTED_BY_POLICY, "Gemma4 Unified image-text config is required"
        if _processor_class_from_files(local_path) != "Gemma4UnifiedProcessor" or _image_processor_type(local_path) != "Gemma4UnifiedImageProcessor":
            return AssetStatus.BLOCKED, "Gemma4UnifiedProcessor and Gemma4UnifiedImageProcessor metadata are required"
        if not _has_safetensors_asset(local_path):
            return AssetStatus.BLOCKED, "Gemma4 Unified safetensors asset is required in the local asset"
        missing = _missing_transformers_classes(("Gemma4UnifiedProcessor", "Gemma4UnifiedForConditionalGeneration", "AutoModelForMultimodalLM"))
        if missing:
            return AssetStatus.BLOCKED, "installed transformers is missing required Gemma4 Unified classes: " + ", ".join(missing)
    if family == "phi3_v":
        if config.get("model_type") != "phi3_v" or "img_processor" not in config:
            return AssetStatus.UNSUPPORTED_BY_POLICY, "Phi-3.5 vision image-text config is required"
        if _processor_class_from_files(local_path) != "Phi3VProcessor" or _image_processor_type(local_path) != "Phi3VImageProcessor":
            return AssetStatus.BLOCKED, "Phi3V image processor metadata is required"
    if family == "phi4mm":
        if config.get("model_type") != "phi4mm" or not _phi4_config_has_image_text_path(config):
            return AssetStatus.UNSUPPORTED_BY_POLICY, "Phi-4 multimodal image-text config is required"
        if _processor_class_from_files(local_path) != "Phi4MMProcessor" or _image_processor_type(local_path) != "Phi4MMImageProcessor":
            return AssetStatus.BLOCKED, "Phi4MM image processor metadata is required"
        missing = _missing_imports(("peft",))
        if missing:
            return (
                AssetStatus.BLOCKED,
                "missing dependency required by Phi4MMForCausalLM local image-text loading: " + ", ".join(missing),
            )
    if family == "llava_v15":
        if config.get("model_type") not in {"llava", "llava_v15"}:
            return AssetStatus.UNSUPPORTED_BY_POLICY, "LLaVA-v1.5 image-text config is required"
        if _llava_v15_vision_tower_incomplete(local_path, config):
            return (
                AssetStatus.BLOCKED,
                "local llava-v1.5-7b is incomplete for local image-text smoke extraction; processor/image processor metadata is missing, vision tower weights are not included in the registered asset, and tokenizer loading also needs missing protobuf or tiktoken",
            )
        if _processor_class_from_files(local_path) is None or _image_processor_type(local_path) is None:
            return AssetStatus.BLOCKED, "processor/image processor metadata is missing for LLaVA-v1.5 local image-text loading"
    return None


def _processor_class_from_files(path: Path) -> str | None:
    for filename in ("processor_config.json", "preprocessor_config.json"):
        candidate = path / filename
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and payload.get("processor_class"):
            return str(payload["processor_class"])
    return None


def _image_processor_type(path: Path) -> str | None:
    for filename in ("preprocessor_config.json", "processor_config.json"):
        candidate = path / filename
        if not candidate.is_file():
            continue
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping) and payload.get("image_processor_type"):
            return str(payload["image_processor_type"])
        nested = payload.get("image_processor") if isinstance(payload, Mapping) else None
        if isinstance(nested, Mapping) and nested.get("image_processor_type"):
            return str(nested["image_processor_type"])
    return None


def _phi4_config_has_image_text_path(config: Mapping[str, object]) -> bool:
    embedding = config.get("embd_layer")
    if not isinstance(embedding, Mapping):
        return False
    image_layer = embedding.get("image_embd_layer")
    if not isinstance(image_layer, Mapping):
        return False
    image_embedding = str(image_layer.get("embedding_cls", "")).lower()
    return "image" in image_embedding


def _gemma4_config_has_image_text_path(config: Mapping[str, object]) -> bool:
    if _has_any_key(config, ("image_token_id", "image_token_index")):
        return True
    modalities = config.get("supported_modalities")
    if isinstance(modalities, Sequence) and not isinstance(modalities, str):
        return "image" in {str(modality).lower() for modality in modalities}
    return False


def _has_safetensors_asset(path: Path) -> bool:
    index_path = path / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        weight_map = payload.get("weight_map")
        if not isinstance(weight_map, Mapping) or not weight_map:
            return False
        shard_names = {str(filename) for filename in weight_map.values()}
        return all((path / shard_name).is_file() for shard_name in shard_names)
    return any(path.glob("*.safetensors"))


def _llava_v15_vision_tower_incomplete(path: Path, config: Mapping[str, object]) -> bool:
    if not config.get("mm_vision_tower"):
        return False
    index_path = _first_existing_path(path, ("model.safetensors.index.json", "pytorch_model.bin.index.json"))
    if index_path is None:
        return False
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return True
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping):
        return True
    return not any(str(name).startswith("model.vision_tower") for name in weight_map)


def _first_existing_path(path: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        candidate = path / name
        if candidate.is_file():
            return candidate
    return None


def _has_any_key(payload: Mapping[str, object], keys: Sequence[str]) -> bool:
    return any(key in payload for key in keys)


def _missing_imports(module_names: Sequence[str]) -> list[str]:
    return [module_name for module_name in module_names if importlib.util.find_spec(module_name) is None]


def _missing_transformers_classes(class_names: Sequence[str]) -> list[str]:
    try:
        import transformers
    except ImportError:
        return [f"transformers.{class_name}" for class_name in class_names]
    return [f"transformers.{class_name}" for class_name in class_names if not hasattr(transformers, class_name)]


def _deterministic_generation_is_valid(asset: AssetModel) -> bool:
    generation = asset.deterministic_generation
    return (
        generation.do_sample is False
        and float(generation.temperature) == 0.0
        and int(generation.max_new_tokens) == 1
    )


def _lookup_path(payload: Mapping[str, object], path: Sequence[str]) -> object | None:
    current: object = payload
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _positive_int_or_none(value: object) -> int | None:
    try:
        integer = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def _require_tensor(entry: Mapping[str, object], key: str) -> torch.Tensor:
    value = entry.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a torch.Tensor")
    return value


def _max_abs_diff(first: torch.Tensor, second: torch.Tensor) -> float:
    if tuple(first.shape) != tuple(second.shape):
        return float("inf")
    return float((first.float() - second.float()).abs().max().item())
