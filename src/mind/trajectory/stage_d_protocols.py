"""Stage D protocol, baseline-tier, and related-method contracts."""

from __future__ import annotations

from typing import Mapping


PRIMARY_STAGE_D_PROTOCOLS = (
    "repope_to_repope",
    "repope_to_pope",
    "repope_to_dashb",
    "pope_to_dashb",
)
OPTIONAL_STAGE_D_PROTOCOLS = ("pope_to_repope",)
STAGE_D_PROTOCOL_DATASETS = {
    "repope_to_repope": ("repope", "repope"),
    "repope_to_pope": ("repope", "pope"),
    "repope_to_dashb": ("repope", "dash-b"),
    "pope_to_dashb": ("pope", "dash-b"),
    "pope_to_repope": ("pope", "repope"),
}
STAGE_D_TIER_A_METHODS = (
    "MIND-main",
    "MIND-param",
    "logistic(z)",
    "final-hidden linear probe",
    "output-confidence",
    "HALP-lite",
)
STAGE_D_TIER_B_METHODS = ("official HALP",)


def stage_d_protocol_contract() -> dict[str, object]:
    """Return the frozen Stage D protocol contract."""

    rows = []
    for name in PRIMARY_STAGE_D_PROTOCOLS + OPTIONAL_STAGE_D_PROTOCOLS:
        source, target = STAGE_D_PROTOCOL_DATASETS[name]
        rows.append(
            {
                "protocol": name,
                "source_dataset": source,
                "target_dataset": target,
                "calibration_split": "source/cal",
                "optional": name in OPTIONAL_STAGE_D_PROTOCOLS,
            }
        )
    return {
        "primary_protocols": list(PRIMARY_STAGE_D_PROTOCOLS),
        "optional_protocols": list(OPTIONAL_STAGE_D_PROTOCOLS),
        "default_protocols": list(PRIMARY_STAGE_D_PROTOCOLS),
        "protocol_rows": rows,
    }


def build_stage_d_calibration_scopes(protocol: str) -> list[dict[str, object]]:
    """Return source and oracle-target calibration scopes for one protocol."""

    if protocol not in STAGE_D_PROTOCOL_DATASETS:
        raise ValueError(f"unknown Stage D protocol: {protocol}")
    return [
        {
            "protocol": protocol,
            "calibration_scope": "source_calibration",
            "diagnostic_only": False,
        },
        {
            "protocol": protocol,
            "calibration_scope": "oracle_target_calibration",
            "diagnostic_only": True,
        },
    ]


def stage_d_baseline_tiers() -> dict[str, object]:
    """Return the frozen Stage D Tier A/Tier B baseline contract."""

    return {
        "tierA": list(STAGE_D_TIER_A_METHODS),
        "tierB": list(STAGE_D_TIER_B_METHODS),
        "roles": {
            "MIND-main": "frozen_main_method",
            "MIND-param": "parametric_secondary",
            "logistic(z)": "same_embedding_supervised_comparator",
            "final-hidden linear probe": "traditional_linear_probe",
            "output-confidence": "cheap_gray_box_baseline",
            "HALP-lite": "fair_same_constraint_baseline",
            "official HALP": "ceiling_broader_access",
        },
    }


def related_method_feasibility_payload() -> dict[str, object]:
    """Build the Stage D transparency artifact for recent related methods."""

    return {
        "stage": "stage_d",
        "purpose": "related_method_feasibility_audit",
        "methods": [
            {
                "method": "HALP",
                "detection_granularity": "sample-level object yes/no hallucination probe",
                "required_supervision": "binary hallucination labels",
                "required_access_type": "pre-generation hidden states",
                "generation_timing": "pre-generation",
                "method_type": "detector",
                "executable_with_current_cache": True,
                "incompatibility_reason": "",
                "stage_d_handling": "HALP-lite in Tier A; official HALP as Tier B ceiling if feasible",
            },
            {
                "method": "EnsemHalDet",
                "detection_granularity": "ensemble over multiple internal representations",
                "required_supervision": "binary or method-specific hallucination labels",
                "required_access_type": "multiple internal feature families",
                "generation_timing": "varies by setup",
                "method_type": "detector",
                "executable_with_current_cache": False,
                "incompatibility_reason": "requires heavier multi-representation ensemble access beyond Stage D fair constraints",
                "stage_d_handling": "feasibility row only",
            },
            {
                "method": "VIB-Probe",
                "detection_granularity": "attention/head-level probe",
                "required_supervision": "probe labels and bottleneck training setup",
                "required_access_type": "attention-head or method-specific internal signals",
                "generation_timing": "pre-generation/internal-state dependent",
                "method_type": "detector",
                "executable_with_current_cache": False,
                "incompatibility_reason": "current full-cache stores full-layer hidden trajectories, not attention-head bottleneck signals",
                "stage_d_handling": "feasibility row only",
            },
            {
                "method": "HaloProbe",
                "detection_granularity": "token-level hallucination posterior",
                "required_supervision": "token-level posterior or compatible annotations",
                "required_access_type": "token-level Bayesian/probe signals",
                "generation_timing": "token-level",
                "method_type": "detector",
                "executable_with_current_cache": False,
                "incompatibility_reason": "requires token-level labels or posterior signals not present in the Stage D cache",
                "stage_d_handling": "feasibility row only",
            },
        ],
    }


def protocol_source_target(protocol: str) -> tuple[str, str]:
    """Return source and target dataset family names for one Stage D protocol."""

    if protocol not in STAGE_D_PROTOCOL_DATASETS:
        raise ValueError(f"unknown Stage D protocol: {protocol}")
    return STAGE_D_PROTOCOL_DATASETS[protocol]


def validate_stage_d_protocols(protocols: list[str] | tuple[str, ...]) -> list[str]:
    """Validate requested Stage D protocols without expanding to all pairwise routes."""

    values = [str(protocol) for protocol in protocols]
    valid = set(PRIMARY_STAGE_D_PROTOCOLS) | set(OPTIONAL_STAGE_D_PROTOCOLS)
    invalid = [protocol for protocol in values if protocol not in valid]
    if invalid:
        raise ValueError("unsupported Stage D protocol(s): " + ", ".join(invalid))
    return values


__all__ = [
    "OPTIONAL_STAGE_D_PROTOCOLS",
    "PRIMARY_STAGE_D_PROTOCOLS",
    "STAGE_D_PROTOCOL_DATASETS",
    "STAGE_D_TIER_A_METHODS",
    "STAGE_D_TIER_B_METHODS",
    "build_stage_d_calibration_scopes",
    "protocol_source_target",
    "related_method_feasibility_payload",
    "stage_d_baseline_tiers",
    "stage_d_protocol_contract",
    "validate_stage_d_protocols",
]
