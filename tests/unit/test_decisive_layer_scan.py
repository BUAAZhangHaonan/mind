from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_decisive_layer_scan.py"
    assert script_path.exists(), "scripts/prepare_decisive_layer_scan.py should exist"
    spec = importlib.util.spec_from_file_location("prepare_decisive_layer_scan", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(
    root: Path,
    *,
    model: str,
    setting: str,
    layer_count: int,
    bank_scope: str = "object",
    payload: dict[str, object],
) -> None:
    name = f"layer-scan-{model}-{setting}-lc{layer_count}-{bank_scope}"
    _write_json(root / name / "baselines.json", payload)


def test_writes_rows_marks_best_layers_and_interprets_full_mind(tmp_path: Path) -> None:
    script = _load_script()
    reports_root = tmp_path / "reports"
    output_path = tmp_path / "docs" / "tables" / "layer_scan.md"

    values = {
        ("popular-object-heldout", "qwen3-vl-8b", 8): (0.70, 0.80),
        ("popular-object-heldout", "qwen3-vl-8b", 12): (0.90, 0.88),
        ("popular-object-heldout", "qwen3-vl-8b", 16): (0.90, 0.87),
        ("dash-b", "qwen3-vl-8b", 8): (0.45, 0.60),
        ("dash-b", "qwen3-vl-8b", 12): (0.60, 0.72),
        ("dash-b", "qwen3-vl-8b", 16): (0.55, 0.70),
        ("popular-object-heldout", "molmo-7b-d-0924", 8): (0.30, 0.50),
        ("popular-object-heldout", "molmo-7b-d-0924", 12): (0.40, 0.65),
        ("dash-b", "molmo-7b-d-0924", 8): (0.52, 0.69),
        ("dash-b", "molmo-7b-d-0924", 12): (0.48, 0.66),
    }
    for (setting, model, layer_count), (pr_auc, roc_auc) in values.items():
        _write_report(
            reports_root,
            model=model,
            setting=setting,
            layer_count=layer_count,
            payload={
                "full": {"pr_auc": pr_auc, "roc_auc": roc_auc},
                "linear_probe": {"pr_auc": pr_auc - 0.10, "roc_auc": roc_auc - 0.10},
                "no_manifold": {"pr_auc": pr_auc - 0.20, "roc_auc": roc_auc - 0.20},
            },
        )

    rows = script.prepare_layer_scan_report(reports_root=reports_root, output_path=output_path)

    assert len(rows) == 30
    markdown = output_path.read_text(encoding="utf-8")
    assert (
        "| setting | model | layer_count | bank_scope | method | pr_auc | roc_auc | best_layer_for_method |"
        in markdown
    )
    assert "| popular-object-heldout | qwen3-vl-8b | 12 | object | full | 0.9000 | 0.8800 | yes |" in markdown
    assert "| popular-object-heldout | qwen3-vl-8b | 16 | object | full | 0.9000 | 0.8700 | yes |" in markdown
    assert "| popular-object-heldout | qwen3-vl-8b | 8 | object | full | 0.7000 | 0.8000 | no |" in markdown
    assert (
        "- qwen3-vl-8b / popular-object-heldout: full MIND best at 12, 16 layers by PR-AUC; "
        "16-layer default is best."
    ) in markdown
    assert (
        "- qwen3-vl-8b / dash-b: full MIND best at 12 layers by PR-AUC; "
        "16-layer default is present but not best."
    ) in markdown
    assert (
        "- molmo-7b-d-0924 / dash-b: full MIND best at 8 layers by PR-AUC; "
        "16-layer default was not evaluated."
    ) in markdown


def test_extracts_required_methods_from_nested_json_shapes(tmp_path: Path) -> None:
    script = _load_script()
    reports_root = tmp_path / "reports"
    output_path = tmp_path / "layer_scan.md"

    for setting in script.SETTINGS:
        for layer_count in (8, 12, 16):
            _write_report(
                reports_root,
                model="qwen3-vl-8b",
                setting=setting,
                layer_count=layer_count,
                payload={
                    "variants": {
                        "full": {"metrics": {"PR-AUC": 0.80 + layer_count / 100, "ROC-AUC": 0.70}},
                        "linear_probe": {"summary": {"prauc": 0.60, "rocauc": 0.65}},
                    },
                    "results": [{"variant": "no_manifold", "scores": {"pr_auc": 0.50, "roc_auc": 0.62}}],
                },
            )

    rows = script.prepare_layer_scan_report(
        reports_root=reports_root,
        output_path=output_path,
        models=("qwen3-vl-8b",),
    )

    methods = {row["method"] for row in rows}
    assert methods == {"full", "linear_probe", "no_manifold"}
    assert (
        "| popular-object-heldout | qwen3-vl-8b | 16 | object | full | 0.9600 | 0.7000 | yes |"
        in output_path.read_text(encoding="utf-8")
    )


def test_missing_full_method_fails_loudly(tmp_path: Path) -> None:
    script = _load_script()
    reports_root = tmp_path / "reports"
    output_path = tmp_path / "layer_scan.md"
    _write_report(
        reports_root,
        model="qwen3-vl-8b",
        setting="popular-object-heldout",
        layer_count=8,
        payload={"linear_probe": {"pr_auc": 0.65, "roc_auc": 0.70}},
    )

    with pytest.raises(ValueError, match="Missing full"):
        script.prepare_layer_scan_report(
            reports_root=reports_root,
            output_path=output_path,
            models=("qwen3-vl-8b",),
            model_counts={"qwen3-vl-8b": (8,)},
            settings=("popular-object-heldout",),
        )
