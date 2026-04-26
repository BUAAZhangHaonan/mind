from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


def _load_script():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "prepare_decisive_bank_identity.py"
    assert script_path.exists(), "scripts/prepare_decisive_bank_identity.py should exist"
    spec = importlib.util.spec_from_file_location("prepare_decisive_bank_identity", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_report(root: Path, name: str, payload: dict[str, object]) -> None:
    _write_json(root / name / "baselines.json", payload)


def test_writes_table_and_interprets_object_first_tied_and_not_first(tmp_path: Path) -> None:
    script = _load_script()
    reports_root = tmp_path / "reports"
    output_path = tmp_path / "docs" / "tables" / "bankid.md"

    values = {
        ("popular-object-heldout", "qwen3-vl-8b", "object"): (0.90, 0.80),
        ("popular-object-heldout", "qwen3-vl-8b", "shared"): (0.70, 0.72),
        ("popular-object-heldout", "qwen3-vl-8b", "shuffled_object"): (0.60, 0.65),
        ("dash-b", "qwen3-vl-8b", "object"): (0.55, 0.62),
        ("dash-b", "qwen3-vl-8b", "shared"): (0.55, 0.60),
        ("dash-b", "qwen3-vl-8b", "shuffled_object"): (0.40, 0.58),
        ("popular-object-heldout", "molmo-7b-d-0924", "object"): (0.30, 0.55),
        ("popular-object-heldout", "molmo-7b-d-0924", "shared"): (0.80, 0.76),
        ("popular-object-heldout", "molmo-7b-d-0924", "shuffled_object"): (0.20, 0.51),
        ("dash-b", "molmo-7b-d-0924", "object"): (0.45, 0.66),
        ("dash-b", "molmo-7b-d-0924", "shared"): (0.40, 0.64),
        ("dash-b", "molmo-7b-d-0924", "shuffled_object"): (0.35, 0.63),
    }
    for (setting, model, bank_scope), (pr_auc, roc_auc) in values.items():
        if setting == "popular-object-heldout":
            name = f"bankid-{model}-popular-{bank_scope}-object-heldout"
        else:
            name = f"bankid-{model}-dash-b-{bank_scope}"
        payload = {"full": {"pr_auc": pr_auc, "roc_auc": roc_auc}}
        _write_report(reports_root, name, payload)

    rows = script.prepare_bank_identity_report(reports_root=reports_root, output_path=output_path)

    assert len(rows) == 12
    markdown = output_path.read_text(encoding="utf-8")
    assert "| setting | model | bank_scope | pr_auc | roc_auc | rank_by_pr_auc |" in markdown
    assert "| popular-object-heldout | qwen3-vl-8b | object | 0.9000 | 0.8000 | 1 |" in markdown
    assert "| dash-b | qwen3-vl-8b | object | 0.5500 | 0.6200 | 1 |" in markdown
    assert "| dash-b | qwen3-vl-8b | shared | 0.5500 | 0.6000 | 1 |" in markdown
    assert "| popular-object-heldout | molmo-7b-d-0924 | object | 0.3000 | 0.5500 | 2 |" in markdown
    assert "- qwen3-vl-8b / popular-object-heldout: object-conditioned ranked first by PR-AUC." in markdown
    assert "- qwen3-vl-8b / dash-b: object-conditioned tied for first by PR-AUC." in markdown
    assert "- molmo-7b-d-0924 / popular-object-heldout: object-conditioned did not rank first by PR-AUC." in markdown


def test_extracts_full_metrics_from_nested_json_shapes(tmp_path: Path) -> None:
    script = _load_script()
    reports_root = tmp_path / "reports"
    output_path = tmp_path / "bankid.md"

    expected_names = script.expected_report_names(models=("qwen3-vl-8b",), bank_scopes=("object", "shared", "shuffled_object"))
    for name in expected_names:
        if "popular-object-object-heldout" in name or name.endswith("dash-b-object"):
            payload = {"variants": {"full": {"metrics": {"PR-AUC": 0.81, "ROC-AUC": 0.71}}}}
        elif "-shared" in name:
            payload = {"results": [{"variant": "full", "pr_auc": 0.60, "roc_auc": 0.65}]}
        else:
            payload = {"baselines": {"full": {"summary": {"pr_auc": 0.50, "roc_auc": 0.61}}}}
        _write_report(reports_root, name, payload)

    rows = script.prepare_bank_identity_report(
        reports_root=reports_root,
        output_path=output_path,
        models=("qwen3-vl-8b",),
    )

    assert [row["rank_by_pr_auc"] for row in rows if row["bank_scope"] == "object"] == [1, 1]
    assert "| popular-object-heldout | qwen3-vl-8b | object | 0.8100 | 0.7100 | 1 |" in output_path.read_text(
        encoding="utf-8"
    )


def test_missing_expected_outputs_fail_loudly_unless_allowed(tmp_path: Path) -> None:
    script = _load_script()
    reports_root = tmp_path / "reports"
    output_path = tmp_path / "bankid.md"
    _write_report(
        reports_root,
        "bankid-qwen3-vl-8b-popular-object-object-heldout",
        {"full": {"pr_auc": 0.75, "roc_auc": 0.70}},
    )

    with pytest.raises(FileNotFoundError, match="Missing expected bank-identity report"):
        script.prepare_bank_identity_report(
            reports_root=reports_root,
            output_path=output_path,
            models=("qwen3-vl-8b",),
        )

    rows = script.prepare_bank_identity_report(
        reports_root=reports_root,
        output_path=output_path,
        models=("qwen3-vl-8b",),
        allow_missing=True,
    )

    assert len(rows) == 1
    assert output_path.exists()
