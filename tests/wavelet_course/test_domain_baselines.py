from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch


def _entry(sample_id: str, split: str, *, with_halp: bool = False, label: int = 1) -> dict[str, object]:
    return {
        "model_name": "qwen3-vl-8b",
        "dataset_name": "repope",
        "subset": "popular",
        "sample_id": sample_id,
        "image_id": f"image-{sample_id}",
        "question": "Is there a cat?",
        "object_name": "cat",
        "label": label,
        "parsed_answer": 1,
        "wavelet_split": split,
        **(_halp_fields(sample_id) if with_halp else {}),
    }


def _entries_and_labels(
    *,
    single_class_train: bool = False,
    with_halp: bool = False,
) -> tuple[list[dict[str, object]], np.ndarray]:
    labels = np.asarray([0, 0 if single_class_train else 1, 0, 1, 0, 1], dtype=np.int64)
    splits = ("train", "train", "validation", "validation", "test", "test")
    entries = [
        _entry(f"{split}-{index}", split, with_halp=with_halp)
        for index, split in enumerate(splits)
    ]
    return entries, labels


def _halp_fields(sample_id: str) -> dict[str, object]:
    offset = float(sum(ord(char) for char in sample_id) % 7)
    full = torch.full((3, 3, 5), offset, dtype=torch.float32)
    return {
        "vision_features": torch.full((2, 5), offset + 0.5, dtype=torch.float32),
        "query_hidden_states": torch.stack(
            [torch.full((5,), offset + layer, dtype=torch.float32) for layer in range(3)]
        ),
        "vision_token_hidden_states": torch.stack(
            [torch.full((5,), offset + layer + 0.25, dtype=torch.float32) for layer in range(3)]
        ),
        "full_hidden_states": full,
        "query_token_index": 1,
        "vision_token_span": [0, 2],
    }


def _compact_halp_fields(sample_id: str) -> dict[str, object]:
    fields = _halp_fields(sample_id)
    fields.pop("full_hidden_states")
    return fields


def _primary_entries() -> list[dict[str, object]]:
    return [
        _entry("train-0", "train", label=1),
        _entry("train-1", "train", label=0),
        _entry("validation-0", "validation", label=1),
        _entry("test-0", "test", label=0),
    ]


def _readout_entry(primary: dict[str, object], *, include_identity_label: bool = True) -> dict[str, object]:
    readout = {
        "model_name": primary["model_name"],
        "dataset_name": primary["dataset_name"],
        "subset": primary["subset"],
        "sample_id": primary["sample_id"],
        **_compact_halp_fields(str(primary["sample_id"])),
    }
    if include_identity_label:
        readout["label"] = primary["label"]
        readout["parsed_answer"] = primary["parsed_answer"]
    return readout


def _official_halp_row_entries() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    labels = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    for index, object_label in enumerate(labels):
        is_hallucination = object_label == 0
        value = 4.0 if is_hallucination else -4.0
        entry = _entry(
            f"row-{index:02d}",
            "train",
            with_halp=False,
            label=object_label,
        )
        entry["image_id"] = f"row-image-{index:02d}"
        entry["parsed_answer"] = 1 if is_hallucination else 0
        entry.update(_constant_halp_fields(value))
        entries.append(entry)
    return entries


def _constant_halp_fields(value: float) -> dict[str, object]:
    return {
        "vision_features": torch.full((2, 5), value, dtype=torch.float32),
        "query_hidden_states": torch.full((3, 5), value, dtype=torch.float32),
        "vision_token_hidden_states": torch.full((3, 5), value, dtype=torch.float32),
        "query_token_index": 1,
        "vision_token_span": [0, 2],
    }


def _load_domain_script() -> ModuleType:
    script_path = Path("scripts/wavelet_course_domain_baselines.py")
    spec = importlib.util.spec_from_file_location("wavelet_course_domain_baselines", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_official_halp_fields_are_recorded_as_failure(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import (
        SplitInfo,
        missing_official_halp_fields,
        run_official_halp,
    )

    entries, labels = _entries_and_labels()
    split_info = SplitInfo.from_entries(entries, labels)

    missing = missing_official_halp_fields(entries)
    row, selection_rows, result_rows = run_official_halp(
        entries,
        labels,
        split_info=split_info,
        seed=0,
        device="cpu",
        output_dir=tmp_path,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        dropout=0.0,
    )

    assert "query_hidden_states" in missing
    assert row["method_family"] == "halp_official"
    assert row["status"] == "failure"
    assert "missing_official_halp_fields" in row["failure_reason"]
    assert selection_rows == []
    assert result_rows == []


def test_official_halp_runs_when_required_cache_fields_exist(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import SplitInfo, run_official_halp

    entries, labels = _entries_and_labels(with_halp=True)
    split_info = SplitInfo.from_entries(entries, labels)

    row, selection_rows, result_rows = run_official_halp(
        entries,
        labels,
        split_info=split_info,
        seed=0,
        device="cpu",
        output_dir=tmp_path,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        dropout=0.0,
    )

    assert row["method_family"] == "halp_official"
    assert row["status"] == "success"
    assert row["selected_probe"]
    assert row["num_halp_candidates"] == 7
    assert selection_rows
    assert result_rows
    assert (tmp_path / "halp_selection.csv").exists()
    assert (tmp_path / "halp_results.csv").exists()


def test_official_halp_row_protocol_recomputes_object_hallucination_labels(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import run_official_halp_row_protocol

    entries = _official_halp_row_entries()
    wrong_course_labels = np.zeros(len(entries), dtype=np.int64)

    row, selection_rows, result_rows = run_official_halp_row_protocol(
        entries,
        labels=wrong_course_labels,
        output_dir=tmp_path,
        seed=13,
        device="cpu",
        hidden_dims=(4,),
        epochs=1,
        batch_size=4,
        test_size=0.25,
    )

    assert row["status"] == "success"
    assert row["train_pos"] > 0
    assert row["test_pos"] > 0
    assert row["best_val_threshold"] == 0.5
    assert row["selection_metric"] == "eval_roc_auc_then_pr_auc"
    assert {result["label"] for result in result_rows} == {0, 1}
    assert selection_rows


def test_official_halp_row_protocol_uses_11_probes_and_old_selection_rule(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import run_official_halp_row_protocol

    entries = _official_halp_row_entries()

    row, selection_rows, result_rows = run_official_halp_row_protocol(
        entries,
        output_dir=tmp_path,
        seed=13,
        device="cpu",
        hidden_dims=(4,),
        epochs=1,
        batch_size=4,
        test_size=0.25,
        layer_indices=[0, 1, 2, 1, 0],
    )

    assert row["status"] == "success"
    assert row["num_halp_candidates"] == 11
    assert len(selection_rows) == 11
    assert sum(1 for item in selection_rows if item["selected"]) == 1
    expected = sorted(
        selection_rows,
        key=lambda item: (item["eval_roc_auc"], item["eval_pr_auc"], item["probe_name"]),
        reverse=True,
    )[0]["probe_name"]
    assert row["selected_probe"] == expected
    assert all(result["prediction"] in {0, 1} for result in result_rows)


def test_compact_official_halp_cache_without_full_hidden_states_is_accepted(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import (
        SplitInfo,
        missing_official_halp_fields,
        run_official_halp,
    )

    entries, labels = _entries_and_labels()
    compact_entries = [
        {**entry, **_compact_halp_fields(str(entry["sample_id"]))}
        for entry in entries
    ]
    split_info = SplitInfo.from_entries(compact_entries, labels)

    assert missing_official_halp_fields(compact_entries) == []
    row, selection_rows, result_rows = run_official_halp(
        compact_entries,
        labels,
        split_info=split_info,
        seed=0,
        device="cpu",
        output_dir=tmp_path,
        hidden_dims=(4,),
        epochs=1,
        batch_size=2,
        learning_rate=1e-3,
        dropout=0.0,
    )

    assert row["status"] == "success"
    assert selection_rows
    assert result_rows


def test_halp_readout_cache_is_key_merged_in_primary_order(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import (
        load_halp_readout_cache_entries,
        merge_halp_readout_cache,
    )

    primary = _primary_entries()
    shuffled_readouts = [_readout_entry(entry) for entry in reversed(primary)]
    cache_path = tmp_path / "halp_readouts.pt"
    torch.save({"entries": shuffled_readouts}, cache_path)

    loaded = load_halp_readout_cache_entries(cache_path)
    merged = merge_halp_readout_cache(primary, loaded)

    assert [entry["sample_id"] for entry in merged] == [entry["sample_id"] for entry in primary]
    assert [entry["sample_id"] for entry in loaded] == [entry["sample_id"] for entry in shuffled_readouts]
    assert all("query_hidden_states" in entry for entry in merged)
    assert all("vision_features" in entry for entry in merged)
    assert [entry["label"] for entry in merged] == [entry["label"] for entry in primary]
    assert [entry["wavelet_split"] for entry in merged] == [entry["wavelet_split"] for entry in primary]


def test_halp_readout_cache_loader_accepts_sharded_directory(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import load_halp_readout_cache_entries

    primary = _primary_entries()
    cache_dir = tmp_path / "halp_cache"
    cache_dir.mkdir()
    first = [_readout_entry(primary[0]), _readout_entry(primary[1])]
    second = [_readout_entry(primary[2]), _readout_entry(primary[3])]
    torch.save(first, cache_dir / "shard-00000.pt")
    torch.save({"entries": second}, cache_dir / "shard-00001.pt")

    loaded = load_halp_readout_cache_entries(cache_dir)

    assert [entry["sample_id"] for entry in loaded] == [entry["sample_id"] for entry in primary]


def test_halp_readout_cache_loader_accepts_partitioned_shard_directory(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import load_halp_readout_cache_entries

    primary = _primary_entries()
    cache_dir = tmp_path / "halp_cache"
    part0 = cache_dir / "part-000"
    part1 = cache_dir / "part-001"
    part0.mkdir(parents=True)
    part1.mkdir(parents=True)
    torch.save([_readout_entry(primary[0]), _readout_entry(primary[2])], part0 / "shard-00000.pt")
    torch.save([_readout_entry(primary[1]), _readout_entry(primary[3])], part1 / "shard-00000.pt")

    loaded = load_halp_readout_cache_entries(cache_dir)

    assert sorted(entry["sample_id"] for entry in loaded) == sorted(entry["sample_id"] for entry in primary)


def test_halp_readout_cache_merge_fails_when_primary_key_is_missing() -> None:
    from mind.wavelet_course.domain_baselines import merge_halp_readout_cache

    primary = _primary_entries()
    readouts = [_readout_entry(entry) for entry in primary[:-1]]

    with pytest.raises(ValueError, match="missing HALP readout"):
        merge_halp_readout_cache(primary, readouts)


def test_halp_readout_cache_merge_fails_on_duplicate_key() -> None:
    from mind.wavelet_course.domain_baselines import merge_halp_readout_cache

    primary = _primary_entries()
    readouts = [_readout_entry(entry) for entry in primary]
    readouts.append(_readout_entry(primary[0]))

    with pytest.raises(ValueError, match="duplicate HALP readout"):
        merge_halp_readout_cache(primary, readouts)


def test_halp_readout_cache_merge_fails_on_label_mismatch() -> None:
    from mind.wavelet_course.domain_baselines import merge_halp_readout_cache

    primary = _primary_entries()
    readouts = [_readout_entry(entry) for entry in primary]
    readouts[0]["label"] = 1 - int(primary[0]["label"])

    with pytest.raises(ValueError, match="label mismatch"):
        merge_halp_readout_cache(primary, readouts)


def test_halp_readout_cache_merge_fails_on_nonfinite_feature() -> None:
    from mind.wavelet_course.domain_baselines import merge_halp_readout_cache

    primary = _primary_entries()
    readouts = [_readout_entry(entry) for entry in primary]
    readouts[0]["query_hidden_states"] = torch.full((3, 5), float("nan"), dtype=torch.float32)

    with pytest.raises(ValueError, match="NaN or Inf"):
        merge_halp_readout_cache(primary, readouts)


def test_domain_baseline_success_row_uses_train_validation_test_fields(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import SplitInfo, _run_feature_logreg, write_domain_baselines_csv

    entries, labels = _entries_and_labels()
    split_info = SplitInfo.from_entries(entries, labels)
    features = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [0.2, 0.1],
            [0.9, 0.8],
            [0.1, 0.2],
            [0.8, 0.9],
        ],
        dtype=np.float32,
    )
    row = _run_feature_logreg(
        "linear_probe_tiny",
        method_family="linear_probe",
        source="linear_probe",
        representation="tiny",
        entries=entries,
        labels=labels,
        split_info=split_info,
        feature_builder=lambda _entries: features,
        seed=0,
        max_iter=200,
    )

    assert row["status"] == "success"
    assert row["train_samples"] == 2
    assert row["validation_samples"] == 2
    assert row["test_samples"] == 2

    output = tmp_path / "domain_baselines.csv"
    write_domain_baselines_csv([row], output)
    with output.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)
    assert "validation_samples" in (reader.fieldnames or [])
    assert "val_samples" not in (reader.fieldnames or [])
    assert written[0]["validation_samples"] == "2"


def test_run_domain_baselines_contains_only_halp_and_linear_probe(tmp_path: Path) -> None:
    from mind.wavelet_course.domain_baselines import run_domain_baselines

    entries, labels = _entries_and_labels()

    result = run_domain_baselines(
        entries,
        labels,
        output_dir=tmp_path,
        model_name="qwen3-vl-8b",
        dataset_name="repope",
        subsets=["popular"],
        seed=0,
        device="cpu",
        logreg_max_iter=200,
        halp_epochs=1,
        halp_batch_size=2,
    )

    families = {row["method_family"] for row in result["rows"]}
    names = {row["baseline_name"] for row in result["rows"]}
    assert families == {"halp_official", "linear_probe"}
    assert {"halp_official_mlp", "halp_official_row_protocol"}.issubset(names)
    assert "mind_stage_a" not in families
    assert "halp_like" not in families


def test_best_rows_keeps_course_grouped_and_row_protocol_halp_separate() -> None:
    from mind.wavelet_course.domain_baselines import best_rows_by_family

    rows = [
        {
            "baseline_name": "halp_official_mlp",
            "method_family": "halp_official",
            "status": "success",
            "test_pr_auc": 0.538,
        },
        {
            "baseline_name": "halp_official_row_protocol",
            "method_family": "halp_official",
            "status": "success",
            "test_pr_auc": 0.904,
        },
        {
            "baseline_name": "linear_probe_final_hidden_logreg",
            "method_family": "linear_probe",
            "status": "success",
            "test_pr_auc": 0.541,
        },
    ]

    rank_names = {row["rank_name"] for row in best_rows_by_family(rows)}

    assert "best_halp_official" not in rank_names
    assert "best_halp_official_mlp" in rank_names
    assert "best_halp_official_row_protocol" in rank_names
    assert "best_linear_probe" in rank_names


def test_single_class_train_returns_explicit_failure_row() -> None:
    from mind.wavelet_course.domain_baselines import SplitInfo, _run_feature_logreg

    entries, labels = _entries_and_labels(single_class_train=True)
    split_info = SplitInfo.from_entries(entries, labels)
    features = np.arange(12, dtype=np.float32).reshape(6, 2)

    row = _run_feature_logreg(
        "linear_probe_single_class",
        method_family="linear_probe",
        source="linear_probe",
        representation="tiny",
        entries=entries,
        labels=labels,
        split_info=split_info,
        feature_builder=lambda _entries: features,
        seed=0,
        max_iter=50,
    )

    assert row["status"] == "failure"
    assert "train_y must contain both classes" in row["failure_reason"]


def test_non_finite_features_return_explicit_failure_row() -> None:
    from mind.wavelet_course.domain_baselines import SplitInfo, _run_feature_logreg

    entries, labels = _entries_and_labels()
    split_info = SplitInfo.from_entries(entries, labels)
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    features[0, 0] = np.inf

    row = _run_feature_logreg(
        "linear_probe_nonfinite",
        method_family="linear_probe",
        source="linear_probe",
        representation="tiny",
        entries=entries,
        labels=labels,
        split_info=split_info,
        feature_builder=lambda _entries: features,
        seed=0,
        max_iter=50,
    )

    assert row["status"] == "failure"
    assert "NaN or Inf" in row["failure_reason"]


def test_summary_section_points_to_wavelet_course_outputs(tmp_path: Path) -> None:
    script = _load_domain_script()
    summary_path = tmp_path / "outputs" / "wavelet_course_v2" / "reports" / "summary.md"
    summary_path.parent.mkdir(parents=True)
    summary_path.write_text("# Existing Summary\n", encoding="utf-8")
    rows = [
        {
            "baseline_name": "halp_official_mlp",
            "method_family": "halp_official",
            "source": "halp_official",
            "status": "success",
            "selected_probe": "query_token_layer_27",
            "best_val_threshold": 0.65,
            "selection_metric": "validation_roc_auc_then_pr_auc",
            "test_pr_auc": 0.538,
            "test_f1": 0.52,
        },
        {
            "baseline_name": "halp_official_row_protocol",
            "method_family": "halp_official",
            "source": "halp_official_legacy_row_protocol",
            "status": "success",
            "selected_probe": "query_token_layer_35",
            "best_val_threshold": 0.5,
            "selection_metric": "eval_roc_auc_then_pr_auc",
            "test_pr_auc": 0.904,
            "test_f1": 0.796,
        },
        {
            "baseline_name": "linear_probe_final_hidden_logreg",
            "method_family": "linear_probe",
            "status": "success",
            "test_pr_auc": 0.5,
            "test_f1": 0.4,
        },
    ]

    script.append_domain_section(
        summary_path,
        domain_csv=summary_path.parent / "domain_baselines.csv",
        domain_summary=summary_path.parent / "domain_baseline_comparison.md",
        rows=rows,
        paired_metrics_long=summary_path.parent / "metrics_long.csv",
    )

    text = summary_path.read_text(encoding="utf-8")
    assert "outputs/wavelet_course_v2/reports/domain_baselines.csv" in text
    assert "outputs/stageA" not in text
    assert "best_linear_probe" in text
    assert "- best_halp_official:" not in text
    assert "- best_halp_official_mlp:" in text
    assert "- best_halp_official_row_protocol:" in text
    assert "course_grouped_halp_policy" in text
    assert "official_row_halp_policy" in text
    assert "halp_official_mlp: protocol=course-grouped" in text
    assert "halp_official_row_protocol: protocol=official-row" in text
    assert "best_mind_stage_a" not in text
    assert "best_halp_like" not in text
