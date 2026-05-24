from __future__ import annotations

import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys
import types
from types import ModuleType

import numpy as np
import pytest
import yaml


def _load_v2_cli_script() -> ModuleType:
    script_path = Path("scripts/wavelet_course_v2_run.py")
    spec = importlib.util.spec_from_file_location("wavelet_course_v2_run", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v2_config_contract_uses_requested_output_root_and_fields() -> None:
    module = _load_v2_cli_script()
    config = yaml.safe_load(
        Path("configs/wavelet_course/repope_qwen3_vl_8b_v2_paired.yaml").read_text(
            encoding="utf-8"
        )
    )

    required_top_level = {
        "experiment_name",
        "runner_version",
        "seed",
        "stage0_root",
        "output_root",
        "model_name",
        "dataset_name",
        "subsets",
        "expected_num_layers",
        "expected_hidden_dim",
        "group_key",
        "split_ratios",
        "require_positive_in_each_split",
        "allow_cpu",
        "allow_no_xgboost",
        "device",
        "paired_wavelet_v2",
        "population_grid",
        "teacher_signal",
        "ours_signal",
        "classifiers",
        "metrics",
        "quick",
    }
    required_paired_fields = {
        "enabled",
        "run_id",
        "description",
        "blocks",
        "expected_blocks",
        "expected_sources",
        "epsilon",
        "runner_module",
        "runner_functions",
        "signal_builders",
        "feature_protocols",
        "transforms",
        "window_strategies",
        "pair_definitions",
    }

    assert config["experiment_name"] == "wavelet_course_repope_qwen3_vl_8b_v2_paired"
    assert config["output_root"] == "outputs/wavelet_course_v2"
    assert module.DEFAULT_OUTPUT_ROOT == Path("outputs/wavelet_course_v2")
    assert required_top_level <= set(config)
    assert required_paired_fields <= set(config["paired_wavelet_v2"])
    assert config["paired_wavelet_v2"]["blocks"] == ["A", "B", "C", "D", "E"]
    assert config["paired_wavelet_v2"]["expected_sources"] == ["Teacher", "Ours"]
    assert config["paired_wavelet_v2"]["num_pair_ids"] == 47
    assert config["paired_wavelet_v2"]["num_pair_rows"] == 94
    assert len(config["paired_wavelet_v2"]["pair_definitions"]) == 47
    pair_definitions = {
        row["pair_id"]: row
        for row in config["paired_wavelet_v2"]["pair_definitions"]
    }
    assert pair_definitions["A_none"]["feature_protocol"] == "wavelet_summary_static_pooled"
    assert pair_definitions["A_cwt_morl_scales_1_16"]["cwt_scales"] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
        8.0,
        12.0,
        16.0,
    ]
    assert config["classifiers"]["logreg"]["max_iter"] == 20000
    assert config["classifiers"]["linear_svm"]["max_iter"] == 20000
    assert config["classifiers"]["rf"]["n_estimators"] >= 1000
    assert config["classifiers"]["rf"]["class_weight"] == "balanced_subsample"
    assert config["classifiers"]["rf"]["max_depth"] == [8, 16, None]
    assert config["classifiers"]["extra_trees"]["n_estimators"] >= 1000
    assert config["classifiers"]["extra_trees"]["max_depth"] == [8, 16, None]
    assert config["classifiers"]["xgboost"]["n_estimators"] == 5000
    assert config["classifiers"]["xgboost"]["early_stopping_rounds"] == 100
    assert config["classifiers"]["xgboost"]["max_depth"] == [2, 3, 4]
    assert config["classifiers"]["xgboost"]["learning_rate"] == [0.03, 0.1]
    for section in config["sequence_models"].values():
        assert section["max_epochs"] == 200
        assert section["patience"] == 20
        assert section["learning_rate"] == [0.001, 0.0003]


def test_sequence_readout_specs_expand_learning_rate_grid() -> None:
    from mind.wavelet_course.paired_runner import _sequence_readout_specs

    specs = _sequence_readout_specs(
        {
            "sequence_models": {
                "lstm_projected": {
                    "enabled": True,
                    "max_epochs": 200,
                    "patience": 20,
                    "batch_size": 16,
                    "learning_rate": [0.001, 0.0003],
                    "hidden_dim": 8,
                    "dropout": 0.0,
                }
            }
        }
    )

    assert [spec.params["learning_rate"] for spec in specs] == [0.001, 0.0003]
    assert [spec.classifier_name for spec in specs] == [
        "lstm_projected_lr0p001",
        "lstm_projected_lr0p0003",
    ]
    assert [spec.config_suffix for spec in specs] == [
        "lstm_projected_lr0p001",
        "lstm_projected_lr0p0003",
    ]


def test_sequence_metric_base_row_records_training_hyperparameters() -> None:
    from mind.wavelet_course.paired_config import PairSpec
    from mind.wavelet_course.paired_runner import ReadoutSpec, _base_metric_row

    pair = PairSpec(
        pair_id="C_raw_sequence_lstm_projected",
        block="C",
        source="Teacher",
        signal_builder="teacher_hidden_dim_signal",
        transform="swt",
        wavelet="db2",
        level=2,
        threshold="universal_soft",
        feature_protocol="raw_sequence",
        window_mode="global",
        window_strategy="full",
        sequence_model="lstm_projected",
    )
    spec = ReadoutSpec(
        kind="sequence",
        train_name="lstm_projected",
        classifier_name="lstm_projected_lr0p0003",
        config_suffix="lstm_projected_lr0p0003",
        params={
            "learning_rate": 0.0003,
            "max_epochs": 200,
            "patience": 20,
        },
    )

    row = _base_metric_row(pair, spec, {}, run_id="paired_wavelet_v2", quick_run=False)

    assert row["classifier"] == "lstm_projected_lr0p0003"
    assert row["sequence_model"] == "lstm_projected"
    assert row["learning_rate"] == pytest.approx(0.0003)
    assert row["max_epochs"] == 200
    assert row["patience"] == 20


def test_sequence_learning_rate_variants_have_distinct_metric_pair_ids() -> None:
    from mind.wavelet_course.paired_config import PairSpec
    from mind.wavelet_course.paired_runner import ReadoutSpec, _metric_pair_id

    pair = PairSpec(
        pair_id="C_raw_sequence_lstm_projected",
        block="C",
        source="Teacher",
        signal_builder="teacher_hidden_dim_signal",
        transform="swt",
        wavelet="db2",
        level=2,
        threshold="universal_soft",
        feature_protocol="raw_sequence",
        window_mode="global",
        window_strategy="full",
        sequence_model="lstm_projected",
    )
    lr_1e3 = ReadoutSpec(
        kind="sequence",
        train_name="lstm_projected",
        classifier_name="lstm_projected_lr0p001",
        config_suffix="lstm_projected_lr0p001",
        params={"learning_rate": 0.001},
    )
    lr_3e4 = ReadoutSpec(
        kind="sequence",
        train_name="lstm_projected",
        classifier_name="lstm_projected_lr0p0003",
        config_suffix="lstm_projected_lr0p0003",
        params={"learning_rate": 0.0003},
    )

    assert _metric_pair_id(pair, lr_1e3) == "C_raw_sequence_lstm_projected__lstm_projected_lr0p001"
    assert _metric_pair_id(pair, lr_3e4) == "C_raw_sequence_lstm_projected__lstm_projected_lr0p0003"


def test_expected_metric_pair_ids_expand_sequence_learning_rate_grid() -> None:
    from mind.wavelet_course.paired_grid import build_paired_grid
    from mind.wavelet_course.paired_runner import _expected_metric_pair_ids, _readout_specs

    config = {
        "sequence_models": {
            "lstm_projected": {
                "enabled": True,
                "max_epochs": 200,
                "patience": 20,
                "batch_size": 16,
                "learning_rate": [0.001, 0.0003],
            }
        },
        "classifiers": {"logreg": {"enabled": False}},
    }
    pairs = build_paired_grid(blocks=["B"])
    readouts = _readout_specs(config, quick_run=False)

    metric_pair_ids = _expected_metric_pair_ids(pairs, readouts, config)

    assert len(metric_pair_ids) == 12
    assert len(set(metric_pair_ids)) == 12
    assert "B_direct_raw_sequence__lstm_projected_lr0p001" in metric_pair_ids
    assert "B_direct_raw_sequence__lstm_projected_lr0p0003" in metric_pair_ids


def test_quick_pair_limit_respects_cli_block_selection() -> None:
    from mind.wavelet_course.paired_grid import build_paired_grid
    from mind.wavelet_course.paired_runner import _resolve_pairs

    config = {
        "paired_wavelet_v2": {
            "blocks": ["B"],
            "block_source": "cli",
            "pairs": [pair.as_dict() for pair in build_paired_grid(blocks=["B"])],
        },
        "quick": {
            "blocks": ["A"],
            "max_pair_ids": 1,
        },
    }

    pairs = _resolve_pairs(config, quick_run=True)

    assert {pair.block for pair in pairs} == {"B"}
    assert {pair.pair_id for pair in pairs} == {"B_direct_raw_sequence"}
    assert {pair.source for pair in pairs} == {"Teacher", "Ours"}


def test_metric_pair_ids_are_valid_completeness_keys_for_learning_rate_grid() -> None:
    from mind.wavelet_course.paired_config import PairSpec
    from mind.wavelet_course.paired_reporting import assert_paired_completeness
    from mind.wavelet_course.paired_runner import ReadoutSpec, _base_metric_row

    specs = [
        ReadoutSpec(
            kind="sequence",
            train_name="lstm_projected",
            classifier_name="lstm_projected_lr0p001",
            config_suffix="lstm_projected_lr0p001",
            params={"learning_rate": 0.001},
        ),
        ReadoutSpec(
            kind="sequence",
            train_name="lstm_projected",
            classifier_name="lstm_projected_lr0p0003",
            config_suffix="lstm_projected_lr0p0003",
            params={"learning_rate": 0.0003},
        ),
    ]
    rows = []
    for source, signal_builder in (
        ("Teacher", "teacher_hidden_dim_signal"),
        ("Ours", "ours_semantic_trace_signal"),
    ):
        pair = PairSpec(
            pair_id="B_direct_raw_sequence",
            block="B",
            source=source,
            signal_builder=signal_builder,
            transform="none",
            threshold="none",
            feature_protocol="raw_sequence",
            window_mode="global",
            window_strategy="full",
            sequence_model="lstm_projected",
        )
        for spec in specs:
            rows.append(_base_metric_row(pair, spec, {}, run_id="paired_wavelet_v2", quick_run=False))

    assert_paired_completeness(
        rows,
        expected_pair_ids=[
            "B_direct_raw_sequence__lstm_projected_lr0p001",
            "B_direct_raw_sequence__lstm_projected_lr0p0003",
        ],
    )
    grouped = Counter(row["pair_id"] for row in rows)
    assert grouped == {
        "B_direct_raw_sequence__lstm_projected_lr0p001": 2,
        "B_direct_raw_sequence__lstm_projected_lr0p0003": 2,
    }


def test_v2_metric_rows_use_required_method_family_values() -> None:
    from mind.wavelet_course.paired_config import PairSpec
    from mind.wavelet_course.paired_runner import ReadoutSpec, _base_metric_row

    common = {
        "pair_id": "A_none",
        "block": "A",
        "transform": "none",
        "feature_protocol": "wavelet_summary_static_pooled",
        "classifier": "logreg",
    }
    teacher = PairSpec(
        **common,
        source="Teacher",
        signal_builder="teacher_hidden_dim_signal",
    )
    ours = PairSpec(
        **common,
        source="Ours",
        signal_builder="ours_semantic_trace_signal",
    )
    spec = ReadoutSpec(
        kind="static",
        train_name="logreg",
        classifier_name="logreg",
        config_suffix="logreg",
    )

    assert _base_metric_row(teacher, spec, {}, run_id="paired_wavelet_v2", quick_run=False)[
        "method_family"
    ] == "teacher_bagua"
    assert _base_metric_row(ours, spec, {}, run_id="paired_wavelet_v2", quick_run=False)[
        "method_family"
    ] == "ours_wavelet"


def test_v2_device_gate_fails_closed_for_cpu_without_allow_cpu() -> None:
    module = _load_v2_cli_script()

    with pytest.raises(RuntimeError, match="allow_cpu=false"):
        module.ensure_device_available("cpu", allow_cpu=False)

    module.ensure_device_available("cpu", allow_cpu=True)


def test_v2_device_gate_validates_cuda_even_when_cpu_is_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_v2_cli_script()
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
        ),
    )

    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available\(\) is false"):
        module.ensure_device_available("cuda:0", allow_cpu=True)


def test_v2_preflight_writes_canonical_paired_grid_artifact(tmp_path: Path) -> None:
    from mind.wavelet_course.paired_grid import build_paired_grid

    module = _load_v2_cli_script()
    rows = [row.as_dict() for row in build_paired_grid(blocks=("A",))]
    config = {
        "paired_wavelet_v2": {
            "run_id": "paired_wavelet_v2",
            "blocks": ["A"],
            "expected_sources": ["Teacher", "Ours"],
            "pairs": rows,
        }
    }

    audit = module.write_paired_grid_artifacts(config, audit_dir=tmp_path)

    paired_grid_path = tmp_path / "paired_grid.json"
    paired_grid_audit_path = tmp_path / "paired_grid_audit.json"
    assert paired_grid_path.exists()
    assert paired_grid_audit_path.exists()

    paired_grid = json.loads(paired_grid_path.read_text(encoding="utf-8"))
    paired_grid_audit = json.loads(paired_grid_audit_path.read_text(encoding="utf-8"))

    assert paired_grid["rows"] == rows
    assert paired_grid["num_pair_rows"] == 38
    assert paired_grid["num_pair_ids"] == 19
    assert paired_grid_audit["paired_grid_path"] == str(paired_grid_path)
    assert paired_grid_audit["num_pair_rows"] == 38
    assert audit["num_pair_rows"] == 38


def test_v2_preflight_writes_stable_sample_grid_artifact(tmp_path: Path) -> None:
    module = _load_v2_cli_script()
    population = _minimal_population()

    audit = module.write_sample_grid_artifacts(population, audit_dir=tmp_path)

    sample_grid_path = tmp_path / "sample_grid.csv"
    sample_grid_audit_path = tmp_path / "sample_grid_audit.json"
    assert sample_grid_path.exists()
    assert sample_grid_audit_path.exists()

    with sample_grid_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        assert handle

    sample_grid_audit = json.loads(sample_grid_audit_path.read_text(encoding="utf-8"))

    assert rows
    assert set(rows[0]) == {
        "row_index",
        "population_key",
        "image_id",
        "subset",
        "split",
        "label",
        "row_order_hash",
    }
    assert len(rows) == len(population.primary_entries)
    assert {row["row_order_hash"] for row in rows} == {audit["row_order_hash"]}
    assert sample_grid_audit["sample_grid_path"] == str(sample_grid_path)
    assert sample_grid_audit["row_order_hash"] == audit["row_order_hash"]
    assert rows[0]["population_key"] == population.primary_entries[0]["wavelet_population_key"]
    assert rows[0]["image_id"] == population.primary_entries[0]["image_id"]
    assert rows[0]["split"] == population.primary_entries[0]["wavelet_split"]
    assert rows[0]["label"] == str(population.labels[0])


def test_runner_sample_grid_validation_rejects_integrity_drift() -> None:
    from mind.wavelet_course import paired_runner

    population = _minimal_population()
    entries = [dict(entry) for entry in population.primary_entries]
    labels = np.asarray(population.labels, dtype=np.int64)
    contract = paired_runner._build_sample_grid_contract(entries, labels)

    paired_runner._validate_entries_against_sample_grid(entries, labels, contract)

    missing_key_entries = [dict(entry) for entry in entries[:-1]]
    with pytest.raises(ValueError, match="row-count mismatch"):
        paired_runner._validate_entries_against_sample_grid(missing_key_entries, labels[:-1], contract)

    duplicate_key_entries = [dict(entry) for entry in entries]
    duplicate_key_entries[1]["wavelet_population_key"] = duplicate_key_entries[0]["wavelet_population_key"]
    with pytest.raises(ValueError, match="duplicate sample grid key"):
        paired_runner._validate_entries_against_sample_grid(duplicate_key_entries, labels, contract)

    missing_grid_key_entries = [dict(entry) for entry in entries]
    missing_grid_key_entries[0]["wavelet_population_key"] = '["missing"]'
    with pytest.raises(ValueError, match="missing sample grid key"):
        paired_runner._validate_entries_against_sample_grid(missing_grid_key_entries, labels, contract)

    split_drift_entries = [dict(entry) for entry in entries]
    split_drift_entries[0]["wavelet_split"] = "test"
    with pytest.raises(ValueError, match="split drift"):
        paired_runner._validate_entries_against_sample_grid(split_drift_entries, labels, contract)

    drifted_labels = labels.copy()
    drifted_labels[0] = 1 - int(drifted_labels[0])
    with pytest.raises(ValueError, match="label drift"):
        paired_runner._validate_entries_against_sample_grid(entries, drifted_labels, contract)


def test_runner_sample_grid_reader_requires_every_row_hash(tmp_path: Path) -> None:
    from mind.wavelet_course import paired_runner

    population = _minimal_population()
    entries = [dict(entry) for entry in population.primary_entries[:2]]
    labels = np.asarray(population.labels[:2], dtype=np.int64)
    rows = paired_runner._sample_grid_rows_from_entries(entries, labels)
    row_order_hash = paired_runner._sample_grid_row_order_hash(rows)
    path = tmp_path / "sample_grid.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_runner.SAMPLE_GRID_FIELDS)
        writer.writeheader()
        writer.writerow({**rows[0], "row_order_hash": row_order_hash})
        writer.writerow({**rows[1], "row_order_hash": ""})

    with pytest.raises(ValueError, match="every row must contain the same non-empty row_order_hash"):
        paired_runner._read_sample_grid_contract(path)


def test_build_split_features_uses_split_memmaps_for_large_sequence_features(tmp_path: Path) -> None:
    from mind.wavelet_course import paired_runner
    from mind.wavelet_course.paired_config import PairSpec

    population = _minimal_population()
    entries = [dict(entry) for entry in population.primary_entries]
    labels = np.asarray(population.labels, dtype=np.int64)
    contract = paired_runner._build_sample_grid_contract(entries, labels)
    pair = PairSpec(
        pair_id="B_win4_s4_window_stat28_sequence_lstm_projected",
        block="B",
        source="Teacher",
        signal_builder="teacher_hidden_dim_signal",
        transform="swt",
        wavelet="db2",
        level=2,
        threshold="universal_soft",
        feature_protocol="window_stat28_sequence",
        window_mode="win4_s4",
        window_strategy="non_overlapping",
        window_size=4,
        stride=4,
        sequence_model="lstm_projected",
    )

    result = paired_runner._build_split_features(
        pair,
        {
            "expected_num_layers": 36,
            "expected_hidden_dim": 4,
            "feature_memmap_min_bytes": 0,
        },
        entries,
        labels,
        sample_grid=contract,
        feature_kind="sequence",
        feature_dir=tmp_path,
    )

    assert not isinstance(result, Exception)
    assert result.feature_storage == "memmap"
    assert result.feature_shape == (6, 9, 112)
    assert result.train_x.shape == (2, 9, 112)
    assert result.validation_x.shape == (2, 9, 112)
    assert result.test_x.shape == (2, 9, 112)
    assert len(result.feature_paths) == 3
    assert all(Path(path).exists() for path in result.feature_paths)
    assert all(isinstance(array, np.memmap) for array in (result.train_x, result.validation_x, result.test_x))

    paired_runner._cleanup_split_feature_arrays(result)

    assert all(not Path(path).exists() for path in result.feature_paths)


def test_v2_resolve_config_populates_ours_yes_no_ids_from_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_v2_cli_script()
    calls: list[dict[str, object]] = []

    class FakeTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return {" yes": 7, " no": 11}.get(token, -1)

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str, **kwargs: object) -> FakeTokenizer:
            calls.append({"model_id": model_id, **kwargs})
            return FakeTokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer),
    )

    args = module.build_parser().parse_args(["--config", "unused", "--quick"])
    resolved = module.resolve_config(
        {"model_id": "local/qwen", "ours_signal": {}},
        args,
    )

    ours_signal = resolved["ours_signal"]
    assert ours_signal["yes_token_id"] == 7
    assert ours_signal["no_token_id"] == 11
    assert ours_signal["chosen_yes_token"] == " yes"
    assert ours_signal["chosen_no_token"] == " no"
    assert ours_signal["yes_no_trace_source"] == "final_broadcast"
    assert {
        "label": "yes",
        "candidate": " yes",
        "token_id": 7,
        "selected": True,
        "source": "convert_tokens_to_ids",
    } in ours_signal["tokenizer_candidate_table"]
    assert {
        "label": "no",
        "candidate": " no",
        "token_id": 11,
        "selected": True,
        "source": "convert_tokens_to_ids",
    } in ours_signal["tokenizer_candidate_table"]
    assert calls == [
        {
            "model_id": "local/qwen",
            "local_files_only": True,
            "trust_remote_code": True,
        }
    ]


def test_v2_resolve_config_rejects_yaml_pair_definition_drift() -> None:
    from mind.wavelet_course.paired_grid import PAIR_DEFINITIONS

    module = _load_v2_cli_script()
    args = module.build_parser().parse_args(["--config", "unused"])
    drifted = [dict(definition) for definition in PAIR_DEFINITIONS]
    drifted[0]["feature_protocol"] = "raw_sequence"

    with pytest.raises(ValueError, match="pair_definitions drift"):
        module.resolve_config(
            {
                "ours_signal": {
                    "yes_token_id": 7,
                    "no_token_id": 11,
                },
                "paired_wavelet_v2": {
                    "pair_definitions": drifted,
                },
            },
            args,
        )


def test_v2_resolved_config_writes_tokenizer_audit(tmp_path: Path) -> None:
    module = _load_v2_cli_script()
    dirs = module.ensure_output_dirs(tmp_path)
    config = {
        "ours_signal": {
            "yes_token_id": 7,
            "no_token_id": 11,
            "chosen_yes_token": " yes",
            "chosen_no_token": " no",
            "tokenizer_candidate_table": [
                {
                    "label": "yes",
                    "candidate": " yes",
                    "token_id": 7,
                    "selected": True,
                    "source": "convert_tokens_to_ids",
                }
            ],
        }
    }

    module.write_resolved_config(config, dirs)

    tokenizer_audit = json.loads((tmp_path / "audit" / "tokenizer_audit.json").read_text(encoding="utf-8"))
    assert tokenizer_audit["yes_token_id"] == 7
    assert tokenizer_audit["no_token_id"] == 11
    assert tokenizer_audit["chosen_yes_token"] == " yes"
    assert tokenizer_audit["chosen_no_token"] == " no"
    assert tokenizer_audit["tokenizer_candidate_table"] == config["ours_signal"]["tokenizer_candidate_table"]


def test_runner_resolves_ours_yes_no_ids_from_tokenizer_when_config_ids_missing() -> None:
    from mind.wavelet_course import paired_runner
    from mind.wavelet_course.paired_config import PairSpec

    class FakeTokenizer:
        def convert_tokens_to_ids(self, token: str) -> int:
            return {" yes": 7, " no": 11}.get(token, -1)

    pair = PairSpec(
        pair_id="A_none_raw_sequence_full",
        block="A",
        source="Ours",
        signal_builder="ours_semantic_trace_signal",
        transform="none",
        feature_protocol="raw_sequence",
    )
    logits = np.zeros(20, dtype=np.float32)
    logits[7] = 2.25
    logits[11] = -0.75
    signal = paired_runner._build_signal(
        pair,
        {
            "layer_vectors": np.arange(36 * 4, dtype=np.float32).reshape(36, 4),
            "first_token_logits": logits,
        },
        {
            "expected_num_layers": 36,
            "expected_hidden_dim": 4,
            "ours_signal": {"tokenizer": FakeTokenizer()},
        },
    )

    from mind.wavelet_course.signal_builders import REQUIRED_OURS_TRACE_NAMES

    assert signal.shape == (len(REQUIRED_OURS_TRACE_NAMES), 36)
    margin_index = REQUIRED_OURS_TRACE_NAMES.index("yes_no_margin_trace")
    np.testing.assert_allclose(signal[margin_index], np.full(36, 3.0, dtype=np.float32))


def test_paired_runner_fails_closed_when_xgboost_missing_and_not_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.common_classifiers import XGBoostNotInstalledError
    from mind.wavelet_course.paired_runner import run_paired_wavelet_experiment

    monkeypatch.setitem(sys.modules, "xgboost", None)

    with pytest.raises(XGBoostNotInstalledError, match="xgboost"):
        run_paired_wavelet_experiment(
            config=_minimal_xgboost_config(allow_no_xgboost=False),
            preflight={"population": _minimal_population()},
            output_root=tmp_path,
        )


def test_paired_runner_retains_xgboost_failure_rows_when_missing_is_allowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.common_classifiers import XGBOOST_NOT_INSTALLED
    from mind.wavelet_course.paired_runner import run_paired_wavelet_experiment

    monkeypatch.setitem(sys.modules, "xgboost", None)

    status = run_paired_wavelet_experiment(
        config=_minimal_xgboost_config(allow_no_xgboost=True),
        preflight={"population": _minimal_population()},
        output_root=tmp_path,
    )

    metrics_long = Path(status["report_paths"]["metrics_long"])
    with metrics_long.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert status["status"] == "success"
    assert status["failure_rows"] == 38
    assert len(rows) == 38
    assert {row["source"] for row in rows} == {"Teacher", "Ours"}
    assert all(row["status"] == "failure" for row in rows)
    assert all(row["failure_reason"] == XGBOOST_NOT_INSTALLED for row in rows)
    ledger = Path(status["metrics_ledger"])
    assert ledger == tmp_path / "reports" / "metrics_ledger.csv"
    with ledger.open(newline="", encoding="utf-8") as handle:
        ledger_rows = list(csv.DictReader(handle))
    assert ledger_rows == rows


def test_paired_runner_reports_configured_and_selected_grid_counts_for_quick(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.paired_runner import run_paired_wavelet_experiment

    monkeypatch.setitem(sys.modules, "xgboost", None)
    config = _minimal_xgboost_config(allow_no_xgboost=True)
    config["quick_run"] = True
    config["quick"] = {"max_pair_ids": 1}

    status = run_paired_wavelet_experiment(
        config=config,
        preflight={"population": _minimal_population()},
        output_root=tmp_path,
    )

    assert status["configured_grid_rows"] == 38
    assert status["configured_grid_pair_ids"] == 19
    assert status["selected_run_grid_rows"] == 2
    assert status["selected_run_grid_pair_ids"] == 1


def test_paired_runner_quick_run_writes_and_uses_selected_sample_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mind.wavelet_course.paired_runner import run_paired_wavelet_experiment

    monkeypatch.setitem(sys.modules, "xgboost", None)
    config = _minimal_xgboost_config(allow_no_xgboost=True)
    config["quick_run"] = True
    config["quick"] = {"max_pair_ids": 1, "max_samples_per_split": 1}

    status = run_paired_wavelet_experiment(
        config=config,
        preflight={"population": _minimal_population()},
        output_root=tmp_path,
    )

    configured_path = tmp_path / "audit" / "configured_sample_grid.csv"
    selected_path = tmp_path / "audit" / "selected_sample_grid.csv"
    assert configured_path.exists()
    assert selected_path.exists()
    assert status["configured_sample_grid_path"] == str(configured_path)
    assert status["configured_sample_grid_rows"] == 6
    assert status["sample_grid_path"] == str(selected_path)
    assert status["sample_grid_rows"] == 3

    configured_rows = list(csv.DictReader(configured_path.open(newline="", encoding="utf-8")))
    selected_rows = list(csv.DictReader(selected_path.open(newline="", encoding="utf-8")))
    assert len(configured_rows) == 6
    assert len(selected_rows) == 3
    assert selected_rows != configured_rows
    selected_hashes = {row["row_order_hash"] for row in selected_rows}
    assert selected_hashes == {status["sample_grid_row_order_hash"]}
    assert "" not in selected_hashes
    assert {row["population_key"] for row in selected_rows} < {
        row["population_key"] for row in configured_rows
    }


def test_main_overwrites_stale_success_reports_on_failed_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_v2_cli_script()
    output_root = tmp_path / "wavelet_course_v2"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "summary.md").write_text("final_status=success\nold success", encoding="utf-8")
    (reports_dir / "metrics_long.csv").write_text(
        "run_id,status,failure_reason\nold,success,\n",
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(output_root),
                "device": "cpu",
                "allow_cpu": True,
                "ours_signal": {"yes_token_id": 7, "no_token_id": 11},
                "paired_wavelet_v2": {"run_id": "failed_test_run"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "ensure_device_available", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        module,
        "run_preflight",
        lambda *args, **kwargs: {
            "cache_audit": {"accepted": True, "num_entries": 6},
            "population_summary": {
                "num_primary_population": 6,
                "num_hard_hallucination": 3,
                "num_correct": 3,
            },
            "split_validation": {
                "valid": True,
                "counts": {
                    "train": {"pos": 1, "neg": 1},
                    "validation": {"pos": 1, "neg": 1},
                    "test": {"pos": 1, "neg": 1},
                },
            },
            "sample_grid_audit": {
                "sample_grid_path": str(output_root / "audit" / "selected_sample_grid.csv"),
                "num_rows": 6,
                "row_order_hash": "hash123",
            },
        },
    )

    def _raise_run_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("runner exploded")

    monkeypatch.setattr(module, "run_full", _raise_run_failure)

    exit_code = module.main(["--config", str(config_path), "--allow-cpu", "--device", "cpu"])

    assert exit_code == 1
    status = json.loads((reports_dir / "full_run_status.json").read_text(encoding="utf-8"))
    summary = (reports_dir / "summary.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((reports_dir / "metrics_long.csv").open(newline="", encoding="utf-8")))
    assert status["status"] == "failed"
    assert status["run_id"] == "failed_test_run"
    assert status["metrics_ledger"] == str(reports_dir / "metrics_ledger.csv")
    assert "final_status: failed" in summary
    assert "runner exploded" in summary
    assert "old success" not in summary
    assert rows == [
        {
            **rows[0],
            "run_id": "failed_test_run",
            "status": "failed",
            "failure_reason": "runner exploded",
        }
    ]


def test_main_cleans_stale_run_outputs_before_new_run(
    tmp_path: Path,
) -> None:
    module = _load_v2_cli_script()
    output_root = tmp_path / "wavelet_course_v2"
    reports_dir = output_root / "reports"
    features_dir = output_root / "features"
    curves_dir = reports_dir / "training_curves"
    curves_dir.mkdir(parents=True)
    features_dir.mkdir(parents=True)
    for name in (
        "metrics_long.csv",
        "metrics_wide_paired.csv",
        "summary.md",
        "failure_report.csv",
        "metrics_ledger.csv",
        "full_run_status.json",
    ):
        (reports_dir / name).write_text("old success\n", encoding="utf-8")
    (curves_dir / "old_curve.csv").write_text("epoch,metric\n", encoding="utf-8")
    (features_dir / "feature_shape_manifest.csv").write_text("old feature manifest\n", encoding="utf-8")

    dirs = module.ensure_output_dirs(output_root)
    removed = module.clean_stale_run_outputs(dirs)

    assert sorted(removed) == sorted(
        [
            str(reports_dir / "metrics_long.csv"),
            str(reports_dir / "metrics_wide_paired.csv"),
            str(reports_dir / "summary.md"),
            str(reports_dir / "failure_report.csv"),
            str(reports_dir / "metrics_ledger.csv"),
            str(reports_dir / "full_run_status.json"),
            str(curves_dir),
            str(features_dir / "feature_shape_manifest.csv"),
        ]
    )
    assert not (reports_dir / "metrics_long.csv").exists()
    assert not curves_dir.exists()
    assert reports_dir.exists()


def test_main_overwrites_stale_success_reports_on_preflight_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_v2_cli_script()
    output_root = tmp_path / "wavelet_course_v2"
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True)
    (reports_dir / "summary.md").write_text("final_status=success\nold success", encoding="utf-8")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "output_root": str(output_root),
                "device": "cpu",
                "allow_cpu": True,
                "ours_signal": {"yes_token_id": 7, "no_token_id": 11},
                "paired_wavelet_v2": {"run_id": "preflight_failed_run"},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(module, "ensure_device_available", lambda *args, **kwargs: {})

    def _raise_preflight_failure(*args: object, **kwargs: object) -> object:
        raise RuntimeError("cache missing")

    monkeypatch.setattr(module, "run_preflight", _raise_preflight_failure)

    exit_code = module.main(["--config", str(config_path), "--allow-cpu", "--device", "cpu"])

    assert exit_code == 2
    status = json.loads((reports_dir / "full_run_status.json").read_text(encoding="utf-8"))
    summary = (reports_dir / "summary.md").read_text(encoding="utf-8")
    assert status["run_id"] == "preflight_failed_run"
    assert status["status"] == "failed"
    assert status["failure_reason"] == "cache missing"
    assert "final_status: failed" in summary
    assert "old success" not in summary


def test_final_terminal_summary_includes_required_v2_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_v2_cli_script()
    output_root = tmp_path / "wavelet_course_v2"
    preflight = {
        "cache_audit": {"accepted": True, "num_entries": 10},
        "population_summary": {
            "num_primary_population": 10,
            "num_hard_hallucination": 4,
            "num_correct": 6,
        },
        "split_validation": {
            "valid": True,
            "counts": {
                "train": {"pos": 3, "neg": 3},
                "validation": {"pos": 1, "neg": 1},
                "test": {"pos": 1, "neg": 1},
            },
        },
        "paired_grid_audit": {
            "paired_grid_path": str(output_root / "audit" / "paired_grid.json"),
            "num_pair_rows": 2,
            "num_pair_ids": 1,
            "requested_blocks": ["A"],
        },
        "sample_grid_audit": {
            "sample_grid_path": str(output_root / "audit" / "selected_sample_grid.csv"),
            "num_rows": 10,
            "row_order_hash": "abc123",
        },
    }
    status = {
        "status": "success",
        "metrics_long_rows": 2,
        "success_rows": 1,
        "failure_rows": 1,
        "pair_rows": 2,
        "pair_ids": 1,
        "report_paths": {
            "metrics_long": str(output_root / "reports" / "metrics_long.csv"),
            "metrics_wide_paired": str(output_root / "reports" / "metrics_wide_paired.csv"),
            "failure_report": str(output_root / "reports" / "failure_report.csv"),
            "summary": str(output_root / "reports" / "summary.md"),
        },
        "sample_grid_path": str(output_root / "audit" / "selected_sample_grid.csv"),
        "sample_grid_rows": 10,
        "sample_grid_row_order_hash": "abc123",
        "metrics_ledger": str(output_root / "reports" / "metrics_ledger.csv"),
    }

    module.print_final_summary(
        config={
            "output_root": str(output_root),
            "population_grid": {"source": "v1_wavelet_population"},
        },
        preflight=preflight,
        status=status,
        output_root=output_root,
    )

    out = capsys.readouterr().out
    assert "final_status=success" in out
    assert "v2_paired_extension=true" in out
    assert "v1_preservation=v1_wavelet_population" in out
    assert "output_root=" + str(output_root) in out
    assert "primary_population=10" in out
    assert "hard_hallucinations=4" in out
    assert "correct=6" in out
    assert "train_pos=3" in out
    assert "train_neg=3" in out
    assert "validation_pos=1" in out
    assert "validation_neg=1" in out
    assert "test_pos=1" in out
    assert "test_neg=1" in out
    assert f"paired_grid_path={output_root / 'audit' / 'paired_grid.json'}" in out
    assert "paired_grid_rows=2" in out
    assert "paired_grid_pair_ids=1" in out
    assert f"metrics_long={output_root / 'reports' / 'metrics_long.csv'}" in out
    assert f"metrics_wide_paired={output_root / 'reports' / 'metrics_wide_paired.csv'}" in out
    assert f"failure_report={output_root / 'reports' / 'failure_report.csv'}" in out
    assert f"summary_md={output_root / 'reports' / 'summary.md'}" in out
    assert f"sample_grid_path={output_root / 'audit' / 'selected_sample_grid.csv'}" in out
    assert "sample_grid_rows=10" in out
    assert "sample_grid_row_order_hash=abc123" in out
    assert f"metrics_ledger={output_root / 'reports' / 'metrics_ledger.csv'}" in out
    assert "failure_rows=1" in out
    assert "limitations=failed configs remain in reports; paired comparison is limited to comparable successes" in out
    assert "conclusion=see summary.md for paired results and wavelet rationale" in out


def test_v2_device_gate_validates_cuda_ordinal_and_writes_memory_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_v2_cli_script()

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 1,
        get_device_name=lambda ordinal: f"Fake GPU {ordinal}",
        mem_get_info=lambda ordinal=None: (123, 456),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))

    audit = module.ensure_device_available("cuda:0", allow_cpu=False, audit_dir=tmp_path)

    assert audit["device"] == "cuda:0"
    assert audit["cuda_ordinal"] == 0
    assert audit["cuda_device_count"] == 1
    assert audit["gpu_name"] == "Fake GPU 0"
    assert audit["memory_free_bytes"] == 123
    assert audit["memory_total_bytes"] == 456
    audit_path = tmp_path / "cuda_device_audit.json"
    assert json.loads(audit_path.read_text(encoding="utf-8")) == audit

    with pytest.raises(RuntimeError, match="cuda ordinal 1"):
        module.ensure_device_available("cuda:1", allow_cpu=False, audit_dir=tmp_path)


def test_v2_device_gate_accepts_multi_gpu_spec_and_writes_memory_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_v2_cli_script()

    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 3,
        get_device_name=lambda ordinal: f"Fake GPU {ordinal}",
        mem_get_info=lambda ordinal=None: (100 + int(ordinal or 0), 200 + int(ordinal or 0)),
    )
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=fake_cuda))

    audit = module.ensure_device_available("cuda:0,1", allow_cpu=False, audit_dir=tmp_path)

    assert audit["device"] == "cuda:0,1"
    assert audit["device_type"] == "cuda"
    assert audit["cuda_ordinal"] == 0
    assert audit["cuda_primary_ordinal"] == 0
    assert audit["cuda_ordinals"] == [0, 1]
    assert audit["cuda_device_count"] == 3
    assert audit["data_parallel"] is True
    assert audit["cuda_devices"] == [
        {
            "cuda_ordinal": 0,
            "gpu_name": "Fake GPU 0",
            "memory_free_bytes": 100,
            "memory_total_bytes": 200,
        },
        {
            "cuda_ordinal": 1,
            "gpu_name": "Fake GPU 1",
            "memory_free_bytes": 101,
            "memory_total_bytes": 201,
        },
    ]
    assert json.loads((tmp_path / "cuda_device_audit.json").read_text(encoding="utf-8")) == audit

    all_audit = module.ensure_device_available("cuda:all", allow_cpu=False, audit_dir=tmp_path)

    assert all_audit["cuda_ordinals"] == [0, 1, 2]
    assert [row["gpu_name"] for row in all_audit["cuda_devices"]] == [
        "Fake GPU 0",
        "Fake GPU 1",
        "Fake GPU 2",
    ]

    with pytest.raises(RuntimeError, match="cuda ordinal 3"):
        module.ensure_device_available("cuda:0,3", allow_cpu=False, audit_dir=tmp_path)


def _minimal_xgboost_config(*, allow_no_xgboost: bool) -> dict[str, object]:
    from mind.wavelet_course.paired_grid import build_paired_grid

    pairs = [row.as_dict() for row in build_paired_grid(blocks=("A",))]
    for row in pairs:
        row["classifier"] = "xgboost"
    return {
        "run_id": "test_v2",
        "model_name": "test-model",
        "dataset_name": "repope",
        "subsets": ["popular"],
        "seed": 7,
        "quick_run": False,
        "expected_num_layers": 36,
        "expected_hidden_dim": 4,
        "allow_no_xgboost": allow_no_xgboost,
        "paired_wavelet_v2": {
            "run_id": "test_v2",
            "blocks": ["A"],
            "expected_sources": ["Teacher", "Ours"],
            "pairs": pairs,
        },
        "ours_signal": {
            "yes_token_id": 1,
            "no_token_id": 2,
            "yes_no_trace_source": "final_broadcast",
        },
        "classifiers": {
            "xgboost": {
                "enabled": True,
                "n_estimators": 1,
                "n_jobs": 1,
            }
        },
    }


def _minimal_population() -> object:
    from mind.wavelet_course.population import WaveletPopulation

    entries: list[dict[str, object]] = []
    labels: list[int] = []
    splits = ("train", "train", "validation", "validation", "test", "test")
    split_labels = (0, 1, 0, 1, 0, 1)
    for index, (split, label) in enumerate(zip(splits, split_labels, strict=True)):
        logits = np.zeros(4, dtype=np.float32)
        logits[1] = 1.0 + float(index)
        logits[2] = -1.0
        entries.append(
            {
                "model_name": "test-model",
                "dataset_name": "repope",
                "subset": "popular",
                "sample_id": f"sample-{index}",
                "image_id": f"image-{index // 2}",
                "wavelet_population_key": json.dumps(
                    ["test-model", "repope", "popular", f"sample-{index}"],
                    separators=(",", ":"),
                ),
                "layer_vectors": (
                    np.arange(36 * 4, dtype=np.float32).reshape(36, 4)
                    + float(index)
                ),
                "first_token_logits": logits,
                "wavelet_split": split,
                "wavelet_label": label,
            }
        )
        labels.append(label)
    return WaveletPopulation(
        primary_entries=entries,
        labels=labels,
        assignments={},
        audit_rows=[],
        split_source="test",
    )
