from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


cache_reference_states = _load_script("cache_reference_states")
compute_drift = _load_script("compute_drift")
run_experiment = _load_script("run_experiment")


def test_build_reference_cache_output_path_uses_model_dataset_split_layout(tmp_path: Path) -> None:
    output_path = cache_reference_states.build_reference_cache_output_path(
        output_root=tmp_path,
        model_name="qwen3-vl-8b",
        dataset_name="pope-reference",
        split="train",
        shard_index=2,
    )

    assert output_path == tmp_path / "qwen3-vl-8b" / "pope-reference" / "train" / "shard-00002.pt"


def test_build_feature_output_path_uses_experiment_and_split_layout(tmp_path: Path) -> None:
    output_path = compute_drift.build_feature_output_path(
        output_root=tmp_path,
        experiment_name="smoke-qwen3-vl",
        split="popular",
    )

    assert output_path == tmp_path / "smoke-qwen3-vl" / "popular.parquet"


def test_parse_stage_list_supports_csv_and_all() -> None:
    assert run_experiment.parse_stage_list("prepare,extract,train") == ["prepare", "extract", "train"]
    assert run_experiment.parse_stage_list("all") == run_experiment.DEFAULT_STAGES
