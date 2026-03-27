from __future__ import annotations

from pathlib import Path

import yaml

from mind.config import ExperimentConfig, load_yaml_config


def test_load_yaml_config_builds_typed_experiment_config(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.yaml"
    payload = {
        "name": "smoke-qwen35",
        "model": {
            "name": "qwen3.5-4b",
            "model_id": "Qwen/Qwen3.5-4B",
            "family": "qwen",
            "dtype": "float16",
            "attn_implementation": "sdpa",
        },
        "dataset": {
            "name": "pope",
            "root": "data/pope",
            "splits": ["popular"],
            "prompt_template": "Answer yes or no: {question}",
        },
        "runtime": {
            "device": "cuda",
            "batch_size": 1,
            "num_workers": 2,
            "selected_layers": 16,
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_yaml_config(config_path, ExperimentConfig)

    assert config.name == "smoke-qwen35"
    assert config.model.model_id == "Qwen/Qwen3.5-4B"
    assert config.dataset.splits == ["popular"]
    assert config.runtime.selected_layers == 16
