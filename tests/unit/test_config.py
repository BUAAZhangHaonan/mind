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
            "image_root": "data/coco/val2014",
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
    assert config.dataset.image_root == "data/coco/val2014"
    assert config.runtime.selected_layers == 16


def test_load_yaml_config_leaves_attention_backend_unset_when_key_missing(tmp_path: Path) -> None:
    config_path = tmp_path / "model.yaml"
    payload = {
        "name": "qwen3-vl-8b",
        "model_id": "Qwen/Qwen3-VL-8B-Instruct",
        "family": "qwen_vl",
        "dtype": "float16",
        "trust_remote_code": True,
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    config = load_yaml_config(config_path, ExperimentConfig.model_fields["model"].annotation)

    assert config.attn_implementation is None
