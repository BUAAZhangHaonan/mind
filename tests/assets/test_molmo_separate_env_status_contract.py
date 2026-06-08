from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch

from mind.models.asset_validation import AssetStatus, build_completion_summary
from mind.models.registry import REQUIRED_MODEL_ALIASES


def _load_acceptance_module():
    path = Path("scripts/asset_accept_separate_env.py")
    spec = importlib.util.spec_from_file_location("asset_accept_separate_env", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_valid_source(root: Path) -> None:
    alias = "molmo-7b-d-0924"
    (root / "smoke_cache" / alias / "pope" / "popular").mkdir(parents=True)
    (root / "smoke_cache" / alias / "repope" / "popular").mkdir(parents=True)
    (root / "smoke_cache" / alias / "dash-b" / "all").mkdir(parents=True)
    summary = {
        "smoke_limit": 2,
        "model_statuses": {alias: AssetStatus.VERIFIED.value},
        "verified_models": [alias],
    }
    (root / "asset_completion_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    smoke_lines = ["model_alias,dataset,subset,status,reason,shard_path,sidecar_path,num_records"]
    validation_lines = ["model_alias,dataset,subset,status,reason,shard_path,num_entries"]
    for dataset, subset in (("pope", "popular"), ("repope", "popular"), ("dash-b", "all")):
        shard = root / "smoke_cache" / alias / dataset / subset / "shard-00000.pt"
        sidecar = Path(str(shard) + ".json")
        entry = {
            "sample_id": "1",
            "selected_layers": list(range(2)),
            "layer_vectors": torch.ones(2, 4),
            "first_token_logits": torch.ones(5),
            "answer_text": "Yes",
            "parsed_answer": 1,
            "token_index": 1,
            "prompt_template_id": "molmo_single_image_raw_question_v1",
        }
        torch.save([entry], shard)
        sidecar.write_text(
            json.dumps(
                {
                    "model_alias": alias,
                    "model_family": "molmo",
                    "local_path": "/models/molmo",
                    "wrapper_class": "MolmoWrapper",
                    "processor_class": "MolmoProcessor",
                    "model_class": "MolmoForCausalLM",
                    "total_layers": 2,
                    "hidden_dim": 4,
                    "hidden_state_index_offset": 1,
                    "hidden_state_count": 3,
                    "selected_layers": list(range(2)),
                    "selected_layer_hidden_state_indices": [1, 2],
                    "token_index": 1,
                    "prompt_template_id": "molmo_single_image_raw_question_v1",
                    "deterministic_generation_kwargs": {"do_sample": False, "temperature": 0, "max_new_tokens": 1},
                    "thinking_disabled": True,
                    "trust_remote_code": True,
                    "validation_commit": "abc123",
                }
            ),
            encoding="utf-8",
        )
        smoke_lines.append(f"{alias},{dataset},{subset},verified,,{shard},{sidecar},2")
        validation_lines.append(f"{alias},{dataset},{subset},verified,,{shard},2")
    (root / "smoke_extraction_report.csv").write_text("\n".join(smoke_lines) + "\n", encoding="utf-8")
    (root / "hidden_state_validation_report.csv").write_text("\n".join(validation_lines) + "\n", encoding="utf-8")
    (root / "validation_checksums.json").write_text(
        json.dumps(
            {
                "determinism": {
                    f"{alias}/pope/popular": {"status": "verified"},
                    f"{alias}/repope/popular": {"status": "verified"},
                    f"{alias}/dash-b/all": {"status": "verified"},
                },
                "image_sensitivity_canary": {
                    alias: {
                        "status": "verified",
                        "dataset": "pope",
                        "subset": "popular",
                        "first_checksum": "a",
                        "second_checksum": "b",
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def test_separate_env_acceptance_marks_molmo_verified_separate_env(tmp_path: Path) -> None:
    module = _load_acceptance_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_valid_source(source)

    report = module.accept_separate_env(model="molmo-7b-d-0924", source_root=source, output_root=output)

    assert report["status"] == "verified_separate_env"
    assert report["copied_tensors"] is False
    assert (output / "molmo_separate_env_acceptance.json").is_file()
    assert (output / "MOLMO_SEPARATE_ENV_ACCEPTANCE.md").is_file()


def test_missing_separate_env_artifacts_keep_molmo_blocked(tmp_path: Path) -> None:
    module = _load_acceptance_module()

    report = module.accept_separate_env(model="molmo-7b-d-0924", source_root=tmp_path / "missing", output_root=tmp_path / "output")

    assert report["status"] == AssetStatus.BLOCKED.value
    assert "not verified" in report["reason"] or "missing" in report["reason"]


def test_separate_env_acceptance_requires_checksums_and_canary(tmp_path: Path) -> None:
    module = _load_acceptance_module()
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_valid_source(source)
    (source / "validation_checksums.json").unlink()

    report = module.accept_separate_env(model="molmo-7b-d-0924", source_root=source, output_root=output)

    assert report["status"] == AssetStatus.BLOCKED.value
    assert "checksum" in report["reason"] or "canary" in report["reason"]


def test_verified_separate_env_is_not_main_env_verified() -> None:
    statuses = {alias: AssetStatus.VERIFIED.value for alias in REQUIRED_MODEL_ALIASES}
    statuses["molmo-7b-d-0924"] = "verified_separate_env"

    summary = build_completion_summary(
        model_statuses=statuses,
        model_reasons={"molmo-7b-d-0924": "accepted from separate env"},
        smoke_datasets=("pope", "repope", "dash-b"),
        smoke_limit=2,
        tests_run=(),
        git_commit="abc123",
    )

    assert "molmo-7b-d-0924" not in summary["verified_models"]
    assert "molmo-7b-d-0924" in summary["verified_separate_env_models"]
    assert summary["final_status"] == "passed"
