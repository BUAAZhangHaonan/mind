from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from mind.data import HallucinationRecord
from mind.models.asset_validation import AssetStatus
from mind.models.registry import REQUIRED_MODEL_ALIASES

from scripts import asset_smoke_extract


def test_smoke_image_path_resolver_preserves_existing_repo_relative_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    existing = tmp_path / "data" / "coco" / "val2014" / "COCO_val2014_000000000001.jpg"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"not a real jpeg")
    record = HallucinationRecord(
        sample_id="s0",
        image_id=1,
        image_path="data/coco/val2014/COCO_val2014_000000000001.jpg",
        question="Is there a dog in the image?",
        label=1,
        object_name="dog",
        split="popular",
        subset="popular",
        source_dataset="pope",
    )

    resolved = asset_smoke_extract.resolve_record_image_path(record, dataset_name="pope")

    assert resolved.image_path == "data/coco/val2014/COCO_val2014_000000000001.jpg"


def test_unsupported_and_blocked_models_are_recorded_but_not_loaded(monkeypatch, tmp_path: Path) -> None:
    datasets = ("pope", "repope", "dash-b")
    loaded: list[str] = []
    rows = []

    def fake_load_asset_registry(path: Path):
        models = [
            SimpleNamespace(alias=alias, model_config_path="unused.yaml", local_path=f"/models/{alias}")
            for alias in REQUIRED_MODEL_ALIASES
        ]
        return SimpleNamespace(models=models)

    class FakeAudit:
        def __init__(self, alias: str, status: AssetStatus, reason: str) -> None:
            self.alias = alias
            self.status = status
            self.reason = reason

    def fake_audit(asset):
        if asset.alias == REQUIRED_MODEL_ALIASES[0]:
            return FakeAudit(asset.alias, AssetStatus.VERIFIED, "ok")
        if asset.alias == REQUIRED_MODEL_ALIASES[1]:
            return FakeAudit(asset.alias, AssetStatus.UNSUPPORTED_BY_WRAPPER, "no wrapper")
        return FakeAudit(asset.alias, AssetStatus.BLOCKED, "blocked")

    def fake_load_config(path, model_type):
        return SimpleNamespace(
            name=REQUIRED_MODEL_ALIASES[0],
            family="qwen3_vl",
            dtype="float16",
            trust_remote_code=False,
            thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        )

    class FakeWrapper:
        def load_processor(self):
            loaded.append("processor")
            return object()

        def load_model(self, *, device: str):
            loaded.append(device)
            return object()

        def resolve_total_layers(self, model):
            return 1

        def resolve_hidden_dim(self, model):
            return 2

        def resolve_hidden_state_index_offset(self):
            return 1

        def prompt_template_id(self):
            return "fake"

        def prompt_template_text(self):
            return "fake"

        def deterministic_generation_kwargs(self, *, max_new_tokens: int):
            return {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0,
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_hidden_states": True,
            }

        def disable_thinking_kwargs(self):
            return {}

    def fake_create_model_wrapper(config):
        return FakeWrapper()

    def fake_extract_entries(**kwargs):
        return (
            [
                {
                    "sample_id": "s0",
                    "token_index": 1,
                    "layer_vectors": asset_smoke_extract.torch.tensor([[1.0, 0.0]]),
                    "first_token_logits": asset_smoke_extract.torch.tensor([1.0]),
                    "answer_text": "Yes",
                    "parsed_answer": 1,
                    "selected_layers": [0],
                }
            ],
            2,
        )

    def fake_save(entries, path, **kwargs):
        rows.append(str(path))
        return {"actual_file_bytes": 1}

    monkeypatch.setattr(asset_smoke_extract, "load_asset_registry", fake_load_asset_registry)
    monkeypatch.setattr(asset_smoke_extract, "audit_asset_metadata", fake_audit)
    monkeypatch.setattr(asset_smoke_extract, "required_dataset_paths", lambda **kwargs: [])
    monkeypatch.setattr(asset_smoke_extract, "load_smoke_records", lambda **kwargs: [SimpleNamespace(question="q", image_path="i")])
    monkeypatch.setattr(asset_smoke_extract, "load_yaml_config", fake_load_config)
    monkeypatch.setattr(asset_smoke_extract, "merge_asset_model_config", lambda config, asset: config)
    monkeypatch.setattr(asset_smoke_extract, "create_model_wrapper", fake_create_model_wrapper)
    monkeypatch.setattr(asset_smoke_extract, "extract_entries", fake_extract_entries)
    monkeypatch.setattr(asset_smoke_extract, "save_prefill_cache_shard", fake_save)
    monkeypatch.setattr(asset_smoke_extract, "merge_top_level_sidecar_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(asset_smoke_extract, "estimate_prefill_cache_tensor_bytes", lambda *args, **kwargs: 1)
    monkeypatch.setattr(asset_smoke_extract, "run_image_sensitivity_canary", lambda **kwargs: {"status": "skipped_with_reason", "reason": "test"})

    result = asset_smoke_extract.run_smoke(
        registry_path=Path("registry.yaml"),
        output_root=tmp_path,
        stage0_root=tmp_path / "stage0",
        datasets=datasets,
        smoke_limit=2,
        device="cuda:0",
    )

    assert result == 0
    assert loaded == ["processor", "cuda:0"]
    report = (tmp_path / "smoke_extraction_report.csv").read_text(encoding="utf-8")
    assert "unsupported_by_wrapper" in report
    assert "blocked" in report
    summary = (tmp_path / "asset_completion_summary.json").read_text(encoding="utf-8")
    assert '"final_status": "blocked"' in summary


def test_scoped_smoke_only_loads_selected_supported_model(monkeypatch, tmp_path: Path) -> None:
    datasets = ("pope", "repope", "dash-b")
    selected = REQUIRED_MODEL_ALIASES[2]
    loaded: list[str] = []

    def fake_load_asset_registry(path: Path):
        models = [
            SimpleNamespace(alias=alias, model_config_path="unused.yaml", local_path=f"/models/{alias}")
            for alias in REQUIRED_MODEL_ALIASES
        ]
        return SimpleNamespace(models=models)

    class FakeAudit:
        def __init__(self, alias: str, status: AssetStatus, reason: str) -> None:
            self.alias = alias
            self.status = status
            self.reason = reason

    def fake_audit(asset):
        return FakeAudit(asset.alias, AssetStatus.VERIFIED, "ok")

    class FakeWrapper:
        def load_processor(self):
            loaded.append("processor")
            return object()

        def load_model(self, *, device: str):
            loaded.append(device)
            return object()

        def resolve_total_layers(self, model):
            return 1

        def resolve_hidden_dim(self, model):
            return 2

        def resolve_hidden_state_index_offset(self):
            return 1

        def prompt_template_id(self):
            return "fake"

        def prompt_template_text(self):
            return "fake"

        def deterministic_generation_kwargs(self, *, max_new_tokens: int):
            return {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0,
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_hidden_states": True,
            }

        def disable_thinking_kwargs(self):
            return {}

    def fake_extract_entries(**kwargs):
        return (
            [
                {
                    "sample_id": "s0",
                    "token_index": 1,
                    "layer_vectors": asset_smoke_extract.torch.tensor([[1.0, 0.0]]),
                    "first_token_logits": asset_smoke_extract.torch.tensor([1.0]),
                    "answer_text": "Yes",
                    "parsed_answer": 1,
                    "selected_layers": [0],
                }
            ],
            2,
        )

    monkeypatch.setattr(asset_smoke_extract, "load_asset_registry", fake_load_asset_registry)
    monkeypatch.setattr(asset_smoke_extract, "audit_asset_metadata", fake_audit)
    monkeypatch.setattr(asset_smoke_extract, "required_dataset_paths", lambda **kwargs: [])
    monkeypatch.setattr(asset_smoke_extract, "load_smoke_records", lambda **kwargs: [SimpleNamespace(question="q", image_path="i")])
    monkeypatch.setattr(
        asset_smoke_extract,
        "load_yaml_config",
        lambda path, model_type: SimpleNamespace(
            name=selected,
            family="qwen3_vl",
            dtype="float16",
            trust_remote_code=False,
            thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        ),
    )
    monkeypatch.setattr(asset_smoke_extract, "merge_asset_model_config", lambda config, asset: config)
    monkeypatch.setattr(asset_smoke_extract, "create_model_wrapper", lambda config: FakeWrapper())
    monkeypatch.setattr(asset_smoke_extract, "extract_entries", fake_extract_entries)
    monkeypatch.setattr(asset_smoke_extract, "save_prefill_cache_shard", lambda *args, **kwargs: {"actual_file_bytes": 1})
    monkeypatch.setattr(asset_smoke_extract, "merge_top_level_sidecar_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(asset_smoke_extract, "estimate_prefill_cache_tensor_bytes", lambda *args, **kwargs: 1)
    monkeypatch.setattr(asset_smoke_extract, "run_image_sensitivity_canary", lambda **kwargs: {"status": "skipped_with_reason", "reason": "test"})

    result = asset_smoke_extract.run_smoke(
        registry_path=Path("registry.yaml"),
        output_root=tmp_path,
        stage0_root=tmp_path / "stage0",
        datasets=datasets,
        smoke_limit=2,
        device="cuda:0",
        models=[selected],
    )

    assert result == 0
    assert loaded == ["processor", "cuda:0"]
    report = (tmp_path / "smoke_extraction_report.csv").read_text(encoding="utf-8")
    assert report.count(",verified,") == 3
    assert "not_attempted_due_to_dependency" in report


def test_molmo_without_separate_env_manifest_is_blocked_not_loaded(monkeypatch, tmp_path: Path) -> None:
    datasets = ("pope", "repope", "dash-b")

    def fake_load_asset_registry(path: Path):
        models = [
            SimpleNamespace(alias=alias, model_config_path="unused.yaml", local_path=f"/models/{alias}")
            for alias in REQUIRED_MODEL_ALIASES
        ]
        return SimpleNamespace(models=models)

    class FakeAudit:
        def __init__(self, alias: str) -> None:
            self.alias = alias
            self.status = AssetStatus.VERIFIED
            self.reason = "ok"

    def fail_if_loaded(config):
        raise AssertionError("Molmo must not be loaded in the main smoke process")

    monkeypatch.setattr(asset_smoke_extract, "load_asset_registry", fake_load_asset_registry)
    monkeypatch.setattr(asset_smoke_extract, "audit_asset_metadata", lambda asset: FakeAudit(asset.alias))
    monkeypatch.setattr(asset_smoke_extract, "required_dataset_paths", lambda **kwargs: [])
    monkeypatch.setattr(asset_smoke_extract, "load_smoke_records", lambda **kwargs: [SimpleNamespace(question="q", image_path="i")])
    monkeypatch.setattr(asset_smoke_extract, "create_model_wrapper", fail_if_loaded)

    result = asset_smoke_extract.run_smoke(
        registry_path=Path("registry.yaml"),
        output_root=tmp_path,
        stage0_root=tmp_path / "stage0",
        datasets=datasets,
        smoke_limit=2,
        device="cuda:0",
        models=["molmo-7b-d-0924"],
    )

    assert result == 0
    report = (tmp_path / "smoke_extraction_report.csv").read_text(encoding="utf-8")
    assert "molmo-7b-d-0924" in report
    assert "main-env smoke loading is disabled" in report
    summary = json.loads((tmp_path / "asset_completion_summary.json").read_text(encoding="utf-8"))
    assert summary["model_statuses"]["molmo-7b-d-0924"] == AssetStatus.BLOCKED.value


def test_gemma4_separate_env_manifest_is_recorded_not_loaded(monkeypatch, tmp_path: Path) -> None:
    datasets = ("pope", "repope", "dash-b")
    alias = "gemma-4-12b-it"
    (tmp_path / "gemma_4_12b_it_separate_env_acceptance.json").write_text(
        json.dumps(
            {
                "model_alias": alias,
                "status": AssetStatus.VERIFIED_SEPARATE_ENV.value,
                "reason": "accepted from Gemma4 separate environment",
            }
        ),
        encoding="utf-8",
    )

    def fake_load_asset_registry(path: Path):
        models = [
            SimpleNamespace(alias=model_alias, model_config_path="unused.yaml", local_path=f"/models/{model_alias}")
            for model_alias in REQUIRED_MODEL_ALIASES
        ]
        return SimpleNamespace(models=models)

    class FakeAudit:
        def __init__(self, model_alias: str) -> None:
            self.alias = model_alias
            self.status = AssetStatus.VERIFIED
            self.reason = "ok"

    def fail_if_loaded(config):
        raise AssertionError("Gemma4 must not be loaded in the main smoke process after separate-env acceptance")

    monkeypatch.setattr(asset_smoke_extract, "load_asset_registry", fake_load_asset_registry)
    monkeypatch.setattr(asset_smoke_extract, "audit_asset_metadata", lambda asset: FakeAudit(asset.alias))
    monkeypatch.setattr(asset_smoke_extract, "required_dataset_paths", lambda **kwargs: [])
    monkeypatch.setattr(asset_smoke_extract, "load_smoke_records", lambda **kwargs: [SimpleNamespace(question="q", image_path="i")])
    monkeypatch.setattr(asset_smoke_extract, "create_model_wrapper", fail_if_loaded)

    result = asset_smoke_extract.run_smoke(
        registry_path=Path("registry.yaml"),
        output_root=tmp_path,
        stage0_root=tmp_path / "stage0",
        datasets=datasets,
        smoke_limit=2,
        device="cuda:0",
        models=[alias],
    )

    assert result == 0
    report = (tmp_path / "smoke_extraction_report.csv").read_text(encoding="utf-8")
    assert f"{alias},pope,popular,verified_separate_env" in report
    summary = json.loads((tmp_path / "asset_completion_summary.json").read_text(encoding="utf-8"))
    assert summary["model_statuses"][alias] == AssetStatus.VERIFIED_SEPARATE_ENV.value


def test_scoped_smoke_preserves_existing_validation_checksums(monkeypatch, tmp_path: Path) -> None:
    datasets = ("pope", "repope", "dash-b")
    selected = REQUIRED_MODEL_ALIASES[2]
    preserved = REQUIRED_MODEL_ALIASES[0]
    checksum_path = tmp_path / "validation_checksums.json"
    checksum_path.write_text(
        json.dumps(
            {
                "determinism": {
                    f"{preserved}/pope/popular": {
                        "status": "verified",
                        "primary": [{"sample_id": "old"}],
                        "repeat": [{"sample_id": "old"}],
                    }
                },
                "image_sensitivity_canary": {
                    preserved: {"status": "skipped_with_reason", "reason": "old canary"}
                },
                "structural": {"old": [{"sample_id": "old"}]},
            }
        ),
        encoding="utf-8",
    )

    def fake_load_asset_registry(path: Path):
        models = [
            SimpleNamespace(alias=alias, model_config_path="unused.yaml", local_path=f"/models/{alias}")
            for alias in REQUIRED_MODEL_ALIASES
        ]
        return SimpleNamespace(models=models)

    class FakeAudit:
        def __init__(self, alias: str, status: AssetStatus, reason: str) -> None:
            self.alias = alias
            self.status = status
            self.reason = reason

    class FakeWrapper:
        def load_processor(self):
            return object()

        def load_model(self, *, device: str):
            return object()

        def resolve_total_layers(self, model):
            return 1

        def resolve_hidden_dim(self, model):
            return 2

        def resolve_hidden_state_index_offset(self):
            return 1

        def prompt_template_id(self):
            return "fake"

        def prompt_template_text(self):
            return "fake"

        def deterministic_generation_kwargs(self, *, max_new_tokens: int):
            return {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": 0,
                "return_dict_in_generate": True,
                "output_scores": True,
                "output_hidden_states": True,
            }

        def disable_thinking_kwargs(self):
            return {}

    def fake_extract_entries(**kwargs):
        return (
            [
                {
                    "sample_id": "s0",
                    "token_index": 1,
                    "layer_vectors": asset_smoke_extract.torch.tensor([[1.0, 0.0]]),
                    "first_token_logits": asset_smoke_extract.torch.tensor([1.0]),
                    "answer_text": "Yes",
                    "parsed_answer": 1,
                    "selected_layers": [0],
                }
            ],
            2,
        )

    monkeypatch.setattr(asset_smoke_extract, "load_asset_registry", fake_load_asset_registry)
    monkeypatch.setattr(
        asset_smoke_extract,
        "audit_asset_metadata",
        lambda asset: FakeAudit(asset.alias, AssetStatus.VERIFIED, "ok"),
    )
    monkeypatch.setattr(asset_smoke_extract, "required_dataset_paths", lambda **kwargs: [])
    monkeypatch.setattr(asset_smoke_extract, "load_smoke_records", lambda **kwargs: [SimpleNamespace(question="q", image_path="i")])
    monkeypatch.setattr(
        asset_smoke_extract,
        "load_yaml_config",
        lambda path, model_type: SimpleNamespace(
            name=selected,
            family="qwen3_vl",
            dtype="float16",
            trust_remote_code=False,
            thinking={"supported": False, "disabled_by_default": True, "disable_argument": None},
        ),
    )
    monkeypatch.setattr(asset_smoke_extract, "merge_asset_model_config", lambda config, asset: config)
    monkeypatch.setattr(asset_smoke_extract, "create_model_wrapper", lambda config: FakeWrapper())
    monkeypatch.setattr(asset_smoke_extract, "extract_entries", fake_extract_entries)
    monkeypatch.setattr(asset_smoke_extract, "save_prefill_cache_shard", lambda *args, **kwargs: {"actual_file_bytes": 1})
    monkeypatch.setattr(asset_smoke_extract, "merge_top_level_sidecar_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(asset_smoke_extract, "estimate_prefill_cache_tensor_bytes", lambda *args, **kwargs: 1)
    monkeypatch.setattr(asset_smoke_extract, "run_image_sensitivity_canary", lambda **kwargs: {"status": "skipped_with_reason", "reason": "test"})

    result = asset_smoke_extract.run_smoke(
        registry_path=Path("registry.yaml"),
        output_root=tmp_path,
        stage0_root=tmp_path / "stage0",
        datasets=datasets,
        smoke_limit=2,
        device="cuda:0",
        models=[selected],
    )

    assert result == 0
    checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
    assert f"{preserved}/pope/popular" in checksums["determinism"]
    assert preserved in checksums["image_sensitivity_canary"]
    assert f"{selected}/pope/popular" in checksums["determinism"]
    assert selected in checksums["image_sensitivity_canary"]
    assert "old" in checksums["structural"]


def test_scoped_smoke_rejects_unknown_or_duplicate_aliases() -> None:
    assert asset_smoke_extract.resolve_model_selection(["qwen3-vl-8b"]) == {"qwen3-vl-8b"}
    try:
        asset_smoke_extract.resolve_model_selection(["qwen3-vl-8b", "qwen3-vl-8b"])
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate alias was accepted")
    try:
        asset_smoke_extract.resolve_model_selection(["missing"])
    except ValueError as error:
        assert "unknown" in str(error)
    else:
        raise AssertionError("unknown alias was accepted")


def test_load_failure_before_smoke_output_is_blocked(monkeypatch, tmp_path: Path) -> None:
    datasets = ("pope", "repope", "dash-b")

    def fake_load_asset_registry(path: Path):
        models = [
            SimpleNamespace(alias=alias, model_config_path="unused.yaml")
            for alias in REQUIRED_MODEL_ALIASES
        ]
        return SimpleNamespace(models=models)

    class FakeAudit:
        def __init__(self, alias: str, status: AssetStatus, reason: str) -> None:
            self.alias = alias
            self.status = status
            self.reason = reason

    def fake_audit(asset):
        if asset.alias == REQUIRED_MODEL_ALIASES[0]:
            return FakeAudit(asset.alias, AssetStatus.VERIFIED, "ok")
        return FakeAudit(asset.alias, AssetStatus.UNSUPPORTED_BY_WRAPPER, "no wrapper")

    class FailingWrapper:
        def load_processor(self):
            return object()

        def load_model(self, *, device: str):
            raise AttributeError("loader compatibility error")

    monkeypatch.setattr(asset_smoke_extract, "load_asset_registry", fake_load_asset_registry)
    monkeypatch.setattr(asset_smoke_extract, "audit_asset_metadata", fake_audit)
    monkeypatch.setattr(asset_smoke_extract, "required_dataset_paths", lambda **kwargs: [])
    monkeypatch.setattr(asset_smoke_extract, "load_smoke_records", lambda **kwargs: [SimpleNamespace(question="q", image_path="i")])
    monkeypatch.setattr(asset_smoke_extract, "load_yaml_config", lambda path, model_type: SimpleNamespace(name=REQUIRED_MODEL_ALIASES[0], family="qwen3_vl"))
    monkeypatch.setattr(asset_smoke_extract, "merge_asset_model_config", lambda config, asset: config)
    monkeypatch.setattr(asset_smoke_extract, "create_model_wrapper", lambda config: FailingWrapper())

    result = asset_smoke_extract.run_smoke(
        registry_path=Path("registry.yaml"),
        output_root=tmp_path,
        stage0_root=tmp_path / "stage0",
        datasets=datasets,
        smoke_limit=2,
        device="cuda:0",
    )

    assert result == 0
    report = (tmp_path / "smoke_extraction_report.csv").read_text(encoding="utf-8")
    assert f"{REQUIRED_MODEL_ALIASES[0]},pope,popular,blocked" in report
    summary = (tmp_path / "asset_completion_summary.json").read_text(encoding="utf-8")
    assert f'"{REQUIRED_MODEL_ALIASES[0]}": "blocked"' in summary
    assert '"num_failed_validation": 0' in summary


def test_smoke_output_validation_error_is_failed_validation(monkeypatch, tmp_path: Path) -> None:
    datasets = ("pope", "repope", "dash-b")

    def fake_load_asset_registry(path: Path):
        models = [
            SimpleNamespace(alias=alias, model_config_path="unused.yaml")
            for alias in REQUIRED_MODEL_ALIASES
        ]
        return SimpleNamespace(models=models)

    class FakeAudit:
        def __init__(self, alias: str, status: AssetStatus, reason: str) -> None:
            self.alias = alias
            self.status = status
            self.reason = reason

    def fake_audit(asset):
        if asset.alias == REQUIRED_MODEL_ALIASES[0]:
            return FakeAudit(asset.alias, AssetStatus.VERIFIED, "ok")
        return FakeAudit(asset.alias, AssetStatus.UNSUPPORTED_BY_WRAPPER, "no wrapper")

    class FakeWrapper:
        def load_processor(self):
            return object()

        def load_model(self, *, device: str):
            return object()

        def resolve_total_layers(self, model):
            return 1

        def resolve_hidden_dim(self, model):
            return 2

        def resolve_hidden_state_index_offset(self):
            return 1

    def fake_extract_entries(**kwargs):
        raise asset_smoke_extract.SmokeOutputValidationError("non-finite logits")

    monkeypatch.setattr(asset_smoke_extract, "load_asset_registry", fake_load_asset_registry)
    monkeypatch.setattr(asset_smoke_extract, "audit_asset_metadata", fake_audit)
    monkeypatch.setattr(asset_smoke_extract, "required_dataset_paths", lambda **kwargs: [])
    monkeypatch.setattr(asset_smoke_extract, "load_smoke_records", lambda **kwargs: [SimpleNamespace(question="q", image_path="i")])
    monkeypatch.setattr(asset_smoke_extract, "load_yaml_config", lambda path, model_type: SimpleNamespace(name=REQUIRED_MODEL_ALIASES[0], family="qwen3_vl"))
    monkeypatch.setattr(asset_smoke_extract, "merge_asset_model_config", lambda config, asset: config)
    monkeypatch.setattr(asset_smoke_extract, "create_model_wrapper", lambda config: FakeWrapper())
    monkeypatch.setattr(asset_smoke_extract, "extract_entries", fake_extract_entries)
    monkeypatch.setattr(asset_smoke_extract, "run_image_sensitivity_canary", lambda **kwargs: {"status": "skipped_with_reason", "reason": "test"})

    result = asset_smoke_extract.run_smoke(
        registry_path=Path("registry.yaml"),
        output_root=tmp_path,
        stage0_root=tmp_path / "stage0",
        datasets=datasets,
        smoke_limit=2,
        device="cuda:0",
    )

    assert result == 0
    report = (tmp_path / "smoke_extraction_report.csv").read_text(encoding="utf-8")
    assert f"{REQUIRED_MODEL_ALIASES[0]},pope,popular,failed_validation" in report
    summary = (tmp_path / "asset_completion_summary.json").read_text(encoding="utf-8")
    assert f'"{REQUIRED_MODEL_ALIASES[0]}": "failed_validation"' in summary


def test_extract_entries_uses_raw_prefill_logits_when_generation_scores_are_masked() -> None:
    record = HallucinationRecord(
        sample_id="s0",
        image_id=1,
        image_path="image.jpg",
        question="Is there a dog in the image?",
        label=1,
        object_name="dog",
        split="popular",
        subset="popular",
        source_dataset="pope",
    )

    class FakeWrapper:
        def prepare_asset_batch_inputs(self, processor, *, questions, image_paths, device):
            return {"input_ids": asset_smoke_extract.torch.tensor([[5]])}

        def generate(self, model, processor, *, model_inputs, max_new_tokens):
            return SimpleNamespace(
                sequences=asset_smoke_extract.torch.tensor([[5, 6]]),
                scores=[asset_smoke_extract.torch.tensor([[float("-inf"), 0.0]])],
            )

        def resolve_prefill_hidden_states(self, model, processor, *, model_inputs, generation_output):
            return [
                asset_smoke_extract.torch.zeros((1, 1, 2)),
                asset_smoke_extract.torch.tensor([[[1.0, 2.0]]]),
            ]

        def resolve_query_token_index(self, processor, *, model_inputs, batch_index):
            return 0

        def resolve_prefill_logits(self, model, processor, *, model_inputs, batch_index, token_index):
            return asset_smoke_extract.torch.tensor([0.25, 0.75])

        def decode_generation(self, processor, *, generated_ids, prompt_input_ids):
            return "Yes"

        def prompt_template_id(self):
            return "fake_prompt"

    entries, hidden_state_count = asset_smoke_extract.extract_entries(
        model=object(),
        processor=object(),
        wrapper=FakeWrapper(),
        records=[record],
        device="cpu",
        total_layers=1,
        offset=1,
        model_config=SimpleNamespace(name="fake-model", family="fake-family"),
        dataset_name="pope",
        subset="popular",
    )

    assert hidden_state_count == 2
    assert asset_smoke_extract.torch.isfinite(entries[0]["first_token_logits"]).all().item()
    assert entries[0]["first_token_logits"].tolist() == [0.25, 0.75]
