from __future__ import annotations

from pathlib import Path

from mind.models.registry import REQUIRED_MODEL_ALIASES, load_asset_registry


REGISTRY_PATH = Path("configs/assets/model_assets.yaml")


def test_every_requested_alias_appears_in_registry() -> None:
    registry = load_asset_registry(REGISTRY_PATH)

    aliases = [model.alias for model in registry.models]

    assert aliases == list(REQUIRED_MODEL_ALIASES)
    assert len(aliases) == len(set(aliases))


def test_registry_models_include_required_contract_fields() -> None:
    registry = load_asset_registry(REGISTRY_PATH)

    for model in registry.models:
        assert model.local_path
        assert model.model_config_path
        assert model.family
        assert model.deterministic_generation.do_sample is False
        assert model.deterministic_generation.temperature == 0
        assert model.deterministic_generation.max_new_tokens == 1
        assert model.policy.allow_moe is False
        assert model.policy.allow_thinking is False
        assert model.policy.allow_video_only is False
        assert model.policy.allow_audio_only is False
