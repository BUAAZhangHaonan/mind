from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = Path("tmp/asset_repair/run_phi4_tmp_smoke.py")
    spec = importlib.util.spec_from_file_location("run_phi4_tmp_smoke", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_tmp_smoke_defaults_to_dry_run() -> None:
    module = _load_module()

    args = module.build_parser().parse_args([])

    assert args.execute is False
    assert args.dry_run is True


def test_prompt_uses_exact_phi4_image_template() -> None:
    module = _load_module()

    prompt = module.build_phi4_prompt("Is there a snowboard in the image?")

    assert prompt == "<|user|><|image_1|>Is there a snowboard in the image?<|end|><|assistant|>"


def test_dry_run_writes_report_without_loading(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    calls = []
    monkeypatch.setattr(module, "run_phi4_smoke", lambda **kwargs: calls.append(kwargs))

    report = module.run(output_root=tmp_path, execute=False)

    assert calls == []
    assert report["status"] == "planned"
    assert report["tmp_only"] is True
    assert (tmp_path / "phi4_tmp_smoke_report.json").is_file()
    assert (tmp_path / "phi4_tmp_smoke_report.md").is_file()


def test_execute_records_smoke_result(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    def fake_smoke(**kwargs):
        return {
            "status": "verified_tmp",
            "answer_text": "Yes",
            "layer_vectors_shape": [32, 3072],
            "hidden_state_index_offset": 1,
            "repeat_layer_vectors_max_abs_diff": 0.0,
            "image_sensitivity_max_abs_diff": 1.0,
        }

    monkeypatch.setattr(module, "run_phi4_smoke", fake_smoke)

    report = module.run(output_root=tmp_path, execute=True)

    assert report["status"] == "verified_tmp"
    assert report["smoke_result"]["hidden_state_index_offset"] == 1


def test_user_site_disable_is_recorded() -> None:
    module = _load_module()

    plan = module.tmp_loading_plan()

    assert plan["python_no_user_site_required"] is True
    assert plan["attention_override"] == "eager"
    assert plan["low_cpu_mem_usage"] is False
    assert plan["peft_prepare_inputs_patch"] == "Phi4MMModel.prepare_inputs_for_generation"
