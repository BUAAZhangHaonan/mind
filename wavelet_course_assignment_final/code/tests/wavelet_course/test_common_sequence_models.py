from __future__ import annotations

import types

import numpy as np
import pytest

from mind.wavelet_course import common_sequence_models as module


class _FakeCuda:
    def __init__(self, *, available: bool = True, count: int = 2) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class _FakeDataParallel:
    def __init__(self, model: object, *, device_ids: list[int], output_device: int) -> None:
        self.module = model
        self.device_ids = list(device_ids)
        self.output_device = int(output_device)


def test_sequence_device_plan_parses_multi_gpu_spec_without_real_gpus() -> None:
    fake_torch = types.SimpleNamespace(cuda=_FakeCuda(count=4))

    plan = module._resolve_sequence_device_plan("cuda:0,1", fake_torch)

    assert plan.target_device == "cuda:0"
    assert plan.cuda_device_ids == (0, 1)
    assert plan.use_data_parallel is True


def test_sequence_device_plan_expands_cuda_all_and_validates_ordinals() -> None:
    fake_torch = types.SimpleNamespace(cuda=_FakeCuda(count=3))

    plan = module._resolve_sequence_device_plan("cuda:all", fake_torch)

    assert plan.target_device == "cuda:0"
    assert plan.cuda_device_ids == (0, 1, 2)
    assert plan.use_data_parallel is True

    with pytest.raises(RuntimeError, match="cuda ordinal 3"):
        module._resolve_sequence_device_plan("cuda:0,3", fake_torch)


def test_sequence_model_wraps_only_multi_gpu_plans() -> None:
    fake_nn = types.SimpleNamespace(DataParallel=_FakeDataParallel)
    model = object()
    parallel_plan = module.SequenceDevicePlan(
        target_device="cuda:0",
        cuda_device_ids=(0, 1),
        use_data_parallel=True,
    )
    single_plan = module.SequenceDevicePlan(
        target_device="cuda:0",
        cuda_device_ids=(0,),
        use_data_parallel=False,
    )

    wrapped = module._wrap_sequence_model_for_device_plan(model, parallel_plan, nn=fake_nn)

    assert isinstance(wrapped, _FakeDataParallel)
    assert wrapped.module is model
    assert wrapped.device_ids == [0, 1]
    assert wrapped.output_device == 0
    assert module._wrap_sequence_model_for_device_plan(model, single_plan, nn=fake_nn) is model


def _tiny_sequence_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.array(
        [
            [[0.0, 0.1], [0.2, 0.3], [0.1, 0.0]],
            [[1.0, 0.9], [0.8, 0.7], [0.9, 1.0]],
            [[0.1, 0.0], [0.3, 0.2], [0.0, 0.1]],
            [[0.9, 1.0], [0.7, 0.8], [1.0, 0.9]],
        ],
        dtype=np.float32,
    )
    train_y = np.array([0, 1, 0, 1], dtype=np.int64)
    validation_x = train_x.copy()
    validation_y = train_y.copy()
    test_x = train_x.copy()
    return train_x, train_y, validation_x, validation_y, test_x


def test_sequence_training_curve_records_required_fields_and_early_stop_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_x, train_y, validation_x, validation_y, test_x = _tiny_sequence_data()
    monkeypatch.setattr(module, "_average_precision", lambda labels, scores: 0.5)

    result = module.train_sequence_model(
        "cnn1d",
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x=test_x,
        device="cpu",
        batch_size=2,
        max_epochs=5,
        patience=1,
        learning_rate=1e-3,
        hidden_dim=4,
        dropout=0.0,
        seed=20260506,
    )

    assert result.best_epoch == 1
    assert result.early_stopped is True
    assert result.converged is True
    assert result.max_epoch_reached is False
    assert result.learning_rate == pytest.approx(1e-3)
    assert len(result.training_curve) == 2
    assert {
        "epoch",
        "train_loss",
        "val_loss",
        "val_pr_auc",
        "val_f1",
    } <= set(result.training_curve[0])


def test_sequence_training_marks_max_epoch_reached_without_convergence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_x, train_y, validation_x, validation_y, test_x = _tiny_sequence_data()
    values = iter([0.1, 0.2, 0.3])
    monkeypatch.setattr(module, "_average_precision", lambda labels, scores: next(values))

    result = module.train_sequence_model(
        "cnn1d",
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x=test_x,
        device="cpu",
        batch_size=2,
        max_epochs=3,
        patience=5,
        learning_rate=3e-4,
        hidden_dim=4,
        dropout=0.0,
        seed=20260506,
    )

    assert result.best_epoch == 3
    assert result.early_stopped is False
    assert result.converged is False
    assert result.max_epoch_reached is True
    assert result.learning_rate == pytest.approx(3e-4)


def test_sequence_evaluation_uses_no_grad_not_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch, nn, _ = module._torch_modules()

    class TinyReadout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 1)

        def forward(self, x: object) -> object:
            return self.linear(x[:, -1, :]).squeeze(-1)

    class _NoGradContext:
        def __enter__(self) -> None:
            return None

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

    def fail_if_used() -> object:
        raise AssertionError("torch.inference_mode must not be used for sequence eval")

    x = np.array(
        [
            [[0.0, 0.1], [0.2, 0.3], [0.1, 0.0]],
            [[1.0, 0.9], [0.8, 0.7], [0.9, 1.0]],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1], dtype=np.int64)
    model = TinyReadout()

    monkeypatch.setattr(torch, "inference_mode", fail_if_used)
    monkeypatch.setattr(torch, "no_grad", lambda: _NoGradContext())

    loss, scores = module._evaluate_sequence(
        model,
        x,
        y,
        loss_fn=nn.BCEWithLogitsLoss(),
        batch_size=2,
        device=torch.device("cpu"),
        torch=torch,
    )

    assert np.isfinite(loss)
    assert scores.shape == (2,)


@pytest.mark.skipif(
    not module._torch_modules()[0].cuda.is_available()
    or module._torch_modules()[0].cuda.device_count() < 2,
    reason="requires at least two CUDA devices",
)
@pytest.mark.parametrize("model_name", ["lstm_projected", "gru_projected"])
def test_recurrent_sequence_model_cuda_dataparallel_smoke(model_name: str) -> None:
    train_x, train_y, validation_x, validation_y, test_x = _tiny_sequence_data()

    result = module.train_sequence_model(
        model_name,
        train_x,
        train_y,
        validation_x,
        validation_y,
        test_x=test_x,
        device="cuda:0,1",
        batch_size=2,
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
        hidden_dim=4,
        dropout=0.0,
        seed=20260506,
    )

    assert result.scores.validation.shape == validation_y.shape
    assert result.training_curve[0]["val_f1"] >= 0.0
