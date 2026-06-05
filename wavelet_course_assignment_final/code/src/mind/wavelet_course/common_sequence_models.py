"""Common sequence readouts for paired-wavelet v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
from sklearn.metrics import average_precision_score

from .common_classifiers import SplitScores

SEQUENCE_MODELS = ("lstm_projected", "gru_projected", "tcn", "cnn1d")
RECURRENT_PROJECTION_DIM = 256


@dataclass(frozen=True)
class SequenceTrainingResult:
    model_name: str
    model: Any
    scores: SplitScores
    training_curve: list[dict[str, float]]
    best_epoch: int
    best_validation_pr_auc: float
    epochs_ran: int
    early_stopped: bool
    converged: bool
    max_epoch_reached: bool
    max_epochs: int
    patience: int
    learning_rate: float


@dataclass(frozen=True)
class SequenceDevicePlan:
    target_device: str
    cuda_device_ids: tuple[int, ...]
    use_data_parallel: bool


def train_sequence_model(
    model_name: str,
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    *,
    test_x: Any | None = None,
    device: str | None = None,
    batch_size: int = 32,
    max_epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
    hidden_dim: int = 128,
    dropout: float = 0.1,
) -> SequenceTrainingResult:
    """Train one PyTorch sequence readout with validation PR-AUC early stopping."""

    torch, nn, data = _torch_modules()
    name = _normalize_model_name(model_name)
    _validate_training_hyperparameters(
        batch_size=batch_size,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )

    x_train = _as_sequence_array(train_x, name="train_x")
    y_train = _as_labels(train_y, expected_size=x_train.shape[0], name="train_y")
    _require_two_classes(y_train, name="train_y")
    x_val = _as_sequence_array(validation_x, name="validation_x")
    y_val = _as_labels(validation_y, expected_size=x_val.shape[0], name="validation_y")
    _require_matching_sequence_shape(x_val, expected_shape=x_train.shape[1:], name="validation_x")
    x_test = _optional_sequence_array(test_x, expected_shape=x_train.shape[1:], name="test_x")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    device_plan = _resolve_sequence_device_plan(device, torch)
    target_device = torch.device(device_plan.target_device)

    model = _build_model(
        name,
        input_dim=int(x_train.shape[2]),
        projection_dim=RECURRENT_PROJECTION_DIM,
        hidden_dim=int(hidden_dim),
        dropout=float(dropout),
        nn=nn,
    ).to(target_device)
    model = _wrap_sequence_model_for_device_plan(model, device_plan, nn=nn)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=target_device)
    )
    loader = _make_train_loader(
        x_train,
        y_train,
        batch_size=int(batch_size),
        seed=int(seed),
        torch=torch,
        data=data,
    )

    training_curve: list[dict[str, float]] = []
    best_epoch = 0
    best_validation_pr_auc = -float("inf")
    best_state: dict[str, Any] | None = None
    stale_epochs = 0
    early_stopped = False

    for epoch in range(1, int(max_epochs) + 1):
        train_loss = _train_epoch(
            model,
            loader,
            optimizer,
            loss_fn,
            device=target_device,
        )
        validation_loss, validation_scores = _evaluate_sequence(
            model,
            x_val,
            y_val,
            loss_fn=loss_fn,
            batch_size=int(batch_size),
            device=target_device,
            torch=torch,
        )
        validation_pr_auc = _average_precision(y_val, validation_scores)
        validation_f1 = _best_f1(y_val, validation_scores)
        training_curve.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(validation_loss),
                "val_pr_auc": float(validation_pr_auc),
                "val_f1": float(validation_f1),
            }
        )
        if validation_pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = float(validation_pr_auc)
            best_epoch = int(epoch)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                early_stopped = True
                break

    if best_state is None:
        raise RuntimeError("sequence training did not produce a best checkpoint")
    model.load_state_dict(best_state)
    model.to(target_device)

    scores = SplitScores(
        train=_predict_sequence_scores(
            model,
            x_train,
            batch_size=int(batch_size),
            device=target_device,
            torch=torch,
        ),
        validation=_predict_sequence_scores(
            model,
            x_val,
            batch_size=int(batch_size),
            device=target_device,
            torch=torch,
        ),
        test=(
            _predict_sequence_scores(
                model,
                x_test,
                batch_size=int(batch_size),
                device=target_device,
                torch=torch,
            )
            if x_test is not None
            else None
        ),
    )
    return SequenceTrainingResult(
        model_name=name,
        model=model,
        scores=scores,
        training_curve=training_curve,
        best_epoch=best_epoch,
        best_validation_pr_auc=best_validation_pr_auc,
        epochs_ran=len(training_curve),
        early_stopped=early_stopped,
        converged=early_stopped,
        max_epoch_reached=(len(training_curve) >= int(max_epochs) and not early_stopped),
        max_epochs=int(max_epochs),
        patience=int(patience),
        learning_rate=float(learning_rate),
    )


def _normalize_model_name(model_name: str) -> str:
    name = str(model_name).strip().lower()
    if name not in SEQUENCE_MODELS:
        raise ValueError(f"model_name must be one of {SEQUENCE_MODELS}, got {model_name!r}")
    return name


def parse_cuda_device_ordinals(
    device: str,
    *,
    device_count: int | None = None,
) -> tuple[int, ...]:
    normalized_device = str(device).strip().lower()
    if normalized_device == "cuda":
        ordinals = (0,)
    else:
        if not normalized_device.startswith("cuda:"):
            raise RuntimeError(
                f"unsupported cuda device {device!r}; expected cuda:N, cuda:N,M, or cuda:all"
            )
        raw = normalized_device.split(":", 1)[1].strip()
        if raw == "all":
            if device_count is None:
                raise RuntimeError("cuda:all requires torch.cuda.device_count()")
            if int(device_count) <= 0:
                raise RuntimeError(
                    f"requested cuda:all, but torch.cuda.device_count() is {int(device_count)}"
                )
            ordinals = tuple(range(int(device_count)))
        else:
            parts = [part.strip() for part in raw.split(",")]
            if not parts or any(not part for part in parts):
                raise RuntimeError(f"invalid cuda ordinal list {raw!r}")
            parsed: list[int] = []
            for part in parts:
                try:
                    ordinal = int(part)
                except ValueError as error:
                    raise RuntimeError(f"invalid cuda ordinal {part!r}") from error
                if ordinal < 0:
                    raise RuntimeError(f"invalid cuda ordinal {ordinal}; ordinal must be non-negative")
                parsed.append(ordinal)
            ordinals = tuple(parsed)
    if len(set(ordinals)) != len(ordinals):
        raise RuntimeError(f"duplicate cuda ordinals are not allowed: {list(ordinals)}")
    if device_count is not None:
        count = int(device_count)
        for ordinal in ordinals:
            if ordinal >= count:
                raise RuntimeError(
                    f"requested cuda ordinal {ordinal}, but torch.cuda.device_count() is {count}"
                )
    return ordinals


def _resolve_sequence_device_plan(device: str | None, torch: Any) -> SequenceDevicePlan:
    requested = "" if device is None else str(device).strip()
    if not requested:
        if torch.cuda.is_available():
            return SequenceDevicePlan(
                target_device="cuda",
                cuda_device_ids=(0,),
                use_data_parallel=False,
            )
        return SequenceDevicePlan(
            target_device="cpu",
            cuda_device_ids=(),
            use_data_parallel=False,
        )

    normalized_device = requested.lower()
    if normalized_device == "cpu":
        return SequenceDevicePlan(
            target_device="cpu",
            cuda_device_ids=(),
            use_data_parallel=False,
        )
    if not normalized_device.startswith("cuda"):
        raise RuntimeError(f"unsupported device {device}; expected cuda device or cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"requested device {device}, but torch.cuda.is_available() is false"
        )
    device_count = int(torch.cuda.device_count())
    ordinals = parse_cuda_device_ordinals(normalized_device, device_count=device_count)
    return SequenceDevicePlan(
        target_device=f"cuda:{ordinals[0]}",
        cuda_device_ids=ordinals,
        use_data_parallel=len(ordinals) > 1,
    )


def _wrap_sequence_model_for_device_plan(model: Any, plan: SequenceDevicePlan, *, nn: Any) -> Any:
    if not plan.use_data_parallel:
        return model
    return nn.DataParallel(
        model,
        device_ids=list(plan.cuda_device_ids),
        output_device=int(plan.cuda_device_ids[0]),
    )


def _validate_training_hyperparameters(
    *,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
) -> None:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")
    if learning_rate <= 0.0 or not np.isfinite(float(learning_rate)):
        raise ValueError("learning_rate must be finite and positive")
    if weight_decay < 0.0 or not np.isfinite(float(weight_decay)):
        raise ValueError("weight_decay must be finite and non-negative")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    if dropout < 0.0 or dropout >= 1.0 or not np.isfinite(float(dropout)):
        raise ValueError("dropout must be finite and in [0, 1)")


def _build_model(
    model_name: str,
    *,
    input_dim: int,
    projection_dim: int,
    hidden_dim: int,
    dropout: float,
    nn: Any,
) -> Any:
    if model_name == "lstm_projected":
        return _projected_recurrent_class(nn)(
            input_dim=input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            recurrent_type="lstm",
        )
    if model_name == "gru_projected":
        return _projected_recurrent_class(nn)(
            input_dim=input_dim,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            recurrent_type="gru",
        )
    if model_name == "tcn":
        return _tcn_class(nn)(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    if model_name == "cnn1d":
        return _cnn1d_class(nn)(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    raise AssertionError(f"unhandled sequence model: {model_name}")


def _projected_recurrent_class(nn: Any) -> type[Any]:
    class ProjectedRecurrentReadout(nn.Module):
        def __init__(
            self,
            *,
            input_dim: int,
            projection_dim: int,
            hidden_dim: int,
            dropout: float,
            recurrent_type: str,
        ) -> None:
            super().__init__()
            self.input_projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, projection_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            recurrent_cls = nn.LSTM if recurrent_type == "lstm" else nn.GRU
            self.recurrent_type = recurrent_type
            self.recurrent = recurrent_cls(
                input_size=projection_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
            )
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: Any) -> Any:
            projected = self.input_projection(x.to(dtype=self.classifier[-1].weight.dtype))
            output = self.recurrent(projected)
            state = output[1]
            if self.recurrent_type == "lstm":
                hidden = state[0][-1]
            else:
                hidden = state[-1]
            return self.classifier(hidden).squeeze(-1)

    return ProjectedRecurrentReadout


def _tcn_class(nn: Any) -> type[Any]:
    class Chomp1d(nn.Module):
        def __init__(self, chomp_size: int) -> None:
            super().__init__()
            self.chomp_size = int(chomp_size)

        def forward(self, x: Any) -> Any:
            if self.chomp_size <= 0:
                return x
            return x[:, :, : -self.chomp_size]

    class TemporalBlock(nn.Module):
        def __init__(
            self,
            *,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            dropout: float,
        ) -> None:
            super().__init__()
            padding = (kernel_size - 1) * dilation
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                Chomp1d(padding),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.downsample = (
                nn.Conv1d(in_channels, out_channels, kernel_size=1)
                if in_channels != out_channels
                else nn.Identity()
            )
            self.activation = nn.GELU()

        def forward(self, x: Any) -> Any:
            return self.activation(self.net(x) + self.downsample(x))

    class TCNReadout(nn.Module):
        def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                TemporalBlock(
                    in_channels=input_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=1,
                    dropout=dropout,
                ),
                TemporalBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=2,
                    dropout=dropout,
                ),
                TemporalBlock(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    kernel_size=3,
                    dilation=4,
                    dropout=dropout,
                ),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> Any:
            features = self.blocks(x.to(dtype=self.classifier.weight.dtype).transpose(1, 2))
            pooled = self.pool(features).squeeze(-1)
            return self.classifier(pooled).squeeze(-1)

    return TCNReadout


def _cnn1d_class(nn: Any) -> type[Any]:
    class CNN1DReadout(nn.Module):
        def __init__(self, *, input_dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.AdaptiveMaxPool1d(1),
            )
            self.classifier = nn.Linear(hidden_dim, 1)

        def forward(self, x: Any) -> Any:
            features = self.features(x.to(dtype=self.classifier.weight.dtype).transpose(1, 2)).squeeze(-1)
            return self.classifier(features).squeeze(-1)

    return CNN1DReadout


def _make_train_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
    torch: Any,
    data: Any,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    dataset = data.TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(y.astype(np.float32, copy=False)),
    )
    return data.DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )


def _train_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    loss_fn: Any,
    *,
    device: Any,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_x)
        loss = loss_fn(logits, batch_y)
        loss.backward()
        optimizer.step()
        batch_size = int(batch_x.shape[0])
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total += batch_size
    return total_loss / max(total, 1)


def _evaluate_sequence(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    loss_fn: Any,
    batch_size: int,
    device: Any,
    torch: Any,
) -> tuple[float, np.ndarray]:
    model.eval()
    losses: list[float] = []
    logits_parts: list[np.ndarray] = []
    total = 0
    # cuDNN RNN modules may flatten weights during eval forward. Under
    # torch.inference_mode(), DataParallel replicas can expose inference tensors
    # to that in-place flatten path, so sequence eval uses no_grad instead.
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            tensor_x = torch.from_numpy(x[start:stop]).to(device)
            tensor_y = torch.from_numpy(y[start:stop].astype(np.float32, copy=False)).to(device)
            logits = model(tensor_x)
            loss = loss_fn(logits, tensor_y)
            count = int(stop - start)
            losses.append(float(loss.detach().cpu().item()) * count)
            logits_parts.append(logits.detach().cpu().numpy().astype(np.float32))
            total += count
    logits_array = np.concatenate(logits_parts, axis=0)
    scores = _sigmoid(logits_array).astype(np.float32, copy=False)
    _raise_if_non_finite(scores, name="sequence_scores")
    return sum(losses) / max(total, 1), scores


def _predict_sequence_scores(
    model: Any,
    x: np.ndarray,
    *,
    batch_size: int,
    device: Any,
    torch: Any,
) -> np.ndarray:
    labels = np.zeros(x.shape[0], dtype=np.float32)
    _, scores = _evaluate_sequence(
        model,
        x,
        labels,
        loss_fn=lambda logits, target: (logits * 0.0).mean(),
        batch_size=batch_size,
        device=device,
        torch=torch,
    )
    return scores


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = float(average_precision_score(labels, scores))
    if not np.isfinite(value):
        raise ValueError("validation PR-AUC must be finite")
    return value


def _best_f1(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.shape[0] != s.shape[0]:
        raise ValueError("validation labels and scores must be aligned for val_f1")
    _raise_if_non_finite(s, name="validation_scores")
    thresholds = np.unique(s)
    if thresholds.size == 0:
        raise ValueError("validation scores must not be empty")
    best = 0.0
    for threshold in thresholds:
        pred = (s >= float(threshold)).astype(np.int64)
        tp = int(np.sum((pred == 1) & (y == 1)))
        fp = int(np.sum((pred == 1) & (y == 0)))
        fn = int(np.sum((pred == 0) & (y == 1)))
        denom = (2 * tp) + fp + fn
        f1 = 0.0 if denom == 0 else (2.0 * tp) / float(denom)
        if f1 > best:
            best = float(f1)
    return best


def _as_sequence_array(values: Any, *, name: str) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError(f"{name} must be a 3D array shaped (samples, sequence, features)")
    if array.shape[0] == 0 or array.shape[1] == 0 or array.shape[2] == 0:
        raise ValueError(f"{name} must have non-empty sample, sequence, and feature dimensions")
    _raise_if_non_finite(array, name=name)
    return array


def _optional_sequence_array(
    values: Any | None,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> np.ndarray | None:
    if values is None:
        return None
    array = _as_sequence_array(values, name=name)
    _require_matching_sequence_shape(array, expected_shape=expected_shape, name=name)
    return array


def _require_matching_sequence_shape(
    array: np.ndarray,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> None:
    if tuple(array.shape[1:]) != tuple(expected_shape):
        raise ValueError(f"{name} sequence and feature dimensions must match train_x")


def _as_labels(values: Any, *, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if array.shape[0] != expected_size:
        raise ValueError(f"{name} length must match sample count")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must not be empty")
    if not set(np.unique(array).tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 labels")
    return array


def _require_two_classes(labels: np.ndarray, *, name: str) -> None:
    if np.unique(labels).shape[0] < 2:
        raise ValueError(f"{name} must contain at least two classes")


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64, copy=False), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")


def _torch_modules() -> tuple[Any, Any, Any]:
    try:
        import torch
        from torch import nn
        from torch.utils import data
    except ImportError as exc:
        raise ImportError("PyTorch is required for paired sequence readouts") from exc
    return torch, nn, data


__all__ = [
    "RECURRENT_PROJECTION_DIM",
    "SEQUENCE_MODELS",
    "SequenceDevicePlan",
    "SequenceTrainingResult",
    "parse_cuda_device_ordinals",
    "train_sequence_model",
]
