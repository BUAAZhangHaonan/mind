"""Baseline feature builders and training helpers for wavelet-course runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import inspect
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

EXPECTED_LAYER_SHAPE = (36, 4096)
TEACHER_SEQUENCE_LEN = 9
TEACHER_INPUT_DIM = 4096 * 28
EPSILON = 1e-12


@dataclass(frozen=True)
class SplitScores:
    train: np.ndarray
    validation: np.ndarray | None
    test: np.ndarray | None


@dataclass(frozen=True)
class LogisticTrainingResult:
    model: Pipeline
    scores: SplitScores


@dataclass(frozen=True)
class XGBoostTrainingResult:
    status: str
    rows: list[dict[str, Any]]
    best_model: Any | None
    scores: SplitScores | None


@dataclass(frozen=True)
class TeacherLSTMTrainingResult:
    model: Any
    history: list[dict[str, float]]
    scores: SplitScores


def final_hidden_logreg(entries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build final-layer hidden vectors for logistic regression."""

    return _stack_entry_features(entries, lambda vectors: vectors[-1])


def mean_layer_hidden_logreg(entries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build mean-pooled hidden vectors across all layers."""

    return _stack_entry_features(entries, lambda vectors: vectors.mean(axis=0, dtype=np.float32))


def norm_traj_logreg(entries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build the 36-point norm trajectory baseline."""

    return _stack_entry_features(entries, lambda vectors: np.linalg.norm(vectors, axis=1))


def sphere_traj_meanpool_logreg(entries: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Build mean-pooled unit-sphere trajectory vectors."""

    def build(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        unit = vectors / np.maximum(norms, EPSILON)
        pooled = unit.mean(axis=0, dtype=np.float32)
        pooled_norm = float(np.linalg.norm(pooled))
        return pooled / max(pooled_norm, EPSILON)

    return _stack_entry_features(entries, build)


def train_logistic_regression(
    train_x: Any,
    train_y: Any,
    *,
    validation_x: Any | None = None,
    test_x: Any | None = None,
    max_iter: int = 1000,
    random_state: int = 0,
) -> LogisticTrainingResult:
    """Train StandardScaler + balanced LogisticRegression and return scores."""

    x_train = _as_2d_float_array(train_x, name="train_x")
    y_train = _as_labels(train_y, expected_size=x_train.shape[0], name="train_y")
    _require_two_classes(y_train, name="train_y")
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=int(max_iter),
                    random_state=int(random_state),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    scores = SplitScores(
        train=_predict_positive_proba(model, x_train),
        validation=_optional_predict(model, validation_x, expected_dim=x_train.shape[1], name="validation_x"),
        test=_optional_predict(model, test_x, expected_dim=x_train.shape[1], name="test_x"),
    )
    return LogisticTrainingResult(model=model, scores=scores)


def train_xgboost_grid(
    train_x: Any,
    train_y: Any,
    *,
    validation_x: Any | None = None,
    validation_y: Any | None = None,
    test_x: Any | None = None,
    allow_no_xgboost: bool = False,
    random_state: int = 0,
) -> XGBoostTrainingResult:
    """Run the fixed XGBoost grid and return continuous split scores."""

    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        if allow_no_xgboost:
            return XGBoostTrainingResult(
                status="failure",
                rows=[{"status": "failure", "reason": "xgboost_not_installed"}],
                best_model=None,
                scores=None,
            )
        raise ImportError("xgboost is required when allow_no_xgboost is false") from exc

    x_train = _as_2d_float_array(train_x, name="train_x")
    y_train = _as_labels(train_y, expected_size=x_train.shape[0], name="train_y")
    _require_two_classes(y_train, name="train_y")
    x_val, y_val = _optional_validation(validation_x, validation_y, feature_dim=x_train.shape[1])
    x_test = _optional_2d(test_x, expected_dim=x_train.shape[1], name="test_x")

    rows: list[dict[str, Any]] = []
    best_model: Any | None = None
    best_score = -float("inf")
    for max_depth in (2, 3):
        for learning_rate in (0.03, 0.1):
            for n_estimators in (100, 300):
                model = XGBClassifier(
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    objective="binary:logistic",
                    eval_metric="aucpr",
                    random_state=int(random_state),
                    n_jobs=1,
                )
                _fit_xgboost(model, x_train, y_train, x_val=x_val, y_val=y_val)
                train_scores = _predict_positive_proba(model, x_train)
                val_scores = _predict_positive_proba(model, x_val) if x_val is not None else None
                score = (
                    float(average_precision_score(y_val, val_scores))
                    if y_val is not None and val_scores is not None
                    else float(average_precision_score(y_train, train_scores))
                )
                rows.append(
                    {
                        "status": "success",
                        "max_depth": max_depth,
                        "learning_rate": learning_rate,
                        "n_estimators": n_estimators,
                        "selection_pr_auc": score,
                    }
                )
                if score > best_score:
                    best_score = score
                    best_model = model

    if best_model is None:
        return XGBoostTrainingResult(status="failure", rows=rows, best_model=None, scores=None)
    scores = SplitScores(
        train=_predict_positive_proba(best_model, x_train),
        validation=_predict_positive_proba(best_model, x_val) if x_val is not None else None,
        test=_predict_positive_proba(best_model, x_test) if x_test is not None else None,
    )
    return XGBoostTrainingResult(status="success", rows=rows, best_model=best_model, scores=scores)


def train_teacher_lstm(
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    *,
    test_x: Any | None = None,
    device: str,
    batch_size: int = 32,
    max_epochs: int = 50,
    patience: int = 3,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> TeacherLSTMTrainingResult:
    """Train the fixed Teacher-Bagua LSTM and return continuous scores."""

    import torch
    from torch import nn
    if not device:
        raise ValueError("device is required; no CPU fallback is applied")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive")
    if patience <= 0:
        raise ValueError("patience must be positive")

    x_train = _as_teacher_sequence(train_x, name="train_x")
    y_train = _as_labels(train_y, expected_size=x_train.shape[0], name="train_y")
    _require_two_classes(y_train, name="train_y")
    x_val = _as_teacher_sequence(validation_x, name="validation_x")
    y_val = _as_labels(validation_y, expected_size=x_val.shape[0], name="validation_y")
    x_test = _as_teacher_sequence(test_x, name="test_x") if test_x is not None else None

    torch.manual_seed(int(seed))
    target_device = torch.device(device)
    model = _TeacherBaguaLSTM().to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    positives = float(np.sum(y_train == 1))
    negatives = float(np.sum(y_train == 0))
    if positives <= 0.0 or negatives <= 0.0:
        raise ValueError("train_y must contain positive and negative labels")
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negatives / positives], dtype=torch.float32, device=target_device)
    )
    loader = _teacher_loader(x_train, y_train, batch_size=batch_size, seed=seed)

    history: list[dict[str, float]] = []
    best_state: dict[str, Any] | None = None
    best_val_loss = float("inf")
    stale = 0
    for epoch in range(1, int(max_epochs) + 1):
        train_loss = _run_lstm_train_epoch(model, loader, optimizer, loss_fn, device=target_device)
        val_loss = _lstm_loss(model, x_val, y_val, loss_fn=loss_fn, device=target_device)
        history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": val_loss})
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(target_device)

    scores = SplitScores(
        train=_lstm_scores(model, x_train, device=target_device),
        validation=_lstm_scores(model, x_val, device=target_device),
        test=_lstm_scores(model, x_test, device=target_device) if x_test is not None else None,
    )
    return TeacherLSTMTrainingResult(model=model, history=history, scores=scores)


class _TeacherBaguaLSTM:
    pass


def _make_teacher_lstm_class() -> type[Any]:
    import torch
    from torch import nn

    class TeacherBaguaLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=TEACHER_INPUT_DIM,
                hidden_size=64,
                num_layers=1,
                dropout=0.0,
                batch_first=True,
            )
            self.classifier = nn.Linear(64, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            _, (hidden, _) = self.lstm(x.to(dtype=torch.float32))
            return self.classifier(hidden[-1]).squeeze(-1)

    return TeacherBaguaLSTM


_TeacherBaguaLSTM = _make_teacher_lstm_class()


def _stack_entry_features(
    entries: Sequence[Mapping[str, Any]],
    builder: Any,
) -> np.ndarray:
    if not entries:
        raise ValueError("entries must not be empty")
    features = [np.asarray(builder(_entry_layer_vectors(entry)), dtype=np.float32) for entry in entries]
    matrix = np.stack(features, axis=0).astype(np.float32, copy=False)
    _raise_if_non_finite(matrix, name="features")
    return matrix


def _entry_layer_vectors(entry: Mapping[str, Any]) -> np.ndarray:
    if "layer_vectors" not in entry:
        raise ValueError("entry must contain layer_vectors")
    return _as_layer_vectors(entry["layer_vectors"])


def _as_layer_vectors(values: Any) -> np.ndarray:
    if hasattr(values, "detach") and callable(values.detach):
        values = values.detach().cpu().numpy()
    array = np.asarray(values, dtype=np.float32)
    if array.shape != EXPECTED_LAYER_SHAPE:
        raise ValueError(f"layer_vectors must have shape {EXPECTED_LAYER_SHAPE}")
    _raise_if_non_finite(array, name="layer_vectors")
    return array


def _as_teacher_sequence(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 3 or array.shape[1:] != (TEACHER_SEQUENCE_LEN, TEACHER_INPUT_DIM):
        raise ValueError(f"{name} must have shape (samples, {TEACHER_SEQUENCE_LEN}, {TEACHER_INPUT_DIM})")
    _raise_if_non_finite(array, name=name)
    return array


def _as_2d_float_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty sample and feature dimensions")
    _raise_if_non_finite(array, name=name)
    return array


def _optional_2d(values: Any | None, *, expected_dim: int, name: str) -> np.ndarray | None:
    if values is None:
        return None
    array = _as_2d_float_array(values, name=name)
    if array.shape[1] != expected_dim:
        raise ValueError(f"{name} feature dimension must match train_x")
    return array


def _as_labels(values: Any, *, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if array.shape[0] != expected_size:
        raise ValueError(f"{name} length must match sample count")
    if not set(np.unique(array).tolist()).issubset({0, 1}):
        raise ValueError(f"{name} must contain only 0/1 labels")
    return array


def _require_two_classes(labels: np.ndarray, *, name: str) -> None:
    if np.unique(labels).shape[0] < 2:
        raise ValueError(f"{name} must contain both classes")


def _predict_positive_proba(model: Any, x: np.ndarray) -> np.ndarray:
    scores = model.predict_proba(x)[:, 1].astype(np.float32)
    _raise_if_non_finite(scores, name="scores")
    return scores


def _optional_predict(
    model: Any,
    values: Any | None,
    *,
    expected_dim: int,
    name: str,
) -> np.ndarray | None:
    array = _optional_2d(values, expected_dim=expected_dim, name=name)
    if array is None:
        return None
    return _predict_positive_proba(model, array)


def _optional_validation(
    validation_x: Any | None,
    validation_y: Any | None,
    *,
    feature_dim: int,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if validation_x is None and validation_y is None:
        return None, None
    if validation_x is None or validation_y is None:
        raise ValueError("validation_x and validation_y must be provided together")
    x_val = _optional_2d(validation_x, expected_dim=feature_dim, name="validation_x")
    if x_val is None:
        raise ValueError("validation_x is required")
    y_val = _as_labels(validation_y, expected_size=x_val.shape[0], name="validation_y")
    return x_val, y_val


def _fit_xgboost(
    model: Any,
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    x_val: np.ndarray | None,
    y_val: np.ndarray | None,
) -> None:
    kwargs: dict[str, Any] = {}
    fit_parameters = inspect.signature(model.fit).parameters
    if x_val is not None and y_val is not None and "eval_set" in fit_parameters:
        kwargs["eval_set"] = [(x_val, y_val)]
        if "verbose" in fit_parameters:
            kwargs["verbose"] = False
        if "early_stopping_rounds" in fit_parameters:
            kwargs["early_stopping_rounds"] = 20
    model.fit(train_x, train_y, **kwargs)


def _teacher_loader(
    x: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    seed: int,
) -> Any:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y.astype(np.float32, copy=False))),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )


def _run_lstm_train_epoch(
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
        total_loss += float(loss.detach().cpu().item()) * int(batch_x.shape[0])
        total += int(batch_x.shape[0])
    return total_loss / max(total, 1)


def _lstm_loss(
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    *,
    loss_fn: Any,
    device: Any,
) -> float:
    import torch

    model.eval()
    with torch.inference_mode():
        tensor_x = torch.from_numpy(x).to(device)
        tensor_y = torch.from_numpy(y.astype(np.float32, copy=False)).to(device)
        loss = loss_fn(model(tensor_x), tensor_y)
    return float(loss.detach().cpu().item())


def _lstm_scores(model: Any, x: np.ndarray, *, device: Any) -> np.ndarray:
    import torch

    model.eval()
    scores: list[np.ndarray] = []
    batch_size = 16
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            tensor_x = torch.from_numpy(x[start : start + batch_size]).to(device)
            scores.append(torch.sigmoid(model(tensor_x)).detach().cpu().numpy().astype(np.float32))
    output = np.concatenate(scores, axis=0)
    _raise_if_non_finite(output, name="lstm_scores")
    return output


def _raise_if_non_finite(array: np.ndarray, *, name: str) -> None:
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
