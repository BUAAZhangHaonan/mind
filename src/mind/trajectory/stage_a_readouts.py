"""Stage A readout helpers for kNN, logistic, and LSTM diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .stage_a_metrics import binary_diagnostic_metrics
from .stage_a_representations import (
    DEFAULT_LSTM_EMBEDDING_DIM,
    DEFAULT_STAGE_A_SEED,
    StageATrajectoryLSTM,
    set_deterministic_seed,
)

_DEFAULT_KNN_CHUNK_SIZE = 1024


@dataclass
class LSTMDiagnosticResult:
    model: StageATrajectoryLSTM
    history: list[dict[str, float]]
    seed: int


@dataclass
class LogisticDiagnosticResult:
    model: Pipeline
    train_scores: np.ndarray
    eval_scores: np.ndarray | None
    train_metrics: dict[str, float]
    eval_metrics: dict[str, float] | None
    seed: int


def compute_knn_scores(
    bank: np.ndarray,
    query: np.ndarray,
    *,
    k: int,
    metric: str,
    backend: str = "auto",
    device: str | torch.device | None = None,
    chunk_size: int | None = None,
) -> np.ndarray:
    """Score queries by mean distance to their k nearest correct samples."""

    bank_array = _as_2d_float_array(bank, name="bank")
    query_array = _as_2d_float_array(query, name="query")
    if bank_array.shape[1] != query_array.shape[1]:
        raise ValueError(
            f"bank hidden_dim {bank_array.shape[1]} does not match query hidden_dim "
            f"{query_array.shape[1]}"
        )
    if k <= 0:
        raise ValueError("k must be positive")
    if bank_array.shape[0] < k:
        raise ValueError("bank has fewer than k correct samples")

    normalized_metric = metric.lower()
    if normalized_metric not in {"euclidean", "angular"}:
        raise ValueError(f"Unsupported kNN metric: {metric}")

    normalized_backend = backend.lower()
    if normalized_backend not in {"auto", "numpy", "torch"}:
        raise ValueError(f"Unsupported kNN backend: {backend}")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if normalized_backend == "auto":
        normalized_backend = "torch" if device is not None else "numpy"

    if normalized_backend == "torch":
        return _torch_knn_scores(
            bank_array,
            query_array,
            k=k,
            metric=normalized_metric,
            device=device,
            chunk_size=chunk_size,
        )

    if normalized_metric == "euclidean":
        distances = _euclidean_distances(query_array, bank_array)
    else:
        distances = _angular_distances(query_array, bank_array)

    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    return nearest.mean(axis=1, dtype=np.float64).astype(np.float32)


def train_logistic_diagnostic(
    train_x: np.ndarray,
    train_y: np.ndarray,
    *,
    eval_x: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
    seed: int = DEFAULT_STAGE_A_SEED,
    max_iter: int = 1000,
    threshold: float = 0.5,
    class_weight: str | dict[int, float] | None = "balanced",
) -> LogisticDiagnosticResult:
    """Fit a deterministic sklearn logistic diagnostic on Stage A features."""

    x_train = _as_2d_float_array(train_x, name="train_x")
    y_train = _as_label_vector(train_y, expected_size=x_train.shape[0], name="train_y")
    if np.unique(y_train).shape[0] < 2:
        raise ValueError("train_y must contain both classes for logistic diagnostics")

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=class_weight,
                    max_iter=int(max_iter),
                    random_state=int(seed),
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    train_scores = model.predict_proba(x_train)[:, 1].astype(np.float32)
    train_metrics = binary_diagnostic_metrics(y_train, train_scores, threshold=threshold)

    eval_scores: np.ndarray | None = None
    eval_metrics: dict[str, float] | None = None
    if eval_x is not None:
        x_eval = _as_2d_float_array(eval_x, name="eval_x")
        if x_eval.shape[1] != x_train.shape[1]:
            raise ValueError("eval_x feature dimension must match train_x")
        eval_scores = model.predict_proba(x_eval)[:, 1].astype(np.float32)
        if eval_y is not None:
            y_eval = _as_label_vector(eval_y, expected_size=x_eval.shape[0], name="eval_y")
            eval_metrics = binary_diagnostic_metrics(y_eval, eval_scores, threshold=threshold)
    elif eval_y is not None:
        raise ValueError("eval_y requires eval_x")

    return LogisticDiagnosticResult(
        model=model,
        train_scores=train_scores,
        eval_scores=eval_scores,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        seed=int(seed),
    )


def train_lstm_diagnostic(
    x: np.ndarray,
    y: np.ndarray,
    *,
    num_layers: int,
    hidden_dim: int,
    epochs: int = 10,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = DEFAULT_STAGE_A_SEED,
    patience: int | None = 3,
    learning_rate: float = 1e-3,
    embedding_dim: int = DEFAULT_LSTM_EMBEDDING_DIM,
) -> LSTMDiagnosticResult:
    """Train the Stage A LSTM diagnostic on trajectory tensors."""

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be positive when provided")

    features = np.asarray(x, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("x must have shape (samples, num_layers, hidden_dim)")
    if int(features.shape[1]) != int(num_layers):
        raise ValueError(f"x has {features.shape[1]} layers, expected {num_layers}")
    if int(features.shape[2]) != int(hidden_dim):
        raise ValueError(f"x has hidden_dim {features.shape[2]}, expected {hidden_dim}")
    labels = _as_label_vector(y, expected_size=features.shape[0], name="y")

    set_deterministic_seed(seed)
    target_device = torch.device(device)
    model = StageATrajectoryLSTM(hidden_dim=int(hidden_dim), embedding_dim=int(embedding_dim))
    model.to(target_device)

    tensor_x = torch.from_numpy(features)
    tensor_y = torch.from_numpy(labels.astype(np.float32, copy=False))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    positives = float(np.sum(labels == 1))
    negatives = float(np.sum(labels == 0))
    pos_weight = None
    if positives > 0.0 and negatives > 0.0:
        pos_weight = torch.tensor([negatives / positives], dtype=torch.float32, device=target_device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        correct = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()

            batch_size_actual = int(batch_x.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size_actual
            total_examples += batch_size_actual
            predictions = (torch.sigmoid(logits.detach()) >= 0.5).to(dtype=torch.int64)
            correct += int((predictions.cpu() == batch_y.cpu().to(dtype=torch.int64)).sum().item())

        epoch_loss = total_loss / max(total_examples, 1)
        history.append(
            {
                "epoch": float(epoch),
                "loss": float(epoch_loss),
                "accuracy": float(correct / max(total_examples, 1)),
            }
        )
        if patience is None:
            continue
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.eval()
    return LSTMDiagnosticResult(model=model, history=history, seed=int(seed))


def _euclidean_distances(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    query_sq = np.sum(np.square(query, dtype=np.float32), axis=1, keepdims=True)
    bank_sq = np.sum(np.square(bank, dtype=np.float32), axis=1, keepdims=True).T
    squared = query_sq + bank_sq - 2.0 * (query @ bank.T)
    return np.sqrt(np.maximum(squared, 0.0), dtype=np.float32)


def _angular_distances(query: np.ndarray, bank: np.ndarray) -> np.ndarray:
    query_norms = np.linalg.norm(query, axis=1, keepdims=True)
    bank_norms = np.linalg.norm(bank, axis=1, keepdims=True)
    if np.any(query_norms <= 0.0) or np.any(bank_norms <= 0.0):
        raise ValueError("angular distance requires non-zero vectors")
    query_unit = query / query_norms
    bank_unit = bank / bank_norms
    cosine = np.clip(query_unit @ bank_unit.T, -1.0, 1.0)
    return np.arccos(cosine) / np.pi


def _torch_knn_scores(
    bank: np.ndarray,
    query: np.ndarray,
    *,
    k: int,
    metric: str,
    device: str | torch.device | None,
    chunk_size: int | None,
) -> np.ndarray:
    target_device = _resolve_knn_torch_device(device)
    batch_size = int(chunk_size) if chunk_size is not None else _DEFAULT_KNN_CHUNK_SIZE
    bank_tensor = torch.as_tensor(bank, dtype=torch.float32, device=target_device)
    query_tensor = torch.as_tensor(query, dtype=torch.float32)
    scores: list[np.ndarray] = []

    with torch.inference_mode():
        if metric == "angular":
            bank_norms = torch.linalg.vector_norm(bank_tensor, dim=1, keepdim=True)
            if torch.any(bank_norms <= 0.0).item():
                raise ValueError("angular distance requires non-zero vectors")
            bank_tensor = bank_tensor / bank_norms
        else:
            bank_sq = torch.sum(torch.square(bank_tensor), dim=1).unsqueeze(0)

        bank_t = bank_tensor.T
        for start in range(0, int(query_tensor.shape[0]), batch_size):
            query_chunk = query_tensor[start : start + batch_size].to(target_device)
            if metric == "euclidean":
                query_sq = torch.sum(torch.square(query_chunk), dim=1, keepdim=True)
                squared = query_sq + bank_sq - 2.0 * (query_chunk @ bank_t)
                distances = torch.sqrt(torch.clamp(squared, min=0.0))
            else:
                query_norms = torch.linalg.vector_norm(query_chunk, dim=1, keepdim=True)
                if torch.any(query_norms <= 0.0).item():
                    raise ValueError("angular distance requires non-zero vectors")
                query_unit = query_chunk / query_norms
                cosine = torch.clamp(query_unit @ bank_t, min=-1.0, max=1.0)
                distances = torch.arccos(cosine) / torch.pi

            nearest = torch.topk(distances, k=k, dim=1, largest=False).values
            chunk_scores = nearest.to(dtype=torch.float64).mean(dim=1).to(dtype=torch.float32)
            scores.append(chunk_scores.detach().cpu().numpy())

    return np.concatenate(scores, axis=0).astype(np.float32, copy=False)


def _resolve_knn_torch_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _as_2d_float_array(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have non-empty sample and feature dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _as_label_vector(values: Any, *, expected_size: int, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if array.shape[0] != expected_size:
        raise ValueError(f"{name} length {array.shape[0]} does not match sample count {expected_size}")
    unique = set(np.unique(array).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{name} must contain only binary labels 0 and 1")
    return array
