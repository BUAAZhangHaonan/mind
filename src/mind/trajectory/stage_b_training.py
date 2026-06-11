"""Stage B training helpers for frozen Sphere-Traj-LSTM objectives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .stage_a_representations import (
    DEFAULT_LSTM_EMBEDDING_DIM,
    DEFAULT_STAGE_A_SEED,
    StageATrajectoryLSTM,
    set_deterministic_seed,
)
from .stage_b_objectives import (
    ALLOWED_STAGE_B_OBJECTIVES,
    bce_with_logits_loss,
    proxy_anchor_loss,
    supcon_loss,
)


@dataclass
class StageBTrainingResult:
    """Trained Stage B encoder and objective history."""

    model: StageATrajectoryLSTM
    objective: str
    history: list[dict[str, float]]
    seed: int


def train_stage_b_lstm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    objective: str,
    num_layers: int,
    hidden_dim: int,
    epochs: int = 20,
    batch_size: int = 128,
    device: str = "cpu",
    seed: int = DEFAULT_STAGE_A_SEED,
    learning_rate: float = 1e-3,
    embedding_dim: int = DEFAULT_LSTM_EMBEDDING_DIM,
    temperature: float = 0.07,
    proxy_margin: float = 0.1,
    proxy_alpha: float = 32.0,
    patience: int | None = 5,
) -> StageBTrainingResult:
    """Train the fixed Stage B encoder with one objective family."""

    if objective not in ALLOWED_STAGE_B_OBJECTIVES:
        raise ValueError(f"unsupported Stage B objective: {objective}")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")

    features = np.asarray(x, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("x must have shape (samples, num_layers, hidden_dim)")
    if int(features.shape[1]) != int(num_layers):
        raise ValueError(f"x has {features.shape[1]} layers, expected {num_layers}")
    if int(features.shape[2]) != int(hidden_dim):
        raise ValueError(f"x has hidden_dim {features.shape[2]}, expected {hidden_dim}")
    labels = _label_vector(y, expected_size=features.shape[0])
    if np.unique(labels).size < 2:
        raise ValueError("Stage B training labels must contain both classes")

    set_deterministic_seed(seed)
    target_device = torch.device(device)
    model = StageATrajectoryLSTM(hidden_dim=int(hidden_dim), embedding_dim=int(embedding_dim))
    model.to(target_device)

    trainable: list[torch.nn.Parameter] = list(model.parameters())
    proxies: torch.nn.Parameter | None = None
    if objective == "proxy_anchor":
        proxies = torch.nn.Parameter(torch.randn(2, int(embedding_dim), device=target_device) * 0.01)
        trainable.append(proxies)

    optimizer = torch.optim.Adam(trainable, lr=float(learning_rate))
    tensor_x = torch.from_numpy(features)
    tensor_y = torch.from_numpy(labels.astype(np.int64, copy=False))
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    effective_batch_size = min(int(batch_size), int(features.shape[0]))
    if objective in {"supcon", "proxy_anchor"}:
        # Metric losses are more stable when each step sees more positives and negatives.
        effective_batch_size = min(max(effective_batch_size, 256), int(features.shape[0]))
    loader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=effective_batch_size,
        shuffle=True,
        generator=generator,
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)
            optimizer.zero_grad(set_to_none=True)
            embeddings, logits = model.embed_and_score(batch_x)
            if objective == "bce":
                positives = float((batch_y == 1).sum().item())
                negatives = float((batch_y == 0).sum().item())
                pos_weight = None
                if positives > 0.0 and negatives > 0.0:
                    pos_weight = torch.tensor(
                        negatives / positives,
                        dtype=torch.float32,
                        device=target_device,
                    )
                loss = bce_with_logits_loss(logits, batch_y.float(), pos_weight=pos_weight)
            elif objective == "supcon":
                loss = supcon_loss(embeddings, batch_y, temperature=temperature)
            else:
                assert proxies is not None
                loss = proxy_anchor_loss(
                    embeddings,
                    batch_y,
                    proxies=proxies,
                    num_classes=2,
                    margin=proxy_margin,
                    alpha=proxy_alpha,
                )
            loss.backward()
            optimizer.step()
            batch_size_actual = int(batch_x.shape[0])
            total_loss += float(loss.detach().cpu().item()) * batch_size_actual
            total_examples += batch_size_actual

        epoch_loss = total_loss / max(total_examples, 1)
        history.append({"epoch": float(epoch), "loss": float(epoch_loss)})
        if patience is None:
            continue
        if epoch_loss < best_loss - 1e-8:
            best_loss = epoch_loss
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(patience):
                break

    model.eval()
    return StageBTrainingResult(
        model=model,
        objective=str(objective),
        history=history,
        seed=int(seed),
    )


def score_stage_b_lstm(
    model: StageATrajectoryLSTM,
    trajectories: np.ndarray,
    *,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized embeddings and BCE-head probabilities for trajectories."""

    values = np.asarray(trajectories, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError("trajectories must have shape (samples, num_layers, hidden_dim)")
    device = next(model.parameters()).device
    embeddings: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, values.shape[0], int(batch_size)):
            batch = torch.from_numpy(values[start : start + int(batch_size)]).to(device)
            emb, logits = model.embed_and_score(batch)
            embeddings.append(emb.detach().cpu().numpy().astype(np.float32))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32))
    embedding_matrix = np.concatenate(embeddings, axis=0)
    return _l2_normalize(embedding_matrix), np.concatenate(probabilities, axis=0)


def _label_vector(values: np.ndarray | Sequence[int], *, expected_size: int) -> np.ndarray:
    labels = np.asarray(values, dtype=np.int64).reshape(-1)
    if labels.shape[0] != int(expected_size):
        raise ValueError("label length must match sample count")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary 0/1")
    return labels


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)
