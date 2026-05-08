"""Stage A trajectory representation builders."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

DEFAULT_STAGE_A_SEED = 20260506
DEFAULT_LSTM_EMBEDDING_DIM = 128
NORM_EPSILON = 1e-12


def set_deterministic_seed(seed: int = DEFAULT_STAGE_A_SEED) -> None:
    """Seed Python, NumPy, and PyTorch for deterministic Stage A diagnostics."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def shuffled_layer_permutation(num_layers: int, seed: int = DEFAULT_STAGE_A_SEED) -> list[int]:
    """Return a deterministic non-identity layer permutation when possible."""

    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    order = np.random.default_rng(seed).permutation(num_layers).astype(int).tolist()
    if num_layers > 1 and order == list(range(num_layers)):
        order = order[1:] + order[:1]
    return order


def build_lstm_trajectory(
    entry: Mapping[str, object],
    *,
    layer_order: Sequence[int] | None = None,
) -> np.ndarray:
    """Build the layer-normalized trajectory used by Stage A LSTM variants."""

    vectors = _layer_vectors(entry)
    if layer_order is not None:
        vectors = vectors[np.asarray(layer_order, dtype=np.int64)]
    return _row_unit_vectors(vectors)


def build_representation(
    entry: Mapping[str, object],
    representation: str,
    *,
    layer_order: Sequence[int] | None = None,
) -> np.ndarray:
    """Build one Stage A representation from a Stage 0 cache entry."""

    vectors = _layer_vectors(entry)
    if layer_order is not None:
        vectors = vectors[np.asarray(layer_order, dtype=np.int64)]

    name = representation.strip()
    if name == "Raw-Static":
        return vectors[-1].copy()
    if name == "Sphere-Static":
        return _unit_vector(vectors[-1])
    if name == "Norm-Static":
        return np.asarray([_safe_log_norm(vectors[-1])], dtype=np.float32)
    if name == "Raw-Traj-MeanPool":
        return vectors.mean(axis=0, dtype=np.float32).astype(np.float32, copy=False)
    if name == "Sphere-Traj-MeanPool":
        pooled = _row_unit_vectors(vectors).mean(axis=0, dtype=np.float32)
        return _unit_vector(pooled)
    if name == "Norm-Traj":
        norms = np.linalg.norm(vectors, axis=1).astype(np.float32, copy=False)
        return np.log(np.maximum(norms, NORM_EPSILON)).astype(np.float32, copy=False)
    raise ValueError(f"Unknown Stage A representation: {representation}")


@dataclass(frozen=True)
class RepresentationBatch:
    name: str
    values: np.ndarray


def build_representation_matrix(
    entries: Sequence[Mapping[str, object]],
    representation: str,
    *,
    layer_order: Sequence[int] | None = None,
) -> RepresentationBatch:
    """Build a stacked matrix for one representation across cache entries."""

    if not entries:
        raise ValueError("entries must not be empty")
    values = [
        build_representation(entry, representation, layer_order=layer_order)
        for entry in entries
    ]
    return RepresentationBatch(
        name=representation,
        values=np.stack(values, axis=0).astype(np.float32, copy=False),
    )


class StageATrajectoryLSTM(nn.Module):
    """One-layer LSTM diagnostic readout over full-layer trajectories."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        embedding_dim: int = DEFAULT_LSTM_EMBEDDING_DIM,
        lstm_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")
        effective_dropout = float(dropout) if lstm_layers > 1 else 0.0
        self.input_hidden_dim = int(hidden_dim)
        self.embedding_dim = int(embedding_dim)
        self.input_projection = nn.Linear(int(hidden_dim), 256)
        self.activation = nn.GELU()
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=int(lstm_layers),
            batch_first=True,
            dropout=effective_dropout,
        )
        self.embedding_projection = nn.Linear(128, int(embedding_dim))
        self.classifier = nn.Linear(int(embedding_dim), 1)

    def embed_and_score(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return final LSTM embeddings and scalar logits."""

        if x.ndim != 3:
            raise ValueError("x must have shape (batch, num_layers, hidden_dim)")
        if int(x.shape[-1]) != self.input_hidden_dim:
            raise ValueError(
                f"x hidden_dim {int(x.shape[-1])} does not match model hidden_dim "
                f"{self.input_hidden_dim}"
            )
        projected = self.activation(self.input_projection(x.to(dtype=torch.float32)))
        _, (hidden, _) = self.lstm(projected)
        embeddings = self.embedding_projection(hidden[-1])
        logits = self.classifier(embeddings).squeeze(-1)
        return embeddings, logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return scalar logits for binary hallucination diagnostics."""

        _, logits = self.embed_and_score(x)
        return logits


def _layer_vectors(entry: Mapping[str, object]) -> np.ndarray:
    value = entry.get("layer_vectors")
    if value is None:
        value = entry.get("full_hidden_states")
    if value is None:
        raise ValueError("entry must contain layer_vectors")
    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().to(dtype=torch.float32).numpy()
    else:
        array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("layer_vectors must have shape (num_layers, hidden_dim)")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("layer_vectors must have non-empty layer and hidden dimensions")
    if not np.isfinite(array).all():
        raise ValueError("layer_vectors must be finite")
    return array.astype(np.float32, copy=False)


def _unit_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return (vector / max(norm, NORM_EPSILON)).astype(np.float32, copy=False)


def _row_unit_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.maximum(norms, NORM_EPSILON)).astype(np.float32, copy=False)


def _safe_log_norm(vector: np.ndarray) -> np.float32:
    norm = float(np.linalg.norm(vector))
    return np.float32(np.log(max(norm, NORM_EPSILON)))
