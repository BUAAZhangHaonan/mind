"""Logistic detector for MIND."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class TorchLogisticDetector:
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray
    intercept_: np.ndarray
    device: str = "cpu"
    classes_: np.ndarray = field(default_factory=lambda: np.asarray([0, 1], dtype=np.int64))

    def _transform(self, features: np.ndarray) -> torch.Tensor:
        feature_array = np.asarray(features, dtype=np.float64)
        scaled = (feature_array - self.mean_) / self.scale_
        augmented = np.concatenate(
            [scaled, np.ones((scaled.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        return torch.as_tensor(augmented, dtype=torch.float64, device=self.device)

    def decision_function(self, features: np.ndarray) -> np.ndarray:
        augmented = self._transform(features)
        weight = torch.as_tensor(
            np.concatenate([self.coef_.reshape(-1), self.intercept_.reshape(-1)]),
            dtype=torch.float64,
            device=self.device,
        )
        logits = augmented @ weight
        return logits.detach().cpu().numpy()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        logits = self.decision_function(features)
        positive = 1.0 / (1.0 + np.exp(-logits))
        return np.column_stack([1.0 - positive, positive])

    def predict(self, features: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(features)[:, 1]
        return (probabilities >= 0.5).astype(np.int64)


def _fit_torch_logistic_detector(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: str,
    max_iter: int = 200,
) -> TorchLogisticDetector:
    feature_array = np.asarray(features, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)

    mean = feature_array.mean(axis=0)
    scale = feature_array.std(axis=0)
    scale[scale == 0.0] = 1.0
    scaled = (feature_array - mean) / scale
    augmented = np.concatenate(
        [scaled, np.ones((scaled.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    signed_labels = label_array.astype(np.float64) * 2.0 - 1.0

    feature_tensor = torch.as_tensor(augmented, dtype=torch.float64, device=device)
    label_tensor = torch.as_tensor(signed_labels, dtype=torch.float64, device=device)
    weight = torch.zeros(augmented.shape[1], dtype=torch.float64, device=device, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [weight],
        lr=1.0,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-14,
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        margins = label_tensor * (feature_tensor @ weight)
        loss = 0.5 * torch.sum(weight * weight) + torch.nn.functional.softplus(-margins).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted_weight = weight.detach().cpu().numpy()
    return TorchLogisticDetector(
        mean_=mean,
        scale_=scale,
        coef_=fitted_weight[:-1].reshape(1, -1),
        intercept_=fitted_weight[-1:].copy(),
        device=device,
    )


def fit_logistic_detector(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    device: str = "cpu",
    backend: str = "auto",
) -> Pipeline | TorchLogisticDetector:
    use_torch_backend = backend == "torch" or (backend == "auto" and device != "cpu")
    if use_torch_backend:
        return _fit_torch_logistic_detector(features, labels, device=device)

    detector = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200, solver="liblinear")),
        ]
    )
    detector.fit(features, labels)
    return detector
