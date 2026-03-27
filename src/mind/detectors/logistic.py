"""Logistic detector for MIND."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic_detector(features: np.ndarray, labels: np.ndarray) -> Pipeline:
    detector = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=200, solver="liblinear")),
        ]
    )
    detector.fit(features, labels)
    return detector
