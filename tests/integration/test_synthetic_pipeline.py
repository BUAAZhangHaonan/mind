from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mind.detectors import fit_logistic_detector
from mind.drift import compute_drift_curve
from mind.evaluation import compute_binary_metrics
from mind.manifolds import build_reference_bank
from mind.visualization import plot_drift_curves, plot_roc_curve
from mind.wavelets import extract_wavelet_features


def test_synthetic_pipeline_runs_end_to_end(tmp_path: Path) -> None:
    reference_entries = [
        {
            "object_name": "dog",
            "selected_layers": [8, 13],
            "layer_vectors": torch.tensor([[0.2, 0.3, 0.0], [0.3, 0.4, 0.0]]),
        },
        {
            "object_name": "dog",
            "selected_layers": [8, 13],
            "layer_vectors": torch.tensor([[0.1, 0.4, 0.0], [0.2, 0.5, 0.0]]),
        },
        {
            "object_name": "dog",
            "selected_layers": [8, 13],
            "layer_vectors": torch.tensor([[0.4, 0.2, 0.0], [0.5, 0.3, 0.0]]),
        },
        {
            "object_name": "dog",
            "selected_layers": [8, 13],
            "layer_vectors": torch.tensor([[0.3, 0.3, 0.0], [0.4, 0.4, 0.0]]),
        },
    ]
    reference_bank = build_reference_bank(reference_entries)

    samples = [
        (torch.tensor([[0.25, 0.35, 0.0], [0.35, 0.45, 0.0]]), 0),
        (torch.tensor([[0.22, 0.38, 0.0], [0.32, 0.48, 0.0]]), 0),
        (torch.tensor([[0.25, 0.35, 0.5], [0.35, 0.45, 0.6]]), 1),
        (torch.tensor([[0.20, 0.30, 0.7], [0.30, 0.40, 0.8]]), 1),
    ]

    feature_rows = []
    labels = []
    curves = []
    for layer_vectors, label in samples:
        curve = compute_drift_curve(
            layer_vectors=layer_vectors,
            selected_layers=[8, 13],
            object_name="dog",
            reference_bank=reference_bank,
            k_neighbors=4,
        )
        curves.append(curve)
        features = extract_wavelet_features(curve)
        feature_rows.append(features)
        labels.append(label)

    feature_frame = pd.DataFrame(feature_rows)
    detector = fit_logistic_detector(feature_frame.to_numpy(dtype=np.float32), np.asarray(labels))
    predictions = detector.predict(feature_frame.to_numpy(dtype=np.float32))
    scores = detector.predict_proba(feature_frame.to_numpy(dtype=np.float32))[:, 1]

    metrics = compute_binary_metrics(y_true=labels, y_pred=predictions.tolist(), y_score=scores.tolist())
    assert metrics["accuracy"] == 1.0

    plot_drift_curves(
        grounded_curve=curves[0],
        hallucinated_curve=curves[-1],
        output_path=tmp_path / "drift.png",
    )
    plot_roc_curve(
        y_true=labels,
        y_score=scores.tolist(),
        output_path=tmp_path / "roc.png",
    )

    assert (tmp_path / "drift.png").exists()
    assert (tmp_path / "roc.png").exists()
