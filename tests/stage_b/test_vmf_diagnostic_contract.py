from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from .conftest import stage_b_attr


def test_vmf_diagnostic_returns_and_writes_finite_direction_concentration_and_scores(
    tmp_path: Path,
) -> None:
    compute_vmf_diagnostic = stage_b_attr(
        "stage_b_vmf",
        "compute_vmf_diagnostic",
    )
    write_vmf_diagnostic = stage_b_attr(
        "stage_b_vmf",
        "write_vmf_diagnostic",
    )
    bank_embeddings = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.8, 0.6, 0.0],
            [0.8, -0.2, 0.2],
        ],
        dtype=np.float32,
    )
    query_embeddings = np.array(
        [
            [1.0, 0.1, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    result = compute_vmf_diagnostic(
        bank_embeddings=bank_embeddings,
        query_embeddings=query_embeddings,
    )
    output = tmp_path / "vmf_diagnostic.json"
    written = write_vmf_diagnostic(result, output)
    payload = json.loads(output.read_text(encoding="utf-8"))

    mean_direction = np.asarray(result["mean_direction"], dtype=np.float32)
    scores = np.asarray(result["scores"], dtype=np.float32)
    assert mean_direction.shape == (3,)
    assert float(np.linalg.norm(mean_direction)) == pytest.approx(1.0, abs=1e-6)
    assert np.isfinite(float(result["concentration_proxy"]))
    assert float(result["concentration_proxy"]) >= 0.0
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert written == payload
    np.testing.assert_allclose(payload["mean_direction"], result["mean_direction"], atol=1e-6)
    assert np.isfinite(float(payload["concentration_proxy"]))
