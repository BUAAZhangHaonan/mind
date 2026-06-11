from __future__ import annotations

import importlib
from typing import Any

from tests.stage_b.conftest import (  # noqa: F401
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    EXPECTED_STAGE_B_K_VALUES,
    write_unified_manifest,
)


def stage_b2_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(f"mind.trajectory.{module_name}")
    return getattr(module, attr_name)


def stage_b2_script_attr(script_name: str, attr_name: str) -> Any:
    module = importlib.import_module(f"scripts.{script_name}")
    return getattr(module, attr_name)
