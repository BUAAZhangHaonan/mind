from __future__ import annotations

import importlib
from typing import Any

from tests.stage_b.conftest import (  # noqa: F401
    GLM_MODEL_ALIAS,
    PANEL_MODELS,
    write_unified_manifest,
)


STAGE_C_GLM_EXCLUSION_REASON = (
    "answer format incompatible with frozen yes/no population rule"
)


def stage_c_attr(module_name: str, attr_name: str) -> Any:
    module = importlib.import_module(f"mind.trajectory.{module_name}")
    return getattr(module, attr_name)


def stage_c_script_attr(script_name: str, attr_name: str) -> Any:
    module = importlib.import_module(f"scripts.{script_name}")
    return getattr(module, attr_name)
