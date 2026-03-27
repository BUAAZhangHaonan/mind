"""YAML configuration loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import BaseModel


ConfigT = TypeVar("ConfigT", bound=BaseModel)


def load_yaml_config(path: str | Path, model_type: type[ConfigT]) -> ConfigT:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return model_type.model_validate(payload)
