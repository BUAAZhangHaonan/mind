#!/usr/bin/env python3
"""Verify the local MIND environment."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Iterable


REQUIRED_MODULES = (
    "mind",
    "torch",
    "transformers",
    "accelerate",
    "datasets",
    "sklearn",
    "pandas",
    "yaml",
    "PIL",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="",
        help="Optional Hugging Face model id used for config and processor checks.",
    )
    return parser.parse_args()


def check_imports(modules: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in modules:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        versions[name] = str(version)
    return versions


def print_versions(versions: dict[str, str]) -> None:
    print("Imported modules:")
    for name, version in versions.items():
        print(f"  - {name}: {version}")


def check_torch_runtime() -> None:
    import torch

    print("Torch runtime:")
    print(f"  - version: {torch.__version__}")
    print(f"  - cuda available: {torch.cuda.is_available()}")
    print(f"  - cuda device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            print(f"  - gpu[{index}]: {torch.cuda.get_device_name(index)}")


def check_huggingface(model_id: str) -> None:
    if not model_id:
        print("Skipping Hugging Face model config check.")
        return

    from transformers import AutoConfig, AutoProcessor

    print(f"Checking Hugging Face model assets for: {model_id}")
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    print(f"  - loaded config class: {config.__class__.__name__}")
    try:
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    except Exception as exc:  # pragma: no cover - network and model dependent
        print(f"  - processor load failed: {exc}")
    else:
        print(f"  - loaded processor class: {processor.__class__.__name__}")


def main() -> int:
    args = parse_args()
    print(f"Python executable: {sys.executable}")
    print(f"HF_ENDPOINT: {os.environ.get('HF_ENDPOINT', '<unset>')}")
    versions = check_imports(REQUIRED_MODULES)
    print_versions(versions)
    check_torch_runtime()
    check_huggingface(args.model_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
