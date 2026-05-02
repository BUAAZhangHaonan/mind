"""Shared model wrapper helpers."""

from __future__ import annotations

import re

from mind.utils.dtypes import resolve_torch_dtype


_YES_NO_PATTERN = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def parse_yes_no_answer(text: str) -> int | None:
    match = _YES_NO_PATTERN.search(text)
    if match is None:
        return None
    return 1 if match.group(1).lower() == "yes" else 0
