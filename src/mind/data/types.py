"""Dataset record types."""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class HallucinationRecord:
    sample_id: str
    image_id: int
    image_path: str
    question: str
    label: int
    object_name: str
    split: str
    subset: str
    source_dataset: str

    def with_label(self, label: int, source_dataset: str) -> "HallucinationRecord":
        return replace(self, label=label, source_dataset=source_dataset)
