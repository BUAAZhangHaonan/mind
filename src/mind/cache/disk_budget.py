"""Disk budget tracking for cache extraction runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import warnings


class BudgetExceededError(RuntimeError):
    """Raised when a projected cache write exceeds the halt threshold."""


_BYTE_UNITS = {
    "": 1,
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}

_BYTE_SIZE_PATTERN = re.compile(r"^\s*(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>[A-Za-z]*)\s*$")


@dataclass(frozen=True)
class BudgetAllocation:
    """A single successful budget reservation."""

    label: str
    estimated_bytes: int
    existing_bytes: int
    allocated_bytes: int
    projected_bytes: int
    max_bytes: int
    warning_bytes: int
    halt_bytes: int


def parse_byte_size(value: str | int | float) -> int:
    """Parse a byte size such as ``10GiB``, ``500 MB``, or ``1024``."""

    if isinstance(value, int):
        parsed = value
    elif isinstance(value, float):
        parsed = int(value)
    else:
        match = _BYTE_SIZE_PATTERN.match(value)
        if match is None:
            raise ValueError(f"Invalid byte size: {value!r}")
        unit = match.group("unit").lower()
        if unit not in _BYTE_UNITS:
            raise ValueError(f"Unsupported byte unit: {match.group('unit')!r}")
        parsed = int(float(match.group("value")) * _BYTE_UNITS[unit])
    if parsed < 0:
        raise ValueError("byte size must be non-negative")
    return parsed


def _directory_size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
    return total


@dataclass
class DiskBudget:
    """Track existing cache usage and reservations for an output root."""

    root: Path
    max_bytes: int
    warn_fraction: float = 0.9
    halt_fraction: float = 1.0
    include_existing: bool = True
    _existing_bytes: int = field(init=False, repr=False)
    _reserved_bytes: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.max_bytes = parse_byte_size(self.max_bytes)
        if self.max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if not 0 <= self.warn_fraction <= self.halt_fraction:
            raise ValueError("warn_fraction must be between 0 and halt_fraction")
        if self.halt_fraction <= 0:
            raise ValueError("halt_fraction must be positive")
        object.__setattr__(
            self,
            "_existing_bytes",
            _directory_size_bytes(self.root) if self.include_existing else 0,
        )

    @property
    def existing_bytes(self) -> int:
        return self._existing_bytes

    @property
    def reserved_bytes(self) -> int:
        return self._reserved_bytes

    @property
    def warning_bytes(self) -> int:
        return int(self.max_bytes * self.warn_fraction)

    @property
    def halt_bytes(self) -> int:
        return int(self.max_bytes * self.halt_fraction)

    @property
    def projected_bytes(self) -> int:
        return self.existing_bytes + self.reserved_bytes

    @property
    def remaining_bytes(self) -> int:
        return max(0, self.halt_bytes - self.projected_bytes)

    def allocate(self, estimated_bytes: int, *, label: str = "cache write") -> BudgetAllocation:
        estimated = parse_byte_size(estimated_bytes)
        projected = self.existing_bytes + self.reserved_bytes + estimated
        if projected > self.halt_bytes:
            raise BudgetExceededError(
                f"{label}: projected usage {projected} bytes exceeds halt threshold "
                f"{self.halt_bytes} bytes for {self.root}"
            )
        self._reserved_bytes += estimated
        allocation = BudgetAllocation(
            label=label,
            estimated_bytes=estimated,
            existing_bytes=self.existing_bytes,
            allocated_bytes=estimated,
            projected_bytes=projected,
            max_bytes=self.max_bytes,
            warning_bytes=self.warning_bytes,
            halt_bytes=self.halt_bytes,
        )
        if projected >= self.warning_bytes:
            warnings.warn(
                f"Disk budget warning for {label}: projected usage {projected} bytes "
                f"has reached warning threshold {self.warning_bytes} bytes",
                ResourceWarning,
                stacklevel=2,
            )
        return allocation

    def record_actual(self, *, estimated_bytes: int, actual_bytes: int, label: str = "cache write") -> None:
        """Adjust reservations after a write reports its actual size."""

        estimated = parse_byte_size(estimated_bytes)
        actual = parse_byte_size(actual_bytes)
        self._reserved_bytes += actual - estimated
        if self.projected_bytes > self.halt_bytes:
            raise BudgetExceededError(
                f"{label}: actual usage {self.projected_bytes} bytes exceeds halt threshold "
                f"{self.halt_bytes} bytes for {self.root}"
            )
