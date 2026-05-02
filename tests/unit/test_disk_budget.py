from __future__ import annotations

from pathlib import Path

import pytest

from mind.cache.disk_budget import BudgetExceededError, DiskBudget, parse_byte_size


def test_parse_byte_size_accepts_binary_and_decimal_units() -> None:
    assert parse_byte_size("1024") == 1024
    assert parse_byte_size("1 KiB") == 1024
    assert parse_byte_size("1.5MiB") == 1_572_864
    assert parse_byte_size("2 MB") == 2_000_000
    assert parse_byte_size("3gib") == 3 * 1024**3


def test_disk_budget_counts_existing_usage_and_running_allocations(tmp_path: Path) -> None:
    existing = tmp_path / "existing.pt"
    existing.write_bytes(b"x" * 100)
    budget = DiskBudget(root=tmp_path, max_bytes=250, warn_fraction=0.8, halt_fraction=1.0)

    allocation = budget.allocate(120, label="first shard")

    assert allocation.existing_bytes == 100
    assert allocation.allocated_bytes == 120
    assert allocation.projected_bytes == 220
    assert budget.reserved_bytes == 120
    assert budget.remaining_bytes == 30


def test_disk_budget_warns_when_projected_usage_crosses_warning_threshold(tmp_path: Path) -> None:
    budget = DiskBudget(root=tmp_path, max_bytes=100, warn_fraction=0.5, halt_fraction=1.0)

    with pytest.warns(ResourceWarning, match="Disk budget warning"):
        budget.allocate(60, label="warning shard")


def test_disk_budget_raises_clear_exception_when_halt_threshold_would_be_exceeded(
    tmp_path: Path,
) -> None:
    budget = DiskBudget(root=tmp_path, max_bytes=100, warn_fraction=0.8, halt_fraction=0.9)

    with pytest.raises(BudgetExceededError, match="projected usage .* exceeds halt threshold"):
        budget.allocate(91, label="too large")

    assert budget.reserved_bytes == 0
