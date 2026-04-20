"""Time-based split tests."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tdwm.splits import SplitConfig


@pytest.fixture
def cfg() -> SplitConfig:
    return SplitConfig(
        train_end=date(2023, 12, 31),
        val_start=date(2024, 1, 1),
        val_end=date(2024, 12, 31),
        test_start=date(2025, 1, 1),
        test_end=None,
        strict_no_crossing=True,
    )


def test_train_val_test_assignment(cfg: SplitConfig) -> None:
    assert cfg.assign(pd.Timestamp("2022-06-15")) == "train"
    assert cfg.assign(pd.Timestamp("2023-12-31")) == "train"
    assert cfg.assign(pd.Timestamp("2024-01-01")) == "val"
    assert cfg.assign(pd.Timestamp("2024-12-31")) == "val"
    assert cfg.assign(pd.Timestamp("2025-01-01")) == "test"
    assert cfg.assign(pd.Timestamp("2026-04-18")) == "test"


def test_no_overlap_across_splits(cfg: SplitConfig) -> None:
    # Train.end + 1 day is val.start — boundaries are aligned.
    assert cfg.train_end < cfg.val_start
    assert cfg.val_end < cfg.test_start


def test_strict_no_crossing_drops_boundary_windows(cfg: SplitConfig) -> None:
    # Window that straddles train→val returns None.
    split = cfg.split_of_trajectory(
        pd.Timestamp("2023-12-20"), pd.Timestamp("2024-01-10")
    )
    assert split is None


def test_non_strict_uses_last_timestamp() -> None:
    cfg = SplitConfig(
        train_end=date(2023, 12, 31),
        val_start=date(2024, 1, 1),
        val_end=date(2024, 12, 31),
        test_start=date(2025, 1, 1),
        test_end=None,
        strict_no_crossing=False,
    )
    split = cfg.split_of_trajectory(
        pd.Timestamp("2023-12-20"), pd.Timestamp("2024-01-10")
    )
    assert split == "val"
