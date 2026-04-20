"""Shared test fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# Add src/ to path so tests can `from tdwm import ...` without install.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture(scope="session")
def synthetic_bars() -> pd.DataFrame:
    """Deterministic synthetic daily bars."""
    rng = np.random.default_rng(42)
    n = 600
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="America/New_York")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame(
        {
            "datetime": idx,
            "symbol": "TEST",
            "timeframe": "1day",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "close_adj": close,
            "volume": volume,
        }
    )
