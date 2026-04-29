"""Regression tests for tdwm.client.TDClient._normalize.

Specifically guards two invariants whose absence corrupted production
data: OHLC consistency (high ≥ {open, low, close}, low ≤ them) after
TD's sub-bp rounding noise, and timezone normalization to NY.
"""
from __future__ import annotations

import pandas as pd

from tdwm.client import FetchRequest, TDClient


def _req(timezone: str = "America/New_York") -> FetchRequest:
    return FetchRequest(symbol="TEST", interval="1day", timezone=timezone)


def _row(open_, high, low, close, ts="2020-01-01 09:30") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp(ts)],
            "open": [open_], "high": [high], "low": [low], "close": [close],
            "volume": [1_000_000],
        }
    )


def test_normalize_clamps_close_above_high():
    """close > high must lift high to close; low untouched."""
    out = TDClient._normalize(_row(100.0, 101.0, 99.0, 101.5), _req())
    assert out.loc[0, "high"] == 101.5
    assert out.loc[0, "low"] == 99.0
    assert out.loc[0, "open"] == 100.0
    assert out.loc[0, "close"] == 101.5


def test_normalize_clamps_close_below_low():
    """close < low must drop low to close; high untouched."""
    out = TDClient._normalize(_row(100.0, 101.0, 99.0, 98.5), _req())
    assert out.loc[0, "low"] == 98.5
    assert out.loc[0, "high"] == 101.0
    assert out.loc[0, "open"] == 100.0
    assert out.loc[0, "close"] == 98.5


def test_normalize_well_formed_unchanged():
    """OHLC already-consistent input passes through with O/H/L/C unchanged."""
    out = TDClient._normalize(_row(100.0, 101.0, 99.0, 100.5), _req())
    assert out.loc[0, "open"] == 100.0
    assert out.loc[0, "high"] == 101.0
    assert out.loc[0, "low"] == 99.0
    assert out.loc[0, "close"] == 100.5


def test_normalize_localizes_naive_datetime():
    """Naive datetime input gets tz-localized to req.timezone (NY)."""
    out = TDClient._normalize(_row(100.0, 101.0, 99.0, 100.5), _req())
    assert str(out["datetime"].dt.tz) == "America/New_York"


def test_normalize_converts_tz_aware_datetime():
    """tz-aware UTC datetime gets converted (not relocalized) to NY."""
    df = pd.DataFrame(
        {
            "datetime": [pd.Timestamp("2020-01-01 14:30", tz="UTC")],
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
            "volume": [1_000_000],
        }
    )
    out = TDClient._normalize(df, _req())
    assert str(out["datetime"].dt.tz) == "America/New_York"
    # 14:30 UTC on a January day is 09:30 EST (UTC-5).
    assert out["datetime"].iloc[0].hour == 9
    assert out["datetime"].iloc[0].minute == 30


def test_normalize_handles_empty_input():
    """Empty input returns the canonical empty schema."""
    out = TDClient._normalize(pd.DataFrame(), _req())
    assert list(out.columns) == ["datetime", "open", "high", "low", "close", "volume"]
    assert len(out) == 0
