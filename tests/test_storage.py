"""Storage tests — partitioning, idempotent merge, restatement."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tdwm.storage import last_datetime, read_bars, write_bars


def _frame(symbol: str, start: str, n: int, close0: float = 100.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B", tz="America/New_York")
    return pd.DataFrame({
        "datetime": idx,
        "symbol": symbol,
        "timeframe": "1day",
        "open": close0 + np.arange(n) * 0.1,
        "high": close0 + np.arange(n) * 0.1 + 1,
        "low": close0 + np.arange(n) * 0.1 - 1,
        "close": close0 + np.arange(n) * 0.1,
        "close_adj": close0 + np.arange(n) * 0.1,
        "volume": np.full(n, 1_000_000),
    })


def test_roundtrip(tmp_path: Path) -> None:
    df = _frame("AAPL", "2024-01-02", 10)
    write_bars(df, tmp_path)
    back = read_bars(tmp_path, timeframe="1day", symbol="AAPL")
    assert len(back) == len(df)
    assert back["close"].iloc[-1] == df["close"].iloc[-1]


def test_idempotent_rewrite(tmp_path: Path) -> None:
    df = _frame("AAPL", "2024-01-02", 10)
    write_bars(df, tmp_path)
    write_bars(df, tmp_path)          # same rows again
    back = read_bars(tmp_path, timeframe="1day", symbol="AAPL")
    assert len(back) == 10            # deduped


def test_restatement_last_write_wins(tmp_path: Path) -> None:
    df = _frame("AAPL", "2024-01-02", 10)
    write_bars(df, tmp_path)
    # Overwrite last row with a "restated" close.
    restated = df.iloc[-1:].copy()
    restated["close"] = 999.0
    restated["close_adj"] = 999.0
    write_bars(restated, tmp_path)
    back = read_bars(tmp_path, timeframe="1day", symbol="AAPL")
    assert len(back) == 10
    assert back["close"].iloc[-1] == 999.0


def test_year_partitioning(tmp_path: Path) -> None:
    df = _frame("AAPL", "2023-12-20", 15)   # spans 2023 → 2024
    write_bars(df, tmp_path)
    p23 = tmp_path / "bars" / "1day" / "symbol=AAPL" / "year=2023" / "part.parquet"
    p24 = tmp_path / "bars" / "1day" / "symbol=AAPL" / "year=2024" / "part.parquet"
    assert p23.exists()
    assert p24.exists()


def test_last_datetime_helper(tmp_path: Path) -> None:
    df = _frame("AAPL", "2024-01-02", 10)
    write_bars(df, tmp_path)
    last = last_datetime(tmp_path, timeframe="1day", symbol="AAPL")
    assert pd.Timestamp(last).date() == df["datetime"].iloc[-1].date()


def test_dot_in_symbol(tmp_path: Path) -> None:
    df = _frame("BRK.B", "2024-01-02", 5)
    write_bars(df, tmp_path)
    back = read_bars(tmp_path, timeframe="1day", symbol="BRK.B")
    assert len(back) == 5
