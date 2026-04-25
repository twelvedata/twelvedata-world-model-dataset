"""Tests for corporate actions detection and re-backfill logic."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from tdwm.corporate_actions import (
    check_corporate_actions,
    clear_symbol_bars,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sdk(splits: dict[str, list[str]], dividends: dict[str, list[str]]) -> MagicMock:
    """Return a fake SDK whose get_splits / get_dividends return canned data.

    Mirrors the real twelvedata SDK's `.as_json()` shape:
        {"meta": {...}, "splits"|"dividends": [{"date"|"ex_date": ...}, ...]}
    """
    sdk = MagicMock()

    def _splits_builder(symbol, **_):
        items = [{"date": d} for d in splits.get(symbol, [])]
        builder = MagicMock()
        builder.as_json.return_value = {"meta": {"symbol": symbol}, "splits": items}
        return builder

    def _dividends_builder(symbol, **_):
        items = [{"ex_date": d} for d in dividends.get(symbol, [])]
        builder = MagicMock()
        builder.as_json.return_value = {"meta": {"symbol": symbol}, "dividends": items}
        return builder

    sdk.get_splits.side_effect = _splits_builder
    sdk.get_dividends.side_effect = _dividends_builder
    return sdk


def _make_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=3), "close": [1, 2, 3]})
    df.to_parquet(path)


# ---------------------------------------------------------------------------
# check_corporate_actions
# ---------------------------------------------------------------------------

def test_no_events_returns_empty(tmp_path: Path) -> None:
    sdk = _make_sdk({}, {})
    dirty = check_corporate_actions(sdk, ["AAPL", "MSFT"], tmp_path)
    assert dirty == set()


def test_split_marks_symbol_dirty(tmp_path: Path) -> None:
    sdk = _make_sdk({"AAPL": ["2024-06-10"]}, {})
    dirty = check_corporate_actions(sdk, ["AAPL", "MSFT"], tmp_path)
    assert dirty == {"AAPL"}


def test_dividend_marks_symbol_dirty(tmp_path: Path) -> None:
    sdk = _make_sdk({}, {"MSFT": ["2024-08-15"]})
    dirty = check_corporate_actions(sdk, ["AAPL", "MSFT"], tmp_path)
    assert dirty == {"MSFT"}


def test_seen_event_not_re_triggered(tmp_path: Path) -> None:
    sdk = _make_sdk({"AAPL": ["2024-06-10"]}, {})
    # First call — event is new.
    dirty1 = check_corporate_actions(sdk, ["AAPL"], tmp_path)
    assert "AAPL" in dirty1
    # Second call — same event already in seen file.
    dirty2 = check_corporate_actions(sdk, ["AAPL"], tmp_path)
    assert dirty2 == set()


def test_persists_seen_file(tmp_path: Path) -> None:
    sdk = _make_sdk({"AAPL": ["2024-06-10"]}, {})
    check_corporate_actions(sdk, ["AAPL"], tmp_path)
    seen_file = tmp_path / "_corporate_actions.json"
    assert seen_file.exists()
    data = json.loads(seen_file.read_text())
    assert "AAPL" in data["seen"]
    assert any("split" in e for e in data["seen"]["AAPL"])


def test_multiple_events_same_symbol(tmp_path: Path) -> None:
    sdk = _make_sdk({"AAPL": ["2024-06-10", "2023-08-01"]}, {"AAPL": ["2024-05-15"]})
    dirty = check_corporate_actions(sdk, ["AAPL"], tmp_path)
    assert dirty == {"AAPL"}


# ---------------------------------------------------------------------------
# clear_symbol_bars
# ---------------------------------------------------------------------------

def test_clear_symbol_bars_removes_partitions(tmp_path: Path) -> None:
    sym = "AAPL"
    safe = "AAPL"
    # Create fake parquet files for two years and two timeframes.
    for tf in ("1day", "1h"):
        for year in (2023, 2024):
            p = tmp_path / "bars" / tf / f"symbol={safe}" / f"year={year}" / "part.parquet"
            _make_parquet(p)

    clear_symbol_bars(tmp_path, sym)

    remaining = list((tmp_path / "bars").glob("**/part.parquet"))
    assert remaining == []


def test_clear_symbol_bars_leaves_other_symbols(tmp_path: Path) -> None:
    for sym in ("AAPL", "MSFT"):
        p = tmp_path / "bars" / "1day" / f"symbol={sym}" / "year=2024" / "part.parquet"
        _make_parquet(p)

    clear_symbol_bars(tmp_path, "AAPL")

    remaining = list((tmp_path / "bars").glob("**/part.parquet"))
    assert len(remaining) == 1
    assert "MSFT" in str(remaining[0])


def test_clear_symbol_bars_noop_if_no_data(tmp_path: Path) -> None:
    # Should not raise if no files exist.
    clear_symbol_bars(tmp_path, "AAPL")


def test_clear_handles_slash_in_symbol(tmp_path: Path) -> None:
    safe = "BRK_B"
    p = tmp_path / "bars" / "1day" / f"symbol={safe}" / "year=2024" / "part.parquet"
    _make_parquet(p)
    clear_symbol_bars(tmp_path, "BRK/B")
    remaining = list((tmp_path / "bars").glob("**/part.parquet"))
    assert remaining == []
