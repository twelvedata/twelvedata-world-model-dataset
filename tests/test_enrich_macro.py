"""Regression tests for tdwm.enrich.

Specifically guards two invariants whose absence corrupted production
data: per-column ffill across mismatched macro grids in
`build_macro_frame`, and backward-merge semantics in `attach_macro`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tdwm.enrich import attach_macro, build_macro_frame


NY = "America/New_York"


def _ts(minute_offset: int) -> pd.Timestamp:
    """Return 2020-03-24 09:30 NY plus N minutes."""
    return pd.Timestamp("2020-03-24 09:30", tz=NY) + pd.Timedelta(minutes=minute_offset)


def _macro(symbol_minutes: dict[int, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [_ts(m) for m in symbol_minutes],
            "close": list(symbol_minutes.values()),
        }
    )


def test_build_macro_frame_ffills_missing_minutes():
    """Sparse VIXY must not poison vix_level on minutes SPY has but VIXY skipped.

    Pre-fix bug: pd.concat(axis=1) on a union index left vix_level NaN
    on minute 3, and merge_asof in attach_macro would happily return
    that NaN even though VIXY's minute-2 close was the last valid
    observation.
    """
    macros = {
        "SPY": _macro({0: 100.0, 1: 101.0, 2: 102.0}),
        "VIXY": _macro({1: 20.0}),
    }
    mf = build_macro_frame(macros).set_index("datetime")

    # Minute 0: pre-VIXY inception → still NaN (correct — nothing to ffill from).
    assert pd.isna(mf.loc[_ts(0), "vix_level"])
    # Minute 1: VIXY's first bar, value attached.
    assert mf.loc[_ts(1), "vix_level"] == 20.0
    # Minute 2: VIXY didn't print but ffill carries forward — was the bug.
    assert mf.loc[_ts(2), "vix_level"] == 20.0

    # SPY logret unaffected by the misalignment.
    assert pd.isna(mf.loc[_ts(0), "spy_logret_1"])
    assert np.isclose(mf.loc[_ts(1), "spy_logret_1"], np.log(101.0 / 100.0))
    assert np.isclose(mf.loc[_ts(2), "spy_logret_1"], np.log(102.0 / 101.0))


def test_build_macro_frame_aligned_grid():
    """When every macro has the same timestamps, ffill is a no-op."""
    macros = {
        "SPY": _macro({0: 100.0, 1: 101.0, 2: 102.0}),
        "VIXY": _macro({0: 20.0, 1: 21.0, 2: 22.0}),
    }
    mf = build_macro_frame(macros).set_index("datetime")

    # vix_level matches VIXY's close exactly at every minute.
    assert mf.loc[_ts(0), "vix_level"] == 20.0
    assert mf.loc[_ts(1), "vix_level"] == 21.0
    assert mf.loc[_ts(2), "vix_level"] == 22.0


def test_attach_macro_backward_merge():
    """Equity bar picks up the most recent macro at-or-before its datetime."""
    macros = {
        "SPY": _macro({1: 100.0, 2: 101.0, 3: 102.0}),
    }
    mf = build_macro_frame(macros)

    # One bar before any macro, one bar after the last macro.
    bars = pd.DataFrame(
        {
            "datetime": [_ts(0), _ts(5)],
            "open": [50.0, 60.0],
            "high": [51.0, 61.0],
            "low": [49.0, 59.0],
            "close": [50.5, 60.5],
            "close_adj": [50.5, 60.5],
            "volume": [100, 200],
            "symbol": "TEST",
            "timeframe": "1min",
        }
    )
    out = attach_macro(bars, mf, sector_etf=None, macros=macros).set_index("datetime")

    # Bar at minute 0: nothing prior in macro_frame → NaN.
    assert pd.isna(out.loc[_ts(0), "spy_logret_1"])
    # Bar at minute 5: backward merge picks up minute-3 logret = log(102/101).
    assert np.isclose(out.loc[_ts(5), "spy_logret_1"], np.log(102.0 / 101.0))


def test_attach_macro_sector_etf():
    """sector_logret_1 is populated when sector_etf is in macros, NaN otherwise."""
    macros = {
        "SPY": _macro({0: 100.0, 1: 101.0, 2: 102.0}),
        "XLK": _macro({0: 50.0, 1: 51.0, 2: 52.0}),
    }
    mf = build_macro_frame(macros)
    bars = pd.DataFrame(
        {
            "datetime": [_ts(2)],
            "open": [10.0], "high": [10.0], "low": [10.0],
            "close": [10.0], "close_adj": [10.0], "volume": [1],
            "symbol": "TEST", "timeframe": "1min",
        }
    )

    with_sector = attach_macro(bars, mf, sector_etf="XLK", macros=macros)
    assert np.isclose(
        with_sector["sector_logret_1"].iloc[0], np.log(52.0 / 51.0)
    )

    without_sector = attach_macro(bars, mf, sector_etf=None, macros=macros)
    assert "sector_logret_1" in without_sector.columns
    assert without_sector["sector_logret_1"].isna().all()
