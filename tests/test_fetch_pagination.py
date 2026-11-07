"""Pagination in fetch_symbol_history at the production page size.

A date-only ``end_date`` excludes that date. Evidence: the Aug 20 2026 XOM
1min full backfill logged 122 page boundaries; a regular-session NYSE
calendar reproduces 120 of them only under that reading (2 under an
inclusive reading). The published XOM 1min bars for every one of those
boundary days in the test split start at ~10:40 instead of 09:30: 12
sessions plus 320 bars fill a 5000-row page, and the next request, ending
on the boundary day, skipped the 70 bars before it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tdwm import fetch as tdfetch
from tdwm.client import TDClient

NY = "America/New_York"
PAGE = 5000


class _TS:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def as_pandas(self) -> pd.DataFrame:
        return self._df.copy()


class _Provider:
    """/time_series over naive NY wall-clock bars: date-only start inclusive,
    date-only end exclusive, newest-first, at most ``outputsize`` rows."""

    def __init__(self, wall: pd.DatetimeIndex) -> None:
        pos = np.arange(len(wall), dtype=float)
        self.bars = pd.DataFrame(
            {"open": pos, "high": pos + 0.5, "low": pos - 0.5, "close": pos, "volume": 1.0},
            index=pd.DatetimeIndex(wall, name="datetime"),
        )
        self.calls: list[tuple[str, str]] = []

    def time_series(self, *, start_date, end_date, outputsize, **_kw):
        self.calls.append((start_date, end_date))
        idx = self.bars.index
        keep = (idx >= pd.Timestamp(start_date)) & (idx < pd.Timestamp(end_date))
        page = self.bars[keep].iloc[::-1].iloc[:outputsize]
        if page.empty:
            raise RuntimeError("No data is available on the specified dates")
        return _TS(page)


def _sessions(first: str, last: str, freq: str) -> pd.DatetimeIndex:
    """Weekday regular sessions, 09:30-16:00 NY, bar start labels."""
    out = []
    for day in pd.bdate_range(first, last):
        out.append(pd.date_range(day + pd.Timedelta("9h30min"),
                                 day + pd.Timedelta("15h59min"), freq=freq))
    return out[0].append(out[1:])


def _history(provider: _Provider, interval: str, start: str, end: str) -> pd.DataFrame:
    client = TDClient(sdk=provider, max_retries=1, retry_base_sleep=0)
    tf = tdfetch.TimeframeCfg(interval=interval, enabled=True, history_years=100,
                              trajectory_windows=[])
    return tdfetch.fetch_symbol_history(client, "SYM", tf, start=start, end=end)


def _assert_complete(out: pd.DataFrame, provider: _Provider, start: str) -> None:
    expected = provider.bars[provider.bars.index >= pd.Timestamp(start)]
    assert out["datetime"].dt.tz_localize(None).tolist() == list(expected.index)
    assert out["close"].tolist() == expected["close"].tolist()


def test_regular_session_minute_backfill_keeps_page_boundary_days():
    # ~90 sessions: several 5000-row pages, each ending mid-session.
    p = _Provider(_sessions("2026-03-02", "2026-07-03", "1min"))
    out = _history(p, "1min", "2026-03-02", "2026-07-04")
    _assert_complete(out, p, "2026-03-02")
    assert len(p.calls) > 3


def test_regular_session_hourly_backfill_keeps_page_boundary_days():
    p = _Provider(_sessions("2020-01-01", "2026-06-30", "1h"))
    out = _history(p, "1h", "2020-01-01", "2026-07-01")
    _assert_complete(out, p, "2020-01-01")
    assert len(p.calls) > 2


def test_round_the_clock_minute_catch_up_keeps_page_boundary_days():
    # 24h instrument, two weeks of minutes (~20k rows); no DST transition.
    wall = pd.date_range("2026-10-05", "2026-10-18 23:59", freq="1min")
    p = _Provider(wall)
    out = _history(p, "1min", "2026-10-05", "2026-10-19")
    _assert_complete(out, p, "2026-10-05")


def test_daily_backfill_across_pages():
    p = _Provider(pd.bdate_range("1990-01-01", "2026-11-06"))
    out = _history(p, "1day", "1990-01-01", "2026-11-07")
    _assert_complete(out, p, "1990-01-01")


def test_bar_at_window_start_is_kept():
    # One bar more than a page: the first page's earliest bar sits one
    # interval after the window start.
    wall = pd.date_range("2026-01-05", periods=PAGE + 1, freq="1min")
    p = _Provider(wall)
    end = str((wall[-1].normalize() + pd.Timedelta(days=1)).date())
    out = _history(p, "1min", "2026-01-05", end)
    _assert_complete(out, p, "2026-01-05")


def test_follow_up_pages_end_the_day_after_the_earliest_bar():
    p = _Provider(_sessions("2026-03-02", "2026-05-29", "1min"))
    _history(p, "1min", "2026-03-02", "2026-05-30")
    first, *rest = p.calls
    assert first == ("2026-03-02", "2026-05-30")
    assert rest
    for (_, prev_end), (start, end) in zip(p.calls, rest):
        assert start == "2026-03-02"
        assert end < prev_end


@pytest.mark.parametrize("sessions_after_start", [10, 12, 13, 14, 20])
def test_incremental_catch_up_after_missed_runs(sessions_after_start):
    # update_daily re-fetches from last_known - 2 days; after missed runs the
    # window can exceed one page.
    days = pd.bdate_range("2026-10-28", periods=sessions_after_start)
    p = _Provider(_sessions(str(days[0].date()), str(days[-1].date()), "1min"))
    end = str((days[-1] + pd.Timedelta(days=1)).date())
    out = _history(p, "1min", "2026-10-28", end)
    _assert_complete(out, p, "2026-10-28")


def test_stalled_pagination_raises_instead_of_looping():
    # More bars in one day than fit in a page: no request can step back.
    wall = pd.date_range("2026-10-05", periods=3 * PAGE, freq="5s")
    p = _Provider(wall)
    with pytest.raises(RuntimeError, match="pagination stalled"):
        _history(p, "1min", "2026-10-05", "2026-10-06")
