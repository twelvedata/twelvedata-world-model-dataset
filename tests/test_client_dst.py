"""DST fall-back handling for naive wall-clock bars from Twelve Data.

Regression for the daily-update failure of Nov 2026:
    Cannot infer dst time from 2026-11-01 01:00:00, try using the
    'ambiguous' argument
Near-24h instruments (XAU/USD, XBR/USD) have bars during the repeated
01:00 hour, which TD returns as naive wall-clock strings. The SDK keeps the
API's newest-first row order.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tdwm import fetch as tdfetch
from tdwm.client import AmbiguousTimestampError, FetchRequest, TDClient

NY = "America/New_York"


def _req(interval: str = "1h", symbol: str = "XAU/USD") -> FetchRequest:
    return FetchRequest(symbol=symbol, interval=interval, timezone=NY)


def _td_frame(instants: pd.DatetimeIndex, *, descending: bool = True) -> pd.DataFrame:
    """Frame shaped like SDK ``as_pandas()``: naive NY wall-clock index.

    ``close`` is the bar's position in true time so tests can tell whether
    each bar kept its own instant.
    """
    wall = instants.tz_convert(NY).tz_localize(None)
    n = len(instants)
    df = pd.DataFrame(
        {
            "open": [float(i) for i in range(n)],
            "high": [float(i) + 0.5 for i in range(n)],
            "low": [float(i) - 0.5 for i in range(n)],
            "close": [float(i) for i in range(n)],
            "volume": [100.0] * n,
        },
        index=pd.DatetimeIndex(wall, name="datetime"),
    )
    return df.iloc[::-1] if descending else df


def _utc_range(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, tz="UTC")


class _FakeSDK:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.calls = 0

    def time_series(self, **_kwargs):
        self.calls += 1
        df = self.df

        class _TS:
            @staticmethod
            def as_pandas():
                return df.copy()

        return _TS()


def _assert_exact(out: pd.DataFrame, instants: pd.DatetimeIndex) -> None:
    assert str(out["datetime"].dt.tz) == NY
    assert out["datetime"].dt.tz_convert("UTC").tolist() == list(instants)
    # Every bar kept its own data: close encodes the true-time position.
    assert out["close"].tolist() == [float(i) for i in range(len(instants))]


def test_hourly_fall_back_newest_first_resolves_both_occurrences():
    # 2026-11-01 00:00 EDT .. 03:00 EST; wall time 01:00 occurs twice.
    instants = _utc_range("2026-11-01 03:00", "2026-11-01 09:00", "1h")
    out = TDClient._normalize(_td_frame(instants), _req())

    _assert_exact(out, instants)
    one_am = out[out["datetime"].dt.hour == 1]["datetime"]
    assert [t.utcoffset().total_seconds() / 3600 for t in one_am] == [-4.0, -5.0]
    assert len(out) == len(instants)


def test_hourly_fall_back_oldest_first_resolves_both_occurrences():
    instants = _utc_range("2026-11-01 03:00", "2026-11-01 09:00", "1h")
    out = TDClient._normalize(_td_frame(instants, descending=False), _req())
    _assert_exact(out, instants)


def test_one_minute_fall_back_overlap_keeps_every_bar():
    # Wall time jumps back 59 minutes inside the overlap for 1min bars.
    instants = _utc_range("2026-11-01 04:30", "2026-11-01 07:30", "1min")
    out = TDClient._normalize(_td_frame(instants), _req("1min"))
    _assert_exact(out, instants)
    assert len(out) == 181


def test_window_cut_inside_repeated_hour_is_rejected_not_guessed():
    # Only the tail of the EDT hour and the head of the EST hour: wall times
    # 01:00-01:10 appear once, so their offset cannot be established.
    instants = _utc_range("2026-11-01 05:50", "2026-11-01 06:10", "1min")
    with pytest.raises(AmbiguousTimestampError):
        TDClient._normalize(_td_frame(instants), _req("1min"))


def test_multiple_fall_back_transitions_in_one_frame():
    a = _utc_range("2025-11-02 03:00", "2025-11-02 09:00", "1h")
    b = _utc_range("2026-11-01 03:00", "2026-11-01 09:00", "1h")
    instants = a.append(b)
    out = TDClient._normalize(_td_frame(instants), _req())
    _assert_exact(out, instants)


def test_unambiguous_naive_input_matches_plain_localize():
    # Spring-forward day: 02:00 does not exist and is simply absent.
    instants = _utc_range("2026-03-08 04:00", "2026-03-08 10:00", "1h")
    frame = _td_frame(instants)
    out = TDClient._normalize(frame, _req())

    expected = (
        pd.to_datetime(frame.index.to_series().reset_index(drop=True))
        .dt.tz_localize(NY)
        .sort_values()
        .reset_index(drop=True)
    )
    assert out["datetime"].tolist() == expected.tolist()
    _assert_exact(out, instants)


def test_daily_bars_unchanged_on_transition_day():
    # 1day rows are naive midnights; midnight is never ambiguous in NY.
    idx = pd.DatetimeIndex(["2026-11-02", "2026-11-01", "2026-10-30"], name="datetime")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx
    )
    out = TDClient._normalize(frame, _req("1day"))
    assert out["datetime"].dt.strftime("%Y-%m-%d %H:%M").tolist() == [
        "2026-10-30 00:00", "2026-11-01 00:00", "2026-11-02 00:00",
    ]
    assert out["datetime"].is_monotonic_increasing


def test_single_occurrence_of_repeated_hour_is_rejected_not_guessed():
    # Only one 01:00 row: EDT vs EST cannot be told apart.
    idx = pd.DatetimeIndex(
        ["2026-11-01 02:00", "2026-11-01 01:00", "2026-11-01 00:00"], name="datetime"
    )
    frame = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx
    )
    with pytest.raises(AmbiguousTimestampError) as err:
        TDClient._normalize(frame, _req())
    assert "XAU/USD 1h" in str(err.value)
    assert "2026-11-01 01:00:00" in str(err.value)


def test_unordered_rows_around_repeated_hour_are_rejected():
    instants = _utc_range("2026-11-01 03:00", "2026-11-01 09:00", "1h")
    frame = _td_frame(instants)
    scrambled = frame.iloc[[3, 0, 5, 1, 6, 2, 4]]
    with pytest.raises(AmbiguousTimestampError):
        TDClient._normalize(scrambled, _req())


def test_nonexistent_spring_forward_time_still_raises():
    idx = pd.DatetimeIndex(["2026-03-08 02:30"], name="datetime")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx
    )
    with pytest.raises(Exception) as err:
        TDClient._normalize(frame, _req())
    assert not isinstance(err.value, AmbiguousTimestampError)


def test_fetch_bars_resolves_fall_back_through_the_client():
    instants = _utc_range("2026-10-31 22:00", "2026-11-01 12:00", "1h")
    sdk = _FakeSDK(_td_frame(instants))
    client = TDClient(sdk=sdk, max_retries=8, retry_base_sleep=0)
    out = client.fetch_bars(_req())
    _assert_exact(out, instants)
    assert sdk.calls == 1


def _flat_frame(stamps: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0},
        index=pd.DatetimeIndex(stamps, name="datetime"),
    )


def test_unresolvable_ambiguity_is_not_retried():
    sdk = _FakeSDK(_flat_frame(["2026-11-01 01:00", "2026-11-01 00:00"]))
    client = TDClient(sdk=sdk, max_retries=8, retry_base_sleep=0)
    with pytest.raises(AmbiguousTimestampError):
        client.fetch_bars(_req())
    assert sdk.calls == 1


def test_other_normalization_errors_keep_existing_retry_behavior():
    # Nonexistent spring-forward time: unchanged from before this fix.
    sdk = _FakeSDK(_flat_frame(["2026-03-08 02:30", "2026-03-08 01:30"]))
    client = TDClient(sdk=sdk, max_retries=3, retry_base_sleep=0)
    with pytest.raises(RuntimeError, match="failed after 3 retries"):
        client.fetch_bars(_req())
    assert sdk.calls == 3


@pytest.mark.parametrize("seed", range(20))
def test_illiquid_one_minute_overlap_resolves_exactly(seed):
    # Random 40% of minutes missing in either pass of the repeated hour.
    full = _utc_range("2026-11-01 04:30", "2026-11-01 07:30", "1min")
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(full), size=int(len(full) * 0.6), replace=False))
    instants = full[keep]
    out = TDClient._normalize(_td_frame(instants), _req("1min"))
    _assert_exact(out, instants)


@pytest.mark.parametrize(
    ("first", "last", "descending"),
    [
        # No bars in the EDT pass of 01:xx: each 01:xx wall time occurs once.
        ("2026-11-01 06:00", "2026-11-01 07:30", True),
        # Frame lies entirely inside the overlap and the passes do not share
        # a wall time, so row order cannot establish the offsets.
        ("2026-11-01 05:30", "2026-11-01 06:10", True),
        ("2026-11-01 05:30", "2026-11-01 06:10", False),
    ],
)
def test_overlap_without_repeated_wall_times_is_rejected(first, last, descending):
    instants = _utc_range(first, last, "1min")
    with pytest.raises(AmbiguousTimestampError):
        TDClient._normalize(_td_frame(instants, descending=descending), _req("1min"))


class _PagedSDK:
    """Date-only start/end as NY wall dates (end inclusive), newest rows
    first, at most ``outputsize`` rows per call."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.calls = 0

    def time_series(self, *, start_date, end_date, outputsize, **_kwargs):
        self.calls += 1
        lo = pd.Timestamp(start_date)
        hi = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        f = self.frame
        page = f[(f.index >= lo) & (f.index < hi)].iloc[:outputsize]

        class _TS:
            @staticmethod
            def as_pandas():
                return page.copy()

        return _TS()


def _paged_history(monkeypatch, instants, outputsize, start, end):
    real = tdfetch.FetchRequest
    monkeypatch.setattr(
        tdfetch, "FetchRequest", lambda **kw: real(**{**kw, "outputsize": outputsize})
    )
    sdk = _PagedSDK(_td_frame(instants))
    client = TDClient(sdk=sdk, max_retries=1, retry_base_sleep=0)
    tf = tdfetch.TimeframeCfg(
        interval="1h", enabled=True, history_years=100, trajectory_windows=[]
    )
    return tdfetch.fetch_symbol_history(client, "XAU/USD", tf, start=start, end=end), sdk


def test_paginated_history_across_fall_back_is_exact(monkeypatch):
    instants = _utc_range("2026-10-29 04:00", "2026-11-04 04:00", "1h")
    out, sdk = _paged_history(monkeypatch, instants, 60, "2026-10-29", "2026-11-04")
    assert sdk.calls > 1
    assert out["datetime"].dt.tz_convert("UTC").tolist() == list(instants)
    assert out["close"].tolist() == [float(i) for i in range(len(instants))]


def test_page_boundary_inside_repeated_hour_fails_closed(monkeypatch):
    # 47 newest rows end between the two 01:00 bars.
    instants = _utc_range("2026-10-29 04:00", "2026-11-04 04:00", "1h")
    with pytest.raises(AmbiguousTimestampError):
        _paged_history(monkeypatch, instants, 47, "2026-10-29", "2026-11-04")


def test_fetch_symbol_history_keeps_both_fall_back_bars():
    """The production path: XAU/USD 1h over a window spanning 2026-11-01."""
    instants = _utc_range("2026-10-29 00:00", "2026-11-04 00:00", "1h")
    sdk = _FakeSDK(_td_frame(instants))
    client = TDClient(sdk=sdk, max_retries=1, retry_base_sleep=0)
    tf = tdfetch.TimeframeCfg(
        interval="1h", enabled=True, history_years=100, trajectory_windows=[]
    )
    out = tdfetch.fetch_symbol_history(
        client, "XAU/USD", tf, start="2026-10-27", end="2026-11-05"
    )
    assert len(out) == len(instants)
    assert out["datetime"].is_unique
    assert out["datetime"].dt.tz_convert("UTC").tolist() == list(instants)
    assert out["close"].tolist() == [float(i) for i in range(len(instants))]
    repeated = out["datetime"].dt.tz_localize(None) == pd.Timestamp("2026-11-01 01:00")
    assert int(repeated.sum()) == 2
