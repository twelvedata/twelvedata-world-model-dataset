"""DST fall-back handling for naive wall-clock bars from Twelve Data.

Regression for the daily-update failure of Nov 2026:
    Cannot infer dst time from 2026-11-01 01:00:00, try using the
    'ambiguous' argument
Requests use timezone=America/New_York, so TD labels bars with naive NY
wall-clock strings; during the fall-back hour those labels repeat. XAU/USD
returns both passes, XBR/USD returns sparse bars with a lone 01:00 label.

``fixtures/td_live_dst`` holds 12 recorded live responses (sanitized, no
credentials): six NY-labelled requests, each paired with a UTC-labelled
request over the same instants. The UTC responses are the independent
oracle. They were recorded with order=asc; the SDK default is desc.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tdwm import fetch as tdfetch
from tdwm.client import AmbiguousTimestampError, FetchRequest, TDClient

NY = "America/New_York"
FIXTURES = Path(__file__).parent / "fixtures" / "td_live_dst"
OHLC = ["open", "high", "low", "close"]


def _req(interval: str = "1h", symbol: str = "XAU/USD") -> FetchRequest:
    return FetchRequest(symbol=symbol, interval=interval, timezone=NY)


def _utc_range(start: str, end: str, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq=freq, tz="UTC")


class _TS:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def as_pandas(self) -> pd.DataFrame:
        return self._df.copy()


# --------------------------------------------------------------------------
# Recorded live responses
# --------------------------------------------------------------------------

LIVE_PAIRS = [
    ("01_fall_XAU-USD_1h_ny", "02_fall_XAU-USD_1h_utc", True),
    ("03_fall_XAU-USD_1min_ny", "04_fall_XAU-USD_1min_utc", True),
    ("05_fall_XBR-USD_1h_ny", "06_fall_XBR-USD_1h_utc", True),
    ("07_fall_XBR-USD_1min_ny", "08_fall_XBR-USD_1min_utc", True),
    ("09_ordinary_XAU-USD_1h_ny", "10_ordinary_XAU-USD_1h_utc", False),
    ("11_spring_XAU-USD_1h_ny", "12_spring_XAU-USD_1h_utc", False),
]


def _load(name: str) -> tuple[dict, pd.DataFrame]:
    """(request parameters, frame shaped like SDK ``as_pandas()``)."""
    doc = json.loads((FIXTURES / f"{name}.json").read_text())
    assert doc["http_status"] == 200 and doc["response"]["status"] == "ok"
    df = pd.DataFrame(doc["response"]["values"])
    df["datetime"] = pd.to_datetime(df["datetime"])
    return doc["parameters"], df.set_index("datetime").astype(float)


class _FixtureSDK:
    """Serves a recorded NY response, and UTC bars from the paired recorded
    UTC response within inclusive explicit bounds."""

    def __init__(self, ny: pd.DataFrame, utc: pd.DataFrame, utc_params: dict, *, desc: bool):
        self.ny, self.utc, self.utc_params, self.desc = ny, utc, utc_params, desc
        self.calls: list[dict] = []

    def time_series(self, **kw):
        self.calls.append(kw)
        if kw["timezone"] == "UTC":
            lo, hi = pd.Timestamp(kw["start_date"]), pd.Timestamp(kw["end_date"])
            # Only answer inside the window that was actually recorded.
            assert pd.Timestamp(self.utc_params["start_date"]) <= lo <= hi
            assert hi <= pd.Timestamp(self.utc_params["end_date"])
            df = self.utc[(self.utc.index >= lo) & (self.utc.index <= hi)]
        else:
            df = self.ny
        return _TS(df.iloc[::-1] if self.desc else df)


@pytest.mark.parametrize("desc", [True, False], ids=["desc", "asc"])
@pytest.mark.parametrize(("ny_name", "utc_name", "has_fold"), LIVE_PAIRS)
def test_live_fixture_matches_independent_utc_response(ny_name, utc_name, has_fold, desc):
    params, ny = _load(ny_name)
    utc_params, utc = _load(utc_name)
    sdk = _FixtureSDK(ny, utc, utc_params, desc=desc)
    client = TDClient(sdk=sdk, max_retries=1, retry_base_sleep=0)
    out = client.fetch_bars(
        FetchRequest(
            symbol=params["symbol"],
            interval=params["interval"],
            start=params["start_date"],
            end=params["end_date"],
        )
    )

    assert str(out["datetime"].dt.tz) == NY
    assert len(out) == len(utc) == len(ny)
    assert out["datetime"].dt.tz_convert("UTC").tolist() == list(
        utc.index.tz_localize("UTC")
    )
    np.testing.assert_array_equal(out[OHLC].to_numpy(), utc[OHLC].to_numpy())
    assert out["datetime"].is_unique and out["datetime"].is_monotonic_increasing

    primary, *extra = sdk.calls
    assert primary["timezone"] == NY
    assert primary["start_date"] == params["start_date"]
    assert primary["end_date"] == params["end_date"]
    if has_fold:
        assert [c["timezone"] for c in extra] == ["UTC"]
    else:
        assert extra == []


def test_live_xbr_lone_one_am_bar_is_the_edt_pass():
    _, ny = _load("05_fall_XBR-USD_1h_ny")
    utc_params, utc = _load("06_fall_XBR-USD_1h_utc")
    client = TDClient(sdk=_FixtureSDK(ny, utc, utc_params, desc=True), max_retries=1)
    out = client.fetch_bars(_req("1h", "XBR/USD"))
    assert [str(t) for t in out["datetime"]] == [
        "2026-11-01 01:00:00-04:00",
        "2026-11-01 02:00:00-05:00",
    ]


def test_live_xau_minute_overlap_keeps_both_passes():
    _, ny = _load("03_fall_XAU-USD_1min_ny")
    utc_params, utc = _load("04_fall_XAU-USD_1min_utc")
    client = TDClient(sdk=_FixtureSDK(ny, utc, utc_params, desc=True), max_retries=1)
    out = client.fetch_bars(_req("1min"))
    wall = out["datetime"].dt.tz_localize(None)
    repeated = wall[(wall >= "2026-11-01 01:00") & (wall < "2026-11-01 02:00")]
    assert repeated.value_counts().eq(2).all() and len(repeated) == 120
    offsets = out.loc[repeated.index, "datetime"].map(lambda t: t.utcoffset())
    assert (offsets == pd.Timedelta(hours=-4)).sum() == 60


def test_live_fall_back_without_utc_reference_is_rejected():
    _, ny = _load("05_fall_XBR-USD_1h_ny")
    with pytest.raises(AmbiguousTimestampError, match="XBR/USD 1h"):
        TDClient._normalize(ny, _req("1h", "XBR/USD"))


# --------------------------------------------------------------------------
# Simulated provider
# --------------------------------------------------------------------------


class _Provider:
    """Simulated /time_series over one true-time series.

    Bounds and labels are in the requested timezone. Explicit-time bounds
    are inclusive (matches the recorded live responses); a date-only end
    excludes that date (matches production page logs). Rows newest-first,
    at most ``outputsize``.
    """

    def __init__(self, instants: pd.DatetimeIndex, values: pd.DataFrame | None = None):
        n = len(instants)
        if values is None:
            # close encodes true-time position so misplacement is visible.
            pos = np.arange(n, dtype=float)
            values = pd.DataFrame(
                {"open": pos, "high": pos + 0.5, "low": pos - 0.5, "close": pos,
                 "volume": 100.0}
            )
        self.bars = values.set_index(instants)
        self.calls: list[dict] = []
        self.utc_override: pd.DataFrame | Exception | None = None

    def time_series(self, *, start_date, end_date, outputsize, timezone, **kw):
        self.calls.append({"timezone": timezone, "start_date": start_date,
                           "end_date": end_date})
        if timezone == "UTC" and self.utc_override is not None:
            if isinstance(self.utc_override, Exception):
                raise self.utc_override
            bars = self.utc_override
        else:
            bars = self.bars
        wall = bars.index.tz_convert(timezone).tz_localize(None)
        keep = np.ones(len(bars), dtype=bool)
        if start_date is not None:
            keep &= wall >= pd.Timestamp(start_date)
        if end_date is not None:
            hi = pd.Timestamp(end_date)
            keep &= wall < hi if len(end_date) == 10 else wall <= hi
        df = bars[keep].set_axis(pd.DatetimeIndex(wall[keep], name="datetime"))
        return _TS(df.iloc[::-1].iloc[:outputsize])

    def expected_close(self) -> list[float]:
        return self.bars["close"].tolist()


def _fetch(provider: _Provider, interval: str = "1h", **client_kw) -> pd.DataFrame:
    client = TDClient(sdk=provider, max_retries=client_kw.pop("max_retries", 1),
                      retry_base_sleep=0, **client_kw)
    return client.fetch_bars(_req(interval))


def _assert_exact(out: pd.DataFrame, provider: _Provider) -> None:
    assert str(out["datetime"].dt.tz) == NY
    assert out["datetime"].dt.tz_convert("UTC").tolist() == list(provider.bars.index)
    assert out["close"].tolist() == provider.expected_close()
    assert out["open"].tolist() == provider.bars["open"].tolist()


def test_hourly_fall_back_resolves_both_occurrences():
    p = _Provider(_utc_range("2026-11-01 03:00", "2026-11-01 09:00", "1h"))
    out = _fetch(p)
    _assert_exact(out, p)
    one_am = out[out["datetime"].dt.hour == 1]["datetime"]
    assert [t.utcoffset().total_seconds() / 3600 for t in one_am] == [-4.0, -5.0]
    assert p.calls[1] == {"timezone": "UTC", "start_date": "2026-11-01T05:00:00",
                          "end_date": "2026-11-01T06:00:00"}


@pytest.mark.parametrize(
    ("first", "last"),
    [
        ("2026-11-01 06:00", "2026-11-01 07:30"),  # EST pass only
        ("2026-11-01 04:30", "2026-11-01 05:59"),  # EDT pass only
        ("2026-11-01 05:50", "2026-11-01 06:10"),  # straddles the transition
    ],
)
def test_partial_overlap_resolves_from_utc_reference(first, last):
    p = _Provider(_utc_range(first, last, "1min"))
    _assert_exact(_fetch(p, "1min"), p)


@pytest.mark.parametrize("seed", range(10))
def test_illiquid_one_minute_overlap_resolves_exactly(seed):
    full = _utc_range("2026-11-01 04:30", "2026-11-01 07:30", "1min")
    rng = np.random.default_rng(seed)
    keep = np.sort(rng.choice(len(full), size=int(len(full) * 0.2), replace=False))
    p = _Provider(full[keep])
    _assert_exact(_fetch(p, "1min"), p)


def test_value_identical_passes_are_both_kept():
    # Flat market: both 01:00 bars carry the same values.
    instants = _utc_range("2026-11-01 04:00", "2026-11-01 07:00", "1h")
    values = pd.DataFrame({"open": [1.0, 2.0, 2.0, 3.0], "high": [1.0, 2.0, 2.0, 3.0],
                           "low": [1.0, 2.0, 2.0, 3.0], "close": [1.0, 2.0, 2.0, 3.0]})
    p = _Provider(instants, values)
    out = _fetch(p)
    assert out["datetime"].dt.tz_convert("UTC").tolist() == list(instants)
    assert out["close"].tolist() == [1.0, 2.0, 2.0, 3.0]


def test_two_transitions_use_one_bounded_lookup_each():
    instants = _utc_range("2025-11-02 03:00", "2025-11-02 09:00", "1h").append(
        _utc_range("2026-11-01 03:00", "2026-11-01 09:00", "1h")
    )
    p = _Provider(instants)
    _assert_exact(_fetch(p), p)
    assert [(c["start_date"], c["end_date"]) for c in p.calls[1:]] == [
        ("2025-11-02T05:00:00", "2025-11-02T06:00:00"),
        ("2026-11-01T05:00:00", "2026-11-01T06:00:00"),
    ]


# ---- fail closed ----------------------------------------------------------


def _lone_one_am_provider() -> _Provider:
    return _Provider(_utc_range("2026-11-01 04:00", "2026-11-01 07:00", "1h")[[0, 1, 3]])


def test_ny_bar_missing_from_utc_reference_is_rejected():
    # UTC has only the other (EST) 01:00 pass, with different values.
    p = _lone_one_am_provider()
    other = p.bars.iloc[[1]].copy()
    other.index = other.index + pd.Timedelta(hours=1)
    other["close"] += 1.0
    p.utc_override = pd.concat([p.bars.iloc[[0]], other, p.bars.iloc[[2]]])
    with pytest.raises(AmbiguousTimestampError, match="match 0 UTC bar"):
        _fetch(p)


def test_empty_utc_reference_is_rejected():
    p = _lone_one_am_provider()
    p.utc_override = p.bars.iloc[[0, 2]]
    with pytest.raises(AmbiguousTimestampError, match="returned no bars"):
        _fetch(p)


def test_value_mismatch_with_utc_reference_is_rejected():
    p = _lone_one_am_provider()
    revised = p.bars.copy()
    revised.loc[revised.index[1], "close"] += 0.01
    p.utc_override = revised
    with pytest.raises(AmbiguousTimestampError):
        _fetch(p)


def test_indistinguishable_passes_with_one_ny_bar_are_rejected():
    # NY has one 01:00 bar, UTC has two with identical values: either could be it.
    instants = _utc_range("2026-11-01 04:00", "2026-11-01 07:00", "1h")
    flat = pd.DataFrame({c: [1.0, 2.0, 2.0, 3.0] for c in OHLC})
    p = _Provider(instants[[0, 1, 3]], flat.iloc[[0, 1, 3]].reset_index(drop=True))
    p.utc_override = flat.set_index(instants)
    with pytest.raises(AmbiguousTimestampError, match="match 2 UTC bar"):
        _fetch(p)


def test_utc_reference_without_data_is_rejected_not_read_as_end_of_history():
    p = _Provider(_utc_range("2026-10-29 04:00", "2026-11-04 04:00", "1h"))
    p.utc_override = RuntimeError("No data is available on the specified dates")
    client = TDClient(sdk=p, max_retries=3, retry_base_sleep=0)
    tf = tdfetch.TimeframeCfg(interval="1h", enabled=True, history_years=100,
                              trajectory_windows=[])
    with pytest.raises(AmbiguousTimestampError, match="returned no bars"):
        tdfetch.fetch_symbol_history(client, "XAU/USD", tf, start="2026-10-29",
                                     end="2026-11-04")
    assert len(p.calls) == 2


# ---- retries --------------------------------------------------------------


def test_unresolvable_ambiguity_is_not_retried():
    p = _lone_one_am_provider()
    p.utc_override = p.bars.iloc[[0, 2]]
    with pytest.raises(AmbiguousTimestampError):
        _fetch(p, max_retries=8)
    assert len(p.calls) == 2


def test_transient_utc_reference_error_retries_whole_fetch():
    p = _Provider(_utc_range("2026-11-01 03:00", "2026-11-01 09:00", "1h"))
    real = p.time_series
    failed = []

    def flaky(**kw):
        if kw["timezone"] == "UTC" and not failed:
            failed.append(True)
            raise RuntimeError('{"code": 500, "message": "internal error"}')
        return real(**kw)

    p.time_series = flaky
    _assert_exact(_fetch(p, max_retries=3), p)
    # The failed UTC attempt raises before it is logged.
    assert failed and [c["timezone"] for c in p.calls] == [NY, NY, "UTC"]


def test_other_normalization_errors_keep_existing_retry_behavior():
    # Nonexistent spring-forward time: unchanged from before this fix.
    idx = pd.DatetimeIndex(["2026-03-08 02:30", "2026-03-08 01:30"], name="datetime")
    frame = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5,
                          "volume": 10.0}, index=idx)

    class SDK:
        calls = 0

        def time_series(self, **_kw):
            SDK.calls += 1
            return _TS(frame)

    client = TDClient(sdk=SDK(), max_retries=3, retry_base_sleep=0)
    with pytest.raises(RuntimeError, match="failed after 3 retries"):
        client.fetch_bars(_req())
    assert SDK.calls == 3


# ---- ordinary input unchanged ----------------------------------------------


@pytest.mark.parametrize(
    ("first", "last", "freq"),
    [
        ("2026-03-08 04:00", "2026-03-08 10:00", "1h"),    # spring forward
        ("2026-06-01 00:00", "2026-06-07 23:00", "1h"),    # ordinary 24h week
        ("2026-11-02 14:30", "2026-11-02 20:59", "1min"),  # session after fall-back
    ],
)
def test_unambiguous_input_matches_plain_localize_with_one_request(first, last, freq):
    p = _Provider(_utc_range(first, last, freq))
    out = _fetch(p, "1min" if freq == "1min" else "1h")
    _assert_exact(out, p)
    wall = p.bars.index.tz_convert(NY).tz_localize(None)
    assert out["datetime"].tolist() == list(wall.tz_localize(NY))
    assert len(p.calls) == 1


def test_daily_bars_unchanged_on_transition_day():
    # 1day rows are naive midnights; midnight is never ambiguous in NY.
    idx = pd.DatetimeIndex(["2026-11-02", "2026-11-01", "2026-10-30"], name="datetime")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx
    )
    out = TDClient._normalize(frame, _req("1day"))
    assert out["datetime"].dt.strftime("%Y-%m-%d %H:%M %z").tolist() == [
        "2026-10-30 00:00 -0400", "2026-11-01 00:00 -0400", "2026-11-02 00:00 -0500",
    ]


def test_tz_aware_input_is_converted_not_relocalized():
    instants = _utc_range("2026-11-01 04:00", "2026-11-01 08:00", "1h")
    frame = pd.DataFrame({"close": np.arange(5.0)}, index=pd.DatetimeIndex(instants,
                                                                           name="datetime"))
    out = TDClient._normalize(frame.iloc[::-1], _req())
    assert out["datetime"].dt.tz_convert("UTC").tolist() == list(instants)


def test_nonexistent_spring_forward_time_still_raises():
    idx = pd.DatetimeIndex(["2026-03-08 02:30"], name="datetime")
    frame = pd.DataFrame(
        {"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 10.0}, index=idx
    )
    with pytest.raises(Exception) as err:
        TDClient._normalize(frame, _req())
    assert not isinstance(err.value, AmbiguousTimestampError)


# ---- pagination -------------------------------------------------------------


def _history(monkeypatch, provider: _Provider, interval: str, outputsize: int,
             start: str, end: str) -> pd.DataFrame:
    real = tdfetch.FetchRequest
    monkeypatch.setattr(
        tdfetch, "FetchRequest", lambda **kw: real(**{**kw, "outputsize": outputsize})
    )
    client = TDClient(sdk=provider, max_retries=1, retry_base_sleep=0)
    tf = tdfetch.TimeframeCfg(interval=interval, enabled=True, history_years=100,
                              trajectory_windows=[])
    return tdfetch.fetch_symbol_history(client, "XAU/USD", tf, start=start, end=end)


def test_hourly_history_across_fall_back_is_exact(monkeypatch):
    p = _Provider(_utc_range("2026-10-29 04:00", "2026-11-04 04:00", "1h"))
    out = _history(monkeypatch, p, "1h", 5000, "2026-10-29", "2026-11-04")
    _assert_exact(out, p)
    assert [c["timezone"] for c in p.calls] == [NY, "UTC"]


def test_minute_history_across_fall_back_is_exact(monkeypatch):
    p = _Provider(_utc_range("2026-10-30 04:00", "2026-11-02 04:00", "1min"))
    out = _history(monkeypatch, p, "1min", 5000, "2026-10-30", "2026-11-02")
    _assert_exact(out, p)
    assert [c["timezone"] for c in p.calls] == [NY, "UTC"]


def test_every_hourly_page_size_across_fall_back_is_exact(monkeypatch):
    # Page sizes larger than the 25-bar fall-back day; some put a page
    # boundary inside the repeated hour, so each page needs its own lookup.
    split_hour = 0
    for outputsize in range(26, 160):
        p = _Provider(_utc_range("2026-10-29 04:00", "2026-11-04 04:00", "1h"))
        with monkeypatch.context() as mp:
            out = _history(mp, p, "1h", outputsize, "2026-10-29", "2026-11-04")
        _assert_exact(out, p)
        split_hour += sum(c["timezone"] == "UTC" for c in p.calls) > 1
    assert split_hour


@pytest.mark.parametrize("outputsize", [1500, 2000, 2500, 3000, 4000])
def test_paginated_minute_history_across_fall_back_is_exact(monkeypatch, outputsize):
    p = _Provider(_utc_range("2026-10-30 04:00", "2026-11-02 04:00", "1min"))
    out = _history(monkeypatch, p, "1min", outputsize, "2026-10-30", "2026-11-02")
    _assert_exact(out, p)
    assert sum(c["timezone"] == NY for c in p.calls) > 1
