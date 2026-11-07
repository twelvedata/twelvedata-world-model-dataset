"""Thin wrapper around the twelvedata SDK.

Centralizes rate limiting, retries, and tz handling so the rest of the
pipeline never touches the SDK directly. Makes testing easier — tests
inject a fake client with the same surface area.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any, Iterable

import pandas as pd

# Twelve Data: retry only 429/5xx. 401 is a dead key, 403 is a plan
# restriction — backing off just burns the job clock (the Sep 2026
# outage spent ~5 min on 8 retries of the same 401).
_NON_RETRYABLE_CODES = {400, 401, 403, 404}
_CODE_RE = re.compile(r'"code"\s*:\s*(\d+)')


def api_status_code(exc: BaseException) -> int | None:
    m = _CODE_RE.search(str(exc))
    return int(m.group(1)) if m else None


class AmbiguousTimestampError(RuntimeError):
    """Naive wall-clock timestamps that cannot be mapped to one UTC instant."""


_VALUE_COLUMNS = ("open", "high", "low", "close", "volume")

# Fetches bars for an inclusive UTC instant range, normalized (tz-aware).
FoldLookup = Callable[[pd.Timestamp, pd.Timestamp], pd.DataFrame]


def _value_keys(df: pd.DataFrame, cols: list[str]) -> list[tuple]:
    return [
        tuple(None if pd.isna(v) else float(v) for v in row)
        for row in df[cols].itertuples(index=False)
    ]


def _resolve_folds(
    df: pd.DataFrame, localized: pd.Series, req: FetchRequest, lookup: FoldLookup | None
) -> pd.Series:
    """Assign UTC instants to naive wall-clock rows in a DST fall-back hour.

    Labels in the repeated hour carry no offset and a sparse series may hold
    only one of the two passes, so neither the label nor row order identifies
    the instant. The instants come from a UTC-labelled request over the
    repeated hour instead: every ambiguous row must match, by its full set of
    values, UTC bars with that same wall-clock label. Anything short of an
    unambiguous match raises rather than dropping or guessing.
    """
    amb = localized.isna() & df["datetime"].notna()
    labels = df.loc[amb, "datetime"]
    sample = ", ".join(str(t) for t in labels.drop_duplicates().head(3))

    def failure(reason: str) -> AmbiguousTimestampError:
        return AmbiguousTimestampError(
            f"{req.symbol} {req.interval}: ambiguous DST wall-clock timestamps in "
            f"{req.timezone} ({sample}): {reason}; refusing to drop or guess offsets"
        )

    if lookup is None:
        raise failure("no UTC reference available")
    # One bounded request per transition day, spanning both passes.
    refs = []
    for _, day in labels.groupby(labels.dt.date):
        first = day.min().tz_localize(req.timezone, ambiguous=True).tz_convert("UTC")
        last = day.max().tz_localize(req.timezone, ambiguous=False).tz_convert("UTC")
        ref_day = lookup(first, last)
        if not ref_day.empty:
            refs.append(ref_day)
    if not refs:
        raise failure("UTC reference returned no bars")
    ref = pd.concat(refs, ignore_index=True)
    ref_wall = ref["datetime"].dt.tz_convert(req.timezone).dt.tz_localize(None)

    cols = [c for c in _VALUE_COLUMNS if c in df.columns]
    if cols != [c for c in _VALUE_COLUMNS if c in ref.columns]:
        raise failure("UTC reference has different value columns")
    rows = df.loc[amb]
    row_key = dict(zip(rows.index, _value_keys(rows, cols)))
    ref_keys = _value_keys(ref, cols)
    resolved = localized.copy()
    for label in labels.drop_duplicates():
        by_key: dict[tuple, list[int]] = {}
        for i in range(len(ref)):
            if ref_wall.iloc[i] == label:
                by_key.setdefault(ref_keys[i], []).append(i)
        mine: dict[tuple, list] = {}
        for pos in rows.index[(rows["datetime"] == label).to_numpy()]:
            mine.setdefault(row_key[pos], []).append(pos)
        for key, positions in mine.items():
            matches = by_key.get(key, [])
            # Equal counts make value-identical bars interchangeable; any other
            # count means a bar is missing upstream or cannot be told apart.
            if len(matches) != len(positions):
                raise failure(
                    f"{len(positions)} bar(s) at {label} match {len(matches)} UTC bar(s)"
                )
            for pos, i in zip(positions, matches):
                resolved.loc[pos] = ref["datetime"].iloc[i].tz_convert(req.timezone)
    if not resolved.dropna().is_unique:
        raise failure("resolved instants are not unique")
    return resolved


# Load .env from repo root if present (no third-party dep required).
_env_file = Path(__file__).resolve().parents[2] / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())


@dataclass
class FetchRequest:
    symbol: str
    interval: str           # "1day", "1h", "5min"
    start: str | None = None
    end: str | None = None
    outputsize: int = 5000
    timezone: str = "America/New_York"


class TDClient:
    """Wrapper.

    Accepts either a real twelvedata.TDClient (passed via `sdk`) or a
    fake object exposing `time_series(...).as_pandas()` — the tests use
    the latter.
    """

    def __init__(
        self,
        api_key: str | None = None,
        sdk: Any = None,
        *,
        max_retries: int = 8,
        retry_base_sleep: float = 1.0,
    ):
        self.api_key = api_key or os.environ.get("TWELVE_DATA_API_KEY")
        self.max_retries = max_retries
        self.retry_base_sleep = retry_base_sleep
        self._sdk = sdk
        if self._sdk is None and self.api_key:
            # Import lazily so tests can run without the SDK installed.
            from twelvedata import TDClient as _TDSdk  # type: ignore
            self._sdk = _TDSdk(apikey=self.api_key)

    # --- public API ---

    def fetch_bars(self, req: FetchRequest) -> pd.DataFrame:
        if self._sdk is None:
            raise RuntimeError(
                "No API key and no injected SDK — cannot fetch live data.\n"
                "Set TWELVE_DATA_API_KEY env var or pass api_key= to TDClient.\n"
                "Get a free key at https://twelvedata.com/account/api-keys"
            )
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                ts = self._sdk.time_series(
                    symbol=req.symbol,
                    interval=req.interval,
                    start_date=req.start,
                    end_date=req.end,
                    outputsize=req.outputsize,
                    timezone=req.timezone,
                )
                df = ts.as_pandas()
                return self._normalize(
                    df, req, lambda first, last: self._utc_reference(req, first, last)
                )
            except AmbiguousTimestampError:
                # Deterministic for a given payload; retrying only burns the
                # job clock.
                raise
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                msg = str(exc)
                if "No data is available" in msg:
                    # Expected when requesting history past a symbol's
                    # inception — not retryable.
                    raise
                code = api_status_code(exc)
                if code in _NON_RETRYABLE_CODES:
                    if code == 401:
                        raise RuntimeError(
                            "TWELVE_DATA_API_KEY was rejected (401). "
                            "Update the GitHub Actions secret and re-run. "
                            f"Last error: {exc}"
                        ) from exc
                    raise RuntimeError(
                        f"fetch_bars got HTTP {code} for "
                        f"{req.symbol} {req.interval}: {exc}"
                    ) from exc
                if "run out of API credits" in msg or code == 429:
                    print(f"  [rate-limit] sleeping 60s before retry (attempt {attempt + 1})")
                    time.sleep(60)
                else:
                    time.sleep(self.retry_base_sleep * (2**attempt))
        raise RuntimeError(
            f"fetch_bars failed after {self.max_retries} retries for "
            f"{req.symbol} {req.interval}: {last_err}"
        )

    def fetch_many(self, reqs: Iterable[FetchRequest]) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for r in reqs:
            out[f"{r.symbol}:{r.interval}"] = self.fetch_bars(r)
        return out

    # --- internal ---

    def _utc_reference(
        self, req: FetchRequest, first: pd.Timestamp, last: pd.Timestamp
    ) -> pd.DataFrame:
        """Bars for ``first..last`` (UTC instants), requested with UTC labels.

        Explicit-time bounds are inclusive at both ends, and with
        ``timezone=UTC`` they and the returned labels are UTC.
        """
        ref_req = FetchRequest(
            symbol=req.symbol,
            interval=req.interval,
            start=first.strftime("%Y-%m-%dT%H:%M:%S"),
            end=last.strftime("%Y-%m-%dT%H:%M:%S"),
            outputsize=req.outputsize,
            timezone="UTC",
        )
        try:
            ts = self._sdk.time_series(
                symbol=ref_req.symbol,
                interval=ref_req.interval,
                start_date=ref_req.start,
                end_date=ref_req.end,
                outputsize=ref_req.outputsize,
                timezone=ref_req.timezone,
            )
            df = ts.as_pandas()
        except Exception as exc:
            # Must not reach fetch_symbol_history, which reads this message
            # as the end of the symbol's history.
            if "No data is available" in str(exc):
                return pd.DataFrame(columns=["datetime", *_VALUE_COLUMNS])
            raise
        return self._normalize(df, ref_req)

    @staticmethod
    def _normalize(
        df: pd.DataFrame, req: FetchRequest, fold_lookup: FoldLookup | None = None
    ) -> pd.DataFrame:
        """Return a DataFrame with a canonical shape regardless of SDK version.

        ``fold_lookup`` supplies UTC-labelled bars used to place naive
        timestamps that fall in a DST fall-back hour; without it such rows
        raise ``AmbiguousTimestampError``.
        """
        if df is None or len(df) == 0:
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
        df = df.copy()
        # Twelve Data returns a DatetimeIndex named "datetime" — move it to a column.
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None) if df.index.tz is None else df.index
            df = df.reset_index().rename(columns={"index": "datetime"})
        df.columns = [c.lower() for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        for c in _VALUE_COLUMNS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        # TD occasionally returns close slightly above high or below low
        # (sub-bp rounding noise from the source pipeline). high must
        # dominate {open, low, close} from above and low must dominate
        # them from below — without this, downstream OHLC sanity checks
        # fail on rows that are otherwise fine.
        if {"open", "high", "low", "close"}.issubset(df.columns):
            ohlc = df[["open", "high", "low", "close"]]
            df["high"] = ohlc.max(axis=1)
            df["low"] = ohlc.min(axis=1)
        # Ensure tz-aware in NY (localize if naive, convert otherwise).
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is None:
            localized = df["datetime"].dt.tz_localize(req.timezone, ambiguous="NaT")
            if localized.isna().sum() != df["datetime"].isna().sum():
                localized = _resolve_folds(df, localized, req, fold_lookup)
            df["datetime"] = localized
        else:
            df["datetime"] = df["datetime"].dt.tz_convert(req.timezone)
        return df.sort_values("datetime").reset_index(drop=True)
