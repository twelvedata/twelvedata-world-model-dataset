"""Thin wrapper around the twelvedata SDK.

Centralizes rate limiting, retries, and tz handling so the rest of the
pipeline never touches the SDK directly. Makes testing easier — tests
inject a fake client with the same surface area.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

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
        max_retries: int = 4,
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
                return self._normalize(df, req)
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                sleep_s = self.retry_base_sleep * (2**attempt)
                time.sleep(sleep_s)
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

    @staticmethod
    def _normalize(df: pd.DataFrame, req: FetchRequest) -> pd.DataFrame:
        """Return a DataFrame with a canonical shape regardless of SDK version."""
        if df is None or len(df) == 0:
            return pd.DataFrame(
                columns=["datetime", "open", "high", "low", "close", "volume"]
            )
        df = df.copy()
        # Twelve Data returns a DatetimeIndex named "datetime" — move it to a column.
        if isinstance(df.index, pd.DatetimeIndex):
            df.index = df.index.tz_localize(None) if df.index.tz is None else df.index
            df = df.reset_index().rename(columns={"index": "datetime"})
        # Lowercase column names.
        df.columns = [c.lower() for c in df.columns]
        if "datetime" not in df.columns and "date" in df.columns:
            df = df.rename(columns={"date": "datetime"})
        # Ensure tz-aware in NY.
        df["datetime"] = pd.to_datetime(df["datetime"])
        if df["datetime"].dt.tz is None:
            df["datetime"] = df["datetime"].dt.tz_localize(req.timezone)
        else:
            df["datetime"] = df["datetime"].dt.tz_convert(req.timezone)
        # Sort ascending.
        df = df.sort_values("datetime").reset_index(drop=True)
        # Cast numeric columns.
        for c in ("open", "high", "low", "close", "volume"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
