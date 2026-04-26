"""High-level fetch helpers built on top of TDClient.

Responsibilities:
- Load symbol + timeframe config.
- Request historical windows for backfill.
- Request just the tail for the daily updater.
- Return DataFrames already normalized by TDClient._normalize.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from .client import FetchRequest, TDClient


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass
class TimeframeCfg:
    interval: str
    enabled: bool
    history_years: int
    trajectory_windows: list[dict[str, int]]


def load_symbols() -> dict[str, list[str]]:
    with open(CONFIG_DIR / "symbols.yaml") as f:
        return yaml.safe_load(f)


def load_sectors() -> dict[str, str]:
    with open(CONFIG_DIR / "sectors.yaml") as f:
        return yaml.safe_load(f)


def load_timeframes() -> list[TimeframeCfg]:
    with open(CONFIG_DIR / "timeframes.yaml") as f:
        raw = yaml.safe_load(f)
    return [TimeframeCfg(**{k: tf[k] for k in TimeframeCfg.__dataclass_fields__})
            for tf in raw["timeframes"] if tf.get("enabled", True)]


def backfill_window(tf: TimeframeCfg, *, now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.utcnow()
    end = now.date().isoformat()
    start = (now - timedelta(days=int(tf.history_years * 365.25))).date().isoformat()
    return start, end


def incremental_window(last_known: datetime, *, now: datetime | None = None) -> tuple[str, str]:
    """Window for the daily updater. Always re-fetches the previous day
    to catch restatements (late prints, corporate actions)."""
    now = now or datetime.utcnow()
    start = (last_known - timedelta(days=2)).date().isoformat()
    end = now.date().isoformat()
    return start, end


_INTERVAL_MINUTES: dict[str, int] = {
    "1min": 1, "5min": 5, "15min": 15, "30min": 30,
    "1h": 60, "2h": 120, "4h": 240, "1day": 1440,
}


def fetch_symbol_history(
    client: TDClient,
    symbol: str,
    tf: TimeframeCfg,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    if start is None or end is None:
        start, end = backfill_window(tf)

    tz = "America/New_York"
    start_dt = pd.Timestamp(start, tz=tz)
    end_dt = pd.Timestamp(end, tz=tz)
    interval_delta = timedelta(minutes=_INTERVAL_MINUTES.get(tf.interval, 1440))

    chunks: list[pd.DataFrame] = []
    chunk_end = end_dt

    while chunk_end > start_dt:
        req = FetchRequest(
            symbol=symbol,
            interval=tf.interval,
            start=start_dt.date().isoformat(),
            end=chunk_end.date().isoformat(),
            outputsize=5000,
        )
        try:
            chunk = client.fetch_bars(req)
        except RuntimeError as e:
            if "No data is available" in str(e):
                break  # hit the API's history limit
            raise
        if chunk.empty:
            break
        chunks.append(chunk)
        earliest = pd.Timestamp(chunk["datetime"].iloc[0]).tz_convert(tz)
        # Reached the requested start — normal termination.
        if earliest <= start_dt:
            break
        # A short page means the next request would return no rows, so
        # stop. Two reasons we end up here, and they look different in
        # logs: a deep backfill that hit the symbol's actual inception
        # (earliest >> start_dt with a full window's worth of unfilled
        # range), vs. an incremental run where the requested window was
        # just smaller than one page.
        if len(chunk) < req.outputsize:
            unfilled = (earliest - start_dt) - interval_delta * len(chunk)
            page_span = interval_delta * req.outputsize
            if unfilled > page_span:
                reason = "inception reached"
            else:
                reason = "window complete"
            print(f"  [page] {symbol} {tf.interval}: got {len(chunk)} rows ({reason})")
            break
        # Step back one interval before the earliest bar we got.
        chunk_end = earliest - interval_delta
        print(f"  [page] {symbol} {tf.interval}: got {len(chunk)} rows, next end={chunk_end.date()}")

    if not chunks:
        return pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime").reset_index(drop=True)
    # Trim to the requested window.
    df = df.loc[pd.to_datetime(df["datetime"]).dt.tz_convert(tz) >= start_dt].reset_index(drop=True)
    return df


def fetch_macro(
    client: TDClient,
    macro_symbols: Iterable[str],
    tf: TimeframeCfg,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    # Fail fast on macro fetch errors: a missing macro silently shifts
    # equity rows to use the previous day's macro values via merge_asof,
    # which is locked in once the equity's last_known advances past the
    # affected date. Investigate manually rather than corrupting bars.
    out: dict[str, pd.DataFrame] = {}
    for sym in macro_symbols:
        out[sym] = fetch_symbol_history(client, sym, tf, start=start, end=end)
    return out
