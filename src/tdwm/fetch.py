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
from typing import Any, Iterable

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
    req = FetchRequest(
        symbol=symbol,
        interval=tf.interval,
        start=start,
        end=end,
        outputsize=5000,
    )
    return client.fetch_bars(req)


def fetch_macro(
    client: TDClient,
    macro_symbols: Iterable[str],
    tf: TimeframeCfg,
    *,
    start: str | None = None,
    end: str | None = None,
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for sym in macro_symbols:
        try:
            out[sym] = fetch_symbol_history(client, sym, tf, start=start, end=end)
        except Exception as exc:  # noqa: BLE001
            # Macro gaps are not fatal — log and continue.
            print(f"[warn] macro fetch failed for {sym}: {exc}")
            out[sym] = pd.DataFrame(columns=["datetime", "open", "high", "low", "close", "volume"])
    return out
