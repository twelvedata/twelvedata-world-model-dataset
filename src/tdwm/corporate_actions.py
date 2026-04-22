"""Corporate actions: splits and dividends.

Detects splits/dividends that occurred since the last check and returns
the set of symbols that need a full re-backfill (because stored prices
are now on a different scale than what the API returns).

Persistence:
    data/_corporate_actions.json  — records the last check date and all
    seen (symbol, date, action) tuples so we don't re-trigger on the same
    event twice.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


_SEEN_FILE = "_corporate_actions.json"


def _load_seen(data_root: Path) -> dict[str, Any]:
    p = data_root / _SEEN_FILE
    if not p.exists():
        return {"last_checked": None, "seen": {}}
    with open(p) as f:
        return json.load(f)


def _save_seen(data_root: Path, record: dict[str, Any]) -> None:
    p = data_root / _SEEN_FILE
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    tmp.replace(p)


def _fetch_splits(sdk: Any, symbol: str, start_date: str, end_date: str) -> list[str]:
    """Return list of split dates for symbol in [start_date, end_date]."""
    try:
        df = sdk.get_splits(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        ).as_pandas()
        if df is None or df.empty:
            return []
        date_col = "date" if "date" in df.columns else df.columns[0]
        return list(df[date_col].astype(str))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] splits fetch failed for {symbol}: {exc}")
        return []


def _fetch_dividends(sdk: Any, symbol: str, start_date: str, end_date: str) -> list[str]:
    """Return list of ex-dividend dates for symbol in [start_date, end_date]."""
    try:
        df = sdk.get_dividends(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        ).as_pandas()
        if df is None or df.empty:
            return []
        date_col = "ex_date" if "ex_date" in df.columns else (
            "date" if "date" in df.columns else df.columns[0]
        )
        return list(df[date_col].astype(str))
    except Exception as exc:  # noqa: BLE001
        print(f"  [warn] dividends fetch failed for {symbol}: {exc}")
        return []


def check_corporate_actions(
    sdk: Any,
    symbols: list[str],
    data_root: Path,
    *,
    lookback_days: int = 7,
) -> set[str]:
    """Check for splits/dividends since last run.

    Returns the set of symbols that had a corporate action and need a
    full re-backfill. Also persists seen events so re-runs are idempotent.
    """
    record = _load_seen(data_root)
    seen: dict[str, list[str]] = record.get("seen", {})

    today = date.today().isoformat()
    last_checked = record.get("last_checked")
    if last_checked:
        # Go back a bit further than last check to catch late-published events.
        start = (
            datetime.fromisoformat(last_checked) - timedelta(days=2)
        ).date().isoformat()
    else:
        start = (date.today() - timedelta(days=lookback_days)).isoformat()

    dirty: set[str] = set()

    for sym in symbols:
        sym_seen = set(seen.get(sym, []))
        new_events: list[str] = []

        splits = _fetch_splits(sdk, sym, start_date=start, end_date=today)
        for d in splits:
            key = f"split:{d}"
            if key not in sym_seen:
                print(f"  [corporate-action] {sym}: split on {d} — scheduling re-backfill")
                new_events.append(key)
                dirty.add(sym)

        dividends = _fetch_dividends(sdk, sym, start_date=start, end_date=today)
        for d in dividends:
            key = f"dividend:{d}"
            if key not in sym_seen:
                print(f"  [corporate-action] {sym}: dividend ex-date {d} — scheduling re-backfill")
                new_events.append(key)
                dirty.add(sym)

        if new_events:
            seen[sym] = list(sym_seen | set(new_events))

    record["last_checked"] = today
    record["seen"] = seen
    _save_seen(data_root, record)

    return dirty


def clear_symbol_bars(data_root: Path, symbol: str) -> None:
    """Delete all parquet partitions for a symbol across all timeframes/years."""
    safe_sym = symbol.replace("/", "_").replace(".", "_")
    bars_root = data_root / "bars"
    if not bars_root.exists():
        return
    for tf_dir in bars_root.iterdir():
        sym_dir = tf_dir / f"symbol={safe_sym}"
        if sym_dir.exists():
            for part in sym_dir.glob("year=*/part.parquet"):
                part.unlink()
                print(f"  [clear] removed {part}")
