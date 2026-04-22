"""Partitioned parquet storage.

Layout:
    data/bars/{timeframe}/symbol={SYM}/year={YYYY}/part.parquet

One file per (timeframe, symbol, year). Daily appends touch at most one
partition; year-end rolls into the next automatically. We keep the write
atomic by writing to a `.tmp` file and renaming.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def _partition_path(root: Path, timeframe: str, symbol: str, year: int) -> Path:
    safe_sym = symbol.replace("/", "_").replace(".", "_")
    return root / "bars" / timeframe / f"symbol={safe_sym}" / f"year={year}" / "part.parquet"


def _atomic_write(df: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)


def write_bars(df: pd.DataFrame, root: Path) -> list[Path]:
    """Append/merge bars into partitioned parquet. Returns written paths.

    Existing rows with the same (datetime, symbol, timeframe) are replaced
    by incoming ones. Useful for restatement handling — the updater can
    safely re-send a day and the later copy wins.
    """
    if df.empty:
        return []
    required = {"datetime", "symbol", "timeframe"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"write_bars missing columns: {missing}")
    df = df.copy()
    df["_year"] = pd.DatetimeIndex(df["datetime"]).year
    written: list[Path] = []
    for (tf, sym, year), group in df.groupby(["timeframe", "symbol", "_year"]):
        dest = _partition_path(root, tf, sym, int(year))
        group = group.drop(columns=["_year"])
        if dest.exists():
            existing = pd.read_parquet(dest)
            combined = pd.concat([existing, group], ignore_index=True)
            combined = (
                combined.sort_values("datetime")
                .drop_duplicates(subset=["datetime", "symbol", "timeframe"], keep="last")
                .reset_index(drop=True)
            )
        else:
            combined = group.sort_values("datetime").reset_index(drop=True)
        _atomic_write(combined, dest)
        written.append(dest)
    return written


def read_bars(
    root: Path,
    *,
    timeframe: str,
    symbol: str,
    year: int | None = None,
) -> pd.DataFrame:
    if year is not None:
        p = _partition_path(root, timeframe, symbol, year)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)
    # Combine all years for the symbol/timeframe.
    safe_sym = symbol.replace("/", "_").replace(".", "_")
    sym_dir = root / "bars" / timeframe / f"symbol={safe_sym}"
    if not sym_dir.exists():
        return pd.DataFrame()
    parts = sorted(sym_dir.glob("year=*/part.parquet"))
    frames = [pd.read_parquet(p) for p in parts]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("datetime").reset_index(drop=True)


def last_datetime(
    root: Path, *, timeframe: str, symbol: str
) -> pd.Timestamp | None:
    df = read_bars(root, timeframe=timeframe, symbol=symbol)
    if df.empty:
        return None
    return pd.Timestamp(df["datetime"].iloc[-1])


def _macro_path(root: Path, timeframe: str, symbol: str) -> Path:
    safe_sym = symbol.replace("/", "_").replace(".", "_")
    return root / "macro" / timeframe / f"{safe_sym}.parquet"


def write_macro(df: pd.DataFrame, root: Path, timeframe: str, symbol: str) -> Path:
    """Merge-write macro bars for one symbol/timeframe. Dedupes by datetime."""
    dest = _macro_path(root, timeframe, symbol)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        existing = pd.read_parquet(dest)
        df = (
            pd.concat([existing, df], ignore_index=True)
            .drop_duplicates(subset=["datetime"], keep="last")
            .sort_values("datetime")
            .reset_index(drop=True)
        )
    _atomic_write(df, dest)
    return dest


def read_macro(root: Path, timeframe: str, symbol: str) -> pd.DataFrame:
    """Read cached macro bars for one symbol/timeframe. Returns empty DF if missing."""
    p = _macro_path(root, timeframe, symbol)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def last_macro_datetime(root: Path, timeframe: str, symbol: str) -> pd.Timestamp | None:
    df = read_macro(root, timeframe, symbol)
    if df.empty:
        return None
    return pd.Timestamp(df["datetime"].iloc[-1])
