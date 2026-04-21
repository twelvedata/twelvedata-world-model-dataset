"""Join macro-context columns onto per-symbol bar DataFrames.

Macro bars are fetched once per run (via fetch.py) and kept as a dict of
{macro_symbol: DataFrame with `datetime` + indicators}. enrich.py merges
the relevant macro returns / levels onto each equity row by datetime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import MACRO_COLUMNS


def _safe_logret(df: pd.DataFrame) -> pd.Series:
    """Compute logret_1 from close_adj (or close) if not already present."""
    if "logret_1" in df.columns:
        return df["logret_1"]
    close = df["close_adj"] if "close_adj" in df.columns else df["close"]
    return np.log(close).diff()


def build_macro_frame(
    macros: dict[str, pd.DataFrame],
    *,
    spy: str = "SPY",
    vix: str = "VIXY",
    tlt: str = "TLT",
    dxy: str = "UUP",
) -> pd.DataFrame:
    """Produce a single frame indexed by datetime with macro columns.

    Uses UUP as the DXY proxy by default (Twelve Data's DXY index coverage
    varies by plan). Override via kwargs.
    """
    parts: list[pd.Series] = []

    def _col(sym: str, name: str, kind: str) -> pd.Series | None:
        if sym not in macros:
            return None
        df = macros[sym]
        if "datetime" not in df.columns:
            return None
        idx = pd.DatetimeIndex(df["datetime"])
        if idx.tz is None:
            idx = idx.tz_localize("America/New_York")
        if kind == "logret":
            s = _safe_logret(df)
        elif kind == "level":
            s = df["close_adj"] if "close_adj" in df.columns else df["close"]
        else:
            raise ValueError(kind)
        s = pd.Series(s.values, index=idx, name=name)
        return s

    parts.append(_col(spy, "spy_logret_1", "logret"))
    parts.append(_col(vix, "vix_level", "level"))
    parts.append(_col(tlt, "tlt_logret_1", "logret"))
    parts.append(_col(dxy, "dxy_logret_1", "logret"))
    parts = [p for p in parts if p is not None]
    if not parts:
        return pd.DataFrame(columns=["datetime", *MACRO_COLUMNS])
    macro = pd.concat(parts, axis=1).sort_index()
    macro.index.name = "datetime"
    return macro.reset_index()


def attach_macro(
    bars: pd.DataFrame,
    macro_frame: pd.DataFrame,
    sector_etf: str | None,
    macros: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Left-join macro context onto `bars` by datetime (nearest past)."""
    out = bars.sort_values("datetime").copy()
    # Normalize datetime precision to microseconds so merge_asof keys match.
    out["datetime"] = pd.to_datetime(out["datetime"]).dt.as_unit("us")
    if not macro_frame.empty:
        # merge_asof: for each bar row take the most recent macro row
        # with datetime <= bar datetime (never future).
        mf = macro_frame.sort_values("datetime").copy()
        mf["datetime"] = pd.to_datetime(mf["datetime"]).dt.as_unit("us")
        out = pd.merge_asof(
            out,
            mf,
            on="datetime",
            direction="backward",
            allow_exact_matches=True,
        )
    else:
        for c in ("spy_logret_1", "vix_level", "tlt_logret_1", "dxy_logret_1"):
            out[c] = np.nan

    # sector_logret_1
    if sector_etf and sector_etf in macros:
        sec = macros[sector_etf]
        sec_ret = _safe_logret(sec)
        sec_dt = pd.to_datetime(sec["datetime"])
        if sec_dt.dt.tz is None:
            sec_dt = sec_dt.dt.tz_localize("America/New_York")
        sec_df = pd.DataFrame(
            {"datetime": sec_dt.dt.as_unit("us"), "sector_logret_1": sec_ret.values}
        ).sort_values("datetime")
        out = pd.merge_asof(out, sec_df, on="datetime", direction="backward")
    else:
        out["sector_logret_1"] = np.nan

    # Make sure every macro column exists even if source was missing.
    for c in MACRO_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out
