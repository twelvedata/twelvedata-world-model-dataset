"""Convert bar+indicator rows into instruction-tuning strings.

Critical invariant: anything placed in `prompt` describes information
available at or before `as_of`. The observed outcome (next-step return)
lives only in `label`. Tests/test_textify.py enforces this by stringified
search — the word "next" and the numeric next-day return may only appear
in label.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Iterable

import numpy as np
import pandas as pd

from .schema import TextRow


def _fmt_num_array(arr: np.ndarray, digits: int) -> list[str]:
    """Format each float with fixed digits, "n/a" for NaN."""
    out = [""] * len(arr)
    for i, x in enumerate(arr):
        if math.isnan(x):
            out[i] = "n/a"
        else:
            out[i] = f"{x:.{digits}f}"
    return out


def _fmt_extra(
    df: pd.DataFrame, col: str, digits: int, template: str
) -> list[str | None]:
    """Return a list of pre-formatted extras for `col`, or None where the
    value is missing/NaN. Returns all-None if the column is absent."""
    n = len(df)
    if col not in df.columns:
        return [None] * n
    arr = df[col].to_numpy(dtype=float)
    out: list[str | None] = [None] * n
    for i, x in enumerate(arr):
        if not math.isnan(x):
            out[i] = template.format(f"{x:.{digits}f}")
    return out


def _fmt_extra_pct(
    df: pd.DataFrame, col: str, digits: int, template: str
) -> list[str | None]:
    n = len(df)
    if col not in df.columns:
        return [None] * n
    arr = df[col].to_numpy(dtype=float)
    out: list[str | None] = [None] * n
    for i, x in enumerate(arr):
        if not math.isnan(x):
            out[i] = template.format(f"{100 * x:+.{digits}f}%")
    return out


def textify_frame(
    df: pd.DataFrame,
    *,
    verbose: bool = True,
) -> list[TextRow]:
    """Produce TextRow records. Assumes df is for a single symbol+timeframe,
    sorted ascending by datetime, and already has indicators + macro.

    Implemented with bulk numpy/pandas ops — avoids the per-row `df.iloc[i]`
    overhead that dominates at 100k+ rows.
    """
    df = df.reset_index(drop=True)
    n = len(df)
    if n == 0:
        return []

    # --- datetime: bulk strftime for the human-readable `when`, per-element
    # isoformat for `as_of` to match legacy output exactly.
    dt_col = pd.to_datetime(df["datetime"])
    tz_aware = dt_col.dt.tz is not None
    if tz_aware:
        # pandas's strftime on a *tz-aware* DatetimeIndex routes through
        # `tslib._format_native_types`, which formats one element at a
        # time with a tz-name lookup per-row. On some us-precision
        # series this hangs at 100% CPU for 10+ minutes (PEP daily).
        # Stripping the tz first puts us on the bulk numpy strftime
        # path, which formats the same 13k rows in ~10ms. The wall-
        # clock values are unchanged; we append the IANA name once.
        tz_name = str(dt_col.dt.tz)
        naive = dt_col.dt.tz_localize(None)
        when = (naive.dt.strftime("%Y-%m-%d %H:%M") + " " + tz_name).tolist()
    else:
        when = dt_col.dt.strftime("%Y-%m-%d").tolist()
    # `.isoformat()` matches the legacy "+HH:MM" offset style that
    # `strftime("%z")` doesn't produce, so iterate.
    as_of = [pd.Timestamp(t).isoformat() for t in dt_col]

    symbols = df["symbol"].astype(str).tolist()
    timeframes = df["timeframe"].astype(str).tolist()

    # --- base sentence ingredients
    o = _fmt_num_array(df["open"].to_numpy(dtype=float), 2)
    h = _fmt_num_array(df["high"].to_numpy(dtype=float), 2)
    lo = _fmt_num_array(df["low"].to_numpy(dtype=float), 2)
    c = _fmt_num_array(df["close"].to_numpy(dtype=float), 2)
    v = _fmt_num_array(df["volume"].to_numpy(dtype=float), 0)

    base = [
        f"On {w}, {s} ({tf}) opened at {oo}, reached a high of {hh}, "
        f"a low of {ll}, and closed at {cc}. Volume: {vv}."
        for w, s, tf, oo, hh, ll, cc, vv in zip(
            when, symbols, timeframes, o, h, lo, c, v
        )
    ]

    # --- verbose extras
    if verbose:
        rsi_s = _fmt_extra(df, "rsi_14", 2, "RSI(14)={}")
        macd_s = _fmt_extra(df, "macd", 3, "MACD={}")
        rv_s = _fmt_extra(df, "rv_20", 4, "rv20={}")
        bb_s = _fmt_extra(df, "bb_pctb", 2, "BB%b={}")
        spy_s = _fmt_extra_pct(df, "spy_logret_1", 2, "SPY ret={}")
        vix_s = _fmt_extra(df, "vix_level", 2, "VIX={}")

        prompts = [""] * n
        for i in range(n):
            extras: list[str] = []
            for s in (rsi_s[i], macd_s[i], rv_s[i], bb_s[i], spy_s[i], vix_s[i]):
                if s is not None:
                    extras.append(s)
            if extras:
                prompts[i] = base[i] + " Indicators: " + ", ".join(extras) + "."
            else:
                prompts[i] = base[i]
    else:
        prompts = base

    # --- labels: next row's logret_1
    if "logret_1" in df.columns:
        logret = df["logret_1"].to_numpy(dtype=float)
    else:
        logret = np.full(n, np.nan)
    labels = [""] * n
    for i in range(n - 1):
        r = logret[i + 1]
        if not math.isnan(r):
            direction = "up" if r > 0 else ("down" if r < 0 else "flat")
            labels[i] = (
                f"Next bar direction: {direction}. "
                f"Next bar log return: {r:+.5f}."
            )

    # --- assemble
    rows: list[TextRow] = [None] * n  # type: ignore[list-item]
    for i in range(n):
        rows[i] = TextRow(
            symbol=symbols[i],
            timeframe=timeframes[i],
            as_of=as_of[i],
            prompt=prompts[i],
            label=labels[i],
            meta={"i": i},
        )
    return rows


def textrows_to_records(rows: Iterable[TextRow]) -> list[dict]:
    return [asdict(r) for r in rows]
