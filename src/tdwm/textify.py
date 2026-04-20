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

import pandas as pd

from .schema import TextRow


def _fmt_num(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.{digits}f}"


def _fmt_pct(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{100 * x:+.{digits}f}%"


def row_to_prompt(row: pd.Series, verbose: bool = True) -> str:
    """Describe a single bar and its *already-observed* indicators."""
    when = pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d %H:%M %Z") \
        if getattr(row["datetime"], "tzinfo", None) is not None \
        else pd.Timestamp(row["datetime"]).strftime("%Y-%m-%d")
    base = (
        f"On {when}, {row['symbol']} ({row['timeframe']}) opened at "
        f"{_fmt_num(row['open'])}, reached a high of {_fmt_num(row['high'])}, "
        f"a low of {_fmt_num(row['low'])}, and closed at "
        f"{_fmt_num(row['close'])}. Volume: {_fmt_num(row['volume'], 0)}."
    )
    if not verbose:
        return base
    extras: list[str] = []
    if "rsi_14" in row and pd.notna(row["rsi_14"]):
        extras.append(f"RSI(14)={_fmt_num(row['rsi_14'])}")
    if "macd" in row and pd.notna(row["macd"]):
        extras.append(f"MACD={_fmt_num(row['macd'], 3)}")
    if "rv_20" in row and pd.notna(row["rv_20"]):
        extras.append(f"rv20={_fmt_num(row['rv_20'], 4)}")
    if "bb_pctb" in row and pd.notna(row["bb_pctb"]):
        extras.append(f"BB%b={_fmt_num(row['bb_pctb'], 2)}")
    if "spy_logret_1" in row and pd.notna(row["spy_logret_1"]):
        extras.append(f"SPY ret={_fmt_pct(row['spy_logret_1'])}")
    if "vix_level" in row and pd.notna(row["vix_level"]):
        extras.append(f"VIX={_fmt_num(row['vix_level'])}")
    if extras:
        return base + " Indicators: " + ", ".join(extras) + "."
    return base


def row_to_label(row: pd.Series, next_row: pd.Series | None) -> str:
    """Next-step outcome string. `None` if no next row (end of series)."""
    if next_row is None:
        return ""
    # Use next row's logret_1 as the forward-looking label.
    r = next_row.get("logret_1")
    if pd.isna(r):
        return ""
    direction = "up" if r > 0 else ("down" if r < 0 else "flat")
    return f"Next bar direction: {direction}. Next bar log return: {r:+.5f}."


def textify_frame(
    df: pd.DataFrame,
    *,
    verbose: bool = True,
) -> list[TextRow]:
    """Produce TextRow records. Assumes df is for a single symbol+timeframe,
    sorted ascending by datetime, and already has indicators + macro."""
    rows: list[TextRow] = []
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        cur = df.iloc[i]
        nxt = df.iloc[i + 1] if i + 1 < len(df) else None
        prompt = row_to_prompt(cur, verbose=verbose)
        label = row_to_label(cur, nxt)
        as_of = pd.Timestamp(cur["datetime"]).isoformat()
        rows.append(
            TextRow(
                symbol=str(cur["symbol"]),
                timeframe=str(cur["timeframe"]),
                as_of=as_of,
                prompt=prompt,
                label=label,
                meta={"i": int(i)},
            )
        )
    return rows


def textrows_to_records(rows: Iterable[TextRow]) -> list[dict]:
    return [asdict(r) for r in rows]
