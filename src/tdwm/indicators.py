"""Causal technical indicators.

Every function here computes values using only rows with index <= i when
producing the value at index i. No centered windows, no forward-fill of
future values, no .shift(-N) anywhere.

This is the single source of truth for indicator logic. New indicators
must be added here AND to tests/test_indicators_causal.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import INDICATOR_COLUMNS


# ---------- primitives ----------

def simple_return(close: pd.Series, horizon: int) -> pd.Series:
    return close.pct_change(periods=horizon)


def log_return(close: pd.Series, horizon: int) -> pd.Series:
    return np.log(close).diff(horizon)


def realized_vol(close: pd.Series, window: int) -> pd.Series:
    """Stdev of log returns over a backward window. Uses t-ddof=0 for stability."""
    lr = np.log(close).diff()
    return lr.rolling(window=window, min_periods=window).std(ddof=0)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window, min_periods=window).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    # Wilder's smoothing uses exponential with alpha=1/window — causal.
    avg_up = up.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_down = down.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_up / avg_down.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = line - sig
    return pd.DataFrame({"macd": line, "macd_signal": sig, "macd_hist": hist})


def momentum(close: pd.Series, window: int) -> pd.Series:
    return close - close.shift(window)


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    sign = np.sign(close.diff()).fillna(0.0)
    return (sign * volume).cumsum()


def volume_zscore(volume: pd.Series, window: int) -> pd.Series:
    mu = volume.rolling(window, min_periods=window).mean()
    sd = volume.rolling(window, min_periods=window).std(ddof=0)
    return (volume - mu) / sd.replace(0.0, np.nan)


def bollinger(close: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = close.rolling(window, min_periods=window).mean()
    sd = close.rolling(window, min_periods=window).std(ddof=0)
    up = mid + k * sd
    lo = mid - k * sd
    pctb = (close - lo) / (up - lo).replace(0.0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_up": up, "bb_lo": lo, "bb_pctb": pctb})


# ---------- assembly ----------

def compute_all(bars: pd.DataFrame) -> pd.DataFrame:
    """Append every indicator in INDICATOR_COLUMNS to `bars` and return a copy.

    `bars` must be sorted ascending by datetime and contain at minimum:
    open, high, low, close, volume, close_adj.
    """
    df = bars.copy()
    if "close_adj" not in df.columns:
        df["close_adj"] = df["close"]

    close = df["close_adj"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    df["ret_1"] = simple_return(close, 1)
    df["logret_1"] = log_return(close, 1)
    df["ret_5"] = simple_return(close, 5)
    df["logret_20"] = log_return(close, 20)

    df["rv_5"] = realized_vol(close, 5)
    df["rv_20"] = realized_vol(close, 20)
    df["rv_60"] = realized_vol(close, 60)
    df["atr_14"] = atr(high, low, close, 14)

    df["rsi_14"] = rsi(close, 14)
    macd_df = macd(close, 12, 26, 9)
    df["macd"] = macd_df["macd"]
    df["macd_signal"] = macd_df["macd_signal"]
    df["macd_hist"] = macd_df["macd_hist"]
    df["mom_10"] = momentum(close, 10)

    df["obv"] = obv(close, vol)
    df["vol_z_20"] = volume_zscore(vol, 20)

    bb = bollinger(close, 20, 2.0)
    df["bb_mid"] = bb["bb_mid"]
    df["bb_up"] = bb["bb_up"]
    df["bb_lo"] = bb["bb_lo"]
    df["bb_pctb"] = bb["bb_pctb"]

    # Ensure every advertised indicator column exists (even if all-NaN
    # when there isn't enough history).
    for c in INDICATOR_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    return df
