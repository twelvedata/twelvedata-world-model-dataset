"""Typed schemas shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------- bars ----------

# Canonical column order for the bars parquet. Adding a new column is a
# schema change — update CHANGELOG.md and metadata/schema.json too.
BAR_CORE_COLUMNS: list[str] = [
    "datetime",     # tz-aware, America/New_York
    "symbol",
    "timeframe",    # 1day | 1h | 1min
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_adj",    # split/dividend-adjusted close (== close for ETFs/no-events)
]

# Indicator columns appended in a deterministic order. indicators.py is
# the single source of truth for which indicators are produced.
INDICATOR_COLUMNS: list[str] = [
    "ret_1", "logret_1", "ret_5", "logret_20",
    "rv_5", "rv_20", "rv_60", "atr_14",
    "rsi_14", "macd", "macd_signal", "macd_hist", "mom_10",
    "obv", "vol_z_20",
    "bb_mid", "bb_up", "bb_lo", "bb_pctb",
]

MACRO_COLUMNS: list[str] = [
    "spy_logret_1",
    "vix_level",
    "tlt_logret_1",
    "dxy_logret_1",
    "sector_logret_1",
]


def full_bar_columns() -> list[str]:
    return BAR_CORE_COLUMNS + INDICATOR_COLUMNS + MACRO_COLUMNS


# ---------- textified rows ----------

@dataclass
class TextRow:
    """A single instruction-tuning example.

    The invariant: `prompt` must NEVER contain information dated strictly
    after `as_of`. `label` may contain the next-step outcome(s).
    """
    symbol: str
    timeframe: str
    as_of: str            # ISO datetime; all info in `prompt` <= this
    prompt: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)


# ---------- trajectories ----------

@dataclass
class Trajectory:
    trajectory_id: str
    symbol: str
    timeframe: str
    feature_names: list[str]
    timestamps: list[str]            # length T
    states: list[list[float]]        # shape (T, F)
    next_states: list[list[float]]   # shape (T, F), shifted by 1
    # Optional scalar reward streams. `None` means the dataset is
    # reward-agnostic; downstream picks a reward definition.
    rewards_logret: list[float] | None = None
    # Split assignment by last-timestamp rule.
    split: str = "train"             # train | val | test
