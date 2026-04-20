"""Time-based train / val / test splits.

The `SplitConfig.assign(ts)` function returns which split a bar belongs
to. For trajectories, the "last-timestamp rule" applies: the split is
determined by the window's final observed timestamp. If
`strict_no_crossing` is True, trajectories whose *first* and *last*
timestamps fall in different splits are dropped entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config" / "splits.yaml"


def _parse_date(x: object) -> Optional[date]:
    if x is None:
        return None
    if isinstance(x, date) and not isinstance(x, datetime):
        return x
    return pd.Timestamp(str(x)).date()


@dataclass
class SplitConfig:
    train_end: date
    val_start: date
    val_end: date
    test_start: date
    test_end: Optional[date]
    strict_no_crossing: bool = True

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "SplitConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f)
        return cls(
            train_end=_parse_date(cfg["train"]["end"]),
            val_start=_parse_date(cfg["val"]["start"]),
            val_end=_parse_date(cfg["val"]["end"]),
            test_start=_parse_date(cfg["test"]["start"]),
            test_end=_parse_date(cfg["test"].get("end")),
            strict_no_crossing=bool(cfg.get("strict_no_crossing", True)),
        )

    def assign(self, ts: pd.Timestamp | datetime | date) -> str:
        d = ts.date() if hasattr(ts, "date") else ts
        if d <= self.train_end:
            return "train"
        if self.val_start <= d <= self.val_end:
            return "val"
        if d >= self.test_start and (self.test_end is None or d <= self.test_end):
            return "test"
        # If we fall through (e.g. tiny gap between val.end and test.start),
        # round up to test.
        return "test"

    def split_of_trajectory(
        self,
        first_ts: pd.Timestamp,
        last_ts: pd.Timestamp,
    ) -> Optional[str]:
        first = self.assign(first_ts)
        last = self.assign(last_ts)
        if self.strict_no_crossing and first != last:
            return None
        return last
