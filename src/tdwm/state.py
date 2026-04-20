"""Resume/backfill state.

A single JSON file at `data/_state.json` keyed by `{timeframe}:{symbol}`
with the last successfully-ingested timestamp and the last run metadata.
The daily updater reads this file, computes missing ranges, and issues
only the necessary fetches — which makes the whole pipeline idempotent
and resumable when GitHub Actions misses a cron window.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class State:
    entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "State":
        if not path.exists():
            return cls()
        with open(path) as f:
            return cls(entries=json.load(f))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.entries, f, indent=2, sort_keys=True, default=str)
        tmp.replace(path)

    @staticmethod
    def key(timeframe: str, symbol: str) -> str:
        return f"{timeframe}:{symbol}"

    def get_last(self, timeframe: str, symbol: str) -> datetime | None:
        v = self.entries.get(self.key(timeframe, symbol))
        if not v:
            return None
        ts = v.get("last_datetime")
        return datetime.fromisoformat(ts) if ts else None

    def record(
        self,
        timeframe: str,
        symbol: str,
        last_datetime: datetime,
        *,
        rows_written: int,
        run_started: datetime,
    ) -> None:
        self.entries[self.key(timeframe, symbol)] = {
            "last_datetime": last_datetime.isoformat(),
            "last_rows_written": int(rows_written),
            "last_run_started": run_started.isoformat(),
        }
