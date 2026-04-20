"""Stand-alone leakage check runnable outside pytest.

Verifies the primary invariant (indicator at t is independent of rows
> t) on a synthetic frame. Used by CI as an extra safety net.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm.indicators import compute_all    # noqa: E402
from tdwm.schema import INDICATOR_COLUMNS  # noqa: E402


def _make_frame(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n, freq="B", tz="America/New_York")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0, 2, n)
    low = close - rng.uniform(0, 2, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n)
    return pd.DataFrame(
        {
            "datetime": idx,
            "symbol": "TEST",
            "timeframe": "1day",
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "close_adj": close,
            "volume": volume,
        }
    )


def main() -> int:
    base = _make_frame()
    full = compute_all(base).reset_index(drop=True)

    rng = np.random.default_rng(123)
    failures: list[str] = []
    for _ in range(10):
        t = int(rng.integers(200, len(base) - 5))
        truncated = compute_all(base.iloc[: t + 1].copy()).reset_index(drop=True)
        for col in INDICATOR_COLUMNS:
            if col not in full.columns or col not in truncated.columns:
                continue
            a = full[col].iloc[t]
            b = truncated[col].iloc[t]
            if pd.isna(a) and pd.isna(b):
                continue
            if pd.isna(a) != pd.isna(b):
                failures.append(f"NaN mismatch t={t} col={col}: full={a} trunc={b}")
                continue
            if not np.isclose(a, b, rtol=1e-9, atol=1e-9):
                failures.append(f"Leakage t={t} col={col}: full={a} trunc={b}")

    if failures:
        for line in failures:
            print("FAIL:", line)
        return 1
    print("OK: no leakage detected across 10 random probe points.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
