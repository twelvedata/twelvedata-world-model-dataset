"""Generate examples/sample_trajectories.jsonl from synthetic data so the
repo ships with a tangible artifact a user can inspect before they even
have an API key.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm.indicators import compute_all            # noqa: E402
from tdwm.textify import textify_frame             # noqa: E402
from tdwm.trajectories import build_trajectories, trajectories_to_records  # noqa: E402


def make_df(symbol: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="America/New_York")
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    vol = rng.integers(1_000_000, 5_000_000, n)
    return pd.DataFrame({
        "datetime": idx,
        "symbol": symbol,
        "timeframe": "1day",
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "close_adj": close,
        "volume": vol,
    })


def main() -> int:
    out_dir = Path(__file__).resolve().parents[1] / "examples"
    out_dir.mkdir(exist_ok=True)

    df = compute_all(make_df("DEMO", 42))
    trajs = build_trajectories(
        df,
        feature_names=[
            "open", "high", "low", "close_adj", "volume",
            "logret_1", "rv_20", "rsi_14", "macd", "macd_hist",
            "atr_14", "bb_pctb", "obv", "vol_z_20",
        ],
        window=30, stride=10,
    )
    records = trajectories_to_records(trajs[:5])
    dest = out_dir / "sample_trajectories.jsonl"
    with open(dest, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {dest} ({len(records)} trajectories)")

    text_rows = textify_frame(df.iloc[-10:])
    from dataclasses import asdict
    dest_t = out_dir / "sample_text.jsonl"
    with open(dest_t, "w") as f:
        for r in text_rows:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"wrote {dest_t} ({len(text_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
