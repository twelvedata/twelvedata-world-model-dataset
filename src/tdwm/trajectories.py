"""Rolling-window trajectory builder for world-model training.

Each trajectory captures T consecutive bars of a single symbol+timeframe.
`states[i]` is the state vector at step i, `next_states[i]` is the state
at i+1. The final row of `next_states` is the state at T (one past the
window end), so the *last* `states` row is still fully observed.

Trajectory ids are stable: f"{symbol}_{interval}_{first_ts}_{T}".
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .schema import Trajectory


def _state_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[feature_names].to_numpy(dtype=float)


def build_trajectories(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    window: int,
    stride: int,
    split_of: callable | None = None,
) -> list[Trajectory]:
    """Build rolling windows.

    `split_of(last_ts) -> "train"|"val"|"test"` assigns a split based on
    the last *observed* timestamp of the window (i.e. df row at position
    start+window-1). If None, all trajectories are "train".
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    # We need one extra row past the window for next_states[-1], so the
    # last valid start index is n - window - 1.
    last_start = n - window - 1
    if last_start < 0:
        return []

    states_all = _state_matrix(df, feature_names)
    timestamps_all = [pd.Timestamp(t).isoformat() for t in df["datetime"]]

    # Pre-extract for optional rewards.
    has_logret = "logret_1" in df.columns
    logret_all = (
        df["logret_1"].to_numpy(dtype=float) if has_logret else None
    )

    out: list[Trajectory] = []
    symbol = str(df["symbol"].iloc[0])
    interval = str(df["timeframe"].iloc[0])

    for start in range(0, last_start + 1, stride):
        end = start + window
        state_slice = states_all[start:end]                 # (T, F)
        next_state_slice = states_all[start + 1:end + 1]    # (T, F)
        ts_slice = timestamps_all[start:end]
        last_ts = df["datetime"].iloc[end - 1]
        split = split_of(last_ts) if split_of else "train"

        rewards = None
        if logret_all is not None:
            # next-step log return for each state in the window.
            rewards = logret_all[start + 1:end + 1].tolist()

        traj_id = f"{symbol}_{interval}_{ts_slice[0]}_{window}"
        out.append(
            Trajectory(
                trajectory_id=traj_id,
                symbol=symbol,
                timeframe=interval,
                feature_names=list(feature_names),
                timestamps=ts_slice,
                states=state_slice.tolist(),
                next_states=next_state_slice.tolist(),
                rewards_logret=rewards,
                split=split,
            )
        )
    return out


def trajectories_to_records(trajs: Iterable[Trajectory]) -> list[dict]:
    return [
        {
            "trajectory_id": t.trajectory_id,
            "symbol": t.symbol,
            "timeframe": t.timeframe,
            "feature_names": t.feature_names,
            "timestamps": t.timestamps,
            "states": t.states,
            "next_states": t.next_states,
            "rewards_logret": t.rewards_logret,
            "split": t.split,
        }
        for t in trajs
    ]
