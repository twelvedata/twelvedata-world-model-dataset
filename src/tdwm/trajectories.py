"""Rolling-window trajectory builder for world-model training.

Each trajectory captures T consecutive bars of a single symbol+timeframe.
`states[i]` is the state vector at step i, `next_states[i]` is the state
at i+1. The final row of `next_states` is the state at T (one past the
window end), so the *last* `states` row is still fully observed.

Trajectory ids are stable: f"{symbol}_{interval}_{first_ts}_{T}".
"""
from __future__ import annotations

from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .schema import Trajectory


def _state_matrix(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[feature_names].to_numpy(dtype=float)


def iter_trajectories(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    window: int,
    stride: int,
    split_of: callable | None = None,
) -> Iterator[Trajectory]:
    """Yield rolling-window trajectories one at a time.

    Streaming variant — peak memory is a single trajectory, not the full
    list. `split_of(last_ts) -> "train"|"val"|"test"` assigns the split
    based on the window's last observed timestamp (defaults to "train").
    """
    df = df.sort_values("datetime").reset_index(drop=True)
    n = len(df)
    # We need one extra row past the window for next_states[-1], so the
    # last valid start index is n - window - 1.
    last_start = n - window - 1
    if last_start < 0:
        return

    states_all = _state_matrix(df, feature_names)
    timestamps_all = [pd.Timestamp(t).isoformat() for t in df["datetime"]]

    has_logret = "logret_1" in df.columns
    logret_all = (
        df["logret_1"].to_numpy(dtype=float) if has_logret else None
    )

    symbol = str(df["symbol"].iloc[0])
    interval = str(df["timeframe"].iloc[0])
    feat_list = list(feature_names)

    for start in range(0, last_start + 1, stride):
        end = start + window
        state_slice = states_all[start:end]
        next_state_slice = states_all[start + 1:end + 1]
        ts_slice = timestamps_all[start:end]
        last_ts = df["datetime"].iloc[end - 1]
        split = split_of(last_ts) if split_of else "train"

        rewards = None
        if logret_all is not None:
            rewards = logret_all[start + 1:end + 1].tolist()

        traj_id = f"{symbol}_{interval}_{ts_slice[0]}_{window}"
        yield Trajectory(
            trajectory_id=traj_id,
            symbol=symbol,
            timeframe=interval,
            feature_names=feat_list,
            timestamps=ts_slice,
            states=state_slice.tolist(),
            next_states=next_state_slice.tolist(),
            rewards_logret=rewards,
            split=split,
        )


def build_trajectories(
    df: pd.DataFrame,
    feature_names: list[str],
    *,
    window: int,
    stride: int,
    split_of: callable | None = None,
) -> list[Trajectory]:
    """Materialize all trajectories for tests and small-scale callers.

    Large pipelines should prefer `iter_trajectories` to avoid holding
    every window in memory.
    """
    return list(
        iter_trajectories(
            df, feature_names, window=window, stride=stride, split_of=split_of
        )
    )


def trajectory_to_record(t: Trajectory) -> dict:
    return {
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


def trajectories_to_records(trajs: Iterable[Trajectory]) -> list[dict]:
    return [trajectory_to_record(t) for t in trajs]
