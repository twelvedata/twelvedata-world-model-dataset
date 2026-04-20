"""Trajectory builder tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tdwm.indicators import compute_all
from tdwm.trajectories import build_trajectories


FEATURES = ["open", "high", "low", "close_adj", "volume", "logret_1", "rsi_14"]


@pytest.fixture
def enriched(synthetic_bars: pd.DataFrame) -> pd.DataFrame:
    return compute_all(synthetic_bars).reset_index(drop=True)


def test_next_states_shifted_by_one(enriched: pd.DataFrame) -> None:
    trajs = build_trajectories(enriched, FEATURES, window=30, stride=30)
    assert len(trajs) > 0
    for t in trajs:
        s = np.asarray(t.states)
        n = np.asarray(t.next_states)
        # next_states[i] must equal the state at position start+i+1 of the
        # source frame — which is s[i+1] for all but the last position.
        assert s.shape == n.shape
        assert np.allclose(s[1:], n[:-1], equal_nan=True), (
            "next_states[:-1] does not equal states[1:] — shift bug"
        )


def test_window_and_stride(enriched: pd.DataFrame) -> None:
    w, stride = 30, 10
    trajs = build_trajectories(enriched, FEATURES, window=w, stride=stride)
    # First traj starts at 0; each subsequent traj starts `stride` later.
    for i, t in enumerate(trajs):
        expected_start = enriched["datetime"].iloc[i * stride]
        assert pd.Timestamp(t.timestamps[0]) == pd.Timestamp(expected_start)
        assert len(t.states) == w
        assert len(t.timestamps) == w


def test_trajectory_id_is_stable(enriched: pd.DataFrame) -> None:
    a = build_trajectories(enriched, FEATURES, window=30, stride=30)
    b = build_trajectories(enriched, FEATURES, window=30, stride=30)
    assert [t.trajectory_id for t in a] == [t.trajectory_id for t in b]


def test_feature_names_are_preserved(enriched: pd.DataFrame) -> None:
    trajs = build_trajectories(enriched, FEATURES, window=30, stride=30)
    for t in trajs:
        assert t.feature_names == FEATURES


def test_too_short_returns_empty() -> None:
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-02", periods=5, freq="B", tz="America/New_York"),
        "symbol": "Q", "timeframe": "1day",
        "open": [1, 2, 3, 4, 5],
        "high": [1, 2, 3, 4, 5],
        "low": [1, 2, 3, 4, 5],
        "close_adj": [1, 2, 3, 4, 5],
        "volume": [1, 2, 3, 4, 5],
        "logret_1": [0, 0, 0, 0, 0],
        "rsi_14": [50, 50, 50, 50, 50],
    })
    trajs = build_trajectories(df, FEATURES, window=30, stride=30)
    assert trajs == []
