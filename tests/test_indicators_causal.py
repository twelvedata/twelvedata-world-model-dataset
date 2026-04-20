"""Leakage tests.

Core invariant: the indicator value at row t must be a function ONLY of
rows 0..t of the input. We test this two ways:

1. **Truncation test.** Compute indicators on df[:t+1]. The value at t
   must equal the value at t when computed on the full frame.

2. **Future-shuffle test.** Take the full frame and shuffle all rows
   with index > t. Recompute indicators. Row t's indicators must be
   unchanged. This is a strictly stronger check — any centered window,
   leaky lookup, or accidental .shift(-k) will break it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tdwm.indicators import compute_all
from tdwm.schema import INDICATOR_COLUMNS


PROBE_POINTS = [150, 250, 400, 500]


@pytest.mark.parametrize("t", PROBE_POINTS)
def test_truncation_invariance(synthetic_bars: pd.DataFrame, t: int) -> None:
    full = compute_all(synthetic_bars).reset_index(drop=True)
    truncated = compute_all(synthetic_bars.iloc[: t + 1].copy()).reset_index(drop=True)
    for col in INDICATOR_COLUMNS:
        a = full[col].iloc[t]
        b = truncated[col].iloc[t]
        if pd.isna(a) and pd.isna(b):
            continue
        assert pd.isna(a) == pd.isna(b), f"NaN mismatch at t={t} col={col}: {a} vs {b}"
        assert np.isclose(a, b, rtol=1e-10, atol=1e-10), (
            f"Leakage at t={t} col={col}: full={a} trunc={b}"
        )


@pytest.mark.parametrize("t", PROBE_POINTS)
def test_future_shuffle_invariance(synthetic_bars: pd.DataFrame, t: int) -> None:
    full = compute_all(synthetic_bars).reset_index(drop=True)

    # Shuffle rows strictly after t; keep 0..t identical.
    rng = np.random.default_rng(0)
    head = synthetic_bars.iloc[: t + 1].reset_index(drop=True)
    tail = synthetic_bars.iloc[t + 1:].sample(frac=1.0, random_state=rng.integers(0, 2**32 - 1))
    # Future-shuffle must preserve 0..t ordering, so we keep head as-is.
    shuffled = pd.concat([head, tail], ignore_index=True)
    # Datetime column is now non-monotonic in the tail — indicator code
    # sorts internally in compute_all? It does NOT (by design, so this
    # test *proves* that indicators only depend on positional order of
    # rows 0..t regardless of what's downstream).
    shuffled_full = compute_all(shuffled).reset_index(drop=True)
    for col in INDICATOR_COLUMNS:
        a = full[col].iloc[t]
        b = shuffled_full[col].iloc[t]
        if pd.isna(a) and pd.isna(b):
            continue
        assert pd.isna(a) == pd.isna(b), f"NaN mismatch at t={t} col={col}"
        assert np.isclose(a, b, rtol=1e-10, atol=1e-10), (
            f"Future-shuffle leakage at t={t} col={col}: full={a} shuffled={b}"
        )


def test_every_advertised_column_present(synthetic_bars: pd.DataFrame) -> None:
    """Schema contract: compute_all must produce every INDICATOR_COLUMNS column."""
    df = compute_all(synthetic_bars)
    missing = [c for c in INDICATOR_COLUMNS if c not in df.columns]
    assert not missing, f"compute_all missing columns: {missing}"


def test_no_all_nan_after_warmup(synthetic_bars: pd.DataFrame) -> None:
    df = compute_all(synthetic_bars).iloc[200:]  # past all warmups
    for col in INDICATOR_COLUMNS:
        assert df[col].notna().any(), f"Indicator {col} is all NaN after warmup"
