"""Prompt/label separation tests.

The `prompt` field of a TextRow must NEVER contain the next-step outcome.
We enforce this by numerical and string-based checks on a synthetic
frame with indicators already attached.
"""
from __future__ import annotations

import re

import pandas as pd
import pytest

from tdwm.indicators import compute_all
from tdwm.textify import textify_frame


@pytest.fixture
def enriched(synthetic_bars: pd.DataFrame) -> pd.DataFrame:
    return compute_all(synthetic_bars).reset_index(drop=True)


def test_prompt_excludes_future_numerics(enriched: pd.DataFrame) -> None:
    rows = textify_frame(enriched)
    assert len(rows) == len(enriched)
    # For each row, the next-step log return must NOT appear in prompt.
    for i in range(len(rows) - 1):
        next_logret = enriched["logret_1"].iloc[i + 1]
        if pd.isna(next_logret):
            continue
        needle = f"{next_logret:+.5f}"
        assert needle not in rows[i].prompt, (
            f"Row {i} prompt leaks next-step logret {needle}"
        )


def test_label_contains_next_logret(enriched: pd.DataFrame) -> None:
    rows = textify_frame(enriched)
    for i in range(len(rows) - 1):
        next_logret = enriched["logret_1"].iloc[i + 1]
        if pd.isna(next_logret):
            continue
        needle = f"{next_logret:+.5f}"
        assert needle in rows[i].label, (
            f"Row {i} label missing next-step logret {needle}"
        )


def test_final_row_label_is_empty(enriched: pd.DataFrame) -> None:
    rows = textify_frame(enriched)
    assert rows[-1].label == "", "Last row should have no forward label"


def test_prompt_does_not_use_forbidden_words() -> None:
    """The word 'Next' appears in labels only, never in prompts.
    This is a blunt-but-useful check to catch future regressions."""
    import numpy as np

    # Minimal custom df so we don't depend on the synthetic fixture here.
    n = 50
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz="America/New_York")
    df = pd.DataFrame({
        "datetime": idx,
        "symbol": "Q",
        "timeframe": "1day",
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100, 110, n),
        "close_adj": np.linspace(100, 110, n),
        "volume": np.full(n, 1_000_000),
    })
    rows = textify_frame(compute_all(df))
    for r in rows:
        assert not re.search(r"\bNext\b", r.prompt), (
            "Prompt contained the word 'Next' — look-ahead leakage risk"
        )
