"""Optional candlestick chart generation.

Kept behind an import guard so the main pipeline doesn't hard-require
plotly/kaleido. The daily updater only calls into here when
--with-charts is passed.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def render_candlestick_png(df: pd.DataFrame, dest: Path, *, title: str = "") -> Path:
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - optional
        raise RuntimeError("plotly is required for chart rendering") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["datetime"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
            )
        ]
    )
    fig.update_layout(
        title=title,
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        width=800,
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.write_image(dest)
    return dest
