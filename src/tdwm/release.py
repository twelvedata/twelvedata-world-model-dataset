"""Assemble a release: read partitioned bars, build text + trajectories,
assign time-based splits, write per-split files ready for HF upload.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml

from . import fetch as fetch_cfg
from .schema import Trajectory, TextRow
from .splits import SplitConfig
from .storage import read_bars
from .textify import textify_frame, textrows_to_records
from .trajectories import build_trajectories, trajectories_to_records


def _iter_symbol_frames(
    root: Path,
    symbols: Iterable[str],
    timeframe: str,
) -> Iterable[pd.DataFrame]:
    for sym in symbols:
        df = read_bars(root, timeframe=timeframe, symbol=sym)
        if not df.empty:
            yield df


def _rows_to_split_tables(
    df_by_split: dict[str, list[pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for split, frames in df_by_split.items():
        if frames:
            out[split] = pd.concat(frames, ignore_index=True)
    return out


def build_bars_splits(
    root: Path,
    symbols: Iterable[str],
    timeframe: str,
    split_cfg: SplitConfig,
) -> dict[str, pd.DataFrame]:
    buckets: dict[str, list[pd.DataFrame]] = {"train": [], "val": [], "test": []}
    for df in _iter_symbol_frames(root, symbols, timeframe):
        splits = df["datetime"].apply(lambda ts: split_cfg.assign(pd.Timestamp(ts)))
        for name in buckets.keys():
            mask = (splits == name).values
            if mask.any():
                buckets[name].append(df.loc[mask].reset_index(drop=True))
    return _rows_to_split_tables(buckets)


def build_text_splits(
    root: Path,
    symbols: Iterable[str],
    timeframe: str,
    split_cfg: SplitConfig,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for df in _iter_symbol_frames(root, symbols, timeframe):
        rows = textify_frame(df, verbose=True)
        for r in rows:
            split = split_cfg.assign(pd.Timestamp(r.as_of))
            buckets[split].append(asdict(r))
    return buckets


def build_trajectory_splits(
    root: Path,
    symbols: Iterable[str],
    timeframe: str,
    feature_names: list[str],
    windows: list[dict],
    split_cfg: SplitConfig,
) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for df in _iter_symbol_frames(root, symbols, timeframe):
        for w in windows:
            def _split_of(last_ts, df=df):
                # `df` captured so we can check the first timestamp of the
                # window for strict-no-crossing.
                return None  # filled below
            # We need the first_ts too; rebuild here instead of split_of
            # callback to avoid coupling trajectories.py to SplitConfig.
            trajs = build_trajectories(
                df, feature_names, window=int(w["size"]), stride=int(w["stride"])
            )
            for t in trajs:
                first_ts = pd.Timestamp(t.timestamps[0])
                last_ts = pd.Timestamp(t.timestamps[-1])
                split = split_cfg.split_of_trajectory(first_ts, last_ts)
                if split is None:
                    continue
                t.split = split
                buckets[split].extend(trajectories_to_records([t]))
    return buckets


def write_splits_parquet(
    tables: dict[str, pd.DataFrame], out_dir: Path
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, df in tables.items():
        dest = out_dir / f"{split}.parquet"
        df.to_parquet(dest, index=False, compression="zstd")
        paths[split] = dest
    return paths


def write_splits_jsonl(
    records: dict[str, list[dict]], out_dir: Path
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for split, rows in records.items():
        dest = out_dir / f"{split}.jsonl"
        with open(dest, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        paths[split] = dest
    return paths
