"""Assemble a release by streaming per-symbol through three sinks.

Bars / text / trajectories are written incrementally as each symbol is
processed, so peak memory stays proportional to one symbol's data rather
than the whole universe. Writers emit the same files as before:

    data/release/bars/{tf}/{split}.parquet
    data/release/text/{tf}/{split}.jsonl
    data/release/trajectories/{tf}/{split}.jsonl
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .splits import SplitConfig
from .storage import read_bars
from .textify import textify_frame
from .trajectories import iter_trajectories, trajectory_to_record


SPLITS = ("train", "val", "test")


# ---------- sinks ----------

class SplitJsonlSink:
    """Three file handles, one per split. `.add(split, record)` writes
    exactly one JSON line. Opens lazily on first use so empty splits
    don't produce zero-byte files."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, object] = {}
        self.paths: dict[str, Path] = {}
        self.counts: dict[str, int] = {s: 0 for s in SPLITS}

    def add(self, split: str, record: dict) -> None:
        if split not in SPLITS:
            return
        fh = self._handles.get(split)
        if fh is None:
            dest = self.out_dir / f"{split}.jsonl"
            fh = open(dest, "w")
            self._handles[split] = fh
            self.paths[split] = dest
        fh.write(json.dumps(record))
        fh.write("\n")
        self.counts[split] += 1

    def close(self) -> dict[str, Path]:
        for fh in self._handles.values():
            fh.close()
        self._handles.clear()
        return self.paths

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SplitParquetRecordSink:
    """Accepts individual record dicts, buffers them per split, and
    flushes to a `ParquetWriter` in fixed-size batches.

    Useful for streaming emitters (e.g. trajectories) where we don't want
    to build a full DataFrame up front. Schema is locked to the first
    flushed batch per split; subsequent batches are cast to match.

    `batch_size` is the number of records buffered per split before a
    write. Bigger = better parquet compression but higher peak RAM.
    """

    def __init__(
        self,
        out_dir: Path,
        *,
        batch_size: int = 64,
        compression: str = "zstd",
    ):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.compression = compression
        self._buffers: dict[str, list[dict]] = {s: [] for s in SPLITS}
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._schemas: dict[str, pa.Schema] = {}
        self.paths: dict[str, Path] = {}
        self.counts: dict[str, int] = {s: 0 for s in SPLITS}

    def add(self, split: str, record: dict) -> None:
        if split not in SPLITS:
            return
        buf = self._buffers[split]
        buf.append(record)
        self.counts[split] += 1
        if len(buf) >= self.batch_size:
            self._flush(split)

    def _flush(self, split: str) -> None:
        buf = self._buffers[split]
        if not buf:
            return
        df = pd.DataFrame(buf)
        table = pa.Table.from_pandas(df, preserve_index=False)
        writer = self._writers.get(split)
        if writer is None:
            dest = self.out_dir / f"{split}.parquet"
            writer = pq.ParquetWriter(
                dest, table.schema, compression=self.compression,
            )
            self._writers[split] = writer
            self._schemas[split] = table.schema
            self.paths[split] = dest
        else:
            table = table.cast(self._schemas[split])
        writer.write_table(table)
        buf.clear()

    def close(self) -> dict[str, Path]:
        for split in SPLITS:
            self._flush(split)
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        return self.paths

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SplitParquetSink:
    """Three ParquetWriters, one per split, opened lazily. `.add(df,
    split_cfg)` assigns each row to a split and appends as a record
    batch. Uses the first batch's arrow schema and casts subsequent
    batches to match — incompatible schemas raise."""

    def __init__(self, out_dir: Path):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._schemas: dict[str, pa.Schema] = {}
        self.paths: dict[str, Path] = {}
        self.counts: dict[str, int] = {s: 0 for s in SPLITS}

    def add(self, df: pd.DataFrame, split_cfg: SplitConfig) -> None:
        if df.empty:
            return
        splits = df["datetime"].map(lambda ts: split_cfg.assign(pd.Timestamp(ts)))
        for name in SPLITS:
            mask = (splits == name).values
            if not mask.any():
                continue
            sub = df.loc[mask].reset_index(drop=True)
            table = pa.Table.from_pandas(sub, preserve_index=False)
            writer = self._writers.get(name)
            if writer is None:
                dest = self.out_dir / f"{name}.parquet"
                writer = pq.ParquetWriter(dest, table.schema, compression="zstd")
                self._writers[name] = writer
                self._schemas[name] = table.schema
                self.paths[name] = dest
            else:
                # Keep schema stable across symbols.
                table = table.cast(self._schemas[name])
            writer.write_table(table)
            self.counts[name] += len(sub)

    def close(self) -> dict[str, Path]:
        for w in self._writers.values():
            w.close()
        self._writers.clear()
        return self.paths

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------- streaming build ----------

def _fmt_counts(counts: dict[str, int]) -> str:
    return "train={train:,} val={val:,} test={test:,}".format(**counts)


def stream_release_for_timeframe(
    root: Path,
    symbols: list[str],
    timeframe: str,
    feature_names: list[str],
    windows: list[dict],
    split_cfg: SplitConfig,
    out_root: Path,
    *,
    log: Callable[[str], None] = print,
) -> dict[str, dict[str, Path]]:
    """Stream bars, text, and trajectories to disk one symbol at a time.

    Returns `{"bars": {split: path}, "text": {...}, "trajectories": {...}}`.
    """
    bars_dir = out_root / "bars" / timeframe
    text_dir = out_root / "text" / timeframe
    traj_dir = out_root / "trajectories" / timeframe

    n_sym = len(symbols)
    tf_started = time.perf_counter()

    with SplitParquetSink(bars_dir) as bars_sink, \
         SplitJsonlSink(text_dir) as text_sink, \
         SplitParquetRecordSink(traj_dir, batch_size=64) as traj_sink:

        for i, sym in enumerate(symbols, 1):
            t0 = time.perf_counter()
            df = read_bars(root, timeframe=timeframe, symbol=sym)
            if df.empty:
                log(f"  [{i}/{n_sym}] {sym}: skip (no data)")
                continue

            # --- bars
            t_bars = time.perf_counter()
            bars_sink.add(df, split_cfg)
            dt_bars = time.perf_counter() - t_bars

            # --- text
            t_text = time.perf_counter()
            text_added = 0
            for row in textify_frame(df, verbose=True):
                split = split_cfg.assign(pd.Timestamp(row.as_of))
                text_sink.add(split, asdict(row))
                text_added += 1
            dt_text = time.perf_counter() - t_text

            # --- trajectories (all configured windows)
            t_traj = time.perf_counter()
            traj_added = 0
            for w in windows:
                for t in iter_trajectories(
                    df,
                    feature_names,
                    window=int(w["size"]),
                    stride=int(w["stride"]),
                ):
                    first_ts = pd.Timestamp(t.timestamps[0])
                    last_ts = pd.Timestamp(t.timestamps[-1])
                    split = split_cfg.split_of_trajectory(first_ts, last_ts)
                    if split is None:
                        continue
                    t.split = split
                    traj_sink.add(split, trajectory_to_record(t))
                    traj_added += 1
            dt_traj = time.perf_counter() - t_traj

            dt = time.perf_counter() - t0
            log(
                f"  [{i}/{n_sym}] {sym}: "
                f"bars={len(df):,} ({dt_bars:.1f}s) "
                f"text={text_added:,} ({dt_text:.1f}s) "
                f"trajs={traj_added:,} ({dt_traj:.1f}s) "
                f"total={dt:.1f}s"
            )

    elapsed = time.perf_counter() - tf_started
    log(f"  [totals] bars {_fmt_counts(bars_sink.counts)}")
    log(f"  [totals] text {_fmt_counts(text_sink.counts)}")
    log(f"  [totals] trajs {_fmt_counts(traj_sink.counts)}")
    log(f"  [done]   {timeframe} in {elapsed:.1f}s")

    return {
        "bars": bars_sink.paths,
        "text": text_sink.paths,
        "trajectories": traj_sink.paths,
    }
