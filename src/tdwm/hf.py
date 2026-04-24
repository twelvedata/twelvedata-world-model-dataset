"""Hugging Face push helpers.

Importing `datasets` / `huggingface_hub` is lazy so the rest of the
pipeline can run (and be tested) without them.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import yaml


def _human_size(path: Path) -> str:
    mb = path.stat().st_size / 1024 / 1024
    return f"{mb:.1f}MB" if mb >= 1 else f"{path.stat().st_size / 1024:.1f}KB"


def load_hf_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def push_parquet_config(
    repo_id: str,
    config_name: str,
    split_files: dict[str, Path],
    *,
    private: bool = True,
    token: str | None = None,
) -> None:
    """Push one config (e.g. "bars") with train/val/test splits to HF."""
    token = token or os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN not set")
    from datasets import Dataset, DatasetDict  # noqa: WPS433 - lazy import
    import pandas as pd

    dd = DatasetDict()
    for split, path in split_files.items():
        if not path.exists() or path.stat().st_size == 0:
            continue
        df = pd.read_parquet(path)
        dd[split] = Dataset.from_pandas(df, preserve_index=False)
    if not dd:
        print(f"  [skip] {config_name}: all splits empty, nothing to push")
        return
    total_rows = sum(len(ds) for ds in dd.values())
    total_size = sum(
        p.stat().st_size for p in split_files.values()
        if p.exists() and p.stat().st_size > 0
    )
    size_str = f"{total_size / 1024 / 1024:.1f}MB"
    print(f"  [push] {config_name}: {total_rows:,} rows, {size_str} → {repo_id}")
    t0 = time.time()
    dd.push_to_hub(
        repo_id,
        config_name=config_name,
        private=private,
        token=token,
    )
    print(f"  [done] {config_name}: uploaded in {time.time() - t0:.0f}s")


def push_jsonl_config(
    repo_id: str,
    config_name: str,
    split_files: dict[str, Path],
    *,
    private: bool = True,
    token: str | None = None,
) -> None:
    token = token or os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN not set")
    from datasets import load_dataset  # noqa: WPS433

    nonempty = {
        split: str(path)
        for split, path in split_files.items()
        if path.exists() and path.stat().st_size > 0
    }
    if not nonempty:
        print(f"  [skip] {config_name}: all splits empty, nothing to push")
        return
    total_size = sum(
        Path(p).stat().st_size for p in nonempty.values()
    )
    size_str = f"{total_size / 1024 / 1024:.1f}MB"
    print(f"  [push] {config_name}: {len(nonempty)} splits, {size_str} → {repo_id}")
    t0 = time.time()
    ds = load_dataset("json", data_files=nonempty)
    total_rows = sum(len(ds[s]) for s in nonempty)
    print(f"  [push] {config_name}: {total_rows:,} rows loaded, uploading …")
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        private=private,
        token=token,
    )
    print(f"  [done] {config_name}: uploaded in {time.time() - t0:.0f}s")
