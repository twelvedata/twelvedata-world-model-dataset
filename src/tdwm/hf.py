"""Hugging Face push helpers.

Importing `datasets` / `huggingface_hub` is lazy so the rest of the
pipeline can run (and be tested) without them.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


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
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        dd[split] = Dataset.from_pandas(df, preserve_index=False)
    dd.push_to_hub(
        repo_id,
        config_name=config_name,
        private=private,
        token=token,
    )


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

    ds = load_dataset(
        "json",
        data_files={split: str(path) for split, path in split_files.items()},
    )
    ds.push_to_hub(
        repo_id,
        config_name=config_name,
        private=private,
        token=token,
    )
