"""Hugging Face push helpers.

Uploads files directly with `HfApi.upload_file` so we never materialize
multi-GB parquet or jsonl files in RAM. The old `datasets.push_to_hub`
path called `load_dataset("json", ...)` which loaded the whole file into
an Arrow table — fine for MB-scale, catastrophic at ~90 GB.

Layout written to the repo:
    {config_name}/{split}.{parquet|jsonl}

This matches HF's auto-discovery layout for multi-config datasets; users
can `load_dataset(repo_id, name=config_name)` or pass explicit
`data_files=...`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import yaml


def load_hf_config(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def clear_repo_contents(
    repo_id: str,
    *,
    token: str | None = None,
    keep: tuple[str, ...] = (".gitattributes", "README.md"),
) -> int:
    """Delete every file in the HF dataset repo except `keep` entries.

    Done as a single commit so it is atomic. Returns the number of files
    removed. No-op if the repo is already empty.
    """
    token = token or os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN not set")
    from huggingface_hub import HfApi, CommitOperationDelete  # noqa: WPS433

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(
            repo_id, repo_type="dataset", token=token,
        )
    except Exception as exc:  # noqa: BLE001
        # Repo may not exist yet — nothing to clean.
        msg = str(exc).lower()
        if "not found" in msg or "404" in msg:
            return 0
        raise

    targets = [f for f in files if f not in keep]
    if not targets:
        return 0

    ops = [CommitOperationDelete(path_in_repo=f) for f in targets]
    api.create_commit(
        repo_id=repo_id,
        repo_type="dataset",
        operations=ops,
        commit_message=f"Clear {len(targets)} files before fresh push",
        token=token,
    )
    return len(targets)


def _human_size(nbytes: int) -> str:
    gb = nbytes / 1024**3
    if gb >= 1:
        return f"{gb:.2f}GB"
    mb = nbytes / 1024**2
    if mb >= 1:
        return f"{mb:.1f}MB"
    return f"{nbytes/1024:.1f}KB"


def push_config(
    repo_id: str,
    config_name: str,
    split_files: dict[str, Path],
    *,
    private: bool = True,
    token: str | None = None,
) -> None:
    """Stream-upload one config's split files to HF. No in-memory copy."""
    token = token or os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN not set")
    from huggingface_hub import HfApi, create_repo  # noqa: WPS433

    api = HfApi(token=token)
    # Idempotent: creates if missing, returns existing otherwise.
    create_repo(
        repo_id,
        token=token,
        private=private,
        repo_type="dataset",
        exist_ok=True,
    )

    nonempty = {
        s: p for s, p in split_files.items()
        if p.exists() and p.stat().st_size > 0
    }
    if not nonempty:
        print(f"  [skip] {config_name}: all splits empty, nothing to push")
        return

    total_bytes = sum(p.stat().st_size for p in nonempty.values())
    print(
        f"  [push] {config_name}: {len(nonempty)} splits, "
        f"{_human_size(total_bytes)} → {repo_id}"
    )
    t_cfg = time.time()
    for split, path in nonempty.items():
        dest = f"{config_name}/{split}{path.suffix}"
        size = _human_size(path.stat().st_size)
        print(f"    uploading {split} ({size}) → {dest}")
        t0 = time.time()
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
        )
        print(f"    [ok]   {split} in {time.time() - t0:.0f}s")
    print(f"  [done] {config_name} in {time.time() - t_cfg:.0f}s")


def push_dataset_card(
    repo_id: str,
    card_path: Path,
    *,
    private: bool = True,
    token: str | None = None,
) -> None:
    """Upload `card_path` as `README.md` at the repo root."""
    token = token or os.environ.get("HF_TOKEN")
    if token is None:
        raise RuntimeError("HF_TOKEN not set")
    if not card_path.exists():
        raise FileNotFoundError(card_path)
    from huggingface_hub import HfApi, create_repo  # noqa: WPS433

    api = HfApi(token=token)
    create_repo(
        repo_id, token=token, private=private,
        repo_type="dataset", exist_ok=True,
    )
    size = _human_size(card_path.stat().st_size)
    print(f"  [push] README.md ({size}) → {repo_id}")
    api.upload_file(
        path_or_fileobj=str(card_path),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        token=token,
    )


