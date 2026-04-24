"""Assemble a release from the partitioned parquet store.

For each timeframe:
- bars:        write_splits_parquet  → data/release/bars/{tf}/{split}.parquet
- text:        write_splits_jsonl    → data/release/text/{tf}/{split}.jsonl
- trajectories: write_splits_jsonl    → data/release/trajectories/{tf}/{split}.jsonl

--dry-run skips the Hugging Face upload.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm import fetch as tdfetch                          # noqa: E402
from tdwm.release import (                                 # noqa: E402
    build_bars_splits,
    build_text_splits,
    build_trajectory_splits,
    write_splits_jsonl,
    write_splits_parquet,
)
from tdwm.splits import SplitConfig                        # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
RELEASE_ROOT = DATA_ROOT / "release"


def _load_features() -> list[str]:
    with open(REPO_ROOT / "config" / "features.yaml") as f:
        cfg = yaml.safe_load(f)
    return list(cfg["trajectory_state"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip Hugging Face upload.")
    parser.add_argument("--timeframes", type=str, default=None)
    args = parser.parse_args()

    tfs = tdfetch.load_timeframes()
    if args.timeframes:
        wanted = set(s.strip() for s in args.timeframes.split(","))
        tfs = [tf for tf in tfs if tf.interval in wanted]
    syms_cfg = tdfetch.load_symbols()
    equities = syms_cfg["equities"]
    split_cfg = SplitConfig.load()
    features = _load_features()

    for tf in tfs:
        print(f"[release] {tf.interval}")

        bars = build_bars_splits(DATA_ROOT, equities, tf.interval, split_cfg)
        p = write_splits_parquet(bars, RELEASE_ROOT / "bars" / tf.interval)
        print(f"  bars: {p}")

        text = build_text_splits(DATA_ROOT, equities, tf.interval, split_cfg)
        p = write_splits_jsonl(text, RELEASE_ROOT / "text" / tf.interval)
        print(f"  text: {p}")

        trajs = build_trajectory_splits(
            DATA_ROOT, equities, tf.interval,
            feature_names=features,
            windows=tf.trajectory_windows,
            split_cfg=split_cfg,
        )
        p = write_splits_jsonl(trajs, RELEASE_ROOT / "trajectories" / tf.interval)
        print(f"  trajectories: {p}")

    if args.dry_run:
        print("[release] dry-run: skipping Hugging Face upload.")
        return 0

    # Real push.
    import time as _time
    from tdwm.hf import load_hf_config, push_parquet_config, push_jsonl_config
    hf_cfg = load_hf_config(REPO_ROOT / "config" / "hf.yaml")
    configs_total = len(tfs) * 3
    configs_done = 0
    t_start = _time.time()

    for tf in tfs:
        suffix = tf.interval
        bars_files = {
            s: RELEASE_ROOT / "bars" / suffix / f"{s}.parquet"
            for s in ("train", "val", "test")
        }
        configs_done += 1
        print(f"\n[push {configs_done}/{configs_total}] bars_{suffix}")
        push_parquet_config(
            hf_cfg["repo_id"], f"bars_{suffix}", bars_files,
            private=hf_cfg.get("private", True),
        )
        text_files = {
            s: RELEASE_ROOT / "text" / suffix / f"{s}.jsonl"
            for s in ("train", "val", "test")
        }
        configs_done += 1
        print(f"\n[push {configs_done}/{configs_total}] text_{suffix}")
        push_jsonl_config(
            hf_cfg["repo_id"], f"text_{suffix}", text_files,
            private=hf_cfg.get("private", True),
        )
        traj_files = {
            s: RELEASE_ROOT / "trajectories" / suffix / f"{s}.jsonl"
            for s in ("train", "val", "test")
        }
        configs_done += 1
        print(f"\n[push {configs_done}/{configs_total}] trajectories_{suffix}")
        push_jsonl_config(
            hf_cfg["repo_id"], f"trajectories_{suffix}", traj_files,
            private=hf_cfg.get("private", True),
        )
    elapsed = _time.time() - t_start
    print(f"\n[release] all {configs_total} configs pushed to HF in {elapsed:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
