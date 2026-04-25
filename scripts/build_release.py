"""Assemble a release from the partitioned parquet store.

For each timeframe:
- bars:         data/release/bars/{tf}/{split}.parquet
- text:         data/release/text/{tf}/{split}.jsonl
- trajectories: data/release/trajectories/{tf}/{split}.jsonl

All three sinks stream per symbol so peak memory stays bounded. Progress
is logged per symbol with counts and per-stage timings.

--dry-run skips the Hugging Face upload.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm import fetch as tdfetch                          # noqa: E402
from tdwm.release import stream_release_for_timeframe      # noqa: E402
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
    parser.add_argument("--push-only", action="store_true",
                        help="Skip the build phase; push existing files in "
                             "data/release/ straight to HF.")
    parser.add_argument("--card-only", action="store_true",
                        help="Skip build and config uploads; only push "
                             "config/dataset_card.md as README.md.")
    parser.add_argument("--clean", action="store_true",
                        help="Delete every file in the HF dataset repo "
                             "before uploading. Requires confirmation.")
    parser.add_argument("--yes", action="store_true",
                        help="Skip the --clean confirmation prompt.")
    parser.add_argument("--timeframes", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run and args.push_only:
        parser.error("--dry-run and --push-only are mutually exclusive.")
    if args.dry_run and args.card_only:
        parser.error("--dry-run and --card-only are mutually exclusive.")
    if args.push_only and args.card_only:
        parser.error("--push-only and --card-only are mutually exclusive.")
    if args.clean and args.card_only:
        parser.error("--clean would wipe data files; not allowed with "
                     "--card-only.")
    if args.clean and args.dry_run:
        parser.error("--clean has no effect with --dry-run.")

    tfs = tdfetch.load_timeframes()
    if args.timeframes:
        wanted = set(s.strip() for s in args.timeframes.split(","))
        tfs = [tf for tf in tfs if tf.interval in wanted]
    syms_cfg = tdfetch.load_symbols()
    equities = syms_cfg["equities"]
    split_cfg = SplitConfig.load()
    features = _load_features()

    if args.card_only:
        from tdwm.hf import load_hf_config, push_dataset_card  # noqa: WPS433
        hf_cfg = load_hf_config(REPO_ROOT / "config" / "hf.yaml")
        card_path = REPO_ROOT / "config" / "dataset_card.md"
        if not card_path.exists():
            print(f"[error] no dataset card at {card_path}")
            return 1
        print("[push] dataset card (README.md)")
        push_dataset_card(
            hf_cfg["repo_id"], card_path,
            private=hf_cfg.get("private", True),
        )
        return 0

    if args.push_only:
        print("[release] push-only: skipping build, using existing "
              f"files in {RELEASE_ROOT}")
    else:
        release_started = time.perf_counter()
        for tf in tfs:
            print(f"[release] {tf.interval}  ({len(equities)} equities)")
            stream_release_for_timeframe(
                DATA_ROOT,
                equities,
                tf.interval,
                feature_names=features,
                windows=tf.trajectory_windows,
                split_cfg=split_cfg,
                out_root=RELEASE_ROOT,
            )
        elapsed = time.perf_counter() - release_started
        print(f"\n[release] all timeframes built in {elapsed:.0f}s")

    if args.dry_run:
        print("[release] dry-run: skipping Hugging Face upload.")
        return 0

    # Real push.
    from tdwm.hf import (
        clear_repo_contents,
        load_hf_config,
        push_dataset_card,
        push_jsonl_config,
        push_parquet_config,
    )
    hf_cfg = load_hf_config(REPO_ROOT / "config" / "hf.yaml")

    if args.clean:
        print(f"[clean] about to delete ALL files in {hf_cfg['repo_id']}")
        if not args.yes:
            resp = input("  type 'yes' to continue: ").strip().lower()
            if resp != "yes":
                print("[clean] aborted.")
                return 1
        removed = clear_repo_contents(hf_cfg["repo_id"])
        print(f"[clean] removed {removed} files from {hf_cfg['repo_id']}")

    configs_total = len(tfs) * 3
    configs_done = 0
    t_push = time.perf_counter()

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
            s: RELEASE_ROOT / "trajectories" / suffix / f"{s}.parquet"
            for s in ("train", "val", "test")
        }
        configs_done += 1
        print(f"\n[push {configs_done}/{configs_total}] trajectories_{suffix}")
        push_parquet_config(
            hf_cfg["repo_id"], f"trajectories_{suffix}", traj_files,
            private=hf_cfg.get("private", True),
        )
    push_elapsed = time.perf_counter() - t_push
    print(f"\n[release] all {configs_total} configs pushed to HF in {push_elapsed:.0f}s")

    card_path = REPO_ROOT / "config" / "dataset_card.md"
    if card_path.exists():
        print("\n[push] dataset card (README.md)")
        push_dataset_card(
            hf_cfg["repo_id"], card_path,
            private=hf_cfg.get("private", True),
        )
    else:
        print(f"[warn] no dataset card at {card_path}; skipping README upload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
