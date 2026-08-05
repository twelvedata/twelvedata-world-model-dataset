"""Fail when the published daily bars are older than --max-age-days.

The pipeline once stopped silently: GitHub disabled the cron for repo
inactivity, so no job ever failed and failure-only alerting stayed quiet.
This compares data on the Hub against the wall clock instead of job status.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm.hf import load_hf_config                                    # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
# Smallest release file that moves on every trading day (~5 MB).
CHECKED_FILE = "bars_1day/test.parquet"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-age-days", type=int, default=4,
                        help="Trading days can gap over weekends and holidays.")
    args = parser.parse_args()

    hf_cfg = load_hf_config(REPO_ROOT / "config" / "hf.yaml")
    local = hf_hub_download(
        hf_cfg["repo_id"], CHECKED_FILE, repo_type="dataset",
    )
    datetimes = pq.read_table(local, columns=["datetime"]).column("datetime")
    last_bar = max(datetimes.to_pylist()).astimezone(timezone.utc)
    age_days = (datetime.now(timezone.utc) - last_bar).days
    print(f"[freshness] {CHECKED_FILE}: last bar {last_bar.isoformat()} ({age_days}d ago)")

    if age_days > args.max_age_days:
        print(
            f"[freshness] STALE: no new bars for {age_days} days "
            f"(limit {args.max_age_days}) — daily update is not reaching the Hub.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
