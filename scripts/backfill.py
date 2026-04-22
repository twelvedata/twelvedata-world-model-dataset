"""Historical backfill.

Usage:
    python scripts/backfill.py [--symbols AAPL,MSFT] [--timeframes 1day,1h]

Fetches the configured history window for each (symbol, timeframe),
computes indicators, joins macro, writes to partitioned parquet, and
records state so the daily updater can resume.

Designed to be safe to re-run: rows are merged by (datetime, symbol,
timeframe) with "last write wins".
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm import fetch as tdfetch                                        # noqa: E402
from tdwm.client import TDClient                                         # noqa: E402
from tdwm.enrich import attach_macro, build_macro_frame                  # noqa: E402
from tdwm.indicators import compute_all                                  # noqa: E402
from tdwm.state import State                                             # noqa: E402
from tdwm.storage import last_macro_datetime, read_macro, write_bars, write_macro  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
STATE_PATH = DATA_ROOT / "_state.json"


def _parse_csv(x: str | None) -> list[str] | None:
    if not x:
        return None
    return [s.strip() for s in x.split(",") if s.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=None,
                        help="CSV override for equity symbols.")
    parser.add_argument("--timeframes", type=str, default=None,
                        help="CSV override for timeframes (e.g. 1day,1h).")
    parser.add_argument("--limit-symbols", type=int, default=None,
                        help="For smoke tests: only fetch first N symbols.")
    args = parser.parse_args()

    syms_cfg = tdfetch.load_symbols()
    sectors = tdfetch.load_sectors()
    tfs = tdfetch.load_timeframes()
    if args.timeframes:
        wanted = set(_parse_csv(args.timeframes) or [])
        tfs = [tf for tf in tfs if tf.interval in wanted]

    equities = _parse_csv(args.symbols) or syms_cfg["equities"]
    macros = syms_cfg["macro"]
    if args.limit_symbols:
        equities = equities[: args.limit_symbols]

    client = TDClient()
    state = State.load(STATE_PATH)
    run_started = datetime.utcnow()

    for tf in tfs:
        start, end = tdfetch.backfill_window(tf)
        print(f"[backfill] {tf.interval} {start}..{end}")
        # Macro caching: only fetch the tail we're missing, then persist.
        macro_bars: dict[str, pd.DataFrame] = {}
        for sym in macros:
            cached = read_macro(DATA_ROOT, tf.interval, sym)
            last_macro = last_macro_datetime(DATA_ROOT, tf.interval, sym)
            if last_macro is not None:
                macro_start = (last_macro - pd.Timedelta(days=2)).date().isoformat()
                print(f"[backfill] macro {sym} {tf.interval}: incremental {macro_start}..{end}")
            else:
                macro_start = start
                print(f"[backfill] macro {sym} {tf.interval}: full {macro_start}..{end}")
            fresh = tdfetch.fetch_macro(client, [sym], tf, start=macro_start, end=end)
            combined_macro = fresh.get(sym, pd.DataFrame())
            if not cached.empty and not combined_macro.empty:
                combined_macro = (
                    pd.concat([cached, combined_macro], ignore_index=True)
                    .drop_duplicates(subset=["datetime"], keep="last")
                    .sort_values("datetime")
                    .reset_index(drop=True)
                )
            elif not cached.empty:
                combined_macro = cached
            if not combined_macro.empty:
                write_macro(combined_macro, DATA_ROOT, tf.interval, sym)
                macro_bars[sym] = combined_macro
        macro_frame = build_macro_frame(macro_bars)

        for sym in equities:
            print(f"[backfill] equity {sym} {tf.interval} …")
            raw = tdfetch.fetch_symbol_history(client, sym, tf, start=start, end=end)
            if raw.empty:
                print(f"  [skip] no data for {sym}")
                continue
            raw["symbol"] = sym
            raw["timeframe"] = tf.interval
            if "close_adj" not in raw.columns:
                raw["close_adj"] = raw["close"]
            enriched = attach_macro(raw, macro_frame, sectors.get(sym), macro_bars)
            final = compute_all(enriched)
            # Re-attach categorical cols clobbered by concat.
            final["symbol"] = sym
            final["timeframe"] = tf.interval
            write_bars(final, DATA_ROOT)
            last_dt = pd.Timestamp(final["datetime"].iloc[-1]).to_pydatetime()
            state.record(
                tf.interval, sym, last_dt,
                rows_written=len(final), run_started=run_started,
            )
            state.save(STATE_PATH)

    print("[backfill] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
