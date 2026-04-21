"""Idempotent daily updater.

Rules:
1. For every (symbol, timeframe) pair, look up the last known datetime
   from state. Fetch from (last - 2 days) to today. This re-fetches the
   previous trading day so late prints / restatements overwrite earlier
   rows (storage.write_bars merges by (datetime, symbol, timeframe)).
2. If state has no record for a pair, fall through to the full history
   window from timeframes.yaml — the job self-heals from empty state.
3. Macro tickers are fetched once per timeframe and reused for all
   equities in that timeframe.

Safe to run multiple times per day. Safe to run after a missed day.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm import fetch as tdfetch                           # noqa: E402
from tdwm.client import TDClient                            # noqa: E402
from tdwm.enrich import attach_macro, build_macro_frame     # noqa: E402
from tdwm.indicators import compute_all                     # noqa: E402
from tdwm.state import State                                # noqa: E402
from tdwm.storage import write_bars                         # noqa: E402


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
        # We need macro history going back to the earliest equity last_known
        # so that merge_asof has something to match. Compute min(last_known)
        # across equities for this timeframe.
        per_sym_last = {
            sym: state.get_last(tf.interval, sym) for sym in equities
        }
        valid_lasts = [v for v in per_sym_last.values() if v is not None]
        if valid_lasts:
            macro_start = (min(valid_lasts) - pd.Timedelta(days=5)).date().isoformat()
        else:
            # No state → fall back to full history.
            macro_start, _ = tdfetch.backfill_window(tf)
        macro_end = datetime.utcnow().date().isoformat()
        print(f"[update] {tf.interval}: macro {macro_start}..{macro_end}")
        macro_bars = tdfetch.fetch_macro(
            client, macros, tf, start=macro_start, end=macro_end
        )
        macro_frame = build_macro_frame(macro_bars)

        for sym in equities:
            last = per_sym_last[sym]
            if last is None:
                start, end = tdfetch.backfill_window(tf)
                print(f"[update] {sym} {tf.interval}: no state → full history {start}..{end}")
            else:
                start, end = tdfetch.incremental_window(last)
                print(f"[update] {sym} {tf.interval}: incremental {start}..{end} (re-fetches prev day)")
            raw = tdfetch.fetch_symbol_history(
                client, sym, tf, start=start, end=end
            )
            if raw.empty:
                print(f"  [skip] no rows returned for {sym}")
                continue
            raw["symbol"] = sym
            raw["timeframe"] = tf.interval
            if "close_adj" not in raw.columns:
                raw["close_adj"] = raw["close"]
            # IMPORTANT: to compute indicators correctly on a short
            # incremental window, we need history too. Pull existing
            # rows, merge, recompute indicators on the combined frame,
            # then write (storage dedupes).
            from tdwm.storage import read_bars
            existing = read_bars(DATA_ROOT, timeframe=tf.interval, symbol=sym)
            if not existing.empty:
                # Drop rows we're about to replace (last 3 days of existing)
                cutoff = pd.Timestamp(raw["datetime"].iloc[0])
                existing = existing.loc[pd.to_datetime(existing["datetime"]) < cutoff]
                combined = pd.concat([existing, raw], ignore_index=True)
            else:
                combined = raw
            combined = combined.sort_values("datetime").reset_index(drop=True)
            enriched = attach_macro(combined, macro_frame, sectors.get(sym), macro_bars)
            final = compute_all(enriched)
            final["symbol"] = sym
            final["timeframe"] = tf.interval
            # Only write the tail — older rows are unchanged.
            if last is not None:
                tail = final.loc[pd.to_datetime(final["datetime"]) >= pd.Timestamp(last) - pd.Timedelta(days=3)]
            else:
                tail = final
            write_bars(tail, DATA_ROOT)
            last_dt = pd.Timestamp(final["datetime"].iloc[-1]).to_pydatetime()
            state.record(
                tf.interval, sym, last_dt,
                rows_written=len(tail), run_started=run_started,
            )
            state.save(STATE_PATH)

    print("[update] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
