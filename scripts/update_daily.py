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
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm import fetch as tdfetch                                        # noqa: E402
from tdwm.client import TDClient                                         # noqa: E402
from tdwm.corporate_actions import check_corporate_actions, clear_symbol_bars  # noqa: E402
from tdwm.enrich import attach_macro, build_macro_frame                  # noqa: E402
from tdwm.indicators import compute_all                                  # noqa: E402
from tdwm.schema import MACRO_COLUMNS                                    # noqa: E402
from tdwm.state import State                                             # noqa: E402
from tdwm.storage import last_macro_datetime, read_macro, write_bars, write_macro  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"
STATE_PATH = DATA_ROOT / "_state.json"
# Marker that flags the next build_release.py run to rebuild ALL splits
# instead of just `test`. Written when a corporate action restates
# historical close_adj on any symbol; build_release.py removes it after
# a successful full push.
FULL_REBUILD_MARKER = DATA_ROOT / "_full_rebuild_required"


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
    run_started = datetime.now(timezone.utc)

    # Detect splits/dividends — re-backfill affected symbols from scratch.
    print("[update] checking corporate actions …")
    dirty = check_corporate_actions(client._sdk, equities, DATA_ROOT)
    for sym in dirty:
        print(f"[update] {sym}: clearing stored bars due to corporate action")
        clear_symbol_bars(DATA_ROOT, sym)
        # Clear state so the equity loop does a full re-backfill.
        for tf in tfs:
            state.entries.pop(State.key(tf.interval, sym), None)
    if dirty:
        state.save(STATE_PATH)
        # Signal the release builder to rebuild ALL splits (train/val
        # close_adj history just shifted under our feet).
        FULL_REBUILD_MARKER.write_text(",".join(sorted(dirty)) + "\n")
        print(f"[update] wrote {FULL_REBUILD_MARKER.name}: full rebuild scheduled")

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
        macro_end = datetime.now(timezone.utc).date().isoformat()
        print(f"[update] {tf.interval}: macro window {macro_start}..{macro_end}")
        # Macro caching: only fetch tail that's missing, then persist.
        macro_bars: dict[str, pd.DataFrame] = {}
        for sym in macros:
            cached = read_macro(DATA_ROOT, tf.interval, sym)
            last_macro = last_macro_datetime(DATA_ROOT, tf.interval, sym)
            if last_macro is not None:
                sym_start = (last_macro - pd.Timedelta(days=2)).date().isoformat()
            else:
                sym_start = macro_start
            fresh = tdfetch.fetch_macro(client, [sym], tf, start=sym_start, end=macro_end)
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
                # Stored rows already carry macro columns from their
                # original write; strip them so attach_macro can re-apply
                # without merge_asof suffix collisions on the combined
                # frame. Indicator columns are also recomputed below, so
                # drop those too to avoid the same kind of collision if
                # we ever re-enrich indicators via merge.
                drop_cols = [c for c in MACRO_COLUMNS if c in existing.columns]
                if drop_cols:
                    existing = existing.drop(columns=drop_cols)
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
