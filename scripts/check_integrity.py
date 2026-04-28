"""Local data-integrity check.

Scans data/bars/ for corruption that would propagate to the HF release
if pushed. Wired into the daily workflow between update_daily.py and
build_release.py — Tier 1 hits exit non-zero and gate the release.

Tier 1 (gating; exits 1 on any hit):
  - Required column missing from a symbol's bars
  - OHLCV / close_adj NaN on a row
  - High/low sanity violations (high < low/open/close, low > open/close)
  - Negative open/high/low/close/volume
  - Macro column NaN on a row dated >= that macro symbol's earliest
    cached datetime — i.e. we had macro data, but attach_macro left
    the row empty anyway (the corruption fixed in fetch.py / update_daily.py)

Tier 2 (warnings; do not gate):
  - Calendar-day gap between consecutive bars exceeds a per-timeframe cap
  - |logret_1| above LOGRET_OUTLIER (single-bar move > 50%)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tdwm import fetch as tdfetch                                     # noqa: E402
from tdwm.storage import read_bars, read_macro                        # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"

LOGRET_OUTLIER = 0.5
GAP_DAYS_BY_TF = {"1day": 7, "1h": 5, "1min": 5}

# Macro column → underlying macro symbol. Mirrors enrich.build_macro_frame
# defaults; sector_logret_1 is per-symbol via sectors.yaml.
DEFAULT_MACRO_MAP = {
    "spy_logret_1": "SPY",
    "vix_level": "VIXY",
    "tlt_logret_1": "TLT",
    "dxy_logret_1": "UUP",
}


def macro_inception(tf: str, sym: str) -> pd.Timestamp | None:
    df = read_macro(DATA_ROOT, tf, sym)
    if df.empty:
        return None
    return pd.Timestamp(df["datetime"].iloc[0])


def _align_tz(ts: pd.Timestamp, ref: pd.Series) -> pd.Timestamp:
    ref_tz = ref.dt.tz
    if ref_tz is not None and ts.tz is None:
        return ts.tz_localize(ref_tz)
    if ref_tz is None and ts.tz is not None:
        return ts.tz_localize(None)
    return ts


def check_symbol(
    df: pd.DataFrame,
    tf: str,
    sym: str,
    sector_etf: str | None,
    macro_inceptions: dict[str, pd.Timestamp | None],
) -> tuple[list[str], list[str]]:
    t1: list[str] = []
    t2: list[str] = []
    n = len(df)
    if n == 0:
        return t1, t2
    dt = pd.to_datetime(df["datetime"])

    # ---- Tier 1: required columns + NaN + negatives + OHLC sanity ----
    for c in ("open", "high", "low", "close", "volume", "close_adj"):
        if c not in df.columns:
            t1.append(f"{tf}/{sym}: missing required column {c}")
            continue
        n_nan = int(df[c].isna().sum())
        if n_nan:
            t1.append(f"{tf}/{sym}: {c} NaN in {n_nan}/{n} rows")

    for c in ("open", "high", "low", "close", "close_adj", "volume"):
        if c in df.columns:
            n_neg = int((df[c] < 0).sum())
            if n_neg:
                t1.append(f"{tf}/{sym}: {c} negative in {n_neg} rows")

    if {"high", "low", "open", "close"}.issubset(df.columns):
        bad = (
            (df["high"] < df["low"])
            | (df["high"] < df["open"])
            | (df["high"] < df["close"])
            | (df["low"] > df["open"])
            | (df["low"] > df["close"])
        )
        n_bad = int(bad.sum())
        if n_bad:
            t1.append(f"{tf}/{sym}: OHLC sanity violations in {n_bad} rows")

    # ---- Tier 1: macro NaN past inception ----
    macros_to_check = dict(DEFAULT_MACRO_MAP)
    if sector_etf:
        macros_to_check["sector_logret_1"] = sector_etf
    for col, macro_sym in macros_to_check.items():
        if col not in df.columns:
            continue
        inception = macro_inceptions.get(macro_sym)
        if inception is None:
            continue
        inception = _align_tz(inception, dt)
        # Logret cols are np.log(close).diff() — the very first bar of
        # the macro series is intrinsically NaN, so the equity row at
        # exactly inception correctly carries NaN. Strict `>` skips
        # that artifact for return columns; level columns (vix_level)
        # use `>=` so we still flag a missing inception-day value.
        if col.endswith("_logret_1"):
            post = dt > inception
        else:
            post = dt >= inception
        nan_post = post & df[col].isna()
        n_nan_post = int(nan_post.sum())
        if n_nan_post:
            first_bad = dt[nan_post].iloc[0]
            last_bad = dt[nan_post].iloc[-1]
            t1.append(
                f"{tf}/{sym}: {col} NaN in {n_nan_post} rows past "
                f"{macro_sym} inception {inception.date()} "
                f"(first={first_bad.date()} last={last_bad.date()})"
            )

    # ---- Tier 2: date gaps ----
    gap_cap = pd.Timedelta(days=GAP_DAYS_BY_TF.get(tf, 7))
    diffs = dt.diff()
    n_gaps = int((diffs > gap_cap).sum())
    if n_gaps:
        worst = diffs.max()
        t2.append(
            f"{tf}/{sym}: {n_gaps} gap(s) > {gap_cap.days}d "
            f"(worst {worst.days}d)"
        )

    # ---- Tier 2: logret outliers ----
    if "logret_1" in df.columns:
        n_out = int((df["logret_1"].abs() > LOGRET_OUTLIER).sum())
        if n_out:
            t2.append(f"{tf}/{sym}: {n_out} |logret_1| > {LOGRET_OUTLIER}")

    return t1, t2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", type=str, default=None,
                        help="CSV override for which equity symbols to scan.")
    parser.add_argument("--timeframes", type=str, default=None,
                        help="CSV override for which timeframes to scan.")
    args = parser.parse_args()

    syms_cfg = tdfetch.load_symbols()
    sectors = tdfetch.load_sectors()
    tfs = tdfetch.load_timeframes()
    if args.timeframes:
        wanted = set(s.strip() for s in args.timeframes.split(","))
        tfs = [tf for tf in tfs if tf.interval in wanted]
    equities = (
        [s.strip() for s in args.symbols.split(",")] if args.symbols
        else syms_cfg["equities"]
    )

    tier1_all: list[str] = []
    tier2_all: list[str] = []

    for tf in tfs:
        macro_syms = set(DEFAULT_MACRO_MAP.values())
        macro_syms.update(v for v in sectors.values() if v)
        inceptions = {s: macro_inception(tf.interval, s) for s in macro_syms}
        missing_inc = [s for s, v in inceptions.items() if v is None]
        if missing_inc:
            print(
                f"[check] {tf.interval}: no macro cache for "
                f"{','.join(sorted(missing_inc))} — those columns won't be "
                "checked for this timeframe"
            )

        print(f"[check] {tf.interval}: scanning {len(equities)} equities")
        for sym in equities:
            df = read_bars(DATA_ROOT, timeframe=tf.interval, symbol=sym)
            if df.empty:
                continue
            t1, t2 = check_symbol(
                df, tf.interval, sym, sectors.get(sym), inceptions
            )
            tier1_all.extend(t1)
            tier2_all.extend(t2)

    print()
    if tier2_all:
        print(f"[warn] {len(tier2_all)} tier-2 finding(s):")
        for line in tier2_all:
            print(f"  {line}")
        print()

    if tier1_all:
        print(f"[FAIL] {len(tier1_all)} tier-1 finding(s):")
        for line in tier1_all:
            print(f"  {line}")
        return 1

    print("[ok] no tier-1 corruption detected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
