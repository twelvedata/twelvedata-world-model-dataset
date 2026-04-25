"""One-shot repair: drop _x / _y suffixed columns from bar parquets.

A pre-fix `update_daily.py` run successfully wrote merge_asof results
that duplicated macro columns into the partitioned parquet store. The
canonical column names are still present (e.g. `spy_logret_1`); the
suffixed copies (`spy_logret_1_x`, `spy_logret_1_y`) are stale at best
and pathologically slow to format at worst (PEP daily strftime hangs).

This script walks `data/bars/**/*.parquet`, drops any `_x`/`_y` columns
where the unsuffixed canonical exists, and rewrites the file in place.
If only a suffixed variant exists, the `_x` copy is renamed to the
canonical name (preferring the original-left-side merge value) and the
`_y` copy is dropped.

Idempotent — re-running on a clean store is a no-op.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "data"


def _atomic_write(df: pd.DataFrame, dest: Path) -> None:
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, tmp, compression="zstd")
    tmp.replace(dest)


def repair_one(path: Path, *, dry_run: bool = False) -> tuple[bool, list[str]]:
    """Return (changed, removed_columns)."""
    df = pd.read_parquet(path)
    cols = list(df.columns)
    removed: list[str] = []
    renamed: list[tuple[str, str]] = []

    for c in cols:
        if c.endswith("_x") or c.endswith("_y"):
            base = c[:-2]
            if base in df.columns:
                # Canonical exists — drop the suffixed copy.
                removed.append(c)
            else:
                # Canonical missing. Prefer _x (left/original) over _y.
                if c.endswith("_x"):
                    renamed.append((c, base))
                else:
                    # _y but no canonical and no _x — promote the _y so
                    # we don't lose the data entirely.
                    if not any(rn[1] == base for rn in renamed):
                        renamed.append((c, base))
                    else:
                        removed.append(c)

    if not removed and not renamed:
        return (False, [])

    if removed:
        df = df.drop(columns=removed)
    for src, dst in renamed:
        df = df.rename(columns={src: dst})

    if not dry_run:
        _atomic_write(df, path)

    return (True, removed + [f"{src}->{dst}" for src, dst in renamed])


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Report changes but don't write.")
    parser.add_argument("--root", type=Path, default=DATA_ROOT,
                        help=f"Data root (default: {DATA_ROOT})")
    args = parser.parse_args()

    bars_root = args.root / "bars"
    if not bars_root.exists():
        print(f"[repair] no bars dir at {bars_root}")
        return 0

    paths = sorted(bars_root.glob("*/symbol=*/year=*/part.parquet"))
    print(f"[repair] scanning {len(paths)} parquet files under {bars_root}")
    if args.dry_run:
        print("[repair] DRY RUN — no files will be written")

    changed_files = 0
    for p in paths:
        try:
            changed, ops = repair_one(p, dry_run=args.dry_run)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {p}: {exc}")
            continue
        if changed:
            changed_files += 1
            rel = p.relative_to(args.root)
            print(f"  [fix] {rel}: {', '.join(ops)}")

    print(f"\n[repair] {changed_files} file(s) repaired "
          f"(out of {len(paths)} scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
