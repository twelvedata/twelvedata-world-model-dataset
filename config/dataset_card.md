---
license: mit
language:
  - en
task_categories:
  - time-series-forecasting
  - text-generation
  - reinforcement-learning
tags:
  - finance
  - equities
  - ohlcv
  - technical-indicators
  - world-model
pretty_name: Twelve Data World Model Dataset
size_categories:
  - 10M<n<100M
configs:
  - config_name: bars_1day
    data_files:
      - split: train
        path: bars_1day/train.parquet
      - split: validation
        path: bars_1day/val.parquet
      - split: test
        path: bars_1day/test.parquet
  - config_name: bars_1h
    data_files:
      - split: train
        path: bars_1h/train.parquet
      - split: validation
        path: bars_1h/val.parquet
      - split: test
        path: bars_1h/test.parquet
  - config_name: bars_1min
    data_files:
      - split: train
        path: bars_1min/train.parquet
      - split: validation
        path: bars_1min/val.parquet
      - split: test
        path: bars_1min/test.parquet
  - config_name: text_1day
    data_files:
      - split: train
        path: text_1day/train.jsonl
      - split: validation
        path: text_1day/val.jsonl
      - split: test
        path: text_1day/test.jsonl
  - config_name: text_1h
    data_files:
      - split: train
        path: text_1h/train.jsonl
      - split: validation
        path: text_1h/val.jsonl
      - split: test
        path: text_1h/test.jsonl
  - config_name: text_1min
    data_files:
      - split: train
        path: text_1min/train.jsonl
      - split: validation
        path: text_1min/val.jsonl
      - split: test
        path: text_1min/test.jsonl
  - config_name: trajectories_1day
    data_files:
      - split: train
        path: trajectories_1day/train.parquet
      - split: validation
        path: trajectories_1day/val.parquet
      - split: test
        path: trajectories_1day/test.parquet
  - config_name: trajectories_1h
    data_files:
      - split: train
        path: trajectories_1h/train.parquet
      - split: validation
        path: trajectories_1h/val.parquet
      - split: test
        path: trajectories_1h/test.parquet
  - config_name: trajectories_1min
    data_files:
      - split: train
        path: trajectories_1min/train.parquet
      - split: validation
        path: trajectories_1min/val.parquet
      - split: test
        path: trajectories_1min/test.parquet
---

# Twelve Data World Model Dataset

A multi-modal financial time-series dataset built from [Twelve Data](https://twelvedata.com/)
market data. Each timeframe is published in three parallel views:

- **`bars_*`** — OHLCV bars enriched with causal technical indicators and macro
  context, in Parquet.
- **`text_*`** — instruction-tuning prompts/labels derived from the bars, in
  JSONL.
- **`trajectories_*`** — fixed-length rolling windows of state vectors plus
  next-state pairs, suitable for world-model / sequence-model training, in
  Parquet.

All views cover the same symbol universe and share the same time-based
train / validation / test splits.

## Splits

| Split | Date range |
| --- | --- |
| `train` | up to 2023-12-31 |
| `validation` | 2024-01-01 → 2024-12-31 |
| `test` | 2025-01-01 → {TEST_END_DATE} |

Splits are assigned by timestamp. For `trajectories_*`, a window is dropped if
it crosses a split boundary (so train and val never overlap). Datetimes are
tz-aware in `America/New_York`.

## Symbol universe

51 large-cap US equities spread across sectors (Tech, Financials, Healthcare,
Consumer, Industrials/Energy/Materials, Communication). Macro context tickers
(SPY, QQQ, VIXY, TLT, sector SPDRs, etc.) are joined as columns onto every
equity row — they are not first-class trainable symbols.

See [`config/symbols.yaml`](https://github.com/) in the source repo for the
exact list.

## Timeframes

The pipeline requests the full history available from Twelve Data per symbol;
actual depth varies by ticker and timeframe and is bounded by the vendor's
historical limits. As a rough guide:

| Interval | Typical depth (older names like AAPL/MSFT) | Trajectory windows (size / stride) |
| --- | --- | --- |
| `1day` | several decades back to listing | 30/5, 60/10, 120/20 |
| `1h` | a few years | 24/6, 120/24 |
| `1min` | a few years | 390/195 (one US session), 1950/390 (one week) |

Newer listings naturally start later. Each row's `datetime` reflects what the
vendor returned — there is no synthetic backfill before a symbol's inception.

## `bars_*` schema

OHLCV plus deterministic, causal indicators and macro joins. Each row is one
bar for one symbol at one timeframe.

### Core
| column | type | description |
| --- | --- | --- |
| `datetime` | timestamp[ns, America/New_York] | bar timestamp (interval-start) |
| `symbol` | string | equity ticker |
| `timeframe` | string | `1day`, `1h`, or `1min` |
| `open`, `high`, `low`, `close` | float64 | OHLC prices |
| `volume` | float64 | traded volume |
| `close_adj` | float64 | split- and dividend-adjusted close (== `close` when no events) |

### Returns
| column | description |
| --- | --- |
| `ret_1` | 1-bar simple return |
| `logret_1` | 1-bar log return |
| `ret_5` | 5-bar simple return |
| `logret_20` | 20-bar log return |

### Volatility
| column | description |
| --- | --- |
| `rv_5`, `rv_20`, `rv_60` | realized volatility (stdev of log returns) over 5/20/60 bars |
| `atr_14` | Average True Range, 14 bars |

### Momentum
| column | description |
| --- | --- |
| `rsi_14` | Relative Strength Index, 14 bars |
| `macd`, `macd_signal`, `macd_hist` | MACD(12,26,9) line, signal, histogram |
| `mom_10` | 10-bar momentum |

### Volume
| column | description |
| --- | --- |
| `obv` | On-Balance Volume |
| `vol_z_20` | volume z-score over 20 bars |

### Bands
| column | description |
| --- | --- |
| `bb_mid`, `bb_up`, `bb_lo` | Bollinger Bands(20, 2σ) middle/upper/lower |
| `bb_pctb` | %B position within the band |

### Macro context (joined by date)
| column | description |
| --- | --- |
| `spy_logret_1` | SPY 1-bar log return |
| `vix_level` | VIX level (proxied via VIXY) |
| `tlt_logret_1` | TLT (20+ yr Treasury) 1-bar log return |
| `dxy_logret_1` | US Dollar index 1-bar log return (UUP proxy) |
| `sector_logret_1` | sector ETF 1-bar log return matching the symbol's sector |

**Causality invariant**: every indicator and macro column at row `t` uses only
information from rows `≤ t`. Tests in `tests/test_indicators_causal.py`
enforce this in CI.

## `text_*` schema

Instruction-tuning records derived from `bars_*`.

| field | type | description |
| --- | --- | --- |
| `symbol` | string | equity ticker |
| `timeframe` | string | `1day` / `1h` / `1min` |
| `as_of` | string (ISO datetime) | the bar's timestamp; **everything in `prompt` is dated ≤ this** |
| `prompt` | string | natural-language description of the bar + already-observed indicators |
| `label` | string | next-bar outcome (direction + log return); empty for the final bar |
| `meta` | object | small bag for provenance (e.g. row index) |

**Leak-prevention invariant**: `prompt` never references information dated
strictly after `as_of`. Only `label` carries the next-step outcome. This is
enforced in `tests/test_textify.py`.

## `trajectories_*` schema

Rolling windows over the bar series, suitable for world-model and
sequence-model training. Each row is one window.

| field | type | description |
| --- | --- | --- |
| `trajectory_id` | string | stable id (`{symbol}_{timeframe}_{window}_{start_ts}`) |
| `symbol` | string | equity ticker |
| `timeframe` | string | `1day` / `1h` / `1min` |
| `feature_names` | list[string] | column order of the state vector (length F) |
| `timestamps` | list[string] | T tz-aware ISO datetimes |
| `states` | list[list[float64]] | shape `(T, F)` |
| `next_states` | list[list[float64]] | shape `(T, F)`, shifted by one step |
| `rewards_logret` | list[float64] \| null | optional scalar reward stream (1-bar log return); `null` means reward-agnostic |
| `split` | string | `train` / `val` / `test` |

The state vector is fixed across all trajectories within the dataset:

```
open, high, low, close_adj, volume,
logret_1, rv_20, rsi_14, macd, macd_signal, macd_hist, atr_14, bb_pctb,
obv, vol_z_20,
spy_logret_1, vix_level, tlt_logret_1, dxy_logret_1, sector_logret_1
```

## How to load

```python
from datasets import load_dataset

# OHLCV + indicators, daily bars
bars = load_dataset("twelvedata/financial-world-model", "bars_1day")

# Instruction-tuning text, hourly
text = load_dataset("twelvedata/financial-world-model", "text_1h")

# Trajectories for world-model training
traj = load_dataset("twelvedata/financial-world-model", "trajectories_1day")
```

For ad-hoc analytics, the parquet files are queryable directly with DuckDB:

```sql
SELECT symbol, datetime, close, rsi_14
FROM 'bars_1day/test.parquet'
WHERE symbol = 'AAPL'
ORDER BY datetime DESC
LIMIT 10;
```

> **Tip on the in-browser SQL Console**: DuckDB-WASM streams parquet over
> HTTPS without local caching, and `1min` configs are millions of rows. Always
> project explicit columns (`SELECT symbol, datetime, close, ...`) instead of
> `SELECT *`, and prefer the `1day` configs for quick exploration. For real
> work, download the parquet and query it from desktop DuckDB.

## Refresh cadence

The dataset is rebuilt by an incremental pipeline that:

1. Fetches the trailing window per symbol/timeframe (re-fetches the previous
   day to capture restatements).
2. Detects splits and dividends and triggers a per-symbol re-backfill when
   needed (so `close_adj` stays correct historically).
3. Recomputes indicators and macro joins.
4. Re-emits all three views and pushes them here.

## Limitations

- US equities only; intraday data is regular-session only (no pre/post).
- Macro context is ETF-proxied (e.g. VIXY for VIX, UUP for DXY) — convenient
  to fetch but not identical to the underlying index.
- Intraday history depth (especially `1min`) is bounded by Twelve Data's
  vendor limits and is much shorter than daily history. Don't assume
  identical date coverage across timeframes for the same symbol.
- The text view is templated, not LLM-generated — it is dense and repetitive
  by design, intended as a substrate for fine-tuning rather than as
  human-style prose.

## License

MIT. Underlying market data is © Twelve Data and redistributed under their
terms; check [twelvedata.com](https://twelvedata.com/) for commercial use.

## Citation

```
@misc{twelvedata-world-model,
  title  = {Twelve Data World Model Dataset},
  author = {Twelve Data},
  year   = {2026},
  url    = {https://huggingface.co/datasets/twelvedata/financial-world-model}
}
```
