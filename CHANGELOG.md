# Changelog

All schema-affecting changes must be reflected here.

## [Unreleased]

## [0.1.0] — 2026-04-21

Initial release.

### Dataset
- 50 large-caps + macro tickers (VIX, SPY, TLT, sector ETFs, FX/commodity proxies)
- Timeframes: `1day`, `1h`, `1min`
- Three output formats: `bars` (Parquet), `text` (JSONL), `trajectories` (JSONL/Parquet)
- Time-based train/val/test splits (train ≤ 2023-12-31, val 2024, test 2025+)

### Pipeline
- Causal indicators with leakage unit tests (`test_indicators_causal.py`)
- Idempotent daily updater with restatement handling (always re-fetches previous day)
- Partitioned parquet storage by `timeframe/symbol/year`
- GitHub Actions: daily cron updater + CI test suite with `pip-audit`

### Schema (`metadata/schema.json` v0.1.0)
- `bars`: OHLCV + causal indicators + macro context columns
- `text`: `prompt`, `label`, `symbol`, `timeframe`, `as_of`, `split`
- `trajectories`: `states`, `next_states`, `feature_names`, `timestamps`, `rewards_logret`
