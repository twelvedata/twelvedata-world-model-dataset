---
license: mit
task_categories:
  - time-series-forecasting
  - text-generation
language:
  - en
tags:
  - finance
  - time-series
  - world-models
  - llm4ts
  - market-data
pretty_name: Twelve Data Financial World-Model Dataset
size_categories:
  - 1M<n<10M
---

# Twelve Data — Financial World-Model Dataset

A curated multi-timeframe dataset of ~50 US large-caps plus macro
context (SPY, QQQ, VIX, TLT, sector ETFs, FX/commodity proxies) designed
for **LLM fine-tuning**, **time-series foundation model** training, and
**world-model / predictive-dynamics** research.

## Configurations

- `bars_1day`, `bars_1h`, `bars_1min` — OHLCV + causal indicators + macro context, as Parquet.
- `text_1day`, `text_1h`, `text_1min` — instruction-style JSONL with explicit `prompt` / `label` separation.
- `trajectories_1day`, `trajectories_1h`, `trajectories_1min` — rolling windows with `states` and `next_states`.

Each configuration has `train`, `val`, `test` splits using time-based
cutoffs (see `config/splits.yaml`). No shuffling across time.

## Causality guarantee

Every indicator value at time `t` is computed using only rows with
timestamps ≤ `t`. The repository's test suite includes truncation and
future-shuffle tests that enforce this invariant.

## Updates

The dataset is updated daily after US market close. Updates are
idempotent: the pipeline always re-fetches the previous trading day to
catch late prints and corporate-action restatements.

## Known limitations

- Adjusted close uses Twelve Data's adjustment; raw close is also kept
  so downstream can re-adjust if needed.
- ETF-based proxies are used for some macros (UUP for DXY, IEF/TLT for
  Treasuries) to keep coverage uniform across accounts.
- Intraday (1h, 1min) history is shorter than daily — see config.

## Citation

```
@misc{twelvedata_financial_world_model,
  title  = {Twelve Data Financial World-Model Dataset},
  author = {Twelve Data},
  year   = {2026},
  url    = {https://huggingface.co/datasets/twelvedata/financial-world-model}
}
```
