# Twelve Data — Financial World-Model Dataset

A reproducible pipeline that turns Twelve Data market bars into a dataset
designed for **LLM fine-tuning**, **time-series foundation models**, and
**world-model / predictive-dynamics** training.

This repo is the code. The data lives on Hugging Face:
`twelvedata/financial-world-model` (set the name in `config/hf.yaml`).

## What you get

- **~50 large-caps + macro tickers** (VIX, SPY, TLT, sector ETFs, 10Y yield proxy)
- **Three timeframes**: `1day`, `1h`, `1min` (intraday where history allows)
- **Three output formats**, all from the same source of truth:
  1. **Parquet** — clean tabular bars + causally-computed indicators
  2. **Textified JSONL** — instruction-style rows with an explicit
     `prompt` / `label` split (no look-ahead in the prompt)
  3. **Trajectory JSONL / Array-Parquet** — rolling windows with
     `states`, `next_states`, optional `rewards`, shaped for world-model
     and RL-style training
- **Multimodal (optional)**: daily candlestick PNGs aligned to rows
- **Time-based train / val / test splits** baked into HF dataset configs
- **Idempotent daily updater** — detects missed days, backfills gaps,
  always re-fetches the previous day to catch late corporate-action and
  price restatements
- **Leakage guards** — unit tests that assert every indicator at time `t`
  depends only on information available at or before `t`

## Repo layout

```
twelvedata-world-model-dataset/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── requirements.txt
├── pyproject.toml
├── .github/workflows/
│   ├── daily-update.yml        # cron updater (idempotent)
│   └── tests.yml               # pytest on PR
├── config/
│   ├── symbols.yaml            # ~50 large-caps + macro tickers
│   ├── features.yaml           # indicator definitions
│   ├── timeframes.yaml         # 1day / 1h / 1min config
│   ├── splits.yaml             # train / val / test cutoffs
│   └── hf.yaml                 # HF dataset repo id + options
├── src/tdwm/                   # core package
│   ├── client.py               # Twelve Data SDK wrapper
│   ├── fetch.py                # bar + macro fetch
│   ├── indicators.py           # causal technical indicators
│   ├── enrich.py               # join macro context onto bars
│   ├── textify.py              # prompt/label textification
│   ├── trajectories.py         # rolling windows for world models
│   ├── splits.py               # time-based splits
│   ├── storage.py              # partitioned parquet append
│   ├── state.py                # resume/backfill state
│   ├── charts.py               # candlestick PNGs (optional)
│   ├── schema.py               # typed schemas
│   ├── hf.py                   # Hugging Face push
│   └── release.py              # build full dataset release
├── scripts/
│   ├── backfill.py             # historical backfill
│   ├── update_daily.py         # daily incremental (idempotent)
│   ├── build_release.py        # assemble + push to HF
│   └── validate_no_leakage.py  # runs the leakage checks as a CLI
├── tests/
│   ├── test_indicators_causal.py
│   ├── test_textify.py
│   ├── test_trajectories.py
│   ├── test_splits.py
│   └── test_storage.py
├── notebooks/
│   ├── 01_llm_finetuning.ipynb
│   ├── 02_world_model_training.ipynb
│   └── 03_multimodal.ipynb
├── examples/
│   ├── sample_trajectories.jsonl
│   └── prompt_templates.md
├── metadata/
│   ├── schema.json
│   └── DATASET_CARD.md
└── data/                        # local parquet store (gitignored)
```

## Quick start

```bash
make install                             # creates .venv and installs deps
cp .env.example .env                     # then fill in TWELVE_DATA_API_KEY
                                         # get a free key at https://twelvedata.com/account/api-keys
make backfill ARGS="--timeframes 1day --limit-symbols 5"  # smoke test: 5 symbols
make test
make release ARGS="--dry-run"            # writes parquet to data/release/, skips HF upload
```

The `--dry-run` flag skips the Hugging Face push and just writes parquet splits to
`data/release/`. The published dataset lives at
[huggingface.co/datasets/twelvedata/financial-world-model](https://huggingface.co/datasets/twelvedata/financial-world-model).

## Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'` (or any other module)**
You're running system Python instead of the project venv. Activate it first:
```bash
source .venv/bin/activate
```
Or prefix every command with `.venv/bin/python`.

**`RuntimeError: No API key and no injected SDK`**
Set your API key: `export TWELVE_DATA_API_KEY=<your_key>`. Get a free key at
https://twelvedata.com/account/api-keys. You can also put it in a `.env` file at the repo root
(see `.env.example`).

**`python scripts/backfill.py` is slow on first run**
That's expected — it fetches years of history for ~50 symbols. Use
`--limit-symbols 5` for a smoke test, or `--timeframes 1day` to skip intraday.

## Design choices (and why)

- **Everything causal.** Indicators are computed left-to-right with no
  centered or two-sided windows. `tests/test_indicators_causal.py`
  shuffles *future* rows and asserts indicator values at time `t` do not
  change — the strongest possible leakage guard.
- **Prompt / label separation.** Textified rows never describe the future
  inside the `prompt` field. Next-day return lives only in `label`, so
  you can concatenate safely for instruction tuning.
- **Adjusted + raw both shipped.** Split/dividend-adjusted series is the
  default (column `close_adj`); raw close is kept (`close`) so downstream
  can recompute if desired.
- **Restatement handling.** The updater always re-fetches the *previous*
  trading day in addition to today. Late prints and corrections are
  captured without manual intervention.
- **Partitioned storage.** Parquet is partitioned by
  `timeframe/symbol/year`, so daily appends touch exactly one file.
- **No rewards by default.** World models predict next state. If you want
  RL rewards, pick one of `log_return`, `vol_adj_return`, or
  `drawdown_penalty` as a post-processing step — the dataset does not
  commit to a reward definition.
- **Time-based splits only.** See `config/splits.yaml`. The default is
  train ≤ 2023-12-31, val 2024, test 2025+.

## Licensing & data redistribution

Code in this repo is MIT. The *data* is sourced from Twelve Data — before
publishing the HF dataset publicly, confirm your redistribution terms
allow it. The default `config/hf.yaml` ships with `private: true`.

## Contributing

- Keep indicators causal. New indicators must add a test to
  `tests/test_indicators_causal.py`.
- Schema changes bump `metadata/schema.json` and add a note to
  `CHANGELOG.md`.
