# Twelve Data — Financial World-Model Dataset

A reproducible pipeline that turns Twelve Data market bars into a multi-modal
dataset for **LLM fine-tuning**, **time-series foundation models**, and
**world-model / predictive-dynamics** training.

> **This repo is the code.** The published dataset lives at
> [huggingface.co/datasets/twelvedata/financial-world-model](https://huggingface.co/datasets/twelvedata/financial-world-model).

## What's in the dataset

51 large-cap US equities × 3 timeframes (`1day`, `1h`, `1min`) × 3 parallel
views, all built from the same source bars and the same time-based splits:

| Config | Format | What it is |
| --- | --- | --- |
| `bars_{tf}` | Parquet | OHLCV + causal indicators + macro context (33+ columns) |
| `text_{tf}` | JSONL | Instruction-tuning prompts with strict prompt/label leak-prevention |
| `trajectories_{tf}` | Parquet | Rolling windows with `states` and `next_states` for world-model / RL training |

**Splits** (time-based, declared in [config/splits.yaml](config/splits.yaml)):
`train ≤ 2023-12-31`, `validation = 2024`, `test = 2025+`. Trajectories that
straddle a split boundary are dropped so train/val never overlap.

**Macro context** (joined by date onto every equity row): SPY, VIX, TLT
(20Y), DXY (UUP proxy), and a sector-matched SPDR ETF.

Full column-level docs live in [config/dataset_card.md](config/dataset_card.md)
— that file is also what gets pushed to HF as the dataset README.

## Loading the data

```python
from datasets import load_dataset

bars = load_dataset("twelvedata/financial-world-model", "bars_1day")
text = load_dataset("twelvedata/financial-world-model", "text_1h")
traj = load_dataset("twelvedata/financial-world-model", "trajectories_1day")
```

Or query the parquet files directly with DuckDB:

```sql
SELECT symbol, datetime, close, rsi_14
FROM 'bars_1day/test.parquet'
WHERE symbol = 'AAPL'
ORDER BY datetime DESC
LIMIT 10;
```

## Quick start (running the pipeline)

```bash
make install                                                # creates .venv and installs deps
cp .env.example .env                                        # then fill in TWELVE_DATA_API_KEY
                                                            # free key at https://twelvedata.com/account/api-keys
make backfill ARGS="--timeframes 1day --limit-symbols 5"    # smoke test: 5 symbols
make test
make release ARGS="--dry-run"                               # writes to data/release/, skips HF upload
```

## `build_release.py` flags

| Flag | What it does |
| --- | --- |
| *(none)* | Build all configs from `data/`, then push everything (configs + README) to HF |
| `--dry-run` | Build only — write parquet/jsonl to `data/release/`, skip HF upload |
| `--push-only` | Skip build, re-upload existing files in `data/release/` (idempotent) |
| `--card-only` | Skip build *and* config uploads — only push `config/dataset_card.md` as `README.md` |
| `--clean` | Delete every file in the HF repo before uploading (preserves `.gitattributes` and `README.md`). Prompts unless `--yes` |
| `--timeframes 1day,1h` | Restrict the build/push to selected timeframes |

## Repo layout

```
twelvedata-world-model-dataset/
├── README.md
├── LICENSE
├── CHANGELOG.md
├── Makefile
├── requirements.txt
├── pyproject.toml
├── .github/workflows/
│   ├── daily-update.yml        # cron updater + auto re-publish
│   └── tests.yml               # pytest on PR
├── config/
│   ├── symbols.yaml            # 51 equities + macro tickers
│   ├── features.yaml           # indicator definitions
│   ├── timeframes.yaml         # 1day / 1h / 1min config
│   ├── splits.yaml             # train / val / test cutoffs
│   ├── sectors.yaml            # symbol → sector ETF mapping
│   ├── hf.yaml                 # HF dataset repo id + options
│   └── dataset_card.md         # README pushed to HF
├── src/tdwm/                   # core package
│   ├── client.py               # Twelve Data SDK wrapper
│   ├── fetch.py                # bar + macro fetch (with pagination)
│   ├── indicators.py           # causal technical indicators
│   ├── enrich.py               # join macro context onto bars
│   ├── corporate_actions.py    # detect splits/dividends, trigger re-backfill
│   ├── textify.py              # vectorized prompt/label textification
│   ├── trajectories.py         # rolling-window state generator
│   ├── splits.py               # time-based splits
│   ├── storage.py              # partitioned parquet append
│   ├── state.py                # resume/backfill state
│   ├── schema.py               # typed schemas
│   ├── charts.py               # candlestick PNG renderer
│   ├── hf.py                   # streaming HF push (HfApi.upload_file)
│   └── release.py              # per-symbol streaming release builder
├── scripts/
│   ├── backfill.py             # historical backfill
│   ├── update_daily.py         # daily incremental (idempotent, multi-day catch-up)
│   ├── build_release.py        # assemble + push to HF
│   └── validate_no_leakage.py  # leakage checks as a CLI
├── tests/
│   ├── test_indicators_causal.py
│   ├── test_textify.py
│   ├── test_trajectories.py
│   ├── test_splits.py
│   ├── test_storage.py
│   └── test_corporate_actions.py
├── notebooks/
│   ├── 01_llm_finetuning.ipynb
│   ├── 02_world_model_training.ipynb
│   └── 03_multimodal.ipynb
├── examples/
│   ├── sample_text.jsonl
│   ├── sample_trajectories.jsonl
│   └── prompt_templates.md
├── metadata/
│   └── schema.json             # versioned schema contract (humans + future CI)
└── data/                        # local parquet store (gitignored)
```

## Daily updates

[.github/workflows/daily-update.yml](.github/workflows/daily-update.yml) runs
`scripts/update_daily.py` on a cron, then rebuilds and pushes the release.
The updater is **idempotent and gap-aware**: each run requests `last_known - 2
days → today`, so a single run automatically catches up however many days
were missed. It also always re-fetches the previous trading day to capture
late prints and corporate-action restatements.

## Design choices (and why)

- **Everything causal.** Indicators are computed left-to-right with no
  centered or two-sided windows. `tests/test_indicators_causal.py` shuffles
  *future* rows and asserts indicator values at time `t` do not change — the
  strongest possible leakage guard.
- **Prompt / label separation.** Textified rows never describe the future
  inside the `prompt` field. Next-bar return lives only in `label`, so you
  can concatenate safely for instruction tuning.
- **Adjusted + raw both shipped.** Split/dividend-adjusted series is the
  default (column `close_adj`); raw `close` is kept so downstream can
  recompute if desired.
- **Restatement handling.** The updater re-fetches the previous trading
  day every run; corporate-action detection in [src/tdwm/corporate_actions.py](src/tdwm/corporate_actions.py)
  triggers a per-symbol re-backfill when a new split/dividend appears, so
  historical `close_adj` stays consistent.
- **Partitioned storage.** Parquet is partitioned by
  `timeframe/symbol/year`, so daily appends touch exactly one file per
  symbol-year.
- **Streaming release builder.** [src/tdwm/release.py](src/tdwm/release.py)
  writes bars/text/trajectories to disk one symbol at a time via
  `ParquetWriter`, so peak RAM stays proportional to one symbol — not the
  whole universe (matters at 1-minute scale).
- **Streaming HF push.** Uploads use `HfApi.upload_file` directly, so
  multi-GB parquet files never get materialized in RAM.
- **No rewards by default.** World models predict next state. If you want
  RL rewards, pick one of `log_return`, `vol_adj_return`, or
  `drawdown_penalty` as a post-processing step — the dataset doesn't commit
  to a reward definition.

## Troubleshooting

**`ModuleNotFoundError: No module named 'yaml'` (or any other module)**
You're running system Python instead of the project venv. Activate it first:
```bash
source .venv/bin/activate
```
Or prefix every command with `.venv/bin/python`.

**`RuntimeError: No API key and no injected SDK`**
Set your API key: `export TWELVE_DATA_API_KEY=<your_key>`. Get a free key at
https://twelvedata.com/account/api-keys, or put it in `.env` (see `.env.example`).

**`python scripts/backfill.py` is slow on first run**
Expected — it fetches years of history for ~50 symbols. Use
`--limit-symbols 5` for a smoke test, or `--timeframes 1day` to skip intraday.

**HF Data Studio query loads forever**
The in-browser DuckDB-WASM streams parquet over HTTPS without local caching.
Always project explicit columns (`SELECT symbol, datetime, close ...`) instead
of `SELECT *`, and prefer the `1day` configs for quick exploration. For real
analytics work, download the parquet locally and query with desktop DuckDB.

## Licensing & data redistribution

Code in this repo is MIT. The *data* is sourced from Twelve Data — confirm
your redistribution terms before publishing the HF dataset publicly. The
default [config/hf.yaml](config/hf.yaml) ships with `private: true`.

## Contributing

- Keep indicators causal. New indicators must add a test to
  [tests/test_indicators_causal.py](tests/test_indicators_causal.py).
- Schema changes bump [metadata/schema.json](metadata/schema.json) and add a
  note to [CHANGELOG.md](CHANGELOG.md).
- The HF dataset card lives in [config/dataset_card.md](config/dataset_card.md).
  Edits there are published with `make release ARGS="--card-only"`.
