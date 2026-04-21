# Prediction Market Research

Research code for testing whether live sports information from ESPN leads prediction-market prices in NBA winner markets, with a focus on Kalshi and some Polymarket ingestion utilities.

The repository does three main things:

1. Collect historical NBA game metadata and event-level win probability series from ESPN.
2. Collect per-game market history from Kalshi and Polymarket, then align those series with ESPN timestamps.
3. Run lead-lag, event-study, and backtest analyses, and save plots/results under `visualizations/` and `data/processed/`.

## Repository Layout

`data_collection/`
Scripts for fetching game IDs and raw historical data from ESPN, Kalshi, and Polymarket.

`analysis/`
Dataset builders, lead-lag tests, event studies, robustness checks, and strategy backtests.

`utils/`
Shared ingestion and timestamp/probability normalization helpers.

`data/raw/`
Raw inputs such as `game_ids.csv` plus per-game parquet outputs under source-specific folders like `espn/`, `kalshi/`, and `polymarket/`.

`data/processed/`
Derived datasets and tabular analysis outputs.

`visualizations/`
Saved charts from the analysis scripts.

`current_empirical_report.md`
Short write-up of the current matched-sample findings and caveats.

## Environment Setup

This repo uses plain Python scripts rather than a package manager.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Core dependencies:

- `pandas`
- `numpy`
- `pyarrow`
- `requests`
- `tqdm`
- `python-dotenv`

## Data Pipeline

### 1. Fetch NBA game IDs

Build a base table of ESPN game IDs for one or more seasons:

```bash
python data_collection/fetch_game_ids.py --seasons 2023 2024 2025
```

This writes `data/raw/game_ids.csv`.

### 2. Fetch raw ESPN and Kalshi history

Collect the full matched-history inputs in one pass:

```bash
python data_collection/collect_multi_season_history.py \
  --seasons 2023 2024 2025 \
  --skip-existing
```

Outputs:

- `data/raw/espn/game_<game_id>_espn.parquet`
- `data/raw/kalshi/<game_id>.parquet`

You can also run the source-specific collectors directly:

```bash
python data_collection/espn_historical_fetch.py --game-ids-file data/raw/game_ids.csv
python data_collection/kalshi_fetch.py --game-ids-file data/raw/game_ids.csv --skip-existing
python data_collection/polymarket_fetch.py --game-ids-file data/raw/game_ids.csv
```

### 3. Build aligned datasets

For the merge-asof lead-lag dataset:

```bash
python analysis/build_lead_lag_dataset.py --market-source kalshi
```

This writes per-game files under `data/processed/lead_lag_dataset/`.

For the raw-time merged dataset used by the newer strategy and timestamp diagnostics:

```bash
python analysis/rebuild_merged_dataset.py --freq 1s
```

This writes `data/processed/merged_games.parquet`.

## Main Analyses

### Lead-lag tests

Merge-asof lead-lag test:

```bash
python analysis/lead_lag_test.py
```

Time-resampled lead-lag test:

```bash
python analysis/time_lead_lag_test.py --resample-frequency 1s --max-lag-seconds 60
```

Independent raw-time lead-lag test:

```bash
python analysis/raw_time_lead_lag_test.py --freq 1s --max-lag-seconds 60
```

### Strategy backtests

Simple spread strategy on raw ESPN vs Kalshi differences:

```bash
python analysis/backtest_strategy.py --freq 1s
```

Shock-following strategy:

```bash
python analysis/backtest_shock_strategy.py --freq 1s
```

Optional liquidity filter:

```bash
python analysis/backtest_shock_strategy.py --freq 1s --min-market-updates-per-minute 1.0
```

### Additional diagnostics and robustness checks

Examples from the current workflow:

```bash
python analysis/timestamp_alignment_check.py
python analysis/control_drift_test.py
python analysis/shock_reaction_curve.py
python analysis/shock_strategy_robustness.py
python analysis/shock_strategy_train_test.py
```

## Current State

The repo already contains generated plots and intermediate outputs from prior runs, plus a current summary in [current_empirical_report.md](/Users/jacobwolf/prediction-market-research/current_empirical_report.md:1).

As of that report:

- the matched Kalshi sample is still small relative to the intended multi-season scope
- the raw-time analyses suggest ESPN shocks lead Kalshi updates on the currently matched sample
- broader out-of-sample validation is still incomplete because historical market coverage has not expanded enough yet

## Notes

- Most scripts assume data lives in the default project directories from `utils/ingestion_utils.py`.
- The code writes parquet outputs aggressively; `--overwrite` is available on most collection scripts if you want to rebuild from scratch.
- `scripts/pm_viability_test.py` is an older exploratory Polymarket-only script and is not part of the main pipeline.
