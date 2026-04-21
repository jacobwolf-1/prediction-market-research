# Prediction Market Research

Research code for an NBA prediction-market study asking whether live ESPN game information appears to lead Kalshi NBA winner-market prices on the subset of games where both series can be matched.

## Current Status

This repository is now structured as a public research artifact rather than a private working directory:

- The main collection and analysis scripts are present and runnable.
- A focused `pytest` suite covers deterministic normalization, matching, parsing, and resampling helpers.
- GitHub Actions CI runs the test suite on Python 3.12.
- The README and report only make claims tied to files that exist in this working tree.

What is still incomplete:

- A fresh clone does not include the large raw and processed datasets because they are intentionally gitignored.
- Some analysis outputs can only be regenerated if the source APIs still return compatible historical data.
- The repo still contains a few exploratory scripts and notes that are not part of the canonical pipeline.

## Research Question

Primary question: when ESPN updates live NBA win probabilities, do Kalshi NBA winner markets react with a measurable delay on the matched sample available in this repository?

Working hypothesis: if ESPN updates arrive before Kalshi market prices fully adjust, then aligned ESPN probability shocks should be associated with short-horizon Kalshi moves in the same direction.

This project does **not** claim broad external validity beyond the matched sample currently available here.

## Data Sources

- ESPN live game and win-probability history, collected with `data_collection/espn_historical_fetch.py`
- Kalshi NBA winner-market history, collected with `data_collection/kalshi_fetch.py`
- Polymarket collection utilities retained as exploratory side work, not part of the main documented pipeline
- Seed NBA game metadata in `data/raw/game_ids.csv`

## Repository Structure

```text
analysis/          Analysis scripts, dataset builders, diagnostics, and backtests
data/              Seed inputs plus ignored raw/processed outputs generated locally
data_collection/   ESPN, Kalshi, and Polymarket collection scripts
data_processing/   Reserved for lightweight processing utilities
docs/              Empirical report and small committed result snapshots
scripts/           Older exploratory or smoke-test scripts
tests/             Pytest suite for deterministic helpers
utils/             Shared ingestion and normalization helpers
visualizations/    Committed plot artifacts from prior analysis runs
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Optional environment file:

```bash
cp .env.example .env
```

`ODDS_API_KEY` is included only as a placeholder for older experiments. The main ESPN/Kalshi pipeline does not currently require environment variables.

## Reproducibility Notes

- `requirements.txt` is pinned to the versions used for this cleanup pass.
- `.env` is ignored and a placeholder `.env.example` is included.
- `data/raw/` and `data/processed/` are ignored by default because the local datasets are large.
- Committed result snapshots in `docs/results/` summarize a local run state from April 21, 2026.
- If a required source dataset is absent in a fresh clone, regenerate it with the commands below instead of assuming it is bundled with the repo.

## How To Run The Pipeline

### 1. Build the seed game list

```bash
python data_collection/fetch_game_ids.py --seasons 2023 2024 2025 2026
```

### 2. Collect raw ESPN and Kalshi histories

```bash
python data_collection/collect_multi_season_history.py \
  --seasons 2023 2024 2025 2026 \
  --skip-existing
```

Or run the collectors individually:

```bash
python data_collection/espn_historical_fetch.py --game-ids-file data/raw/game_ids.csv
python data_collection/kalshi_fetch.py --game-ids-file data/raw/game_ids.csv --skip-existing
```

### 3. Build aligned datasets

```bash
python analysis/build_lead_lag_dataset.py --market-source kalshi
python analysis/rebuild_merged_dataset.py --freq 1s
```

### 4. Run the main analyses

```bash
python analysis/time_lead_lag_test.py --resample-frequency 1s --max-lag-seconds 60
python analysis/raw_time_lead_lag_test.py --freq 1s --max-lag-seconds 60
python analysis/backtest_shock_strategy.py --freq 1s
python analysis/shock_strategy_train_test.py --train-seasons 2025 --test-seasons 2026
python analysis/kalshi_discovery_audit_summary.py
```

## What Outputs Currently Exist

Committed artifacts in the repo:

- `visualizations/` contains 34 PNG figures from prior analysis runs.
- `docs/empirical_report.md` summarizes the current evidence conservatively.
- `docs/results/` contains small committed CSV snapshots derived from local generated outputs.
- `data/raw/game_ids.csv` is the committed seed metadata file.

Generated locally but not committed by default:

- `data/raw/espn/*.parquet`
- `data/raw/kalshi/*.parquet`
- `data/raw/polymarket/*.parquet`
- `data/processed/*.csv`
- `data/processed/*.parquet`

Because those generated datasets are ignored, a fresh clone should be assumed to have the code and documentation, not the full historical output bundle.

## Claims Supported By Current Artifacts

The current artifacts support the following restrained claims:

- The repo contains real data collection code for ESPN and Kalshi historical series.
- The project includes nontrivial team extraction and matching logic, plus audit outputs for unmatched Kalshi events.
- The committed plots and result snapshots support describing this as a matched-sample lead-lag study.
- The committed train/test snapshot documents one feasible `2025 -> 2026` seasonal split on the local matched sample.

The current artifacts do **not** support stronger claims such as:

- stable profitability across all seasons
- venue-wide persistence across all Kalshi NBA history
- robustness beyond the matched sample currently available here
- generalizable production trading performance

## Limitations

- Coverage is constrained by the overlap between ESPN historical availability and Kalshi event discovery/matching.
- The matched sample is incomplete and uneven across seasons.
- Some plots are committed, but many large intermediate tables are intentionally not.
- The Polymarket scripts are exploratory and should not be treated as part of the main empirical pipeline.

## Tests And CI

Run the local test suite with:

```bash
python -m pytest
```

CI lives in `.github/workflows/ci.yml` and runs the same test command on GitHub Actions.

## Next Steps

- Add a single manifest-driven pipeline command for end-to-end regeneration.
- Publish a small sample dataset for smoke tests in fresh clones.
- Expand tests to dataset-building scripts and summary tables.
- Further separate exploratory scripts from the canonical pipeline.
