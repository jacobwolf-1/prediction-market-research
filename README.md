# Prediction Market Research

Matched-sample research code for an NBA market microstructure study: on games where both data sources can be aligned, do live ESPN win-probability updates appear to lead Kalshi NBA winner-market prices over short horizons?

This repository is a research workflow, not a claim of deployable trading alpha. The evidence here is limited to the matched sample that can be collected and aligned with the included code.

## Scope And Non-Claims

What this repo is:

- a real ESPN/Kalshi data collection and matching workflow
- a reproducible public smoke-test path from a fresh clone
- a matched-sample analysis pipeline for lead-lag and short-horizon reaction studies

What this repo is not:

- a venue-wide census of all NBA prediction markets
- proof of persistent market edge
- a production trading system

## Canonical Workflow

The canonical public workflow lives in `pipeline/` and uses the ESPN/Kalshi matched-sample path only.

- `pipeline/run_full_pipeline.py`: canonical end-to-end data build for users who want to regenerate the matched-sample research dataset and have access to the external data sources
- `pipeline/run_smoke_test.py`: small synthetic smoke test that validates the pipeline shape from a clean clone without historical data

The main analysis modules used by that workflow are:

- `data_collection/fetch_game_ids.py`
- `data_collection/collect_multi_season_history.py`
- `data_collection/espn_historical_fetch.py`
- `data_collection/kalshi_fetch.py`
- `analysis/build_lead_lag_dataset.py`
- `analysis/rebuild_merged_dataset.py`
- `analysis/time_lead_lag_test.py`
- `analysis/raw_time_lead_lag_test.py`
- `analysis/backtest_shock_strategy.py`
- `analysis/shock_strategy_train_test.py`
- `analysis/kalshi_discovery_audit_summary.py`

Older one-off analyses and side experiments were moved into exploratory directories so the public path is easier to read:

- `analysis/exploratory/`
- `scripts/exploratory/`
- `data_collection/exploratory/`

## Repository Structure

```text
pipeline/                 Canonical public entrypoints
analysis/                 Core dataset builders and main matched-sample analyses
analysis/exploratory/     Retained side analyses and one-off research scripts
data_collection/          Canonical ESPN/Kalshi collection code
data_collection/exploratory/ Older exploratory collection work
sample_data/smoke/        Tiny synthetic fixtures for public smoke tests
data/                     Seed metadata plus locally generated raw/processed outputs
docs/                     Empirical report and small committed result snapshots
tests/                    Unit and integration-style tests
visualizations/           Committed figures from prior local runs
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

`ODDS_API_KEY` remains only as a placeholder for older experiments. The canonical ESPN/Kalshi workflow does not currently require environment variables.

## Public Reproducibility

There are two different reproducibility levels in this repo.

### 1. Public Smoke Test

This works from a fresh clone and does not require historical data downloads:

```bash
make smoke-test
```

Equivalent direct command:

```bash
python pipeline/run_smoke_test.py
```

What it does:

- materializes tiny synthetic ESPN/Kalshi fixtures from `sample_data/smoke/`
- builds the aligned lead-lag dataset shape
- rebuilds the merged raw-time dataset shape
- writes a small summary under `artifacts/smoke_test/`

What it does not do:

- reproduce the empirical report
- regenerate the committed figures
- validate any historical effect size

### 2. Full Matched-Sample Rebuild

This is the canonical full workflow, but it depends on external APIs and on the continued availability of compatible historical responses:

```bash
python pipeline/run_full_pipeline.py \
  --seasons 2023 2024 2025 2026 \
  --skip-existing
```

That command:

1. builds or updates `data/raw/game_ids.csv`
2. collects ESPN histories
3. discovers and matches Kalshi events
4. writes the Kalshi discovery audit
5. builds `data/processed/lead_lag_dataset/`
6. rebuilds `data/processed/merged_games.parquet`

Large raw and processed datasets are intentionally not part of the public artifact. A clean clone should be assumed to contain code, synthetic smoke fixtures, documentation, and small result snapshots, not the full historical output bundle.

## Main Outputs

Committed outputs:

- `docs/empirical_report.md`
- `docs/results/shock_train_test_results_snapshot.csv`
- `docs/results/shock_train_test_diagnostics_snapshot.csv`
- `docs/results/kalshi_discovery_status_snapshot.csv`
- `visualizations/`

Generated locally by the full pipeline:

- `data/raw/espn/*.parquet`
- `data/raw/kalshi/*.parquet`
- `data/processed/lead_lag_dataset/*.parquet`
- `data/processed/merged_games.parquet`
- additional CSV tables produced by the analysis scripts

## Tests And CI

Run the local test suite after installing dependencies:

```bash
python -m pytest
```

The test suite includes:

- deterministic helper tests
- Kalshi matching tests
- dataset loading and resampling tests
- pipeline-level smoke tests on synthetic fixtures

Minimal CI lives in `.github/workflows/ci.yml` and runs the smoke test plus `pytest` on push and pull request.

## Evidence Supported By This Repo

The current repository supports the following restrained claims:

- the project contains working collection code for ESPN and Kalshi NBA histories
- the repo implements nontrivial event matching and audit logic
- the main empirical workflow is a matched-sample lead-lag study with committed snapshots and figures
- the train/test snapshot documents one feasible seasonal split on one local matched sample

The repository does not support stronger claims such as:

- stable profitability across all seasons
- venue-wide persistence across all Kalshi NBA history
- robustness outside the matched sample currently available here
- production-ready trading performance

## Known Limitations

- Full historical reproduction depends on external APIs and may break if those APIs change.
- Coverage is constrained by overlap between ESPN historical availability and Kalshi event discovery/matching.
- The matched sample is incomplete and uneven across seasons.
- The synthetic smoke fixtures validate code paths only; they do not validate the empirical findings.
- Exploratory scripts are retained for transparency, but they are not part of the canonical workflow.
