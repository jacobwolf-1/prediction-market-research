# Empirical Report

## Question

Does ESPN's live NBA win-probability feed appear to move before Kalshi NBA winner-market prices on the subset of games for which both time series are available?

## Data

- ESPN historical win-probability series collected with the scripts in `data_collection/`.
- Kalshi NBA winner-market history collected and matched to ESPN games with `data_collection/kalshi_fetch.py`.
- Seed game metadata in `data/raw/game_ids.csv`.
- Committed plot artifacts in `visualizations/`.
- Small committed result snapshots derived from local generated outputs on April 21, 2026:
  - `docs/results/shock_train_test_results_snapshot.csv`
  - `docs/results/shock_train_test_diagnostics_snapshot.csv`
  - `docs/results/kalshi_discovery_status_snapshot.csv`

The large raw and processed datasets are intentionally gitignored because they are too large for a lightweight public portfolio repository.

## Methodology

1. Fetch ESPN game IDs and raw event histories.
2. Discover Kalshi NBA events, extract teams from market metadata or tickers, and match them to ESPN games using team overlap and date proximity.
3. Build merged game-level datasets and resampled lead-lag inputs.
4. Run descriptive lead-lag tests and simple backtests on the matched sample.

This is a matched-sample study. It is not a venue-wide census of all NBA markets and it is not evidence of stable performance across all Kalshi eras.

For the public artifact:

- `pipeline/run_smoke_test.py` provides a synthetic smoke-test path from a fresh clone.
- `pipeline/run_full_pipeline.py` is the canonical end-to-end entrypoint for rebuilding the matched-sample dataset when the external sources are available.

## Outputs Produced

Committed outputs:

- 34 PNG figures in `visualizations/`.
- This report and the three CSV snapshots under `docs/results/`.

Generated-but-ignored outputs used to produce the snapshots:

- `data/processed/merged_games.parquet`
- `data/processed/shock_train_test_results.csv`
- `data/processed/shock_train_test_diagnostics.csv`
- `data/processed/kalshi_discovery_audit.csv`

## Main Findings Supported By Current Artifacts

- The repository contains a working pipeline for collecting ESPN and Kalshi histories, matching them, and producing aligned datasets and figures.
- The local merged raw-time dataset used for the current snapshot contains 4,329,372 rows, covering 293 games and 515 game-team groups.
- The committed train/test snapshot shows one concrete seasonal split on the matched sample: train on 2025 and test on 2026.
- The committed diagnostics snapshot shows that this split is feasible on the current local sample, with 68 games in the 2025 train bucket and 244 games in the 2026 test bucket.
- The committed Kalshi audit snapshot shows that the discovery process is partial: 254 events were queued for download, 116 matched games already had outputs, and 57 events remained unmatched in the local audit snapshot.

These findings support a careful framing: the project demonstrates nontrivial collection, matching, and analysis infrastructure, plus a matched-sample empirical workflow. They do not justify broad claims about generalizable trading profitability or persistent lead-lag effects across all seasons or venues.

## Limitations

- The matched sample is incomplete and coverage is uneven across seasons.
- Large generated outputs are not committed, so a fresh clone does not ship with the full processed dataset.
- Some README-era claims depended on local-only files; those claims should be treated as reproducible only after regenerating the ignored outputs.
- Strategy outputs are descriptive research artifacts, not evidence of deployable production trading performance.

## Next Work

- Expand smoke fixtures if additional public validation paths become useful.
- Add more integration tests around report tables and audit summaries.
- Keep exploratory scripts isolated from the canonical pipeline as the repo evolves.
