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

## Backtest Metrics, Transaction Costs, And Statistical Caveats

The shock-following backtest (`analysis/backtest_shock_strategy.py`, reused by
`analysis/shock_strategy_train_test.py`) reports the following, so their exact
meaning should not be over-read.

### The `sharpe` column is a t-statistic, not a Sharpe ratio

The column labeled `sharpe` in `docs/results/shock_train_test_results_snapshot.csv`
is computed as `sqrt(N) * mean(net_return) / std(net_return)` over the per-trade
net-return series. That expression is the **one-sample t-statistic** of the mean
per-trade return against zero, i.e. `mean / (std / sqrt(N))`. It is **not** a
conventional Sharpe ratio: it has no risk-free rate, is not annualized, and it
*increases with the square root of the number of trades*. The committed snapshot
values (`≈11.05` on the 2025 train split with 2,502 trades and `≈14.23` on the
2026 test split with 7,659 trades) are large precisely because N is large, not
because the risk-adjusted edge is extreme. Read them as evidence that mean
per-trade net return is statistically distinguishable from zero **on this
matched sample under these assumptions**, not as an annualized Sharpe.

A future revision should rename this metric to `mean_return_tstat` and, if a
risk-adjusted-per-trade figure is wanted, report `mean/std` separately. That
rename is deferred here because it should be done together with regenerating the
committed snapshots from the (gitignored) local dataset, to avoid a label that
disagrees with the stored numbers.

### Return units and the transaction-cost model

- Prices are win probabilities in `[0, 1]`; a Kalshi YES contract settles at
  `$1`, so a return of `0.01` equals one cent of PnL per one-contract position.
- Entry is the market price one tick after the shock (`next_market_price`); exit
  is the price at the chosen horizon. A per-game/team guard
  (`next_allowed_time = exit_time`) prevents overlapping trades within a group.
- Costs are a single flat constant: `TRADE_COST = 0.01` per side, so
  `net_return = raw_return - 2 * TRADE_COST` for a round trip. This is a stand-in
  for half-spread plus fees; the reported PnL, `avg_return`, `win_rate`, and the
  t-statistic are all net of this constant.

### What the cost model does not capture

The flat-cost assumption is optimistic, and the headline numbers are sensitive to
it. In particular the backtest does **not** model:

- **Bid/ask spread** — the true spread on Kalshi NBA winner markets is
  time-varying and frequently wider than 1 cent, especially away from 50/50; a
  constant 1-cent half-cost understates cost in many states.
- **Order-book depth and market impact** — every fill is assumed to occur at the
  observed price regardless of size; there is no depth or impact model.
- **Fill probability** — fills are assumed certain whenever a next-tick price
  exists; there is no queue position, partial fill, or adverse-selection model.
- **Capacity** — each trade is a single unit with no position sizing, notional,
  or capital constraint, so `total_pnl` is a sum of per-contract moves, not a
  return on deployable capital.

### Why the significance is likely overstated

- **Clustered observations.** Trades are not independent: many fire within the
  same game/team as the win probability moves, so returns are serially
  correlated. The t-statistic assumes i.i.d. draws, so the effective sample size
  is smaller than N and the statistic overstates significance.
- **Repeated signals within games.** The threshold can trigger repeatedly in one
  game; `ignored_signals` only skips signals while a trade is still open, so
  same-direction repeats in a game are still correlated.
- **Uneven seasonal coverage.** The matched sample is concentrated in 2026 (the
  test split has 244 games vs. 68 in the 2025 train split), so the single
  train-2025 / test-2026 split is one feasible split on an uneven sample, not a
  robustness study across balanced seasons.

### Reproducibility: code path vs. empirical result

- **Code-path reproducibility** is public and CI-tested: `make smoke-test` and
  `python -m pytest` run from a fresh clone on synthetic fixtures with no
  external calls.
- **Empirical-result reproducibility** is *not* guaranteed: the numbers above
  were produced from local ESPN/Kalshi pulls that are gitignored, and depend on
  external APIs whose historical responses may change. Regenerating identical
  figures/snapshots requires re-collecting compatible upstream data.

## Limitations

- The matched sample is incomplete and coverage is uneven across seasons.
- Large generated outputs are not committed, so a fresh clone does not ship with the full processed dataset.
- Some README-era claims depended on local-only files; those claims should be treated as reproducible only after regenerating the ignored outputs.
- Strategy outputs are descriptive research artifacts, not evidence of deployable production trading performance.

## Next Work

- Expand smoke fixtures if additional public validation paths become useful.
- Add more integration tests around report tables and audit summaries.
- Keep exploratory scripts isolated from the canonical pipeline as the repo evolves.
