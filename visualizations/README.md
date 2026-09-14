# Visualizations index

Committed figures from prior local runs. Large raw/processed datasets are
gitignored, so these are snapshots, not outputs a fresh clone regenerates (the
public `make smoke-test` path validates code shape only — see the top-level
README and `docs/empirical_report.md`).

The **canonical** current pipeline is the raw-time ESPN/Kalshi path, so prefer
the `raw_time_*` figures. Read every figure alongside the caveats in
`docs/empirical_report.md` (small effect sizes; incomplete, uneven sample; the
"sharpe" column is a t-statistic, not a Sharpe ratio; flat transaction-cost
model).

> ⚠️ **Stale labels from an earlier iteration.** The project originally used
> Polymarket before switching to Kalshi. `lead_lag_curve.png` is still titled
> "ESPN vs Polymarket". The other non-`raw_time_` figures predate the switch and
> should be verified or regenerated before reuse; the `raw_time_*` figures are
> the ones that match the current Kalshi code.

## Lead–lag (canonical)
- `raw_time_lead_lag_curve.png` — ESPN vs Kalshi cross-correlation by lag (headline).
- `raw_time_reaction_histogram.png` — distribution of per-observation reactions.
- `raw_time_spread_alpha_curve.png`, `raw_time_spread_alpha_heatmap.png` — spread/alpha sweeps.
- `raw_time_example_game_*.png` — per-game ESPN vs Kalshi traces (3 games).

## Shock-following backtest (descriptive; net of a flat cost)
- `shock_reaction_curve.png`, `event_study_market_reaction.png` — market move after ESPN shocks.
- `shock_strategy_cumulative_pnl.png`, `shock_strategy_pnl_vs_threshold.png`, `shock_strategy_pnl_vs_horizon.png`.
- `shock_strategy_*_liquidity_1_0.png` — same, restricted to game-teams with ≥1 market update/min.

## Earlier iteration (verify labels before reuse)
- `lead_lag_curve.png` — **titled "ESPN vs Polymarket"**; superseded by the raw-time version.
- `spread_alpha_curve.png`, `spread_alpha_heatmap.png`, `mean_reversion_curve.png`,
  `control_drift_curve.png`, `per_game_reaction_histogram.png`,
  `espn_market_delay_distribution.png`, `example_game_*.png`,
  `backtest_cumulative_pnl.png`, `backtest_pnl_vs_threshold.png`, `backtest_pnl_vs_horizon.png`.
