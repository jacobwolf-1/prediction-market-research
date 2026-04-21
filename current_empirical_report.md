# Current Empirical Report

Date: March 10, 2026

## Status

The multi-season rebuild is still in progress and is not yet informative on the key external-validity question. At the last check, ESPN raw coverage had expanded from 4,199 files to 4,310 files, but Kalshi remained at 68 raw files. As a result, the matched market-history sample has not expanded beyond the original set of games, so the 3-5 season profitability question is still unresolved.

On the currently matched sample, the analysis pipeline was extended with the following scripts:

- `analysis/timestamp_alignment_check.py`
- `analysis/control_drift_test.py`
- `analysis/rebuild_merged_dataset.py`
- `analysis/shock_strategy_robustness.py`
- `analysis/shock_strategy_train_test.py`

The lead-lag and shock analyses were also refreshed to use the shared helper in `analysis/shock_strategy_utils.py`.

## Dataset

- Matched Kalshi sample: 68 games
- Rebuilt merged dataset: `data/processed/merged_games.parquet`
- Rebuilt merged rows: 1,110,402

The train/test split intended for a historical out-of-sample check is not yet feasible because the matched Kalshi sample still contains no 2021-2023 rows. The current split reported:

- `train_rows = 0`
- `test_rows = 1,110,278`

That means there is still no valid multi-season out-of-sample test.

## Timestamp Validation

Timestamp alignment is the strongest artifact check so far.

For raw ESPN shocks with absolute size greater than 0.05, the delay to the first subsequent Kalshi update was:

- mean: 5.96s
- median: 2.84s
- std: 11.55s
- p5: 0.12s
- p95: 21.94s

For all ESPN updates, the delay to the nearest Kalshi update in either direction was:

- mean: -1.09s
- median: +0.07s
- p5: -30.09s
- p95: +26.21s

Interpretation: the nearest-update distribution is centered close to zero, so the clocks do not appear to be systematically offset in a way that would mechanically generate the lead. But conditional on ESPN shocks, the next Kalshi update arrives later on average, which supports a real lead-lag effect rather than a timestamp artifact.

Primary figure:

- `visualizations/espn_market_delay_distribution.png`

## Lead-Lag and Shock Response

The rebuilt lead-lag result is unchanged: peak correlation remains at +5 seconds.

The shock reaction curve also remains strong. Average signed market return after raw ESPN shocks larger than 0.05 in absolute value was:

- 5s: 0.0099
- 10s: 0.0215
- 20s: 0.0324
- 30s: 0.0366
- 60s: 0.0382
- 120s: 0.0389

There is no clear reversal through 120 seconds.

Sample counts:

- Raw shocks above 0.05: 3,598
- Reaction-curve events with usable future path: 3,491

Primary outputs:

- `data/processed/shock_reaction_curve.csv`
- `visualizations/shock_reaction_curve.png`

## Control Test

The control test supports specificity. Average drift after shock timestamps at 30 seconds was 0.0366, versus only 0.0098 for sampled no-shock control timestamps.

Interpretation: the post-shock drift is materially larger than baseline drift at comparable points in the sample, which is consistent with a shock-specific reaction rather than a generic upward bias in market movement.

Primary figure:

- `visualizations/control_drift_curve.png`

## Backtest Results

The best current backtest on the matched sample remains:

- threshold: 0.05
- horizon: 30s
- trades: 2,502
- ignored_signals: 983
- mean return: 0.013094
- win rate: 60.11%
- total PnL: 32.76

With a liquidity filter of 1.0 market updates per minute, the strategy remains positive but weaker. The best filtered configuration was:

- threshold: 0.05
- horizon: 20s
- trades: 786
- total PnL: 8.23

Primary outputs:

- `data/processed/shock_backtest_results.csv`
- `data/processed/shock_backtest_results_liquidity_1_0.csv`
- `visualizations/shock_strategy_pnl_vs_threshold.png`
- `visualizations/shock_strategy_pnl_vs_horizon.png`
- `visualizations/shock_strategy_cumulative_pnl.png`
- `visualizations/shock_strategy_pnl_vs_threshold_liquidity_1_0.png`
- `visualizations/shock_strategy_pnl_vs_horizon_liquidity_1_0.png`
- `visualizations/shock_strategy_cumulative_pnl_liquidity_1_0.png`

## Robustness

Robustness checks on the current sample are directionally consistent.

Profitability is concentrated later in games:

- Q4 PnL: 18.70
- Last 8 minutes PnL: 19.16
- Last 5 minutes PnL: 14.18
- Last 2 minutes PnL: 7.08

Profitability also rises with shock size:

- 0.05-0.07 bin PnL: 10.30
- 0.07-0.10 bin PnL: 10.19
- Greater than 0.10 bin PnL: 12.27

Trade-return distribution:

- median: 0.01
- 95th percentile: 0.10

Primary outputs:

- `data/processed/shock_phase_breakdown.csv`
- `data/processed/shock_shock_size_robustness.csv`
- `data/processed/shock_trade_return_distribution.csv`
- `data/processed/shock_game_pnl_distribution.csv`

## Conclusion

On the currently available matched sample, the ESPN shock signal looks real and does not look like a simple timestamp-misalignment artifact. The timestamp diagnostics, control test, reaction curve, and profitable shock backtest all point in the same direction.

What is not yet established is whether the signal remains profitable on a genuinely larger multi-season Kalshi sample. The correct answer to that question remains: not yet known. The market-history expansion has not completed, and Kalshi coverage has not yet expanded beyond the original 68 matched games.

The current evidence supports the narrower claim that ESPN appears to lead Kalshi on this matched sample and that a shock-following strategy is profitable within that sample. It does not yet support a strong claim about persistence across a materially larger historical dataset.
