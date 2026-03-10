# Raw-Time Results

## Dataset Size

- Games analyzed: 68
- Game-team groups: 124
- Raw Kalshi rows available: 605,452
- Raw-time merged observations after independent 1-second resampling: 1,110,278

## Lead-Lag Result

- Raw-time lead-lag peak correlation: 0.031099
- Lag of peak: +5 seconds
- Mean per-game correlation at peak lag: 0.027588
- Reaction-time distribution:
  - Mean peak lag: 9.78 seconds
  - Median peak lag: 8.00 seconds
  - 90th percentile: 24.4 seconds

Interpretation: after removing the timestamp-collapse artifact from the old merged pipeline, ESPN still appears to lead Kalshi, but the effect is materially smaller than the earlier +20 second estimate.

## Spread Alpha

Raw-time spread alpha remains directionally mean-reverting:

- 5s horizon: beta = -0.019899, R^2 = 0.016771
- 10s horizon: beta = -0.031646, R^2 = 0.026066
- 20s horizon: beta = -0.045241, R^2 = 0.031166
- 30s horizon: beta = -0.051630, R^2 = 0.029200
- 60s horizon: beta = -0.061006, R^2 = 0.022740

Interpretation: when market probability is above ESPN, future market returns tend to be negative; when market probability is below ESPN, future market returns tend to be positive.

## Plots

- `visualizations/raw_time_lead_lag_curve.png`
- `visualizations/raw_time_reaction_histogram.png`
- `visualizations/raw_time_spread_alpha_curve.png`
- `visualizations/raw_time_spread_alpha_heatmap.png`
- `visualizations/raw_time_example_game_401766124.png`
- `visualizations/raw_time_example_game_401767910.png`
- `visualizations/raw_time_example_game_401767818.png`
