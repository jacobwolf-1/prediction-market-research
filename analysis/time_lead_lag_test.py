from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ingestion_utils import PROCESSED_DIR, PROJECT_ROOT, ensure_directory
from analysis.time_series_utils import (
    build_input_diagnostics,
    build_lag_correlation_table,
    build_resampled_diagnostics,
    build_resampled_groups,
    frequency_to_seconds,
    lagged_views,
    load_lead_lag_datasets,
    normalize_frequency_label,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-based lead-lag correlation analysis for ESPN vs market prices.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR / "lead_lag_dataset",
        help="Directory containing merged lead-lag parquet files.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "time_lead_lag_curve.png",
        help="Path for the time lead-lag correlation plot.",
    )
    parser.add_argument(
        "--resample-frequency",
        default="1s",
        help="Fixed time grid used for resampling. Defaults to 1S.",
    )
    parser.add_argument(
        "--freq",
        dest="resample_frequency",
        help="Alias for --resample-frequency.",
    )
    parser.add_argument(
        "--max-lag-seconds",
        type=int,
        default=60,
        help="Maximum lag in seconds on each side of zero. Defaults to 60.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Print diagnostics about duplicate timestamps and zero-change shares after resampling.",
    )
    return parser.parse_args()


def build_per_game_summary(groups: list[dict[str, object]], peak_lag: int) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for group in groups:
        x, y = lagged_views(group["d_espn"], group["d_market"], peak_lag)
        if len(x) < 2:
            corr = float("nan")
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        rows.append(
            {
                "game_id": group["game_id"],
                "team": group["team"],
                "correlation_at_peak_lag": corr,
            }
        )
    per_team = pd.DataFrame(rows)
    return per_team.groupby("game_id", as_index=False)["correlation_at_peak_lag"].mean()


def plot_lag_curve(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["lag_seconds"], results["correlation"], marker="o", linewidth=1.5, markersize=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Time Lead-Lag Relationship: ESPN vs Market")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.resample_frequency)
    frequency_seconds = frequency_to_seconds(frequency)
    frame = load_lead_lag_datasets(args.input_dir)
    if frame.empty:
        print("No lead-lag dataset files found.")
        return

    if args.diagnostics:
        input_diag = build_input_diagnostics(frame)
        print("input diagnostics")
        print(f"rows: {input_diag['num_rows']}")
        print(f"games: {input_diag['num_games']}")
        print(f"game-team groups: {input_diag['num_groups']}")
        print(f"duplicate game/team/timestamp rows: {input_diag['duplicate_game_team_timestamps']}")
        print("note: the processed lead-lag dataset is built with merge_asof onto ESPN timestamps, so market timing has already been collapsed to ESPN event times.")

    groups = build_resampled_groups(frame, frequency)
    if not groups:
        print("No groups remain after resampling.")
        return

    group_diag = build_resampled_diagnostics(groups)
    print(f"frequency: {frequency}")
    print(f"number of games analyzed: {group_diag['num_games']}")
    print(f"number of game-team groups analyzed: {group_diag['num_groups']}")
    print(f"number of observations: {group_diag['num_observations']}")
    if args.diagnostics:
        print(f"share of zero d_espn: {group_diag['share_zero_d_espn']:.6f}")
        print(f"share of zero d_market: {group_diag['share_zero_d_market']:.6f}")

    results = build_lag_correlation_table(groups, args.max_lag_seconds, frequency_seconds=frequency_seconds)
    print("lag_seconds | correlation")
    for row in results.itertuples(index=False):
        corr_text = "nan" if pd.isna(row.correlation) else f"{row.correlation:.6f}"
        print(f"{row.lag_seconds:>3} | {corr_text}")

    plot_lag_curve(results, args.output_plot)

    valid_results = results.dropna(subset=["correlation"])
    if valid_results.empty:
        print("No valid lag correlations computed.")
        return

    peak_row = valid_results.loc[valid_results["correlation"].idxmax()]
    peak_lag = int(peak_row["lag_seconds"])
    peak_lag_periods = int(peak_row["lag_periods"])
    per_game = build_per_game_summary(groups, peak_lag_periods)
    mean_corr = per_game["correlation_at_peak_lag"].dropna().mean()

    print(f"peak correlation: {peak_row['correlation']:.6f}")
    print(f"lag of peak: {peak_lag:+d} seconds")
    print(f"mean per-game correlation: {mean_corr:.6f}")
    print(f"paired observations at peak lag: {int(peak_row['num_observations'])}")
    print(f"Plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
