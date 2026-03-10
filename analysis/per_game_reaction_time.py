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

from analysis.time_series_utils import (
    build_lag_correlation_table,
    build_resampled_groups,
    frequency_to_seconds,
    load_lead_lag_datasets,
    normalize_frequency_label,
)
from utils.ingestion_utils import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute per-game peak reaction times for ESPN vs market changes.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR / "lead_lag_dataset",
        help="Directory containing merged lead-lag parquet files.",
    )
    parser.add_argument(
        "--freq",
        default="1s",
        help="Fixed time grid used for resampling. Defaults to 1S.",
    )
    parser.add_argument(
        "--max-lag-seconds",
        type=int,
        default=60,
        help="Maximum lag in seconds on each side of zero. Defaults to 60.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "per_game_reaction_times.csv",
        help="CSV path for per-game reaction-time summary.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "per_game_reaction_histogram.png",
        help="Histogram path for per-game peak lags.",
    )
    return parser.parse_args()


def build_per_game_table(groups: list[dict[str, object]], max_lag_seconds: int, frequency_seconds: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    grouped_by_game: dict[str, list[dict[str, object]]] = {}
    for group in groups:
        grouped_by_game.setdefault(str(group["game_id"]), []).append(group)

    for game_id, game_groups in sorted(grouped_by_game.items()):
        results = build_lag_correlation_table(game_groups, max_lag_seconds, frequency_seconds=frequency_seconds)
        valid = results.dropna(subset=["correlation"])
        if valid.empty:
            peak_corr = np.nan
            peak_lag = np.nan
            peak_obs = 0
        else:
            peak = valid.loc[valid["correlation"].idxmax()]
            peak_corr = float(peak["correlation"])
            peak_lag = int(peak["lag_seconds"])
            peak_obs = int(peak["num_observations"])
        rows.append(
            {
                "game_id": game_id,
                "teams": "/".join(sorted(str(group["team"]) for group in game_groups)),
                "peak_correlation": peak_corr,
                "lag_seconds_of_peak": peak_lag,
                "num_observations": peak_obs,
            }
        )
    frame = pd.DataFrame(rows)
    return frame.sort_values(["game_id"]).reset_index(drop=True)


def plot_histogram(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    valid = frame["lag_seconds_of_peak"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valid, bins=range(int(valid.min()) - 1, int(valid.max()) + 2), alpha=0.8, edgecolor="black")
    ax.axvline(valid.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={valid.mean():.2f}s")
    ax.set_xlabel("Peak Lag (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Per-Game Peak Reaction-Time Distribution")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    frequency_seconds = frequency_to_seconds(frequency)
    frame = load_lead_lag_datasets(args.input_dir)
    if frame.empty:
        print("No lead-lag dataset files found.")
        return

    groups = build_resampled_groups(frame, frequency)
    if not groups:
        print("No groups remain after resampling.")
        return

    per_game = build_per_game_table(groups, args.max_lag_seconds, frequency_seconds)
    ensure_directory(args.output_csv.parent)
    per_game.to_csv(args.output_csv, index=False)
    plot_histogram(per_game, args.output_plot)

    valid_lags = per_game["lag_seconds_of_peak"].dropna()
    quantiles = valid_lags.quantile([0.10, 0.25, 0.50, 0.75, 0.90])

    print(f"frequency: {frequency}")
    print(f"rows written: {len(per_game)}")
    print(f"games analyzed: {per_game['game_id'].nunique()}")
    print(f"mean peak lag: {valid_lags.mean():.6f}")
    print(f"median peak lag: {valid_lags.median():.6f}")
    print(f"std dev peak lag: {valid_lags.std():.6f}")
    print(f"10% quantile: {quantiles.loc[0.10]:.6f}")
    print(f"25% quantile: {quantiles.loc[0.25]:.6f}")
    print(f"50% quantile: {quantiles.loc[0.50]:.6f}")
    print(f"75% quantile: {quantiles.loc[0.75]:.6f}")
    print(f"90% quantile: {quantiles.loc[0.90]:.6f}")
    print(f"CSV saved to: {args.output_csv}")
    print(f"Histogram saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
