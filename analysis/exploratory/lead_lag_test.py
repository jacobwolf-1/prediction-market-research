from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.ingestion_utils import PROCESSED_DIR, PROJECT_ROOT, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lead-lag correlation analysis for ESPN vs Polymarket.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR / "lead_lag_dataset",
        help="Directory containing merged lead-lag parquet files.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "lead_lag_curve.png",
        help="Path for the lead-lag correlation plot.",
    )
    parser.add_argument(
        "--min-delta-espn",
        type=float,
        default=0.03,
        help="Absolute minimum ESPN probability change to keep.",
    )
    return parser.parse_args()


def load_datasets(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def clean_dataset(frame: pd.DataFrame, min_delta_espn: float) -> pd.DataFrame:
    cleaned = frame.dropna(subset=["delta_espn", "delta_market"]).copy()
    cleaned = cleaned.loc[cleaned["delta_espn"].abs() > min_delta_espn].copy()
    return cleaned.sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def correlation_at_lag(frame: pd.DataFrame, lag: int) -> float:
    shifted = frame.groupby(["game_id", "team"], group_keys=False)["delta_espn"].shift(lag)
    return shifted.corr(frame["delta_market"])


def build_lag_correlation_table(frame: pd.DataFrame, lag_min: int = -30, lag_max: int = 30) -> pd.DataFrame:
    rows = []
    for lag in range(lag_min, lag_max + 1):
        rows.append({"lag": lag, "correlation": correlation_at_lag(frame, lag)})
    return pd.DataFrame(rows)


def build_per_game_summary(frame: pd.DataFrame, lag_results: pd.DataFrame) -> pd.DataFrame:
    peak_row = lag_results.loc[lag_results["correlation"].idxmax()]
    peak_lag = int(peak_row["lag"])
    rows = []
    for game_id, game_frame in frame.groupby("game_id", sort=True):
        corr = correlation_at_lag(game_frame.sort_values(["team", "timestamp"]), peak_lag)
        rows.append({"game_id": str(game_id), "correlation_at_peak_lag": corr})
    return pd.DataFrame(rows)


def plot_lag_curve(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["lag"], results["correlation"], marker="o", linewidth=1.5, markersize=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Lead-Lag Relationship: ESPN vs Polymarket")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_datasets(args.input_dir)
    if frame.empty:
        print("No lead-lag dataset files found.")
        return

    cleaned = clean_dataset(frame, args.min_delta_espn)
    if cleaned.empty:
        print("No rows remain after cleaning.")
        return

    print(f"total rows: {len(cleaned)}")
    print(f"number of games: {cleaned['game_id'].nunique()}")
    print(f"number of teams: {cleaned[['game_id', 'team']].drop_duplicates().shape[0]}")

    results = build_lag_correlation_table(cleaned)
    print("lag | correlation")
    for row in results.itertuples(index=False):
        corr_text = "nan" if pd.isna(row.correlation) else f"{row.correlation:.6f}"
        print(f"{row.lag:>3} | {corr_text}")

    plot_lag_curve(results, args.output_plot)

    valid_results = results.dropna(subset=["correlation"])
    if valid_results.empty:
        print("No valid lag correlations computed.")
        return
    peak_row = valid_results.loc[valid_results["correlation"].idxmax()]
    print(f"Peak correlation: {peak_row['correlation']:.6f}")
    print(f"Lag: {int(peak_row['lag']):+d}")

    per_game = build_per_game_summary(cleaned, valid_results)
    mean_corr = per_game["correlation_at_peak_lag"].dropna().mean()
    print(f"Mean per-game correlation at peak lag: {mean_corr:.6f}")
    print(f"Plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
