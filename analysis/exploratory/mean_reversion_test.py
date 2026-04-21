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
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.ingestion_utils import PROCESSED_DIR, PROJECT_ROOT, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test Polymarket mean reversion toward ESPN probabilities.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR / "lead_lag_dataset",
        help="Directory containing lead-lag parquet files.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=20,
        help="Maximum forward lag to test.",
    )
    parser.add_argument(
        "--min-spread",
        type=float,
        default=0.0,
        help="Optional absolute spread filter.",
    )
    parser.add_argument(
        "--curve-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "mean_reversion_curve.png",
        help="Path for the beta-by-lag plot.",
    )
    parser.add_argument(
        "--scatter-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "spread_vs_future_return.png",
        help="Path for the spread vs future-return scatter plot.",
    )
    return parser.parse_args()


def load_dataset(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def prepare_dataset(frame: pd.DataFrame, min_spread: float) -> pd.DataFrame:
    prepared = frame.dropna(subset=["market_probability", "espn_probability", "delta_market"]).copy()
    prepared["spread"] = prepared["market_probability"] - prepared["espn_probability"]
    if min_spread > 0:
        prepared = prepared.loc[prepared["spread"].abs() >= min_spread].copy()
    return prepared.sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def fit_simple_regression(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return np.nan, np.nan, len(valid)
    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=float)
    x_mean = x_values.mean()
    y_mean = y_values.mean()
    x_centered = x_values - x_mean
    y_centered = y_values - y_mean
    denom = np.dot(x_centered, x_centered)
    if denom == 0:
        return np.nan, np.nan, len(valid)
    beta = float(np.dot(x_centered, y_centered) / denom)
    intercept = y_mean - beta * x_mean
    fitted = intercept + beta * x_values
    ss_res = float(np.sum((y_values - fitted) ** 2))
    ss_tot = float(np.sum((y_values - y_mean) ** 2))
    r_squared = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return beta, r_squared, len(valid)


def build_regression_table(frame: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    rows = []
    grouped_market = frame.groupby(["game_id", "team"], group_keys=False)["delta_market"]
    for lag in range(1, max_lag + 1):
        future_market_move = grouped_market.shift(-lag)
        beta, r_squared, num_observations = fit_simple_regression(frame["spread"], future_market_move)
        rows.append(
            {
                "lag": lag,
                "beta": beta,
                "r_squared": r_squared,
                "num_observations": num_observations,
            }
        )
    return pd.DataFrame(rows)


def plot_beta_curve(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["lag"], results["beta"], marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Beta")
    ax.set_title("Mean Reversion: Future Market Move vs Spread")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_scatter(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    scatter_frame = frame.copy()
    scatter_frame["future_market_move_5"] = scatter_frame.groupby(["game_id", "team"], group_keys=False)[
        "delta_market"
    ].shift(-5)
    scatter_frame = scatter_frame.dropna(subset=["spread", "future_market_move_5"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(scatter_frame["spread"], scatter_frame["future_market_move_5"], alpha=0.25, s=12)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Spread")
    ax.set_ylabel("Future Market Move (lag 5)")
    ax.set_title("Spread vs Future Market Return (lag 5)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_dataset(args.input_dir)
    if frame.empty:
        print("No lead-lag dataset files found.")
        return

    prepared = prepare_dataset(frame, args.min_spread)
    if prepared.empty:
        print("No rows remain after spread filtering.")
        return

    print(f"mean spread: {prepared['spread'].mean():.6f}")
    print(f"std spread: {prepared['spread'].std():.6f}")
    print(f"number of rows: {len(prepared)}")
    print(f"number of games: {prepared['game_id'].nunique()}")

    results = build_regression_table(prepared, args.max_lag)
    print("lag | beta | r_squared | num_observations")
    for row in results.itertuples(index=False):
        beta_text = "nan" if pd.isna(row.beta) else f"{row.beta:.6f}"
        r2_text = "nan" if pd.isna(row.r_squared) else f"{row.r_squared:.6f}"
        print(f"{row.lag:>3} | {beta_text:>8} | {r2_text:>9} | {row.num_observations}")

    plot_beta_curve(results, args.curve_plot)
    plot_scatter(prepared, args.scatter_plot)
    print(f"Beta curve saved to: {args.curve_plot}")
    print(f"Scatter plot saved to: {args.scatter_plot}")


if __name__ == "__main__":
    main()
