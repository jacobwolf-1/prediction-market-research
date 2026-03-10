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
    build_resampled_frame,
    fit_simple_regression,
    frequency_to_seconds,
    load_lead_lag_datasets,
    normalize_frequency_label,
)
from utils.ingestion_utils import DATA_DIR, PROCESSED_DIR, PROJECT_ROOT, ensure_directory


HORIZONS_SECONDS = [5, 10, 20, 30, 60]
SPREAD_BINS = np.arange(-0.50, 0.51, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test spread alpha using time-resampled ESPN and market probabilities.")
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
        "--output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "spread_alpha_table.csv",
        help="CSV path for spread-bin alpha summary.",
    )
    parser.add_argument(
        "--curve-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "spread_alpha_curve.png",
        help="Path for the expected-return-vs-spread plot.",
    )
    parser.add_argument(
        "--heatmap-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "spread_alpha_heatmap.png",
        help="Path for the spread-vs-horizon heatmap.",
    )
    return parser.parse_args()


def prepare_resampled_frame(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    resampled = build_resampled_frame(frame, frequency)
    if resampled.empty:
        return resampled
    resampled["spread"] = resampled["market_probability"] - resampled["espn_probability"]
    grouped_market = resampled.groupby(["game_id", "team"], group_keys=False)["market_probability"]
    frequency_seconds = frequency_to_seconds(frequency)
    for horizon in HORIZONS_SECONDS:
        horizon_periods = horizon // frequency_seconds
        if horizon_periods < 1:
            horizon_periods = 1
        resampled[f"future_market_return_{horizon}s"] = grouped_market.shift(-horizon_periods) - resampled["market_probability"]
    return resampled


def build_spread_alpha_table(frame: pd.DataFrame) -> pd.DataFrame:
    labels = [f"[{SPREAD_BINS[i]:.2f},{SPREAD_BINS[i + 1]:.2f})" for i in range(len(SPREAD_BINS) - 1)]
    frame = frame.copy()
    frame["spread_bin"] = pd.cut(frame["spread"], bins=SPREAD_BINS, labels=labels, right=False, include_lowest=True)
    rows: list[dict[str, float | int | str]] = []
    for horizon in HORIZONS_SECONDS:
        col = f"future_market_return_{horizon}s"
        grouped = frame.dropna(subset=["spread_bin", col]).groupby("spread_bin", observed=False)
        for spread_bin, group in grouped:
            rows.append(
                {
                    "spread_bin": str(spread_bin),
                    "horizon_seconds": horizon,
                    "avg_future_market_return": float(group[col].mean()),
                    "count": int(len(group)),
                }
            )
    return pd.DataFrame(rows)


def plot_spread_curve(alpha_table: pd.DataFrame, output_path: Path, horizon: int = 20) -> None:
    ensure_directory(output_path.parent)
    plot_frame = alpha_table.loc[alpha_table["horizon_seconds"] == horizon].copy()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_frame["spread_bin"], plot_frame["avg_future_market_return"], marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Spread Bin")
    ax.set_ylabel("Average Future Market Return")
    ax.set_title(f"Expected Future Market Return vs Spread ({horizon}s horizon)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_heatmap(alpha_table: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    pivot = alpha_table.pivot(index="spread_bin", columns="horizon_seconds", values="avg_future_market_return")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Spread Bin")
    ax.set_title("Spread Alpha Heatmap")
    fig.colorbar(im, ax=ax, label="Average Future Market Return")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def run_regressions(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for horizon in HORIZONS_SECONDS:
        col = f"future_market_return_{horizon}s"
        beta, r_squared, n = fit_simple_regression(frame["spread"], frame[col])
        rows.append({"horizon_seconds": horizon, "beta": beta, "r_squared": r_squared, "n": n})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    frame = load_lead_lag_datasets(args.input_dir)
    if frame.empty:
        print("No lead-lag dataset files found.")
        return

    prepared = prepare_resampled_frame(frame, frequency)
    if prepared.empty:
        print("No rows remain after resampling.")
        return

    alpha_table = build_spread_alpha_table(prepared)
    ensure_directory(args.output_csv.parent)
    alpha_table.to_csv(args.output_csv, index=False)
    plot_spread_curve(alpha_table, args.curve_plot, horizon=20)
    plot_heatmap(alpha_table, args.heatmap_plot)

    regressions = run_regressions(prepared)

    print(f"frequency: {frequency}")
    print(f"rows analyzed: {len(prepared)}")
    print("horizon_seconds | beta | r_squared | n")
    for row in regressions.itertuples(index=False):
        beta_text = "nan" if pd.isna(row.beta) else f"{row.beta:.6f}"
        r2_text = "nan" if pd.isna(row.r_squared) else f"{row.r_squared:.6f}"
        print(f"{row.horizon_seconds:>15} | {beta_text:>8} | {r2_text:>9} | {row.n}")
    print(f"CSV saved to: {args.output_csv}")
    print(f"Curve plot saved to: {args.curve_plot}")
    print(f"Heatmap saved to: {args.heatmap_plot}")


if __name__ == "__main__":
    main()
