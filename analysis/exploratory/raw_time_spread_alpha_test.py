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

from analysis.raw_time_utils import build_raw_time_groups, fit_simple_regression, frequency_to_seconds, normalize_frequency_label
from utils.ingestion_utils import DATA_DIR, PROJECT_ROOT, RAW_DIR, ensure_directory


HORIZONS_SECONDS = [5, 10, 20, 30, 60]
SPREAD_BINS = np.arange(-0.50, 0.51, 0.10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw-time spread alpha analysis.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "raw_time_spread_alpha_table.csv",
    )
    parser.add_argument(
        "--curve-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "raw_time_spread_alpha_curve.png",
    )
    parser.add_argument(
        "--heatmap-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "raw_time_spread_alpha_heatmap.png",
    )
    return parser.parse_args()


def prepare_frame(groups: list[dict[str, object]], frequency_seconds: int) -> pd.DataFrame:
    frames = []
    for group in groups:
        frame = group["frame"].copy()
        frame["spread"] = frame["value_market"] - frame["value_espn"]
        for horizon in HORIZONS_SECONDS:
            periods = max(1, horizon // frequency_seconds)
            frame[f"future_market_return_{horizon}s"] = frame["value_market"].shift(-periods) - frame["value_market"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_alpha_table(frame: pd.DataFrame) -> pd.DataFrame:
    labels = [f"[{SPREAD_BINS[i]:.2f},{SPREAD_BINS[i + 1]:.2f})" for i in range(len(SPREAD_BINS) - 1)]
    working = frame.copy()
    working["spread_bin"] = pd.cut(working["spread"], bins=SPREAD_BINS, labels=labels, right=False, include_lowest=True)
    rows = []
    for horizon in HORIZONS_SECONDS:
        col = f"future_market_return_{horizon}s"
        grouped = working.dropna(subset=["spread_bin", col]).groupby("spread_bin", observed=False)
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


def plot_curve(table: pd.DataFrame, output_path: Path, horizon: int = 20) -> None:
    ensure_directory(output_path.parent)
    plot_frame = table.loc[table["horizon_seconds"] == horizon]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(plot_frame["spread_bin"], plot_frame["avg_future_market_return"], marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Spread Bin")
    ax.set_ylabel("Average Future Market Return")
    ax.set_title(f"Raw-Time Spread Alpha ({horizon}s horizon)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    pivot = table.pivot(index="spread_bin", columns="horizon_seconds", values="avg_future_market_return")
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="coolwarm", origin="lower")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(col) for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Spread Bin")
    ax.set_title("Raw-Time Spread Alpha Heatmap")
    fig.colorbar(im, ax=ax, label="Average Future Market Return")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    groups, _ = build_raw_time_groups(args.espn_dir, args.market_dir, frequency)
    if not groups:
        print("No raw-time groups available.")
        return
    frame = prepare_frame(groups, frequency_to_seconds(frequency))
    table = build_alpha_table(frame)
    ensure_directory(args.output_csv.parent)
    table.to_csv(args.output_csv, index=False)
    plot_curve(table, args.curve_plot)
    plot_heatmap(table, args.heatmap_plot)

    print(f"rows analyzed: {len(frame)}")
    print("horizon_seconds | beta | r_squared | n")
    for horizon in HORIZONS_SECONDS:
        col = f"future_market_return_{horizon}s"
        beta, r_squared, n = fit_simple_regression(frame["spread"], frame[col])
        beta_text = "nan" if pd.isna(beta) else f"{beta:.6f}"
        r2_text = "nan" if pd.isna(r_squared) else f"{r_squared:.6f}"
        print(f"{horizon:>15} | {beta_text:>8} | {r2_text:>9} | {n}")
    print(f"csv saved to: {args.output_csv}")
    print(f"curve plot saved to: {args.curve_plot}")
    print(f"heatmap saved to: {args.heatmap_plot}")


if __name__ == "__main__":
    main()
