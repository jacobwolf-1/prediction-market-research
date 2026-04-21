from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from analysis.raw_time_dataset import load_raw_time_dataset
from utils.ingestion_utils import PROJECT_ROOT, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate raw timestamp alignment between ESPN shocks and Kalshi updates.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--shock-threshold", type=float, default=0.05)
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "espn_market_delay_distribution.png",
    )
    return parser.parse_args()


def summarize_delays(values: pd.Series) -> dict[str, float]:
    clean = values.dropna()
    if clean.empty:
        return {"count": 0, "mean": np.nan, "median": np.nan, "std": np.nan, "p5": np.nan, "p95": np.nan}
    return {
        "count": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=1)) if len(clean) > 1 else np.nan,
        "p5": float(clean.quantile(0.05)),
        "p95": float(clean.quantile(0.95)),
    }


def compute_delay_tables(dataset: dict[tuple[str, str], dict[str, pd.DataFrame]], shock_threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    shock_rows: list[dict[str, object]] = []
    nearest_rows: list[dict[str, object]] = []

    for (game_id, team), item in dataset.items():
        espn = item["espn"].sort_values("timestamp").copy()
        market = item["market"].sort_values("timestamp").copy()
        if espn.empty or market.empty:
            continue

        espn["shock"] = espn["espn_probability"].diff()
        espn = espn.dropna(subset=["timestamp", "shock"])
        market_times = market["timestamp"].to_numpy(dtype="datetime64[ns]")
        if len(market_times) == 0:
            continue

        for row in espn.itertuples(index=False):
            espn_time = row.timestamp.to_datetime64()
            idx = np.searchsorted(market_times, espn_time, side="left")

            prev_delay = np.nan
            next_delay = np.nan
            nearest_delay = np.nan

            if idx > 0:
                prev_delay = float((pd.Timestamp(market_times[idx - 1]).tz_localize("UTC") - row.timestamp).total_seconds())
            if idx < len(market_times):
                next_delay = float((pd.Timestamp(market_times[idx]).tz_localize("UTC") - row.timestamp).total_seconds())

            candidates = [delay for delay in (prev_delay, next_delay) if pd.notna(delay)]
            if candidates:
                nearest_delay = min(candidates, key=lambda value: abs(value))

            nearest_rows.append(
                {
                    "game_id": str(game_id),
                    "team": str(team),
                    "espn_time": row.timestamp,
                    "nearest_market_delay_seconds": nearest_delay,
                    "prev_market_delay_seconds": prev_delay,
                    "next_market_delay_seconds": next_delay,
                }
            )

            if abs(float(row.shock)) <= shock_threshold or pd.isna(next_delay):
                continue

            shock_rows.append(
                {
                    "game_id": str(game_id),
                    "team": str(team),
                    "espn_time": row.timestamp,
                    "shock": float(row.shock),
                    "first_subsequent_market_delay_seconds": next_delay,
                    "nearest_market_delay_seconds": nearest_delay,
                }
            )

    return pd.DataFrame(shock_rows), pd.DataFrame(nearest_rows)


def plot_distributions(shock_delays: pd.Series, nearest_delays: pd.Series, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(shock_delays.dropna(), bins=50, color="#2b6cb0", alpha=0.8)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("Shock to First Subsequent Market Update")
    axes[0].set_xlabel("Delay (seconds)")
    axes[0].set_ylabel("Count")

    axes[1].hist(nearest_delays.dropna(), bins=50, color="#c05621", alpha=0.8)
    axes[1].axvline(0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("ESPN Update to Nearest Market Update")
    axes[1].set_xlabel("Signed Delay (seconds)")
    axes[1].set_ylabel("Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_raw_time_dataset(args.espn_dir, args.market_dir)
    shock_table, nearest_table = compute_delay_tables(dataset, args.shock_threshold)

    shock_summary = summarize_delays(shock_table["first_subsequent_market_delay_seconds"]) if not shock_table.empty else summarize_delays(pd.Series(dtype=float))
    nearest_summary = summarize_delays(nearest_table["nearest_market_delay_seconds"]) if not nearest_table.empty else summarize_delays(pd.Series(dtype=float))
    plot_distributions(
        shock_table["first_subsequent_market_delay_seconds"] if not shock_table.empty else pd.Series(dtype=float),
        nearest_table["nearest_market_delay_seconds"] if not nearest_table.empty else pd.Series(dtype=float),
        args.output_plot,
    )

    print("shock_to_first_subsequent_market_update")
    print(pd.Series(shock_summary).to_string())
    print("")
    print("espn_update_to_nearest_market_update")
    print(pd.Series(nearest_summary).to_string())
    print("")
    print(f"shock_count: {len(shock_table)}")
    print(f"espn_update_count: {len(nearest_table)}")
    print(f"plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
