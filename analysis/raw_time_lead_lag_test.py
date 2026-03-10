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

from analysis.raw_time_utils import (
    build_lag_correlation_table,
    build_raw_time_groups,
    frequency_to_seconds,
    lagged_views,
    normalize_frequency_label,
)
from utils.ingestion_utils import PROJECT_ROOT, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run raw-time lead-lag analysis using independent ESPN and Kalshi timestamps.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--max-lag-seconds", type=int, default=60)
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "raw_time_lead_lag_curve.png",
    )
    return parser.parse_args()


def build_per_game_summary(groups: list[dict[str, object]], peak_lag_periods: int) -> pd.DataFrame:
    rows = []
    for group in groups:
        x, y = lagged_views(group["d_espn"], group["d_market"], peak_lag_periods)
        if len(x) < 2:
            corr = np.nan
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        rows.append({"game_id": group["game_id"], "correlation_at_peak_lag": corr})
    return pd.DataFrame(rows).groupby("game_id", as_index=False)["correlation_at_peak_lag"].mean()


def plot_lag_curve(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["lag_seconds"], results["correlation"], marker="o", linewidth=1.5, markersize=3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Raw-Time Lead-Lag Relationship: ESPN vs Kalshi")
    ax.set_xlabel("Lag (seconds)")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    frequency_seconds = frequency_to_seconds(frequency)
    groups, summary = build_raw_time_groups(args.espn_dir, args.market_dir, frequency)
    if not groups:
        print("No raw-time groups available.")
        return

    results = build_lag_correlation_table(groups, args.max_lag_seconds, frequency_seconds=frequency_seconds)
    plot_lag_curve(results, args.output_plot)
    valid = results.dropna(subset=["correlation"])
    peak = valid.loc[valid["correlation"].idxmax()]
    per_game = build_per_game_summary(groups, int(peak["lag_periods"]))

    print(f"number of games: {summary['games']}")
    print(f"number of groups: {summary['groups']}")
    print(f"total observations: {sum(group['n_obs'] for group in groups)}")
    print(f"peak correlation: {float(peak['correlation']):.6f}")
    print(f"lag of peak (seconds): {int(peak['lag_seconds']):+d}")
    print(f"mean per-game correlation: {per_game['correlation_at_peak_lag'].dropna().mean():.6f}")
    print(f"plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
