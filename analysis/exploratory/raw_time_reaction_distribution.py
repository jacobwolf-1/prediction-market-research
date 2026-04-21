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

from analysis.raw_time_utils import build_per_group_peak_summary, build_raw_time_groups, frequency_to_seconds, normalize_frequency_label
from utils.ingestion_utils import DATA_DIR, PROJECT_ROOT, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute raw-time reaction-time distribution.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--max-lag-seconds", type=int, default=60)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "raw_time_reaction_times.csv",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "raw_time_reaction_histogram.png",
    )
    return parser.parse_args()


def plot_histogram(frame: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    valid = frame["peak_lag_seconds"].dropna()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(valid, bins=range(int(valid.min()) - 1, int(valid.max()) + 2), alpha=0.8, edgecolor="black")
    ax.axvline(valid.mean(), color="black", linestyle="--", linewidth=1, label=f"mean={valid.mean():.2f}s")
    ax.set_xlabel("Peak Lag (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Raw-Time Reaction Distribution")
    ax.legend()
    ax.grid(True, alpha=0.25)
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
    per_group = build_per_group_peak_summary(
        groups,
        max_lag_seconds=args.max_lag_seconds,
        frequency_seconds=frequency_to_seconds(frequency),
    )
    ensure_directory(args.output_csv.parent)
    per_group.to_csv(args.output_csv, index=False)
    plot_histogram(per_group, args.output_plot)

    valid = per_group["peak_lag_seconds"].dropna()
    quantiles = valid.quantile([0.10, 0.25, 0.50, 0.75, 0.90])
    print(f"mean peak lag: {valid.mean():.6f}")
    print(f"median peak lag: {valid.median():.6f}")
    print(f"std dev: {valid.std():.6f}")
    print(f"10% quantile: {quantiles.loc[0.10]:.6f}")
    print(f"25% quantile: {quantiles.loc[0.25]:.6f}")
    print(f"50% quantile: {quantiles.loc[0.50]:.6f}")
    print(f"75% quantile: {quantiles.loc[0.75]:.6f}")
    print(f"90% quantile: {quantiles.loc[0.90]:.6f}")
    print(f"csv saved to: {args.output_csv}")
    print(f"plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
