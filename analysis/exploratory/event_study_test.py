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
    parser = argparse.ArgumentParser(description="Run an event study on ESPN shocks and future Polymarket moves.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR / "lead_lag_dataset",
        help="Directory containing per-game lead-lag parquet files.",
    )
    parser.add_argument(
        "--shock-threshold",
        type=float,
        default=0.05,
        help="Absolute ESPN delta threshold used to define a shock event.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=20,
        help="Maximum number of future rows to evaluate after a shock.",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "event_study_market_reaction.png",
        help="Path for the event-study reaction curve plot.",
    )
    return parser.parse_args()


def load_dataset(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    return frame.sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def clean_dataset(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.dropna(subset=["delta_espn", "delta_market"]).copy()
    return cleaned.sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def build_event_study_table(frame: pd.DataFrame, shock_threshold: float, max_lag: int) -> tuple[pd.DataFrame, int]:
    shocks = frame.loc[frame["delta_espn"].abs() >= shock_threshold].copy()
    rows = []
    for lag in range(1, max_lag + 1):
        future_move = shocks.groupby(["game_id", "team"], group_keys=False)["delta_market"].shift(-lag)
        rows.append(
            {
                "lag": lag,
                "avg_future_market_move": future_move.mean(),
                "num_events": int(future_move.notna().sum()),
            }
        )
    return pd.DataFrame(rows), len(shocks)


def plot_event_study(results: pd.DataFrame, output_path: Path, shock_threshold: float) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(results["lag"], results["avg_future_market_move"], marker="o", linewidth=1.5, markersize=4)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Average Future Market Move")
    ax.set_title(f"Market Reaction After ESPN Shocks (threshold={shock_threshold:.3f})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frame = load_dataset(args.input_dir)
    if frame.empty:
        print("No lead-lag datasets found.")
        return

    cleaned = clean_dataset(frame)
    if cleaned.empty:
        print("No rows remain after dropping missing deltas.")
        return

    results, shock_count = build_event_study_table(cleaned, args.shock_threshold, args.max_lag)

    print(f"total rows: {len(cleaned)}")
    print(f"shock events: {shock_count}")
    print("lag | avg_future_market_move | num_events")
    for row in results.itertuples(index=False):
        move_text = "nan" if pd.isna(row.avg_future_market_move) else f"{row.avg_future_market_move:.6f}"
        print(f"{row.lag:>3} | {move_text:>22} | {row.num_events}")

    plot_event_study(results, args.output_plot, args.shock_threshold)
    print(f"Plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
