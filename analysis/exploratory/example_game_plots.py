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

from analysis.time_series_utils import load_lead_lag_datasets
from utils.ingestion_utils import PROCESSED_DIR, PROJECT_ROOT, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot example games for visual ESPN vs market sanity checks.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROCESSED_DIR / "lead_lag_dataset",
        help="Directory containing merged lead-lag parquet files.",
    )
    parser.add_argument(
        "--market-dir",
        type=Path,
        default=RAW_DIR / "kalshi",
        help="Directory containing raw Kalshi parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "visualizations",
        help="Directory for example game plots.",
    )
    parser.add_argument(
        "--shock-threshold",
        type=float,
        default=0.05,
        help="Absolute ESPN change threshold to mark on the plot.",
    )
    return parser.parse_args()


def raw_market_row_counts(market_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(market_dir.glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["timestamp"])
        rows.append({"game_id": path.stem, "raw_market_rows": len(frame)})
    return pd.DataFrame(rows)


def choose_example_games(counts: pd.DataFrame) -> list[str]:
    ordered = counts.sort_values("raw_market_rows").reset_index(drop=True)
    low = ordered.iloc[0]["game_id"]
    median = ordered.iloc[len(ordered) // 2]["game_id"]
    high = ordered.iloc[-1]["game_id"]
    return [str(high), str(median), str(low)]


def plot_game(game_frame: pd.DataFrame, output_path: Path, shock_threshold: float) -> None:
    ensure_directory(output_path.parent)
    ordered = game_frame.sort_values("timestamp").copy()
    ordered["delta_espn"] = ordered.groupby("team")["espn_probability"].diff()
    shocks = ordered.loc[ordered["delta_espn"].abs() >= shock_threshold].copy()

    fig, ax = plt.subplots(figsize=(11, 5))
    for team, team_frame in ordered.groupby("team", sort=True):
        ax.plot(team_frame["timestamp"], team_frame["espn_probability"], linewidth=1.5, label=f"{team} ESPN")
        ax.plot(team_frame["timestamp"], team_frame["market_probability"], linewidth=1.5, linestyle="--", label=f"{team} Market")
    if not shocks.empty:
        ax.scatter(shocks["timestamp"], shocks["espn_probability"], color="black", s=14, alpha=0.6, label="Large ESPN jump")
    ax.set_title(f"ESPN vs Market Probability: Game {ordered['game_id'].iloc[0]}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    merged = load_lead_lag_datasets(args.input_dir)
    if merged.empty:
        print("No lead-lag dataset files found.")
        return

    counts = raw_market_row_counts(args.market_dir)
    if counts.empty:
        print("No raw market files found.")
        return

    chosen_game_ids = choose_example_games(counts)
    labels = ["high", "median", "low"]
    for label, game_id in zip(labels, chosen_game_ids, strict=True):
        game_frame = merged.loc[merged["game_id"].astype(str) == str(game_id)].copy()
        if game_frame.empty:
            continue
        output_path = args.output_dir / f"example_game_{game_id}.png"
        plot_game(game_frame, output_path, args.shock_threshold)
        raw_rows = int(counts.loc[counts["game_id"] == str(game_id), "raw_market_rows"].iloc[0])
        print(f"{label}-liquidity game: {game_id} raw_rows={raw_rows} plot={output_path}")


if __name__ == "__main__":
    main()
