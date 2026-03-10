from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.raw_time_dataset import load_raw_time_dataset
from analysis.raw_time_utils import choose_example_games
from utils.ingestion_utils import PROJECT_ROOT, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot raw-time ESPN vs Kalshi probabilities for example games.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "visualizations")
    return parser.parse_args()


def plot_game(game_items: dict[tuple[str, str], dict[str, object]], game_id: str, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(11, 5))
    for (_, team), item in sorted(game_items.items()):
        espn = item["espn"]
        market = item["market"]
        ax.plot(espn["timestamp"], espn["espn_probability"], linewidth=1.5, label=f"{team} ESPN")
        ax.plot(market["timestamp"], market["market_probability"], linewidth=1.2, linestyle="--", label=f"{team} Market")
    ax.set_title(f"Raw-Time ESPN vs Kalshi: Game {game_id}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Probability")
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    dataset = load_raw_time_dataset(args.espn_dir, args.market_dir)
    if not dataset:
        print("No raw-time dataset available.")
        return
    chosen = choose_example_games(args.market_dir)
    by_game: dict[str, dict[tuple[str, str], dict[str, object]]] = {}
    for key, item in dataset.items():
        game_id, team = key
        by_game.setdefault(game_id, {})[(game_id, team)] = item

    for game_id in chosen:
        if game_id not in by_game:
            continue
        output_path = args.output_dir / f"raw_time_example_game_{game_id}.png"
        plot_game(by_game[game_id], game_id, output_path)
        print(f"plot saved: {output_path}")


if __name__ == "__main__":
    main()
