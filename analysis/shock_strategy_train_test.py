from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.backtest_shock_strategy import HORIZONS_SECONDS, SHOCK_THRESHOLDS, prepare_shock_frame, simulate_trades, summarize_strategy
from analysis.raw_time_utils import frequency_to_seconds, normalize_frequency_label
from analysis.shock_strategy_utils import build_shock_groups, filter_groups_by_liquidity
from utils.ingestion_utils import DATA_DIR, RAW_DIR, load_game_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/test evaluation for the shock strategy.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--game-ids-file", type=Path, default=RAW_DIR / "game_ids.csv")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--min-market-updates-per-minute", type=float, default=None)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "shock_train_test_results.csv",
    )
    return parser.parse_args()


def attach_season(frame: pd.DataFrame, game_ids_file: Path) -> pd.DataFrame:
    mapping = load_game_ids(game_ids_file)[["game_id", "season"]].copy()
    mapping["game_id"] = mapping["game_id"].astype(str)
    return frame.merge(mapping, on="game_id", how="left")


def evaluate_grid(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[tuple[float, int], pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    trades_by_key: dict[tuple[float, int], pd.DataFrame] = {}
    for threshold in SHOCK_THRESHOLDS:
        for horizon in HORIZONS_SECONDS:
            trades, diagnostics = simulate_trades(frame, threshold=threshold, horizon=horizon)
            trades_by_key[(threshold, horizon)] = trades
            rows.append(summarize_strategy(trades, threshold=threshold, horizon=horizon, diagnostics=diagnostics))
    return pd.DataFrame(rows), trades_by_key


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    frequency_seconds = frequency_to_seconds(frequency)
    groups, _ = build_shock_groups(args.espn_dir, args.market_dir, frequency)
    groups = filter_groups_by_liquidity(groups, args.min_market_updates_per_minute)
    if not groups:
        print("No groups available after liquidity filter.")
        return

    frame = prepare_shock_frame(groups, frequency_seconds)
    frame = attach_season(frame, args.game_ids_file)

    train = frame.loc[frame["season"].isin([2021, 2022, 2023])].copy()
    test = frame.loc[frame["season"].isin([2024, 2025])].copy()

    if train.empty or test.empty:
        print("Insufficient season coverage for train/test split.")
        print(f"train_rows: {len(train)}")
        print(f"test_rows: {len(test)}")
        return

    train_results, _ = evaluate_grid(train)
    best_train = train_results.loc[train_results["total_pnl"].idxmax()]
    best_key = (float(best_train["threshold"]), int(best_train["horizon"]))

    test_trades, test_diag = simulate_trades(test, threshold=best_key[0], horizon=best_key[1])
    test_summary = summarize_strategy(test_trades, threshold=best_key[0], horizon=best_key[1], diagnostics=test_diag)

    output = pd.DataFrame(
        [
            {"split": "train_best", **best_train.to_dict()},
            {"split": "test_fixed", **test_summary},
        ]
    )
    output.to_csv(args.output_csv, index=False)

    print(output.to_string(index=False))
    print(f"output saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
