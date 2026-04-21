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
from utils.ingestion_utils import DATA_DIR, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leave-one-game-out cross-validation for the shock strategy.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--min-market-updates-per-minute", type=float, default=None)
    parser.add_argument("--max-held-out-games", type=int, default=None)
    parser.add_argument(
        "--fold-output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "shock_cv_folds.csv",
    )
    parser.add_argument(
        "--summary-output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "shock_cv_summary.csv",
    )
    return parser.parse_args()


def evaluate_grid(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for threshold in SHOCK_THRESHOLDS:
        for horizon in HORIZONS_SECONDS:
            trades, diagnostics = simulate_trades(frame, threshold=threshold, horizon=horizon)
            rows.append(summarize_strategy(trades, threshold=threshold, horizon=horizon, diagnostics=diagnostics))
    return pd.DataFrame(rows)


def choose_best_train_config(results: pd.DataFrame) -> pd.Series | None:
    valid = results.loc[results["trades"] > 0].copy()
    if valid.empty:
        return None
    valid = valid.sort_values(["total_pnl", "avg_return", "trades"], ascending=[False, False, False]).reset_index(drop=True)
    return valid.iloc[0]


def build_summary(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame(columns=["metric", "value"])
    rows = [
        {"metric": "num_games", "value": int(folds["held_out_game_id"].nunique())},
        {"metric": "num_positive_test_games", "value": int((folds["test_total_pnl"] > 0).sum())},
        {"metric": "mean_test_total_pnl", "value": float(folds["test_total_pnl"].mean())},
        {"metric": "median_test_total_pnl", "value": float(folds["test_total_pnl"].median())},
        {"metric": "sum_test_total_pnl", "value": float(folds["test_total_pnl"].sum())},
        {"metric": "mean_test_avg_return", "value": float(folds["test_avg_return"].mean())},
        {"metric": "mean_test_win_rate", "value": float(folds["test_win_rate"].mean())},
        {"metric": "mean_test_trades", "value": float(folds["test_trades"].mean())},
    ]

    best_threshold = folds["selected_threshold"].mode(dropna=True)
    if not best_threshold.empty:
        rows.append({"metric": "most_common_threshold", "value": float(best_threshold.iloc[0])})
    best_horizon = folds["selected_horizon"].mode(dropna=True)
    if not best_horizon.empty:
        rows.append({"metric": "most_common_horizon", "value": int(best_horizon.iloc[0])})
    return pd.DataFrame(rows)


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
    if frame.empty:
        print("No shock frame available.")
        return

    game_ids = sorted(frame["game_id"].astype(str).unique().tolist())
    if args.max_held_out_games is not None:
        game_ids = game_ids[: args.max_held_out_games]
    fold_rows: list[dict[str, object]] = []

    for held_out_game_id in game_ids:
        train = frame.loc[frame["game_id"] != held_out_game_id].copy()
        test = frame.loc[frame["game_id"] == held_out_game_id].copy()
        if train.empty or test.empty:
            continue

        train_results = evaluate_grid(train)
        best_train = choose_best_train_config(train_results)
        if best_train is None:
            continue

        threshold = float(best_train["threshold"])
        horizon = int(best_train["horizon"])
        test_trades, test_diag = simulate_trades(test, threshold=threshold, horizon=horizon)
        test_summary = summarize_strategy(test_trades, threshold=threshold, horizon=horizon, diagnostics=test_diag)

        fold_rows.append(
            {
                "held_out_game_id": held_out_game_id,
                "selected_threshold": threshold,
                "selected_horizon": horizon,
                "train_total_pnl": float(best_train["total_pnl"]),
                "train_avg_return": float(best_train["avg_return"]),
                "train_win_rate": float(best_train["win_rate"]),
                "train_trades": int(best_train["trades"]),
                "test_total_pnl": float(test_summary["total_pnl"]),
                "test_avg_return": float(test_summary["avg_return"]) if pd.notna(test_summary["avg_return"]) else float("nan"),
                "test_win_rate": float(test_summary["win_rate"]) if pd.notna(test_summary["win_rate"]) else float("nan"),
                "test_trades": int(test_summary["trades"]),
                "test_ignored_signals": int(test_summary["ignored_signals"]),
            }
        )

    folds = pd.DataFrame(fold_rows).sort_values("held_out_game_id").reset_index(drop=True) if fold_rows else pd.DataFrame()
    summary = build_summary(folds)

    ensure_directory(args.fold_output_csv.parent)
    folds.to_csv(args.fold_output_csv, index=False)
    summary.to_csv(args.summary_output_csv, index=False)

    if folds.empty:
        print("No valid folds evaluated.")
        print(f"fold output saved to: {args.fold_output_csv}")
        print(f"summary output saved to: {args.summary_output_csv}")
        return

    print(summary.to_string(index=False))
    print(f"fold output saved to: {args.fold_output_csv}")
    print(f"summary output saved to: {args.summary_output_csv}")


if __name__ == "__main__":
    main()
