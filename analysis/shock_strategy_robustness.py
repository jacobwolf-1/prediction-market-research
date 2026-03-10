from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.raw_time_utils import frequency_to_seconds, normalize_frequency_label
from analysis.shock_strategy_utils import build_shock_groups, filter_groups_by_liquidity
from utils.ingestion_utils import DATA_DIR, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness diagnostics for a fixed shock strategy.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--min-market-updates-per-minute", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed",
    )
    return parser.parse_args()


def phase_tags(seconds_remaining: float, quarter: float) -> list[str]:
    tags: list[str] = []
    if pd.notna(quarter):
        tags.append(f"Q{int(quarter)}")
    if pd.notna(seconds_remaining):
        if seconds_remaining <= 8 * 60:
            tags.append("last_8m")
        if seconds_remaining <= 5 * 60:
            tags.append("last_5m")
        if seconds_remaining <= 2 * 60:
            tags.append("last_2m")
    return tags


def simulate_trades(groups: list[dict[str, object]], horizon: int, frequency_seconds: int, threshold: float) -> pd.DataFrame:
    periods = max(1, horizon // frequency_seconds)
    rows: list[dict[str, object]] = []
    for group in groups:
        frame = group["frame"].copy().sort_values("timestamp").reset_index(drop=True)
        frame["shock"] = frame["delta_espn"]
        frame["entry_price"] = frame["market_probability"].shift(-1)
        frame["exit_price"] = frame["market_probability"].shift(-(periods + 1))
        frame["exit_time"] = frame["timestamp"].shift(-(periods + 1))
        next_allowed_time = pd.Timestamp.min.tz_localize("UTC")

        for row in frame.itertuples(index=False):
            if abs(float(row.shock)) <= threshold:
                continue
            if row.timestamp < next_allowed_time:
                continue
            if pd.isna(row.entry_price) or pd.isna(row.exit_price) or pd.isna(row.exit_time):
                continue
            direction = 1 if row.shock > 0 else -1
            net_return = ((float(row.exit_price) - float(row.entry_price)) if direction == 1 else (float(row.entry_price) - float(row.exit_price))) - 0.02
            rows.append(
                {
                    "game_id": row.game_id,
                    "team": row.team,
                    "timestamp": row.timestamp,
                    "quarter": row.quarter,
                    "seconds_remaining": row.total_seconds_remaining,
                    "abs_shock": abs(float(row.shock)),
                    "net_return": net_return,
                }
            )
            next_allowed_time = row.exit_time
    return pd.DataFrame(rows)


def summarize_phase_rows(trades: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    expanded: list[dict[str, object]] = []
    for row in trades.itertuples(index=False):
        for tag in phase_tags(row.seconds_remaining, row.quarter):
            expanded.append({"phase": tag, "net_return": row.net_return})
    phase_frame = pd.DataFrame(expanded)
    for phase, group in phase_frame.groupby("phase", sort=True):
        rows.append(
            {
                "phase": phase,
                "trades": int(len(group)),
                "avg_return": float(group["net_return"].mean()),
                "median_return": float(group["net_return"].median()),
                "win_rate": float((group["net_return"] > 0).mean()),
                "total_pnl": float(group["net_return"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("phase").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    groups, _ = build_shock_groups(args.espn_dir, args.market_dir, frequency)
    groups = filter_groups_by_liquidity(groups, args.min_market_updates_per_minute)
    trades = simulate_trades(groups, args.horizon, frequency_to_seconds(frequency), args.threshold)
    if trades.empty:
        print("No trades generated.")
        return

    game_pnl = trades.groupby("game_id", as_index=False)["net_return"].sum().rename(columns={"net_return": "total_pnl"})
    shock_bins = pd.cut(
        trades["abs_shock"],
        bins=[0.03, 0.05, 0.07, 0.10, np.inf],
        labels=["0.03-0.05", "0.05-0.07", "0.07-0.10", ">0.10"],
        right=False,
        include_lowest=True,
    )
    shock_size = (
        trades.assign(shock_bin=shock_bins)
        .groupby("shock_bin", as_index=False)["net_return"]
        .agg(["mean", "median", "count", "sum"])
        .reset_index()
        .rename(columns={"mean": "avg_return", "median": "median_return", "count": "trades", "sum": "total_pnl"})
    )
    trade_dist = trades["net_return"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).to_frame(name="value").reset_index().rename(columns={"index": "metric"})
    phase = summarize_phase_rows(trades)

    ensure_directory(args.output_dir)
    game_pnl.to_csv(args.output_dir / "shock_game_pnl_distribution.csv", index=False)
    shock_size.to_csv(args.output_dir / "shock_shock_size_robustness.csv", index=False)
    trade_dist.to_csv(args.output_dir / "shock_trade_return_distribution.csv", index=False)
    phase.to_csv(args.output_dir / "shock_phase_breakdown.csv", index=False)

    print("pnl per game")
    print(game_pnl["total_pnl"].describe(percentiles=[0.05, 0.5, 0.95]).to_string())
    print("")
    print("phase breakdown")
    print(phase.to_string(index=False))
    print("")
    print("shock size bins")
    print(shock_size.to_string(index=False))
    print("")
    print("trade return distribution")
    print(trade_dist.to_string(index=False))


if __name__ == "__main__":
    main()
