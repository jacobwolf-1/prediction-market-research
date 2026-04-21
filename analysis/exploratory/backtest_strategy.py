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

from analysis.raw_time_utils import build_raw_time_groups, frequency_to_seconds, normalize_frequency_label
from utils.ingestion_utils import DATA_DIR, PROJECT_ROOT, RAW_DIR, ensure_directory


THRESHOLDS = [0.02, 0.03, 0.05, 0.10]
HORIZONS_SECONDS = [5, 10, 20, 30, 60]
TRADE_COST = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest ESPN-vs-Kalshi spread strategy on the raw-time dataset.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "backtest_results.csv",
    )
    parser.add_argument(
        "--pnl-threshold-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "backtest_pnl_vs_threshold.png",
    )
    parser.add_argument(
        "--pnl-horizon-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "backtest_pnl_vs_horizon.png",
    )
    parser.add_argument(
        "--cumulative-pnl-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "backtest_cumulative_pnl.png",
    )
    return parser.parse_args()


def prepare_backtest_frame(groups: list[dict[str, object]], frequency_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group in groups:
        frame = group["frame"].copy()
        frame["game_id"] = str(group["game_id"])
        frame["team"] = str(group["team"])
        frame["spread"] = frame["value_espn"] - frame["value_market"]
        for horizon in HORIZONS_SECONDS:
            periods = max(1, horizon // frequency_seconds)
            frame[f"exit_price_{horizon}s"] = frame["value_market"].shift(-periods)
            frame[f"exit_time_{horizon}s"] = frame["timestamp"].shift(-periods)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def simulate_group_trades(group_frame: pd.DataFrame, threshold: float, horizon: int) -> list[dict[str, object]]:
    exit_price_column = f"exit_price_{horizon}s"
    exit_time_column = f"exit_time_{horizon}s"
    trades: list[dict[str, object]] = []
    next_allowed_time = pd.Timestamp.min.tz_localize("UTC")

    for row in group_frame.itertuples(index=False):
        timestamp = row.timestamp
        if timestamp < next_allowed_time:
            continue

        spread = float(row.spread)
        direction = 0
        if spread > threshold:
            direction = 1
        elif spread < -threshold:
            direction = -1
        if direction == 0:
            continue

        exit_price = getattr(row, exit_price_column)
        exit_time = getattr(row, exit_time_column)
        if pd.isna(exit_price) or pd.isna(exit_time):
            continue

        entry_price = float(row.value_market)
        raw_return = float(exit_price - entry_price) if direction == 1 else float(entry_price - exit_price)
        net_return = raw_return - (2 * TRADE_COST)
        trades.append(
            {
                "game_id": str(row.game_id),
                "team": str(row.team),
                "entry_time": timestamp,
                "exit_time": exit_time,
                "horizon": int(horizon),
                "threshold": float(threshold),
                "direction": "long_yes" if direction == 1 else "short_yes",
                "spread_entry": spread,
                "entry_price": entry_price,
                "exit_price": float(exit_price),
                "entry_price_adjusted": entry_price + TRADE_COST if direction == 1 else entry_price - TRADE_COST,
                "exit_price_adjusted": float(exit_price) - TRADE_COST if direction == 1 else float(exit_price) + TRADE_COST,
                "raw_return": raw_return,
                "net_return": net_return,
                "holding_seconds": int((exit_time - timestamp).total_seconds()),
            }
        )
        next_allowed_time = exit_time

    return trades


def simulate_trades(frame: pd.DataFrame, threshold: float, horizon: int) -> pd.DataFrame:
    trades: list[dict[str, object]] = []
    for (_, _), group_frame in frame.groupby(["game_id", "team"], sort=True):
        trades.extend(simulate_group_trades(group_frame, threshold=threshold, horizon=horizon))
    if not trades:
        return pd.DataFrame(
            columns=[
                "game_id",
                "team",
                "entry_time",
                "exit_time",
                "horizon",
                "threshold",
                "direction",
                "spread_entry",
                "entry_price",
                "exit_price",
                "entry_price_adjusted",
                "exit_price_adjusted",
                "raw_return",
                "net_return",
                "holding_seconds",
            ]
        )
    return pd.DataFrame(trades).sort_values(["entry_time", "game_id", "team"]).reset_index(drop=True)


def sharpe_ratio(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    if std == 0:
        return float("nan")
    return float(np.sqrt(len(returns)) * returns.mean() / std)


def average_time_between_trades(trades: pd.DataFrame) -> float:
    if len(trades) < 2:
        return float("nan")
    ordered = trades.sort_values("entry_time").reset_index(drop=True)
    gaps = ordered["entry_time"].diff().dropna().dt.total_seconds()
    if gaps.empty:
        return float("nan")
    return float(gaps.mean())


def summarize_strategy(trades: pd.DataFrame, threshold: float, horizon: int) -> dict[str, object]:
    if trades.empty:
        return {
            "threshold": threshold,
            "horizon": horizon,
            "trades": 0,
            "avg_return": np.nan,
            "median_return": np.nan,
            "win_rate": np.nan,
            "total_pnl": 0.0,
            "sharpe": np.nan,
            "avg_spread_entry": np.nan,
            "avg_time_between_trades_seconds": np.nan,
        }

    returns = trades["net_return"]
    return {
        "threshold": threshold,
        "horizon": horizon,
        "trades": int(len(trades)),
        "avg_return": float(returns.mean()),
        "median_return": float(returns.median()),
        "win_rate": float((returns > 0).mean()),
        "total_pnl": float(returns.sum()),
        "sharpe": sharpe_ratio(returns),
        "avg_spread_entry": float(trades["spread_entry"].mean()),
        "avg_time_between_trades_seconds": average_time_between_trades(trades),
    }


def plot_pnl_vs_threshold(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    plot_frame = results.groupby("threshold", as_index=False)["total_pnl"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_frame["threshold"], plot_frame["total_pnl"], marker="o", linewidth=1.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total PnL Across Horizons")
    ax.set_title("Backtest PnL vs Threshold")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_pnl_vs_horizon(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    plot_frame = results.groupby("horizon", as_index=False)["total_pnl"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_frame["horizon"], plot_frame["total_pnl"], marker="o", linewidth=1.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Total PnL Across Thresholds")
    ax.set_title("Backtest PnL vs Horizon")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cumulative_pnl(trades: pd.DataFrame, output_path: Path, threshold: float, horizon: int) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    if trades.empty:
        ax.set_title("Cumulative PnL (No Trades)")
        ax.set_xlabel("Exit Time")
        ax.set_ylabel("Cumulative PnL")
    else:
        curve = trades.sort_values("exit_time")[["exit_time", "net_return"]].copy()
        curve["cumulative_pnl"] = curve["net_return"].cumsum()
        ax.plot(curve["exit_time"], curve["cumulative_pnl"], linewidth=1.5)
        ax.set_title(f"Cumulative PnL (threshold={threshold:.2f}, horizon={horizon}s)")
        ax.set_xlabel("Exit Time")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_trade_diagnostics(trades: pd.DataFrame) -> None:
    print("top 10 most profitable trades")
    if trades.empty:
        print("no trades")
    else:
        top = trades.nlargest(10, "net_return")[
            ["game_id", "team", "entry_time", "exit_time", "direction", "spread_entry", "entry_price", "exit_price", "net_return"]
        ]
        print(top.to_string(index=False))

    print("worst 10 trades")
    if trades.empty:
        print("no trades")
    else:
        bottom = trades.nsmallest(10, "net_return")[
            ["game_id", "team", "entry_time", "exit_time", "direction", "spread_entry", "entry_price", "exit_price", "net_return"]
        ]
        print(bottom.to_string(index=False))


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    frequency_seconds = frequency_to_seconds(frequency)
    groups, summary = build_raw_time_groups(args.espn_dir, args.market_dir, frequency)

    if not groups:
        print("No raw-time groups available.")
        return

    frame = prepare_backtest_frame(groups, frequency_seconds)
    strategy_rows: list[dict[str, object]] = []
    strategy_trades: dict[tuple[float, int], pd.DataFrame] = {}

    for threshold in THRESHOLDS:
        for horizon in HORIZONS_SECONDS:
            trades = simulate_trades(frame, threshold=threshold, horizon=horizon)
            strategy_trades[(threshold, horizon)] = trades
            strategy_rows.append(summarize_strategy(trades, threshold=threshold, horizon=horizon))

    results = pd.DataFrame(strategy_rows).sort_values(["threshold", "horizon"]).reset_index(drop=True)
    ensure_directory(args.output_csv.parent)
    results[["threshold", "horizon", "trades", "avg_return", "win_rate", "total_pnl", "sharpe"]].to_csv(args.output_csv, index=False)

    plot_pnl_vs_threshold(results, args.pnl_threshold_plot)
    plot_pnl_vs_horizon(results, args.pnl_horizon_plot)

    best_idx = results["total_pnl"].idxmax()
    best_row = results.loc[best_idx]
    best_key = (float(best_row["threshold"]), int(best_row["horizon"]))
    best_trades = strategy_trades[best_key]
    plot_cumulative_pnl(best_trades, args.cumulative_pnl_plot, threshold=best_key[0], horizon=best_key[1])

    pnl_by_threshold = results.groupby("threshold", as_index=False)["total_pnl"].sum().sort_values("threshold")
    pnl_by_horizon = results.groupby("horizon", as_index=False)["total_pnl"].sum().sort_values("horizon")

    print(f"games: {summary['games']}")
    print(f"groups: {summary['groups']}")
    print(f"rows analyzed: {len(frame)}")
    print("")
    print("strategy summary")
    print(results[["threshold", "horizon", "trades", "avg_return", "median_return", "win_rate", "total_pnl", "sharpe"]].to_string(index=False))
    print("")
    print("pnl by threshold")
    print(pnl_by_threshold.to_string(index=False))
    print("")
    print("pnl by horizon")
    print(pnl_by_horizon.to_string(index=False))
    print("")
    print("best strategy")
    print(f"threshold: {best_key[0]:.2f}")
    print(f"horizon_seconds: {best_key[1]}")
    print(f"trades: {int(best_row['trades'])}")
    print(f"avg_return: {best_row['avg_return']:.6f}")
    print(f"median_return: {best_row['median_return']:.6f}")
    print(f"win_rate: {best_row['win_rate']:.4f}")
    print(f"total_pnl: {best_row['total_pnl']:.6f}")
    print(f"sharpe: {best_row['sharpe']:.6f}" if pd.notna(best_row["sharpe"]) else "sharpe: nan")
    print(f"average_spread_at_entry: {best_row['avg_spread_entry']:.6f}")
    if pd.notna(best_row["avg_time_between_trades_seconds"]):
        print(f"average_time_between_trades_seconds: {best_row['avg_time_between_trades_seconds']:.2f}")
    else:
        print("average_time_between_trades_seconds: nan")
    print_trade_diagnostics(best_trades)
    print("")
    print(f"results csv saved to: {args.output_csv}")
    print(f"threshold plot saved to: {args.pnl_threshold_plot}")
    print(f"horizon plot saved to: {args.pnl_horizon_plot}")
    print(f"cumulative pnl plot saved to: {args.cumulative_pnl_plot}")


if __name__ == "__main__":
    main()
