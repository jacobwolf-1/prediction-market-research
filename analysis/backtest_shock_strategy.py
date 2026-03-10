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
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.raw_time_utils import frequency_to_seconds, normalize_frequency_label
from analysis.shock_strategy_utils import build_shock_groups, filter_groups_by_liquidity
from utils.ingestion_utils import DATA_DIR, PROJECT_ROOT, RAW_DIR, ensure_directory


SHOCK_THRESHOLDS = [0.03, 0.05, 0.08]
HORIZONS_SECONDS = [5, 10, 20, 30]
TRADE_COST = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest an ESPN shock-following strategy on the raw-time dataset.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument(
        "--min-market-updates-per-minute",
        type=float,
        default=None,
        help="Optional liquidity filter applied at the game-team level.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "shock_backtest_results.csv",
    )
    parser.add_argument(
        "--pnl-threshold-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "shock_strategy_pnl_vs_threshold.png",
    )
    parser.add_argument(
        "--pnl-horizon-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "shock_strategy_pnl_vs_horizon.png",
    )
    parser.add_argument(
        "--cumulative-pnl-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "shock_strategy_cumulative_pnl.png",
    )
    return parser.parse_args()


def prepare_shock_frame(groups: list[dict[str, object]], frequency_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group in groups:
        frame = group["frame"].copy()
        frame["shock"] = frame["delta_espn"]
        frame["next_market_price"] = frame["market_probability"].shift(-1)
        frame["entry_time"] = frame["timestamp"].shift(-1)
        frame["entry_delay_seconds"] = (frame["entry_time"] - frame["timestamp"]).dt.total_seconds()
        for horizon in HORIZONS_SECONDS:
            periods = max(1, horizon // frequency_seconds)
            frame[f"exit_price_{horizon}s"] = frame["market_probability"].shift(-(periods + 1))
            frame[f"exit_time_{horizon}s"] = frame["timestamp"].shift(-(periods + 1))
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def first_reaction_delay_seconds(group_frame: pd.DataFrame, start_idx: int, direction: int) -> float:
    baseline_price = float(group_frame.iloc[start_idx]["market_probability"])
    shock_time = group_frame.iloc[start_idx]["timestamp"]
    future = group_frame.iloc[start_idx + 1 :]
    if future.empty:
        return float("nan")

    move = direction * (future["market_probability"] - baseline_price)
    reacted = future.loc[move > 0]
    if reacted.empty:
        return float("nan")
    first_time = reacted.iloc[0]["timestamp"]
    return float((first_time - shock_time).total_seconds())


def simulate_group_trades(group_frame: pd.DataFrame, threshold: float, horizon: int) -> tuple[list[dict[str, object]], int, list[float], list[float], int]:
    exit_price_column = f"exit_price_{horizon}s"
    exit_time_column = f"exit_time_{horizon}s"
    trades: list[dict[str, object]] = []
    ignored_signals = 0
    shock_sizes: list[float] = []
    reaction_delays: list[float] = []
    reaction_hits = 0
    next_allowed_time = pd.Timestamp.min.tz_localize("UTC")

    for idx, row in enumerate(group_frame.itertuples(index=False)):
        shock = float(row.shock)
        if abs(shock) <= threshold:
            continue

        direction = 1 if shock > 0 else -1
        shock_sizes.append(abs(shock))
        reaction_delay = first_reaction_delay_seconds(group_frame, idx, direction)
        if pd.notna(reaction_delay):
            reaction_delays.append(float(reaction_delay))
            reaction_hits += 1

        if row.timestamp < next_allowed_time:
            ignored_signals += 1
            continue

        entry_price = row.next_market_price
        entry_time = row.entry_time
        exit_price = getattr(row, exit_price_column)
        exit_time = getattr(row, exit_time_column)
        if pd.isna(entry_price) or pd.isna(entry_time) or pd.isna(exit_price) or pd.isna(exit_time):
            continue

        raw_return = float(exit_price - entry_price) if direction == 1 else float(entry_price - exit_price)
        net_return = raw_return - (2 * TRADE_COST)
        trades.append(
            {
                "game_id": str(row.game_id),
                "team": str(row.team),
                "shock_time": row.timestamp,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "threshold": float(threshold),
                "horizon": int(horizon),
                "direction": "long_yes" if direction == 1 else "short_yes",
                "shock": shock,
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "entry_price_adjusted": float(entry_price) + TRADE_COST if direction == 1 else float(entry_price) - TRADE_COST,
                "exit_price_adjusted": float(exit_price) - TRADE_COST if direction == 1 else float(exit_price) + TRADE_COST,
                "raw_return": raw_return,
                "net_return": net_return,
                "entry_delay_seconds": float(row.entry_delay_seconds) if pd.notna(row.entry_delay_seconds) else float("nan"),
                "reaction_delay_seconds": reaction_delay,
            }
        )
        next_allowed_time = exit_time

    return trades, ignored_signals, shock_sizes, reaction_delays, reaction_hits


def simulate_trades(frame: pd.DataFrame, threshold: float, horizon: int) -> tuple[pd.DataFrame, dict[str, float | int]]:
    trades: list[dict[str, object]] = []
    ignored_signals = 0
    shock_sizes: list[float] = []
    reaction_delays: list[float] = []
    reaction_hits = 0

    for (_, _), group_frame in frame.groupby(["game_id", "team"], sort=True):
        group_trades, group_ignored, group_shocks, group_reaction_delays, group_reaction_hits = simulate_group_trades(
            group_frame.reset_index(drop=True),
            threshold=threshold,
            horizon=horizon,
        )
        trades.extend(group_trades)
        ignored_signals += group_ignored
        shock_sizes.extend(group_shocks)
        reaction_delays.extend(group_reaction_delays)
        reaction_hits += group_reaction_hits

    trade_frame = pd.DataFrame(trades).sort_values(["entry_time", "game_id", "team"]).reset_index(drop=True) if trades else pd.DataFrame(
        columns=[
            "game_id",
            "team",
            "shock_time",
            "entry_time",
            "exit_time",
            "threshold",
            "horizon",
            "direction",
            "shock",
            "entry_price",
            "exit_price",
            "entry_price_adjusted",
            "exit_price_adjusted",
            "raw_return",
            "net_return",
            "entry_delay_seconds",
            "reaction_delay_seconds",
        ]
    )
    diagnostics = {
        "ignored_signals": int(ignored_signals),
        "avg_shock_size": float(np.mean(shock_sizes)) if shock_sizes else float("nan"),
        "avg_reaction_delay_seconds": float(np.mean(reaction_delays)) if reaction_delays else float("nan"),
        "reaction_observations": int(reaction_hits),
        "total_signals": int(len(shock_sizes)),
    }
    return trade_frame, diagnostics


def sharpe_ratio(returns: pd.Series) -> float:
    if len(returns) < 2:
        return float("nan")
    std = float(returns.std(ddof=1))
    if std == 0:
        return float("nan")
    return float(np.sqrt(len(returns)) * returns.mean() / std)


def summarize_strategy(trades: pd.DataFrame, threshold: float, horizon: int, diagnostics: dict[str, float | int]) -> dict[str, object]:
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
            "ignored_signals": int(diagnostics["ignored_signals"]),
            "total_signals": int(diagnostics["total_signals"]),
            "avg_shock_size": diagnostics["avg_shock_size"],
            "avg_entry_delay_seconds": np.nan,
            "avg_reaction_delay_seconds": diagnostics["avg_reaction_delay_seconds"],
            "reaction_observations": int(diagnostics["reaction_observations"]),
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
        "ignored_signals": int(diagnostics["ignored_signals"]),
        "total_signals": int(diagnostics["total_signals"]),
        "avg_shock_size": diagnostics["avg_shock_size"],
        "avg_entry_delay_seconds": float(trades["entry_delay_seconds"].mean()),
        "avg_reaction_delay_seconds": diagnostics["avg_reaction_delay_seconds"],
        "reaction_observations": int(diagnostics["reaction_observations"]),
    }


def plot_pnl_vs_threshold(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    plot_frame = results.groupby("threshold", as_index=False)["total_pnl"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(plot_frame["threshold"], plot_frame["total_pnl"], marker="o", linewidth=1.6)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Shock Threshold")
    ax.set_ylabel("Total PnL Across Horizons")
    ax.set_title("Shock Strategy PnL vs Threshold")
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
    ax.set_title("Shock Strategy PnL vs Horizon")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_cumulative_pnl(trades: pd.DataFrame, output_path: Path, threshold: float, horizon: int) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    if trades.empty:
        ax.set_title("Shock Strategy Cumulative PnL (No Trades)")
        ax.set_xlabel("Exit Time")
        ax.set_ylabel("Cumulative PnL")
    else:
        curve = trades.sort_values("exit_time")[["exit_time", "net_return"]].copy()
        curve["cumulative_pnl"] = curve["net_return"].cumsum()
        ax.plot(curve["exit_time"], curve["cumulative_pnl"], linewidth=1.5)
        ax.set_title(f"Shock Strategy Cumulative PnL (threshold={threshold:.2f}, horizon={horizon}s)")
        ax.set_xlabel("Exit Time")
        ax.set_ylabel("Cumulative PnL")
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def print_trade_diagnostics(trades: pd.DataFrame) -> None:
    print("top 10 best trades")
    if trades.empty:
        print("no trades")
    else:
        top = trades.nlargest(10, "net_return")[
            ["game_id", "team", "shock_time", "entry_time", "exit_time", "direction", "shock", "entry_price", "exit_price", "net_return"]
        ]
        print(top.to_string(index=False))

    print("top 10 worst trades")
    if trades.empty:
        print("no trades")
    else:
        bottom = trades.nsmallest(10, "net_return")[
            ["game_id", "team", "shock_time", "entry_time", "exit_time", "direction", "shock", "entry_price", "exit_price", "net_return"]
        ]
        print(bottom.to_string(index=False))


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    frequency_seconds = frequency_to_seconds(frequency)
    groups, summary = build_shock_groups(args.espn_dir, args.market_dir, frequency)
    filtered_groups = filter_groups_by_liquidity(groups, args.min_market_updates_per_minute)
    if not filtered_groups:
        print("No raw-time groups available.")
        return

    frame = prepare_shock_frame(filtered_groups, frequency_seconds)
    strategy_rows: list[dict[str, object]] = []
    strategy_trades: dict[tuple[float, int], pd.DataFrame] = {}

    for threshold in SHOCK_THRESHOLDS:
        for horizon in HORIZONS_SECONDS:
            trades, diagnostics = simulate_trades(frame, threshold=threshold, horizon=horizon)
            strategy_trades[(threshold, horizon)] = trades
            strategy_rows.append(summarize_strategy(trades, threshold=threshold, horizon=horizon, diagnostics=diagnostics))

    results = pd.DataFrame(strategy_rows).sort_values(["threshold", "horizon"]).reset_index(drop=True)
    ensure_directory(args.output_csv.parent)
    results.to_csv(args.output_csv, index=False)

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
    print(f"groups_after_liquidity_filter: {len(filtered_groups)}")
    print(f"liquidity_filter_min_updates_per_minute: {args.min_market_updates_per_minute if args.min_market_updates_per_minute is not None else 'none'}")
    print(f"rows analyzed: {len(frame)}")
    print("")
    print("strategy summary")
    print(
        results[
            [
                "threshold",
                "horizon",
                "trades",
                "ignored_signals",
                "avg_return",
                "median_return",
                "win_rate",
                "total_pnl",
                "sharpe",
                "avg_shock_size",
                "avg_reaction_delay_seconds",
            ]
        ].to_string(index=False)
    )
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
    print(f"ignored_signals: {int(best_row['ignored_signals'])}")
    print(f"avg_return: {best_row['avg_return']:.6f}")
    print(f"median_return: {best_row['median_return']:.6f}")
    print(f"win_rate: {best_row['win_rate']:.4f}")
    print(f"total_pnl: {best_row['total_pnl']:.6f}")
    print(f"sharpe: {best_row['sharpe']:.6f}" if pd.notna(best_row["sharpe"]) else "sharpe: nan")
    print(f"average_shock_size: {best_row['avg_shock_size']:.6f}")
    print(f"average_market_reaction_time_seconds: {best_row['avg_reaction_delay_seconds']:.2f}" if pd.notna(best_row["avg_reaction_delay_seconds"]) else "average_market_reaction_time_seconds: nan")
    print_trade_diagnostics(best_trades)
    print("")
    print(f"results csv saved to: {args.output_csv}")
    print(f"threshold plot saved to: {args.pnl_threshold_plot}")
    print(f"horizon plot saved to: {args.pnl_horizon_plot}")
    print(f"cumulative pnl plot saved to: {args.cumulative_pnl_plot}")


if __name__ == "__main__":
    main()
