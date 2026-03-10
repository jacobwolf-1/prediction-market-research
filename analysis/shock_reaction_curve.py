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


PRE_EVENT_SECONDS = 60
POST_EVENT_SECONDS = 120
REVERSAL_HORIZONS = [5, 10, 20, 30, 60, 120]
SHOCK_BINS = [0.03, 0.05, 0.07, 0.10, np.inf]
SHOCK_BIN_LABELS = ["0.03-0.05", "0.05-0.07", "0.07-0.10", ">0.10"]
TRADE_COST = 0.01


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze the market reaction curve after ESPN win-probability shocks.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--shock-threshold", type=float, default=0.05)
    parser.add_argument("--strategy-horizon", type=int, default=30)
    parser.add_argument("--min-market-updates-per-minute", type=float, default=None)
    parser.add_argument(
        "--curve-output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "shock_reaction_curve.csv",
    )
    parser.add_argument(
        "--size-output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "shock_size_analysis.csv",
    )
    parser.add_argument(
        "--phase-output-csv",
        type=Path,
        default=DATA_DIR / "processed" / "shock_game_phase_analysis.csv",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "shock_reaction_curve.png",
    )
    return parser.parse_args()


def prepare_frame(groups: list[dict[str, object]], frequency_seconds: int, strategy_horizon: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    strategy_periods = max(1, strategy_horizon // frequency_seconds)
    for group in groups:
        frame = group["frame"].copy()
        frame["shock"] = frame["delta_espn"]
        frame["next_market_price"] = frame["market_probability"].shift(-1)
        frame["entry_time"] = frame["timestamp"].shift(-1)
        frame["strategy_exit_price"] = frame["market_probability"].shift(-(strategy_periods + 1))
        frame["strategy_exit_time"] = frame["timestamp"].shift(-(strategy_periods + 1))
        frame["market_updates_per_minute"] = float(group["market_updates_per_minute"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def build_event_records(frame: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    events: list[dict[str, object]] = []
    path_rows: list[dict[str, object]] = []

    for (_, _), group_frame in frame.groupby(["game_id", "team"], sort=True):
        group_frame = group_frame.reset_index(drop=True)
        for idx, row in group_frame.iterrows():
            shock = float(row["shock"])
            if abs(shock) <= threshold:
                continue
            direction = 1 if shock > 0 else -1
            base_market = float(row["market_probability"])
            abs_shock = abs(shock)
            shock_bin = pd.cut(
                pd.Series([abs_shock]),
                bins=SHOCK_BINS,
                labels=SHOCK_BIN_LABELS,
                right=False,
                include_lowest=True,
            ).iloc[0]

            next_market_price = row["next_market_price"]
            exit_price = row["strategy_exit_price"]
            if pd.notna(next_market_price) and pd.notna(exit_price):
                strategy_return = (float(exit_price) - float(next_market_price)) if direction == 1 else (float(next_market_price) - float(exit_price))
                strategy_return -= 2 * TRADE_COST
            else:
                strategy_return = float("nan")

            events.append(
                {
                    "game_id": row["game_id"],
                    "team": row["team"],
                    "timestamp": row["timestamp"],
                    "shock": shock,
                    "abs_shock": abs_shock,
                    "direction": direction,
                    "shock_bin": str(shock_bin) if pd.notna(shock_bin) else None,
                    "game_phase": row["game_phase"],
                    "strategy_return": strategy_return,
                }
            )

            for offset in range(-PRE_EVENT_SECONDS, POST_EVENT_SECONDS + 1):
                target_idx = idx + offset
                if target_idx < 0 or target_idx >= len(group_frame):
                    continue
                market_at_offset = float(group_frame.iloc[target_idx]["market_probability"])
                path_rows.append(
                    {
                        "game_id": row["game_id"],
                        "team": row["team"],
                        "event_time": row["timestamp"],
                        "seconds_from_event": int(offset),
                        "shock": shock,
                        "abs_shock": abs_shock,
                        "shock_bin": str(shock_bin) if pd.notna(shock_bin) else None,
                        "game_phase": row["game_phase"],
                        "signed_market_return": direction * (market_at_offset - base_market),
                    }
                )

    return pd.DataFrame(events), pd.DataFrame(path_rows)


def build_reaction_curve(path_frame: pd.DataFrame) -> pd.DataFrame:
    if path_frame.empty:
        return pd.DataFrame(columns=["seconds_from_event", "avg_signed_market_return", "median_signed_market_return", "count"])
    curve = (
        path_frame.groupby("seconds_from_event", as_index=False)["signed_market_return"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "avg_signed_market_return",
                "median": "median_signed_market_return",
                "count": "count",
            }
        )
        .sort_values("seconds_from_event")
    )
    return curve


def build_reversal_table(path_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for horizon in REVERSAL_HORIZONS:
        horizon_frame = path_frame.loc[path_frame["seconds_from_event"] == horizon]
        rows.append(
            {
                "horizon_seconds": horizon,
                "avg_signed_market_return": float(horizon_frame["signed_market_return"].mean()) if not horizon_frame.empty else float("nan"),
                "median_signed_market_return": float(horizon_frame["signed_market_return"].median()) if not horizon_frame.empty else float("nan"),
                "count": int(len(horizon_frame)),
            }
        )
    return pd.DataFrame(rows)


def build_shock_size_table(path_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    valid = path_frame.loc[path_frame["seconds_from_event"].isin(REVERSAL_HORIZONS)].copy()
    for (shock_bin, horizon), group in valid.groupby(["shock_bin", "seconds_from_event"], sort=True):
        rows.append(
            {
                "shock_bin": shock_bin,
                "horizon_seconds": int(horizon),
                "avg_signed_market_return": float(group["signed_market_return"].mean()),
                "median_signed_market_return": float(group["signed_market_return"].median()),
                "count": int(len(group)),
            }
        )
    return pd.DataFrame(rows).sort_values(["shock_bin", "horizon_seconds"]).reset_index(drop=True)


def build_phase_table(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["game_phase", "trades", "avg_strategy_return", "median_strategy_return", "win_rate", "total_pnl"])
    valid = events.dropna(subset=["strategy_return", "game_phase"]).copy()
    rows: list[dict[str, object]] = []
    for phase, group in valid.groupby("game_phase", sort=True):
        rows.append(
            {
                "game_phase": phase,
                "trades": int(len(group)),
                "avg_strategy_return": float(group["strategy_return"].mean()),
                "median_strategy_return": float(group["strategy_return"].median()),
                "win_rate": float((group["strategy_return"] > 0).mean()),
                "total_pnl": float(group["strategy_return"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("game_phase").reset_index(drop=True)


def build_strategy_phase_table(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    trade_rows: list[dict[str, object]] = []
    for (_, _), group_frame in frame.groupby(["game_id", "team"], sort=True):
        next_allowed_time = pd.Timestamp.min.tz_localize("UTC")
        for row in group_frame.itertuples(index=False):
            shock = float(row.shock)
            if abs(shock) <= threshold:
                continue
            if row.timestamp < next_allowed_time:
                continue
            if pd.isna(row.next_market_price) or pd.isna(row.strategy_exit_price) or pd.isna(row.strategy_exit_time):
                continue
            direction = 1 if shock > 0 else -1
            net_return = (
                float(row.strategy_exit_price) - float(row.next_market_price)
                if direction == 1
                else float(row.next_market_price) - float(row.strategy_exit_price)
            ) - (2 * TRADE_COST)
            trade_rows.append(
                {
                    "game_phase": row.game_phase,
                    "net_return": net_return,
                }
            )
            next_allowed_time = row.strategy_exit_time

    if not trade_rows:
        return pd.DataFrame(columns=["game_phase", "trades", "avg_strategy_return", "median_strategy_return", "win_rate", "total_pnl"])

    trades = pd.DataFrame(trade_rows).dropna(subset=["game_phase"])
    rows: list[dict[str, object]] = []
    for phase, group in trades.groupby("game_phase", sort=True):
        rows.append(
            {
                "game_phase": phase,
                "trades": int(len(group)),
                "avg_strategy_return": float(group["net_return"].mean()),
                "median_strategy_return": float(group["net_return"].median()),
                "win_rate": float((group["net_return"] > 0).mean()),
                "total_pnl": float(group["net_return"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("game_phase").reset_index(drop=True)


def plot_reaction_curve(curve: pd.DataFrame, output_path: Path, threshold: float) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(curve["seconds_from_event"], curve["avg_signed_market_return"], linewidth=1.8)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.axhline(0, color="black", linestyle=":", linewidth=1)
    ax.set_xlabel("Seconds From ESPN Shock")
    ax.set_ylabel("Average Signed Market Return")
    ax.set_title(f"Shock Reaction Curve (threshold={threshold:.2f})")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    frequency_seconds = frequency_to_seconds(frequency)
    groups, summary = build_shock_groups(args.espn_dir, args.market_dir, frequency)
    filtered_groups = filter_groups_by_liquidity(groups, args.min_market_updates_per_minute)
    if not filtered_groups:
        print("No shock groups available.")
        return

    frame = prepare_frame(filtered_groups, frequency_seconds, args.strategy_horizon)
    events, path_frame = build_event_records(frame, args.shock_threshold)
    curve = build_reaction_curve(path_frame)
    reversal = build_reversal_table(path_frame)
    shock_size_table = build_shock_size_table(path_frame)
    phase_table = build_strategy_phase_table(frame, args.shock_threshold)

    ensure_directory(args.curve_output_csv.parent)
    curve.to_csv(args.curve_output_csv, index=False)
    shock_size_table.to_csv(args.size_output_csv, index=False)
    phase_table.to_csv(args.phase_output_csv, index=False)
    plot_reaction_curve(curve, args.output_plot, args.shock_threshold)

    optimal_exit = reversal.dropna(subset=["avg_signed_market_return"])
    best_horizon = int(optimal_exit.loc[optimal_exit["avg_signed_market_return"].idxmax(), "horizon_seconds"]) if not optimal_exit.empty else None
    avg_reaction_time = float(
        path_frame.loc[(path_frame["seconds_from_event"] > 0) & (path_frame["signed_market_return"] > 0)].groupby(["game_id", "team", "event_time"]).agg(
            first_positive_second=("seconds_from_event", "min")
        )["first_positive_second"].mean()
    ) if not path_frame.empty else float("nan")

    print(f"games: {summary['games']}")
    print(f"groups: {summary['groups']}")
    print(f"groups_after_liquidity_filter: {len(filtered_groups)}")
    print(f"shock_threshold: {args.shock_threshold:.2f}")
    print(f"events analyzed: {len(events)}")
    print("")
    print("reversal table")
    print(reversal.to_string(index=False))
    print("")
    if best_horizon is not None:
        print(f"optimal_exit_horizon_seconds: {best_horizon}")
    else:
        print("optimal_exit_horizon_seconds: none")
    print(f"average_market_reaction_time_seconds: {avg_reaction_time:.2f}" if pd.notna(avg_reaction_time) else "average_market_reaction_time_seconds: nan")
    print("")
    print("profit by shock size")
    print(shock_size_table.to_string(index=False))
    print("")
    print("profit by game phase")
    print(phase_table.to_string(index=False))
    print("")
    print(f"reaction curve saved to: {args.curve_output_csv}")
    print(f"shock size analysis saved to: {args.size_output_csv}")
    print(f"game phase analysis saved to: {args.phase_output_csv}")
    print(f"plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
