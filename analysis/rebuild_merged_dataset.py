from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.raw_time_dataset import load_raw_time_dataset
from analysis.shock_strategy_utils import total_seconds_remaining
from utils.ingestion_utils import DATA_DIR, RAW_DIR, ensure_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild a merged ESPN/Kalshi raw-time dataset with consistent timestamp normalization.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DATA_DIR / "processed" / "merged_games.parquet",
    )
    return parser.parse_args()


def resample_ffill(frame: pd.DataFrame, time_col: str, value_col: str, freq: str) -> pd.DataFrame:
    series = frame[[time_col, value_col]].copy()
    series[time_col] = pd.to_datetime(series[time_col], utc=True, errors="coerce")
    series = series.dropna(subset=[time_col]).sort_values(time_col)
    series = series.drop_duplicates(subset=[time_col], keep="last")
    if series.empty:
        return pd.DataFrame(columns=["timestamp", value_col])
    out = series.set_index(time_col)[[value_col]].resample(freq).ffill().reset_index()
    return out.rename(columns={time_col: "timestamp"})


def build_group_frame(game_id: str, team: str, espn_path: Path, market_frame: pd.DataFrame, freq: str) -> pd.DataFrame:
    espn = pd.read_parquet(
        espn_path,
        columns=[
            "timestamp",
            "team",
            "espn_probability",
            "quarter",
            "game_clock_seconds_remaining",
            "home_score",
            "away_score",
        ],
    )
    espn = espn.loc[espn["team"] == team].copy()
    if espn.empty or market_frame.empty:
        return pd.DataFrame()
    espn["seconds_remaining"] = espn.apply(
        lambda row: total_seconds_remaining(row["quarter"], row["game_clock_seconds_remaining"]),
        axis=1,
    )

    parts = [
        resample_ffill(espn, "timestamp", "espn_probability", freq),
        resample_ffill(espn, "timestamp", "quarter", freq),
        resample_ffill(espn, "timestamp", "seconds_remaining", freq),
        resample_ffill(espn, "timestamp", "home_score", freq),
        resample_ffill(espn, "timestamp", "away_score", freq),
        resample_ffill(market_frame, "timestamp", "market_probability", freq),
    ]
    merged = parts[0]
    for part in parts[1:]:
        merged = merged.merge(part, on="timestamp", how="inner")
    merged["game_id"] = str(game_id)
    merged["team"] = str(team)
    return merged[
        [
            "timestamp",
            "game_id",
            "team",
            "espn_probability",
            "market_probability",
            "quarter",
            "seconds_remaining",
            "home_score",
            "away_score",
        ]
    ].sort_values("timestamp")


def main() -> None:
    args = parse_args()
    dataset = load_raw_time_dataset(args.espn_dir, args.market_dir)
    frames: list[pd.DataFrame] = []

    for (game_id, team), item in dataset.items():
        espn_path = args.espn_dir / f"game_{game_id}_espn.parquet"
        market_frame = item["market"][["timestamp", "market_probability"]].copy()
        frame = build_group_frame(game_id, team, espn_path, market_frame, args.freq)
        if not frame.empty:
            frames.append(frame)

    merged = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=[
            "timestamp",
            "game_id",
            "team",
            "espn_probability",
            "market_probability",
            "quarter",
            "seconds_remaining",
            "home_score",
            "away_score",
        ]
    )
    ensure_directory(args.output_path.parent)
    merged.to_parquet(args.output_path, index=False)
    print(f"rows: {len(merged)}")
    print(f"games: {merged['game_id'].nunique() if not merged.empty else 0}")
    print(f"teams: {merged[['game_id', 'team']].drop_duplicates().shape[0] if not merged.empty else 0}")
    print(f"output saved to: {args.output_path}")


if __name__ == "__main__":
    main()
