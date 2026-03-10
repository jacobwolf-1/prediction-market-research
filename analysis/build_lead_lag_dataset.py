from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ingestion_utils import PROCESSED_DIR, RAW_DIR, ensure_directory, write_parquet


ESPN_FILE_PATTERN = re.compile(r"^game_(?P<game_id>\d+)_espn\.parquet$")


def normalize_timestamp_column(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build merged ESPN/market lead-lag datasets.")
    parser.add_argument(
        "--espn-dir",
        type=Path,
        default=RAW_DIR / "espn",
        help="Directory containing ESPN parquet files.",
    )
    parser.add_argument(
        "--market-source",
        choices=["polymarket", "kalshi"],
        default="polymarket",
        help="Market source to align with ESPN data.",
    )
    parser.add_argument(
        "--market-dir",
        type=Path,
        default=None,
        help="Optional market parquet directory override. Defaults to data/raw/<market-source>",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR / "lead_lag_dataset",
        help="Directory for merged lead-lag parquet files.",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Optional cap on number of games to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    return parser.parse_args()


def discover_espn_files(espn_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(espn_dir.glob("*.parquet")):
        match = ESPN_FILE_PATTERN.match(path.name)
        if not match:
            continue
        files[match.group("game_id")] = path
    return files


def discover_market_files(market_dir: Path) -> dict[str, Path]:
    return {path.stem: path for path in sorted(market_dir.glob("*.parquet"))}


def load_espn_frame(path: Path, game_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = ["timestamp", "team", "espn_probability"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing ESPN columns: {missing}")
    frame = frame[required]
    frame["timestamp"] = normalize_timestamp_column(frame["timestamp"])
    frame["team"] = frame["team"].astype(str)
    frame["game_id"] = str(game_id)
    frame = frame.dropna(subset=["timestamp", "team", "espn_probability"])
    return frame.sort_values(["team", "timestamp"]).reset_index(drop=True)


def load_market_frame(path: Path, game_id: str) -> pd.DataFrame:
    frame = pd.read_parquet(path).copy()
    required = ["timestamp", "team", "market_probability"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} missing market columns: {missing}")
    frame = frame[required]
    frame["timestamp"] = normalize_timestamp_column(frame["timestamp"])
    frame["team"] = frame["team"].astype(str)
    frame["game_id"] = str(game_id)
    frame = frame.dropna(subset=["timestamp", "team", "market_probability"])
    return frame.sort_values(["team", "timestamp"]).reset_index(drop=True)


def align_game_frames(espn_frame: pd.DataFrame, market_frame: pd.DataFrame) -> pd.DataFrame:
    aligned_parts: list[pd.DataFrame] = []
    for team in sorted(set(espn_frame["team"]).intersection(market_frame["team"])):
        espn_team = espn_frame.loc[espn_frame["team"] == team].sort_values("timestamp").copy()
        market_team = market_frame.loc[market_frame["team"] == team].sort_values("timestamp").copy()
        aligned = pd.merge_asof(
            espn_team,
            market_team[["timestamp", "market_probability"]],
            on="timestamp",
            direction="backward",
        )
        aligned_parts.append(aligned)

    if not aligned_parts:
        return pd.DataFrame(columns=["timestamp", "game_id", "team", "espn_probability", "market_probability"])

    frame = pd.concat(aligned_parts, ignore_index=True)
    frame = frame.sort_values(["team", "timestamp"]).reset_index(drop=True)
    frame["delta_espn"] = frame.groupby("team")["espn_probability"].diff()
    frame["delta_market"] = frame.groupby("team")["market_probability"].diff()
    frame["delta_espn_lag_1"] = frame.groupby("team")["delta_espn"].shift(1)
    frame["delta_espn_lag_5"] = frame.groupby("team")["delta_espn"].shift(5)
    frame["delta_espn_lag_10"] = frame.groupby("team")["delta_espn"].shift(10)
    return frame[
        [
            "timestamp",
            "game_id",
            "team",
            "espn_probability",
            "market_probability",
            "delta_espn",
            "delta_market",
            "delta_espn_lag_1",
            "delta_espn_lag_5",
            "delta_espn_lag_10",
        ]
    ]


def build_game_dataset(game_id: str, espn_path: Path, market_path: Path) -> pd.DataFrame:
    espn_frame = load_espn_frame(espn_path, game_id)
    market_frame = load_market_frame(market_path, game_id)

    print(f"{game_id} ESPN rows loaded: {len(espn_frame)}")
    print(f"{game_id} market rows loaded: {len(market_frame)}")

    aligned = align_game_frames(espn_frame, market_frame)
    print(f"{game_id} aligned rows: {len(aligned)}")
    print(f"{game_id} output rows: {len(aligned)}")
    return aligned


def main() -> None:
    args = parse_args()
    ensure_directory(args.output_dir)

    market_dir = args.market_dir or (RAW_DIR / args.market_source)
    espn_files = discover_espn_files(args.espn_dir)
    market_files = discover_market_files(market_dir)
    common_game_ids = sorted(set(espn_files).intersection(market_files))
    if args.max_games is not None:
        common_game_ids = common_game_ids[: args.max_games]

    total_written = 0
    for game_id in common_game_ids:
        output_path = args.output_dir / f"{game_id}.parquet"
        if output_path.exists() and not args.overwrite:
            continue
        try:
            frame = build_game_dataset(game_id, espn_files[game_id], market_files[game_id])
        except Exception as exc:
            print(f"{game_id} skipped: {exc}")
            continue
        if frame.empty:
            continue
        write_parquet(frame, output_path)
        total_written += 1

    print(f"games processed: {len(common_game_ids)}")
    print(f"datasets written: {total_written}")


if __name__ == "__main__":
    main()
