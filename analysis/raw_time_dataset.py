from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.ingestion_utils import RAW_DIR


def load_raw_time_dataset(
    espn_dir: Path = RAW_DIR / "espn",
    market_dir: Path = RAW_DIR / "kalshi",
) -> dict[tuple[str, str], dict[str, pd.DataFrame]]:
    espn_files = {path.name.removeprefix("game_").removesuffix("_espn.parquet"): path for path in sorted(espn_dir.glob("game_*_espn.parquet"))}
    market_files = {path.stem: path for path in sorted(market_dir.glob("*.parquet"))}
    common_game_ids = sorted(set(espn_files).intersection(market_files))

    dataset: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    for game_id in common_game_ids:
        espn = pd.read_parquet(espn_files[game_id], columns=["timestamp", "game_id", "team", "espn_probability"]).copy()
        market = pd.read_parquet(
            market_files[game_id],
            columns=["timestamp", "market_probability", "team", "market_id", "event_id", "game_id"],
        ).copy()

        espn["timestamp"] = pd.to_datetime(espn["timestamp"], utc=True, errors="coerce")
        market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True, errors="coerce")

        espn = espn.dropna(subset=["timestamp", "game_id", "team", "espn_probability"]).sort_values(["team", "timestamp"]).reset_index(drop=True)
        market = market.dropna(subset=["timestamp", "game_id", "team", "market_probability"]).sort_values(["team", "timestamp"]).reset_index(drop=True)

        shared_teams = sorted(set(espn["team"]).intersection(market["team"]))
        for team in shared_teams:
            espn_team = espn.loc[espn["team"] == team, ["timestamp", "game_id", "team", "espn_probability"]].copy()
            market_team = market.loc[
                market["team"] == team,
                ["timestamp", "game_id", "team", "market_probability", "market_id", "event_id"],
            ].copy()
            if espn_team.empty or market_team.empty:
                continue
            dataset[(str(game_id), str(team))] = {
                "espn": espn_team.sort_values("timestamp").reset_index(drop=True),
                "market": market_team.sort_values("timestamp").reset_index(drop=True),
            }
    return dataset


def dataset_summary(dataset: dict[tuple[str, str], dict[str, pd.DataFrame]]) -> dict[str, int]:
    if not dataset:
        return {"games": 0, "groups": 0, "espn_rows": 0, "market_rows": 0}
    game_ids = {game_id for game_id, _ in dataset}
    espn_rows = sum(len(item["espn"]) for item in dataset.values())
    market_rows = sum(len(item["market"]) for item in dataset.values())
    return {
        "games": len(game_ids),
        "groups": len(dataset),
        "espn_rows": espn_rows,
        "market_rows": market_rows,
    }
