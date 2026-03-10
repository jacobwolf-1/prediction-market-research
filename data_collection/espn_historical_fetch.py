from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.ingestion_utils import (
    RAW_DIR,
    clock_to_seconds,
    coerce_probability,
    load_game_ids,
    probability_columns_valid,
    request_json,
    standardize_clock,
    timestamps_monotonic,
    write_parquet,
)


SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ESPN historical live win probability series for NBA games.")
    parser.add_argument("--game-ids", nargs="*", help="ESPN game IDs to fetch.")
    parser.add_argument(
        "--game-ids-file",
        type=Path,
        default=RAW_DIR / "game_ids.csv",
        help="CSV file with a game_id column. Defaults to data/raw/game_ids.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR / "espn",
        help="Output directory for per-game parquet files. Defaults to data/raw/espn",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing parquet files for the selected game IDs.",
    )
    return parser.parse_args()


def resolve_game_ids(args: argparse.Namespace) -> list[str]:
    if args.game_ids:
        return [str(game_id) for game_id in args.game_ids]
    if args.game_ids_file.exists():
        return load_game_ids(args.game_ids_file)["game_id"].astype(str).tolist()
    raise ValueError("Provide --game-ids or create data/raw/game_ids.csv first.")


def extract_teams(payload: dict) -> tuple[dict, dict]:
    competition = (payload.get("header", {}).get("competitions") or [{}])[0]
    competitors = competition.get("competitors") or []
    home = next((team for team in competitors if team.get("homeAway") == "home"), {})
    away = next((team for team in competitors if team.get("homeAway") == "away"), {})
    return home, away


def score_deltas_valid(frame: pd.DataFrame) -> None:
    ordered = frame.sort_values(["timestamp", "team"])
    score_frame = ordered[["timestamp", "home_score", "away_score"]].drop_duplicates().sort_values("timestamp")
    frame["home_score"] = frame["home_score"].cummax()
    frame["away_score"] = frame["away_score"].cummax()


def build_frame(payload: dict, game_id: str) -> pd.DataFrame:
    plays = {str(play.get("id")): play for play in payload.get("plays", [])}
    winprob = payload.get("winprobability") or []
    home, away = extract_teams(payload)
    home_team = home.get("team", {})
    away_team = away.get("team", {})

    rows: list[dict] = []
    for entry in winprob:
        play = plays.get(str(entry.get("playId")))
        if not play:
            continue

        home_probability = coerce_probability(entry.get("homeWinPercentage"))
        away_probability = None if home_probability is None else round(1.0 - home_probability, 6)
        period = play.get("period", {}).get("number")
        clock_display = play.get("clock", {}).get("displayValue")
        clock_seconds = clock_to_seconds(clock_display)
        play_timestamp = pd.to_datetime(play.get("wallclock"), utc=True)

        base_row = {
            "timestamp": play_timestamp,
            "game_id": game_id,
            "home_score": play.get("homeScore"),
            "away_score": play.get("awayScore"),
            "quarter": period,
            "game_clock": standardize_clock(clock_display),
            "game_clock_seconds_remaining": clock_seconds,
            "seconds_remaining": clock_seconds,
            "play_id": str(play.get("id")),
            "play_text": play.get("text"),
            "play_type": play.get("type", {}).get("text"),
        }

        rows.append(
            {
                **base_row,
                "team": home_team.get("abbreviation"),
                "team_display_name": home_team.get("displayName"),
                "team_home_away": "home",
                "espn_probability": home_probability,
            }
        )
        rows.append(
            {
                **base_row,
                "team": away_team.get("abbreviation"),
                "team_display_name": away_team.get("displayName"),
                "team_home_away": "away",
                "espn_probability": away_probability,
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(["timestamp", "team"]).reset_index(drop=True)
    # Total NBA regulation time = 48 minutes
    TOTAL_GAME_SECONDS = 48 * 60
    frame["seconds_since_game_start"] = TOTAL_GAME_SECONDS - frame["seconds_remaining"]
    frame["espn_probability_delta"] = frame.groupby("team")["espn_probability"].diff()
    probability_columns_valid(frame, ["espn_probability"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    score_deltas_valid(frame)
    return frame


def fetch_game(game_id: str) -> dict:
    return request_json(SUMMARY_URL, params={"event": game_id})


def main() -> None:
    args = parse_args()
    game_ids = resolve_game_ids(args)

    for game_id in tqdm(game_ids, desc="espn games"):
        output_path = args.output_dir / f"game_{game_id}_espn.parquet"
        if output_path.exists() and not args.overwrite:
            continue

        payload = fetch_game(game_id)
        frame = build_frame(payload, game_id=game_id)
        write_parquet(frame, output_path)

    print(f"Wrote ESPN parquet files to {args.output_dir}")


if __name__ == "__main__":
    main()
