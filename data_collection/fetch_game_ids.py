from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from utils.ingestion_utils import RAW_DIR, ensure_directory, request_json


SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


@dataclass(frozen=True)
class SeasonWindow:
    season: int
    start_date: date
    end_date: date


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch ESPN NBA game IDs for one or more dates or seasons.")
    parser.add_argument("--seasons", nargs="*", type=int, help="Season start years, for example: 2023 2024")
    parser.add_argument("--start-date", help="Inclusive start date in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Inclusive end date in YYYY-MM-DD format.")
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_DIR / "game_ids.csv",
        help="Output CSV path. Defaults to data/raw/game_ids.csv",
    )
    parser.add_argument(
        "--season-start-month",
        type=int,
        default=10,
        help="Month to start each season window. Defaults to October.",
    )
    parser.add_argument(
        "--season-end-month",
        type=int,
        default=6,
        help="Month to end each season window in the following calendar year. Defaults to June.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending to existing rows.",
    )
    return parser.parse_args()


def iter_dates(start_date: date, end_date: date) -> list[date]:
    current = start_date
    values: list[date] = []
    while current <= end_date:
        values.append(current)
        current += timedelta(days=1)
    return values


def build_windows(args: argparse.Namespace) -> list[SeasonWindow]:
    windows: list[SeasonWindow] = []
    if args.seasons:
        for season in args.seasons:
            end_day = calendar.monthrange(season + 1, args.season_end_month)[1]
            windows.append(
                SeasonWindow(
                    season=season,
                    start_date=date(season, args.season_start_month, 1),
                    end_date=date(season + 1, args.season_end_month, end_day),
                )
            )
    if args.start_date and args.end_date:
        start = pd.Timestamp(args.start_date).date()
        end = pd.Timestamp(args.end_date).date()
        windows.append(SeasonWindow(season=start.year, start_date=start, end_date=end))
    if not windows:
        raise ValueError("Provide either --seasons or both --start-date and --end-date.")
    return windows


def extract_game_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    for      event in payload.get("events", []):
        competition = (event.get("competitions") or [{}])[0]
        competitors = competition.get("competitors") or []
        home = next((team for team in competitors if team.get("homeAway") == "home"), {})
        away = next((team for team in competitors if team.get("homeAway") == "away"), {})
        rows.append(
            {
                "game_id": str(event.get("id")),
                "date": pd.to_datetime(event.get("date"), utc=True),
                "season": event.get("season", {}).get("year"),
                "season_type": event.get("season", {}).get("slug"),
                "status": competition.get("status", {}).get("type", {}).get("name"),
                "neutral_site": competition.get("neutralSite"),
                "venue": competition.get("venue", {}).get("fullName"),
                "home_team_abbr": home.get("team", {}).get("abbreviation"),
                "home_team_display_name": home.get("team", {}).get("displayName"),
                "home_team_short_name": home.get("team", {}).get("shortDisplayName"),
                "home_team_location": home.get("team", {}).get("location"),
                "home_team_name": home.get("team", {}).get("name"),
                "away_team_abbr": away.get("team", {}).get("abbreviation"),
                "away_team_display_name": away.get("team", {}).get("displayName"),
                "away_team_short_name": away.get("team", {}).get("shortDisplayName"),
                "away_team_location": away.get("team", {}).get("location"),
                "away_team_name": away.get("team", {}).get("name"),
            }
        )
    return rows


def fetch_games_for_date(game_date: date) -> list[dict]:
    payload = request_json(SCOREBOARD_URL, params={"dates": game_date.strftime("%Y%m%d")})
    return extract_game_rows(payload)


def main() -> None:
    args = parse_args()
    windows = build_windows(args)

    rows: list[dict] = []
    for window in windows:
        dates = iter_dates(window.start_date, window.end_date)
        for game_date in tqdm(dates, desc=f"season {window.season}"):
            rows.extend(fetch_games_for_date(game_date))

    if rows:
        frame = pd.DataFrame(rows).drop_duplicates(subset=["game_id"]).sort_values("date")
    else:
        frame = pd.DataFrame(
            columns=[
                "game_id",
                "date",
                "season",
                "season_type",
                "status",
                "neutral_site",
                "venue",
                "home_team_abbr",
                "home_team_display_name",
                "home_team_short_name",
                "home_team_location",
                "home_team_name",
                "away_team_abbr",
                "away_team_display_name",
                "away_team_short_name",
                "away_team_location",
                "away_team_name",
            ]
        )

    if args.output.exists() and not args.overwrite:
        existing = pd.read_csv(args.output, dtype={"game_id": str})
        frame = pd.concat([existing, frame], ignore_index=True).drop_duplicates(subset=["game_id"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")

    ensure_directory(args.output.parent)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} game IDs to {args.output}")


if __name__ == "__main__":
    main()
