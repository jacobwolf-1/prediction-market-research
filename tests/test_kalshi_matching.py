from __future__ import annotations

import pandas as pd

from data_collection.kalshi_fetch import (
    abbreviations_from_event_ticker,
    extract_event_teams,
    match_game,
    normalize_abbreviation,
    normalize_team_name,
    parse_teams_from_matchup_text,
)


def build_games_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "401869369",
                "date": pd.Timestamp("2026-04-20 00:00:00+00:00"),
                "home_team_abbr": "NYK",
                "away_team_abbr": "ATL",
                "home_team_display_name": "New York Knicks",
                "away_team_display_name": "Atlanta Hawks",
                "home_team_short_name": "Knicks",
                "away_team_short_name": "Hawks",
                "home_team_location": "New York",
                "away_team_location": "Atlanta",
                "home_team_name": "Knicks",
                "away_team_name": "Hawks",
            }
        ]
    )


def test_team_normalization_handles_aliases() -> None:
    assert normalize_abbreviation("GS") == "GSW"
    assert normalize_team_name("Golden State Warriors") == "GSW"
    assert normalize_team_name("NO Pelicans") == "NOP"


def test_parse_teams_from_matchup_text_handles_descriptors() -> None:
    parsed = parse_teams_from_matchup_text("Game 2: Atlanta at New York Winner?")

    assert parsed == ("ATL", "NYK")


def test_extract_event_teams_uses_ticker_fallback_when_titles_are_noisy() -> None:
    detail = {
        "event": {
            "event_ticker": "KXNBAGAME-26APR20ATLNYK",
            "title": "Winner market",
        }
    }

    assert abbreviations_from_event_ticker(detail["event"]["event_ticker"]) == ("ATL", "NYK")
    assert extract_event_teams(detail) == ("ATL", "NYK")


def test_match_game_uses_alias_overlap_when_exact_codes_are_unavailable() -> None:
    games = build_games_frame()
    detail = {
        "event": {"last_updated_ts": "2026-04-21T03:00:00Z"},
        "markets": [],
    }
    teams = ("new york knicks", "atlanta hawks")

    matched = match_game(detail, teams, games)

    assert matched is not None
    assert matched["game_id"] == "401869369"


def test_match_game_rejects_distant_dates_even_with_matching_teams() -> None:
    games = build_games_frame()
    detail = {
        "event": {"last_updated_ts": "2026-04-30T03:00:00Z"},
        "markets": [],
    }

    matched = match_game(detail, ("NYK", "ATL"), games)

    assert matched is None
