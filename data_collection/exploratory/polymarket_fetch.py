from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any
import requests

import pandas as pd
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    import data_collection.espn_historical_fetch as espn_historical_fetch
else:
    from data_collection import espn_historical_fetch
from utils.ingestion_utils import (
    NBA_TEAM_ALIASES,
    RAW_DIR,
    ensure_directory,
    load_game_ids,
    normalize_text,
    parse_json_field,
    parse_timestamp,
    request_json,
    write_parquet,
)
NBA_TEAM_ALIASES["PHI"].update({"76ers", "sixers"})
NBA_TEAM_ALIASES["POR"].update({"blazers"})
NBA_TEAM_ALIASES["OKC"].update({"okc"})
NBA_TEAM_ALIASES["NYK"].update({"knicks"})
NBA_TEAM_ALIASES["MIN"].update({"wolves"})
NBA_TEAM_ALIASES["NOP"].update({"pelicans"})
NBA_TEAM_ALIASES["SAS"].update({"spurs"})
NBA_TEAM_ALIASES["DAL"].update({"mavs", "mavericks"})
NBA_TEAM_ALIASES["MEM"].update({"grizzlies"})


GAMMA_EVENTS_URL = "https://gamma-api.polymarket.com/events"
GAMMA_EVENT_DETAIL_URL = "https://gamma-api.polymarket.com/events/{event_id}"
DATA_API_TRADES_URL = "https://data-api.polymarket.com/trades"
PRICE_HISTORY_URL = "https://clob.polymarket.com/prices-history"

WINNER_PATTERNS = (
    re.compile(r"\bwho will win\b", re.I),
    re.compile(r"\bwill\s+.+?\s+or\s+.+?\s+win\b", re.I),
    re.compile(r"\bin-game trading\b.*\bwho will win\b", re.I),
)

SPREAD_PATTERNS = (
    re.compile(r"\bwill\s+.+?\s+beat\s+.+?\s+by\s+more\s+than\s+[-\d\.]+\s+points", re.I),
)

HARD_REJECT_PATTERNS = (
    re.compile(r"\bover\b", re.I),
    re.compile(r"\bunder\b", re.I),
    re.compile(r"\b\+\b"),
    re.compile(r"\bgrammys?\b", re.I),
    re.compile(r"\bcrypto\b", re.I),
    re.compile(r"\bchess\b", re.I),
    re.compile(r"\bnfl\b", re.I),
    re.compile(r"\bnhl\b", re.I),
    re.compile(r"\bncaa\b", re.I),
)
TEAM_SPLIT_VS_PATTERN = re.compile(r"(?P<left>.+?)\s+(?:vs|v)\.?\s+(?P<right>.+)", re.I)
TEAM_SPLIT_OR_PATTERN = re.compile(r"will\s+(?P<left>.+?)\s+or\s+(?P<right>.+?)\s+win", re.I)
TEAM_SPLIT_BEAT_PATTERN = re.compile(r"will\s+(?P<left>.+?)\s+beat\s+(?P<right>.+?)\s+by\s+more\s+than", re.I)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Polymarket NBA winner-market trade history.")
    parser.add_argument(
        "--game-ids-file",
        type=Path,
        default=RAW_DIR / "game_ids.csv",
        help="CSV created by fetch_game_ids.py. Defaults to data/raw/game_ids.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR / "polymarket",
        help="Output directory for per-game parquet files. Defaults to data/raw/polymarket",
    )
    parser.add_argument(
        "--espn-output-dir",
        type=Path,
        default=RAW_DIR / "espn",
        help="Existing ESPN parquet directory. Defaults to data/raw/espn",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=100,
        help="Maximum number of games to collect. Defaults to 100.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Polymarket parquet files.",
    )
    return parser.parse_args()


def normalize_abbreviation(value: str | None) -> str:
    normalized = normalize_text("" if pd.isna(value) else str(value)).upper()
    aliases = {
        "GS": "GSW",
        "NO": "NOP",
        "SA": "SAS",
        "UTAH": "UTA",
    }
    return aliases.get(normalized, normalized)


def build_team_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for abbr, aliases in NBA_TEAM_ALIASES.items():
        lookup[normalize_text(abbr)] = abbr
        for alias in aliases:
            lookup[normalize_text(alias)] = abbr
    return lookup


TEAM_LOOKUP = build_team_lookup()


def load_games(path: Path) -> pd.DataFrame:
    games = load_game_ids(path).copy()
    games["date"] = pd.to_datetime(games["date"], utc=True, errors="coerce")
    for side in ("home", "away"):
        games[f"{side}_team_abbr"] = games[f"{side}_team_abbr"].map(normalize_abbreviation)
    return games

def classify_market_question(question: str) -> str | None:
    if any(p.search(question) for p in WINNER_PATTERNS):
        return "winner"
    if any(p.search(question) for p in SPREAD_PATTERNS):
        return "spread"
    return None


def sanitize_market_question(question: str) -> str:
    text = normalize_text(question)
    text = re.sub(r"^\(?(?:in game trading|in-game trading)\)?:?\s*", "", text)
    text = re.sub(r"^nba:\s*", "", text)
    text = re.sub(r"^\d{4}\s+nba\s+playoffs,\s*round\s+\d+:\s*", "", text)
    text = re.sub(r"\?\s*$", "", text)
    return text.strip()


def clean_team_fragment(value: str) -> str:
    fragment = value.strip(" :-?,")
    fragment = re.sub(r"^the\s+", "", fragment)
    fragment = re.sub(r"\s+game(?:\s+on|\s+scheduled for).*", "", fragment)
    fragment = re.sub(r",\s*scheduled for.*", "", fragment)
    fragment = re.sub(r"\s+scheduled for.*", "", fragment)
    fragment = re.sub(r"\s+on\s+[a-z]+\s+\d{1,2}(?:st|nd|rd|th)?.*", "", fragment)
    fragment = re.sub(r"\s*:\s*.*", "", fragment)
    return fragment.strip(" :-?,")


def extract_teams_from_title(title: str, market: dict[str, Any]) -> tuple[str, str] | None:
    cleaned = sanitize_market_question(title)

    winner_prompt = cleaned
    winner_prompt = re.sub(r"^who will win\s+", "", winner_prompt)
    winner_prompt = re.sub(r"^will\s+the\s+", "", winner_prompt)
    winner_prompt = re.sub(r"^will\s+", "", winner_prompt)

    # --- FIRST: try regex parsing (existing logic) ---
    for candidate in (cleaned, winner_prompt):
        for pattern in (TEAM_SPLIT_OR_PATTERN, TEAM_SPLIT_BEAT_PATTERN, TEAM_SPLIT_VS_PATTERN):
            match = pattern.search(candidate)
            if not match:
                continue
            left = clean_team_fragment(match.group("left"))
            right = clean_team_fragment(match.group("right"))
            if left and right:
                return left, right

    direct_vs_match = TEAM_SPLIT_VS_PATTERN.search(winner_prompt)
    if direct_vs_match:
        left = clean_team_fragment(direct_vs_match.group("left"))
        right = clean_team_fragment(direct_vs_match.group("right"))
        if left and right:
            return left, right

    for pattern in (TEAM_SPLIT_OR_PATTERN, TEAM_SPLIT_BEAT_PATTERN, TEAM_SPLIT_VS_PATTERN):
        match = pattern.search(cleaned)
        if match:
            left = clean_team_fragment(match.group("left"))
            right = clean_team_fragment(match.group("right"))
            if left and right:
                return left, right

    # --- FALLBACK: use market outcomes ---
    outcomes = parse_json_field(market.get("outcomes"))

    if isinstance(outcomes, list) and len(outcomes) == 2:
        return outcomes[0], outcomes[1]

    return None


def normalize_team_name(team_name: str) -> str | None:
    normalized = normalize_text(team_name)
    if normalized in TEAM_LOOKUP:
        return TEAM_LOOKUP[normalized]
    for alias, abbr in TEAM_LOOKUP.items():
        if normalized == alias or normalized.endswith(f" {alias}") or alias.endswith(f" {normalized}"):
            return abbr
    return None


#def fetch_nba_events(limit: int = 1000) -> list[dict[str, Any]]:
 ###     params={"limit": limit, "order": "id", "ascending": "false"},
    #)
    #if not isinstance(payload, list):
    #    return []
    #return payload


def fetch_event_detail(event_id: str | int) -> dict[str, Any]:
    payload = request_json(GAMMA_EVENT_DETAIL_URL.format(event_id=event_id))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected event detail payload for {event_id}")
    return payload


def extract_markets(detail: dict[str, Any]) -> list[dict[str, Any]]:
    markets = detail.get("markets")
    if isinstance(markets, list):
        return markets
    return []


def choose_winner_market(detail: dict[str, Any]) -> dict[str, Any] | None:
    for market in extract_markets(detail):
        question = normalize_text(market.get("question") or market.get("title") or "")
        if not market.get("enableOrderBook"):
            continue
        if not market.get("clobTokenIds"):
            continue
        market_type = classify_market_question(question)
        if market_type == "spread":
            continue
        teams = extract_teams_from_title(question, market)
        if not teams:
            continue
        normalized_teams = [normalize_team_name(team) for team in teams]
        if any(team is None for team in normalized_teams):
            continue
        return {
            **market,
            "parsed_teams": tuple(normalized_teams),
            "market_type": market_type or "winner",
            "match_time": market.get("gameStartTime") or market.get("startDate") or detail.get("startDate"),
        }
    return None


def build_game_team_aliases(game_row: pd.Series, side: str) -> set[str]:
    alias_values = {
        game_row.get(f"{side}_team_abbr"),
        game_row.get(f"{side}_team_display_name"),
        game_row.get(f"{side}_team_short_name"),
        game_row.get(f"{side}_team_location"),
        game_row.get(f"{side}_team_name"),
    }
    aliases: set[str] = set()
    canonical_abbr = normalize_abbreviation(game_row.get(f"{side}_team_abbr"))
    aliases.add(canonical_abbr)
    aliases.update(NBA_TEAM_ALIASES.get(canonical_abbr, set()))
    for value in alias_values:
        normalized = normalize_text(value)
        if normalized:
            aliases.add(normalized)
    return aliases


def match_game(detail: dict[str, Any], teams: tuple[str, str], games: pd.DataFrame) -> pd.Series | None:
    event_date = pd.to_datetime(
        detail.get("gameStartTime") or detail.get("startDate") or detail.get("endDate"),
        utc=True,
        errors="coerce",
    )
    candidates: list[tuple[int, pd.Timestamp, pd.Series]] = []
    for _, game_row in games.iterrows():
        team_set = set(teams)
        row_set = {
            normalize_abbreviation(game_row["home_team_abbr"]),
            normalize_abbreviation(game_row["away_team_abbr"]),
        }
        if team_set != row_set:
            continue
        if pd.isna(event_date) or pd.isna(game_row["date"]):
            distance = 0
        else:
            distance = abs((event_date.normalize() - game_row["date"].normalize()).days)
        if distance > 1:
            continue
        candidates.append((distance, game_row["date"], game_row))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def normalize_trade_team(outcome: Any, game_row: pd.Series) -> str | None:
    outcome_text = normalize_text(str(outcome or ""))
    home_aliases = build_game_team_aliases(game_row, "home")
    away_aliases = build_game_team_aliases(game_row, "away")
    if outcome_text in home_aliases:
        return normalize_abbreviation(game_row["home_team_abbr"])
    if outcome_text in away_aliases:
        return normalize_abbreviation(game_row["away_team_abbr"])
    for alias in home_aliases:
        if alias and alias in outcome_text:
            return normalize_abbreviation(game_row["home_team_abbr"])
    for alias in away_aliases:
        if alias and alias in outcome_text:
            return normalize_abbreviation(game_row["away_team_abbr"])
    return None


def build_market_token_team_map(market: dict[str, Any], game_row: pd.Series) -> dict[str, str]:
    outcomes = parse_json_field(market.get("outcomes")) or []
    token_ids = parse_json_field(market.get("clobTokenIds")) or []
    if not isinstance(outcomes, list) or not isinstance(token_ids, list):
        return {}
    token_team_map: dict[str, str] = {}
    for outcome, token_id in zip(outcomes, token_ids, strict=False):
        team = normalize_trade_team(outcome, game_row)
        if team is not None and token_id is not None:
            token_team_map[str(token_id)] = team
    return token_team_map


def fetch_price_history(token_id: str, fidelity: int = 1) -> list[dict[str, Any]]:
    payload = request_json(
        PRICE_HISTORY_URL,
        params={"market": token_id, "interval": "max", "fidelity": fidelity},
    )
    if not isinstance(payload, dict):
        return []
    history = payload.get("history") or []
    return history if isinstance(history, list) else []


def build_price_frame(
    token_histories: dict[str, list[dict[str, Any]]],
    event_id: str,
    game_id: str,
    game_row: pd.Series,
    token_team_map: dict[str, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for token_id, history in token_histories.items():
        team = token_team_map.get(str(token_id))
        if team is None:
            continue
        for point in history:
            timestamp = parse_timestamp(point.get("t") or point.get("timestamp") or point.get("time"))
            probability = pd.to_numeric(point.get("p") or point.get("price"), errors="coerce")
            if pd.isna(timestamp) or pd.isna(probability):
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "market_probability": probability,
                    "token_id": str(token_id),
                    "event_id": str(event_id),
                    "game_id": str(game_id),
                    "team": team,
                    "source": "polymarket",
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["team", "timestamp"]).reset_index(drop=True)


def fetch_market_prices(
    market: dict[str, Any],
    event_id: str,
    game_row: pd.Series,
) -> pd.DataFrame:
    token_team_map = build_market_token_team_map(market, game_row)
    token_ids = parse_json_field(market.get("clobTokenIds")) or []
    if not isinstance(token_ids, list):
        return pd.DataFrame()
    token_histories: dict[str, list[dict[str, Any]]] = {}
    for token_id in token_ids:
        token_id_str = str(token_id)
        history = fetch_price_history(token_id_str, fidelity=1)
        if not history:
            history = fetch_price_history(token_id_str, fidelity=1440)
        token_histories[token_id_str] = history
    frame = build_price_frame(
        token_histories=token_histories,
        event_id=event_id,
        game_id=str(game_row["game_id"]),
        game_row=game_row,
        token_team_map=token_team_map,
    )
    print(f"PRICE ROWS: {len(frame)}")
    return frame


def ensure_espn_file(game_id: str, output_dir: Path) -> None:
    output_path = output_dir / f"game_{game_id}_espn.parquet"
    if output_path.exists():
        return
    payload = espn_historical_fetch.fetch_game(game_id)
    frame = espn_historical_fetch.build_frame(payload, game_id=game_id)
    espn_historical_fetch.write_parquet(frame, output_path)


def discover_games(games: pd.DataFrame, max_games: int) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()

    min_game_date = games["date"].min()
    max_game_date = games["date"].max()

    for offset in range(0, 5000, 500):
        events = request_json(
            GAMMA_EVENTS_URL,
            params={
                "series_slug": "nba",
                "limit": 500,
                "offset": offset,
            },
        )
        if not isinstance(events, list) or not events:
            break
        for event in tqdm(events, desc="polymarket discovery"):
            event_id = str(event.get("id") or "")
            if not event_id or event_id in seen_event_ids:
                continue
            if normalize_text(event.get("seriesSlug")) != "nba":
                continue
            event_date = pd.to_datetime(event.get("startDate") or event.get("endDate"), utc=True, errors="coerce")
            if not pd.isna(min_game_date) and not pd.isna(event_date) and event_date < (min_game_date - pd.Timedelta(days=1)):
                continue
            if not pd.isna(max_game_date) and not pd.isna(event_date) and event_date > (max_game_date + pd.Timedelta(days=1)):
                continue
            detail = fetch_event_detail(event_id)
            market = choose_winner_market(detail)
            if market is None:
                continue
            title = detail.get("title") or event.get("title") or ""
            print(f"CLOB MARKET FOUND: {title}")
            teams = market["parsed_teams"]
            detail_for_match = {**detail, "gameStartTime": market.get("match_time")}
            game_row = match_game(detail_for_match, teams, games)
            if game_row is None:
                print(f"match_game failed for teams={teams} event_id={event_id}")
                continue
            print(f"MATCHED GAME: {game_row['game_id']}")
            token_ids = parse_json_field(market.get("clobTokenIds")) or []
            if isinstance(token_ids, list) and len(token_ids) >= 2:
                print(f"TOKENS: {token_ids[0]} {token_ids[1]}")
            seen_event_ids.add(event_id)
            discovered.append(
                {
                    "detail": detail,
                    "market": market,
                    "game_row": game_row,
                }
            )
            if len(discovered) >= max_games:
                return discovered
    return discovered


def main() -> None:
    args = parse_args()
    games = load_games(args.game_ids_file)
    ensure_directory(args.output_dir)
    ensure_directory(args.espn_output_dir)

    discovered = discover_games(games=games, max_games=args.max_games)

    games_downloaded = 0
    total_rows_collected = 0

    for item in tqdm(discovered, desc="polymarket trades"):
        detail = item["detail"]
        market = item["market"]
        game_row = item["game_row"]
        event_id = str(detail["id"])
        output_path = args.output_dir / f"{game_row['game_id']}.parquet"
        if output_path.exists() and not args.overwrite:
            continue

        ensure_espn_file(str(game_row["game_id"]), args.espn_output_dir)
        frame = fetch_market_prices(
            market=market,
            event_id=event_id,
            game_row=game_row,
        )
        if frame.empty:
            continue

        write_parquet(frame, output_path)
        games_downloaded += 1
        total_rows_collected += len(frame)

    print(f"games discovered: {len(discovered)}")
    print(f"games downloaded: {games_downloaded}")
    print(f"total rows collected: {total_rows_collected}")


if __name__ == "__main__":
    main()
