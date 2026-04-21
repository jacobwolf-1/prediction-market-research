from __future__ import annotations

import argparse
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    import data_collection.espn_historical_fetch as espn_historical_fetch
else:
    from data_collection import espn_historical_fetch

from utils.ingestion_utils import (
    DATA_DIR,
    NBA_TEAM_ALIASES,
    RAW_DIR,
    coerce_probability,
    ensure_directory,
    load_game_ids,
    normalize_text,
    parse_timestamp,
    request_json,
    write_parquet,
)


KALSHI_API_BASE = "https://api.elections.kalshi.com/trade-api/v2"
EVENTS_URL = f"{KALSHI_API_BASE}/events"
EVENT_DETAIL_URL = f"{KALSHI_API_BASE}/events/{{event_ticker}}"
TRADES_URL = f"{KALSHI_API_BASE}/markets/trades"
CANDLESTICKS_URL = f"{KALSHI_API_BASE}/series/{{series_ticker}}/markets/{{market_ticker}}/candlesticks"
NBA_SERIES_TICKER = "KXNBAGAME"
MAX_PAGE_SIZE = 200
MAX_TRADE_PAGES = 250
EVENT_TICKER_DATE_FORMAT = "%y%b%d"
MATCHUP_SEPARATOR_PATTERN = re.compile(r"\s+(?:vs\.?|v\.?|@|at)\s+", re.I)
LEADING_DESCRIPTOR_PATTERN = re.compile(
    r"^(?:basketball\s+)?(?:(?:east|west)\s+)?play-?in\s*:\s*|^game\s+\d+\s*:\s*|^nba\s*:\s*",
    re.I,
)
PARENTHETICAL_PATTERN = re.compile(r"\([^)]*\)")
TRAILING_DESCRIPTOR_PATTERN = re.compile(
    r"\b(?:winner|loser|team|matchup|series|play-?in|playoffs?)\b.*$",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Kalshi NBA winner-market history.")
    parser.add_argument(
        "--game-ids-file",
        type=Path,
        default=RAW_DIR / "game_ids.csv",
        help="CSV created by fetch_game_ids.py. Defaults to data/raw/game_ids.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RAW_DIR / "kalshi",
        help="Output directory for per-game parquet files. Defaults to data/raw/kalshi",
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
        default=None,
        help="Maximum number of matched Kalshi games to collect.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional cap on number of event-list pages to scan.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip downloading games that already have a raw Kalshi parquet file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Kalshi parquet files.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DATA_DIR / "processed" / "kalshi_discovery_audit.csv",
        help="CSV path for Kalshi discovery audit output.",
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


def normalize_team_name(team_name: str | None) -> str | None:
    normalized = normalize_text(team_name)
    if normalized in TEAM_LOOKUP:
        return TEAM_LOOKUP[normalized]
    for alias, abbr in TEAM_LOOKUP.items():
        if normalized == alias or normalized.endswith(f" {alias}") or alias.endswith(f" {normalized}"):
            return abbr
    return None


def kalshi_request_json(url: str, *, params: dict[str, Any] | None = None) -> Any:
    return request_json(
        url,
        params=params,
        retries=6,
        backoff_seconds=5.0,
        timeout=30,
    )


def parse_event_ticker_date(event_ticker: str) -> pd.Timestamp:
    parts = event_ticker.split("-")
    if len(parts) < 2:
        return pd.NaT
    raw_date = parts[1][:7]
    try:
        parsed = datetime.strptime(raw_date, EVENT_TICKER_DATE_FORMAT)
    except ValueError:
        return pd.NaT
    return pd.Timestamp(parsed, tz="UTC")


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


def team_aliases_for_abbr(team_abbr: str | None) -> set[str]:
    canonical = normalize_abbreviation(team_abbr)
    aliases = {normalize_text(canonical)}
    aliases.update({normalize_text(alias) for alias in NBA_TEAM_ALIASES.get(canonical, set())})
    return {alias for alias in aliases if alias}


def abbreviations_from_event_ticker(event_ticker: str | None) -> tuple[str, str] | None:
    if not event_ticker:
        return None
    parts = str(event_ticker).split("-")
    if len(parts) < 2:
        return None
    suffix = parts[1][7:].upper()
    if len(suffix) != 6:
        return None
    left = normalize_abbreviation(suffix[:3])
    right = normalize_abbreviation(suffix[3:])
    if left in NBA_TEAM_ALIASES and right in NBA_TEAM_ALIASES and left != right:
        return left, right
    return None


def clean_matchup_fragment(value: Any) -> str:
    # Kalshi titles often prepend round descriptors and append market descriptors like "Winner?".
    text = normalize_text(value)
    if not text:
        return ""
    text = LEADING_DESCRIPTOR_PATTERN.sub("", text)
    text = PARENTHETICAL_PATTERN.sub(" ", text)
    text = re.sub(r"\b\d+\s*-\s*\d+\b", " ", text)
    text = re.sub(r"\b\d+(?:st|nd|rd|th)?\b", " ", text)
    text = re.sub(r"[?:!,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_team_side(value: str) -> str:
    text = clean_matchup_fragment(value)
    text = TRAILING_DESCRIPTOR_PATTERN.sub("", text).strip()
    text = re.sub(r"^(?:the|seed\s+\d+\s+)", "", text).strip()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_teams_from_matchup_text(value: Any) -> tuple[str, str] | None:
    cleaned = clean_matchup_fragment(value)
    if not cleaned:
        return None
    parts = MATCHUP_SEPARATOR_PATTERN.split(cleaned, maxsplit=1)
    if len(parts) != 2:
        return None
    left = normalize_team_name(clean_team_side(parts[0]))
    right = normalize_team_name(clean_team_side(parts[1]))
    if left is None or right is None or left == right:
        return None
    return left, right


def extract_teams_from_market_metadata(market: dict[str, Any]) -> list[str]:
    values = [
        market.get("title"),
        market.get("question"),
        market.get("subtitle"),
        market.get("yes_sub_title"),
        market.get("yes_title"),
    ]
    teams: list[str] = []
    for value in values:
        parsed = parse_teams_from_matchup_text(value)
        if parsed is not None:
            for team in parsed:
                if team not in teams:
                    teams.append(team)
        single = normalize_team_name(value)
        if single is not None and single not in teams:
            teams.append(single)
    return teams


def extract_event_teams(detail: dict[str, Any]) -> tuple[str, str] | None:
    # Prefer direct market metadata first because ticker suffixes are the least ambiguous when present.
    markets = detail.get("markets") or []
    teams: list[str] = []
    for market in markets:
        ticker = str(market.get("ticker") or "")
        suffix = ticker.rsplit("-", 1)[-1]
        team = normalize_abbreviation(suffix)
        if team in NBA_TEAM_ALIASES and team not in teams:
            teams.append(team)
            continue
        team = normalize_team_name(market.get("yes_sub_title"))
        if team is not None and team not in teams:
            teams.append(team)
        for parsed_team in extract_teams_from_market_metadata(market):
            if parsed_team not in teams:
                teams.append(parsed_team)
    if len(teams) >= 2:
        return teams[0], teams[1]

    # Fall back to event-level titles for playoff/play-in titles and standard "A vs B" or "A at B" formats.
    event = detail.get("event") or {}
    event_values = [
        detail.get("title"),
        event.get("title"),
        event.get("subtitle"),
        event.get("name"),
    ]
    for value in event_values:
        parsed = parse_teams_from_matchup_text(value)
        if parsed is not None:
            return parsed

    # Finally, compact event ticker suffixes like "...-25APR15ATLORL" can recover the teams even when titles are noisy.
    event_ticker = str(event.get("event_ticker") or detail.get("event_ticker") or "")
    parsed_from_ticker = abbreviations_from_event_ticker(event_ticker)
    if parsed_from_ticker is not None:
        return parsed_from_ticker

    return None


def event_match_time(detail: dict[str, Any]) -> pd.Timestamp:
    markets = detail.get("markets") or []
    for field in ("expected_expiration_time", "close_time", "settlement_ts", "open_time"):
        for market in markets:
            timestamp = parse_timestamp(market.get(field))
            if not pd.isna(timestamp):
                return timestamp
    event = detail.get("event") or {}
    return parse_timestamp(event.get("last_updated_ts"))


def event_date_distance_days(event_time: pd.Timestamp, game_date: pd.Timestamp) -> int:
    if pd.isna(event_time) or pd.isna(game_date):
        return 0
    return abs((event_time.normalize() - game_date.normalize()).days)


def team_overlap_count(teams: tuple[str, str], game_row: pd.Series) -> int:
    event_aliases = {normalize_text(team) for team in teams if team}
    row_aliases = (
        team_aliases_for_abbr(game_row["home_team_abbr"])
        | team_aliases_for_abbr(game_row["away_team_abbr"])
        | build_game_team_aliases(game_row, "home")
        | build_game_team_aliases(game_row, "away")
    )
    matched_event_teams = 0
    for team in event_aliases:
        if team in row_aliases:
            matched_event_teams += 1
    return matched_event_teams


def match_score(
    teams: tuple[str, str],
    game_row: pd.Series,
    event_time: pd.Timestamp,
) -> tuple[int, int, int]:
    team_set = set(teams)
    row_set = {
        normalize_abbreviation(game_row["home_team_abbr"]),
        normalize_abbreviation(game_row["away_team_abbr"]),
    }
    exact_team_match = int(team_set == row_set)
    overlap = team_overlap_count(teams, game_row)
    distance = event_date_distance_days(event_time, game_row["date"])
    return (exact_team_match, overlap, -distance)


def find_match_candidates(
    teams: tuple[str, str],
    games: pd.DataFrame,
    event_time: pd.Timestamp,
) -> list[tuple[int, int, int, pd.Timestamp, pd.Series]]:
    candidates: list[tuple[int, int, int, pd.Timestamp, pd.Series]] = []
    for _, game_row in games.iterrows():
        exact_team_match, overlap, neg_distance = match_score(teams, game_row, event_time)
        distance = -neg_distance
        if overlap == 0:
            continue
        if exact_team_match:
            if distance > 1:
                continue
        else:
            if overlap < 2 or distance > 2:
                continue
        candidates.append((exact_team_match, overlap, distance, game_row["date"], game_row))
    candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    return candidates


def choose_match_candidate(
    candidates: list[tuple[int, int, int, pd.Timestamp, pd.Series]],
) -> tuple[pd.Series, int, str] | None:
    if not candidates:
        return None
    exact_team_match, overlap, distance, _, game_row = candidates[0]
    if exact_team_match:
        return game_row, distance, "exact_team_match"
    if overlap >= 2 and distance <= 2:
        return game_row, distance, "alias_fallback_match"
    return None


def match_game(detail: dict[str, Any], teams: tuple[str, str], games: pd.DataFrame) -> pd.Series | None:
    event_time = event_match_time(detail)
    candidates = find_match_candidates(teams=teams, games=games, event_time=event_time)
    chosen = choose_match_candidate(candidates)
    if chosen is None:
        return None
    return chosen[0]


def format_timestamp(value: pd.Timestamp) -> str | None:
    if value is None or pd.isna(value):
        return None
    return pd.Timestamp(value).isoformat()


def build_audit_row(
    *,
    event_ticker: str,
    title: str,
    status: str,
    reason: str,
    event_date: pd.Timestamp,
    event_time: pd.Timestamp,
    teams: tuple[str, str] | None = None,
    matched_game_id: str | None = None,
    match_distance_days: int | None = None,
) -> dict[str, object]:
    return {
        "event_ticker": event_ticker,
        "title": title,
        "status": status,
        "reason": reason,
        "event_date": format_timestamp(event_date),
        "event_time": format_timestamp(event_time),
        "team_1": teams[0] if teams else None,
        "team_2": teams[1] if teams else None,
        "matched_game_id": matched_game_id,
        "match_distance_days": match_distance_days,
    }


def write_discovery_audit(rows: list[dict[str, object]], output_path: Path) -> None:
    ensure_directory(output_path.parent)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "event_ticker",
                "title",
                "status",
                "reason",
                "event_date",
                "event_time",
                "team_1",
                "team_2",
                "matched_game_id",
                "match_distance_days",
            ]
        )
    frame.to_csv(output_path, index=False)


def fetch_event_detail(event_ticker: str) -> dict[str, Any]:
    payload = kalshi_request_json(EVENT_DETAIL_URL.format(event_ticker=event_ticker))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected event detail payload for {event_ticker}")
    return payload


def discover_games(
    games: pd.DataFrame,
    max_games: int | None,
    max_pages: int | None = None,
    existing_game_ids: set[str] | None = None,
    skip_existing: bool = False,
) -> dict[str, Any]:
    queued: list[dict[str, Any]] = []
    audit_rows: list[dict[str, object]] = []
    seen_event_tickers: set[str] = set()
    min_game_date = games["date"].min()
    max_game_date = games["date"].max()
    pages_scanned = 0
    events_considered = 0
    matched_games = 0
    skipped_existing_count = 0

    for status in ("settled",):
        cursor: str | None = None
        while True:
            if max_pages is not None and pages_scanned >= max_pages:
                return {
                    "queued": queued,
                    "audit_rows": audit_rows,
                    "pages_scanned": pages_scanned,
                    "events_considered": events_considered,
                    "matched_games": matched_games,
                    "skipped_existing": skipped_existing_count,
                }
            params: dict[str, Any] = {
                "series_ticker": NBA_SERIES_TICKER,
                "status": status,
                "limit": MAX_PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor
            payload = kalshi_request_json(EVENTS_URL, params=params)
            pages_scanned += 1
            events = payload.get("events") or []
            if not isinstance(events, list) or not events:
                break

            for summary in events:
                events_considered += 1
                event_ticker = str(summary.get("event_ticker") or "")
                title = str(summary.get("title") or event_ticker)
                event_date = parse_event_ticker_date(event_ticker)
                if not event_ticker or event_ticker in seen_event_tickers:
                    audit_rows.append(
                        build_audit_row(
                            event_ticker=event_ticker,
                            title=title,
                            status="skipped",
                            reason="missing_or_duplicate_event_ticker",
                            event_date=event_date,
                            event_time=pd.NaT,
                        )
                    )
                    continue
                if not pd.isna(max_game_date) and not pd.isna(event_date) and event_date > (max_game_date + pd.Timedelta(days=2)):
                    audit_rows.append(
                        build_audit_row(
                            event_ticker=event_ticker,
                            title=title,
                            status="skipped",
                            reason="event_date_after_game_window",
                            event_date=event_date,
                            event_time=pd.NaT,
                        )
                    )
                    continue
                if not pd.isna(min_game_date) and not pd.isna(event_date) and event_date < (min_game_date - pd.Timedelta(days=2)):
                    return {
                        "queued": queued,
                        "audit_rows": audit_rows,
                        "pages_scanned": pages_scanned,
                        "events_considered": events_considered,
                        "matched_games": matched_games,
                        "skipped_existing": skipped_existing_count,
                    }
                detail = fetch_event_detail(event_ticker)
                title = (
                    (detail.get("event") or {}).get("title")
                    or summary.get("title")
                    or event_ticker
                )
                teams = extract_event_teams(detail)
                if teams is None:
                    audit_rows.append(
                        build_audit_row(
                            event_ticker=event_ticker,
                            title=title,
                            status="unmatched",
                            reason="could_not_extract_two_teams",
                            event_date=event_date,
                            event_time=event_match_time(detail),
                        )
                    )
                    continue
                event_time = event_match_time(detail)
                if not pd.isna(max_game_date) and not pd.isna(event_time) and event_time > (max_game_date + pd.Timedelta(days=2)):
                    audit_rows.append(
                        build_audit_row(
                            event_ticker=event_ticker,
                            title=title,
                            status="skipped",
                            reason="event_time_after_game_window",
                            event_date=event_date,
                            event_time=event_time,
                            teams=teams,
                        )
                    )
                    continue
                if not pd.isna(min_game_date) and not pd.isna(event_time) and event_time < (min_game_date - pd.Timedelta(days=2)):
                    return {
                        "queued": queued,
                        "audit_rows": audit_rows,
                        "pages_scanned": pages_scanned,
                        "events_considered": events_considered,
                        "matched_games": matched_games,
                        "skipped_existing": skipped_existing_count,
                    }
                print(f"KALSHI MARKET FOUND: {title}")
                candidates = find_match_candidates(teams=teams, games=games, event_time=event_time)
                chosen = choose_match_candidate(candidates)
                if chosen is None:
                    audit_rows.append(
                        build_audit_row(
                            event_ticker=event_ticker,
                            title=title,
                            status="unmatched",
                            reason="no_exact_team_and_date_match",
                            event_date=event_date,
                            event_time=event_time,
                            teams=teams,
                        )
                    )
                    continue
                game_row, distance, match_kind = chosen
                print(f"MATCHED GAME: {game_row['game_id']}")
                matched_games += 1
                seen_event_tickers.add(event_ticker)
                game_id = str(game_row["game_id"])
                if skip_existing and existing_game_ids and game_id in existing_game_ids:
                    skipped_existing_count += 1
                    audit_rows.append(
                        build_audit_row(
                            event_ticker=event_ticker,
                            title=title,
                            status="matched_existing",
                            reason=f"{match_kind}_existing_output_skipped",
                            event_date=event_date,
                            event_time=event_time,
                            teams=teams,
                            matched_game_id=game_id,
                            match_distance_days=distance,
                        )
                    )
                    continue
                audit_rows.append(
                    build_audit_row(
                        event_ticker=event_ticker,
                        title=title,
                        status="queued",
                        reason=f"{match_kind}_queued_for_download",
                        event_date=event_date,
                        event_time=event_time,
                        teams=teams,
                        matched_game_id=game_id,
                        match_distance_days=distance,
                    )
                )
                queued.append(
                    {
                        "detail": detail,
                        "game_row": game_row,
                    }
                )
                if max_games is not None and len(queued) >= max_games:
                    return {
                        "queued": queued,
                        "audit_rows": audit_rows,
                        "pages_scanned": pages_scanned,
                        "events_considered": events_considered,
                        "matched_games": matched_games,
                        "skipped_existing": skipped_existing_count,
                    }

            cursor = payload.get("cursor")
            if not cursor:
                break
    return {
        "queued": queued,
        "audit_rows": audit_rows,
        "pages_scanned": pages_scanned,
        "events_considered": events_considered,
        "matched_games": matched_games,
        "skipped_existing": skipped_existing_count,
    }


def fetch_trades_for_market(
    market_ticker: str,
    team: str,
    event_id: str,
    game_id: str,
    min_ts: int | None = None,
    max_ts: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None

    for _ in range(MAX_TRADE_PAGES):
        params: dict[str, Any] = {"ticker": market_ticker, "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        if min_ts is not None:
            params["min_ts"] = min_ts
        if max_ts is not None:
            params["max_ts"] = max_ts
        payload = kalshi_request_json(TRADES_URL, params=params)
        trades = payload.get("trades") or []
        if not isinstance(trades, list) or not trades:
            break
        for trade in trades:
            timestamp = parse_timestamp(trade.get("created_time"))
            probability = coerce_probability(
                trade.get("yes_price_dollars")
                or trade.get("yes_price")
                or trade.get("price")
            )
            if pd.isna(timestamp) or probability is None:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "market_probability": probability,
                    "team": team,
                    "market_id": market_ticker,
                    "event_id": event_id,
                    "game_id": game_id,
                    "source": "kalshi",
                }
            )
        cursor = payload.get("cursor")
        if not cursor:
            break

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["team", "timestamp"]).reset_index(drop=True)


def fetch_candlesticks_for_market(
    series_ticker: str,
    market_ticker: str,
    team: str,
    event_id: str,
    game_id: str,
    start_ts: int,
    end_ts: int,
) -> pd.DataFrame:
    total_minutes = max(1, math.ceil((end_ts - start_ts) / 60))
    period_interval = max(1, math.ceil(total_minutes / 5000))
    payload = kalshi_request_json(
        CANDLESTICKS_URL.format(series_ticker=series_ticker, market_ticker=market_ticker),
        params={
            "start_ts": start_ts,
            "end_ts": end_ts,
            "period_interval": period_interval,
        },
    )
    candlesticks = payload.get("candlesticks") or []
    if not isinstance(candlesticks, list):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for candle in candlesticks:
        price = candle.get("price") or {}
        timestamp = parse_timestamp(candle.get("end_period_ts"))
        probability = coerce_probability(price.get("close_dollars") or price.get("close"))
        if pd.isna(timestamp) or probability is None:
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "market_probability": probability,
                "team": team,
                "market_id": market_ticker,
                "event_id": event_id,
                "game_id": game_id,
                "source": "kalshi",
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["team", "timestamp"]).reset_index(drop=True)


def fetch_market_history(detail: dict[str, Any], game_row: pd.Series) -> pd.DataFrame:
    event = detail.get("event") or {}
    event_id = str(event.get("event_ticker") or "")
    series_ticker = str(event.get("series_ticker") or NBA_SERIES_TICKER)
    game_id = str(game_row["game_id"])
    frames: list[pd.DataFrame] = []

    for market in detail.get("markets") or []:
        market_ticker = str(market.get("ticker") or "")
        if not market_ticker:
            continue
        team = normalize_abbreviation(market_ticker.rsplit("-", 1)[-1])
        if team not in NBA_TEAM_ALIASES:
            team = normalize_team_name(market.get("yes_sub_title"))
        if team is None:
            continue

        open_ts = parse_timestamp(market.get("open_time"))
        end_ts = parse_timestamp(
            market.get("settlement_ts")
            or market.get("close_time")
            or market.get("expected_expiration_time")
        )
        min_ts = int(open_ts.timestamp()) if not pd.isna(open_ts) else None
        max_ts = int(end_ts.timestamp()) if not pd.isna(end_ts) else None

        trade_frame = fetch_trades_for_market(
            market_ticker=market_ticker,
            team=team,
            event_id=event_id,
            game_id=game_id,
            min_ts=min_ts,
            max_ts=max_ts,
        )
        if not trade_frame.empty:
            frames.append(trade_frame)
            continue

        if min_ts is None or max_ts is None or min_ts >= max_ts:
            continue
        candle_frame = fetch_candlesticks_for_market(
            series_ticker=series_ticker,
            market_ticker=market_ticker,
            team=team,
            event_id=event_id,
            game_id=game_id,
            start_ts=min_ts,
            end_ts=max_ts,
        )
        if not candle_frame.empty:
            frames.append(candle_frame)

    if not frames:
        return pd.DataFrame(
            columns=["timestamp", "market_probability", "team", "market_id", "event_id", "game_id", "source"]
        )

    frame = pd.concat(frames, ignore_index=True)
    frame = frame.dropna(subset=["timestamp", "market_probability", "team"])
    frame = frame.drop_duplicates(subset=["timestamp", "team", "market_id", "market_probability"])
    frame = frame.sort_values(["team", "timestamp"]).reset_index(drop=True)
    print(f"ROWS COLLECTED: {len(frame)}")
    return frame


def ensure_espn_file(game_id: str, output_dir: Path) -> None:
    output_path = output_dir / f"game_{game_id}_espn.parquet"
    if output_path.exists():
        return
    payload = espn_historical_fetch.fetch_game(game_id)
    frame = espn_historical_fetch.build_frame(payload, game_id=game_id)
    espn_historical_fetch.write_parquet(frame, output_path)


def main() -> None:
    args = parse_args()
    games = load_games(args.game_ids_file)
    ensure_directory(args.output_dir)
    ensure_directory(args.espn_output_dir)
    existing_game_ids = {path.stem for path in args.output_dir.glob("*.parquet")}

    discovery = discover_games(
        games=games,
        max_games=args.max_games,
        max_pages=args.max_pages,
        existing_game_ids=existing_game_ids,
        skip_existing=(args.skip_existing and not args.overwrite),
    )
    write_discovery_audit(discovery["audit_rows"], args.audit_output)
    queued = discovery["queued"]

    games_downloaded = 0
    total_rows_collected = 0

    for item in queued:
        detail = item["detail"]
        game_row = item["game_row"]
        output_path = args.output_dir / f"{game_row['game_id']}.parquet"
        if output_path.exists() and not args.overwrite:
            continue

        ensure_espn_file(str(game_row["game_id"]), args.espn_output_dir)
        frame = fetch_market_history(detail=detail, game_row=game_row)
        if frame.empty:
            continue

        write_parquet(frame, output_path)
        games_downloaded += 1
        total_rows_collected += len(frame)

    print(f"pages scanned: {discovery['pages_scanned']}")
    print(f"events considered: {discovery['events_considered']}")
    print(f"games discovered: {len(queued)}")
    print(f"games matched to ESPN: {discovery['matched_games']}")
    print(f"games skipped existing: {discovery['skipped_existing']}")
    print(f"games downloaded: {games_downloaded}")
    print(f"total rows collected: {total_rows_collected}")
    print(f"audit saved to: {args.audit_output}")


if __name__ == "__main__":
    main()
