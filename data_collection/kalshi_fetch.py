from __future__ import annotations

import argparse
import math
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
        default=100,
        help="Maximum number of matched Kalshi games to collect. Defaults to 100.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing Kalshi parquet files.",
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
    try:
        parsed = datetime.strptime(parts[1], EVENT_TICKER_DATE_FORMAT)
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


def extract_event_teams(detail: dict[str, Any]) -> tuple[str, str] | None:
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
    if len(teams) >= 2:
        return teams[0], teams[1]
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


def match_game(detail: dict[str, Any], teams: tuple[str, str], games: pd.DataFrame) -> pd.Series | None:
    event_time = event_match_time(detail)
    candidates: list[tuple[int, pd.Timestamp, pd.Series]] = []
    for _, game_row in games.iterrows():
        team_set = set(teams)
        row_set = {
            normalize_abbreviation(game_row["home_team_abbr"]),
            normalize_abbreviation(game_row["away_team_abbr"]),
        }
        if team_set != row_set:
            continue
        if pd.isna(event_time) or pd.isna(game_row["date"]):
            distance = 0
        else:
            distance = abs((event_time.normalize() - game_row["date"].normalize()).days)
        if distance > 1:
            continue
        candidates.append((distance, game_row["date"], game_row))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def fetch_event_detail(event_ticker: str) -> dict[str, Any]:
    payload = kalshi_request_json(EVENT_DETAIL_URL.format(event_ticker=event_ticker))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected event detail payload for {event_ticker}")
    return payload


def discover_games(games: pd.DataFrame, max_games: int) -> list[dict[str, Any]]:
    discovered: list[dict[str, Any]] = []
    seen_event_tickers: set[str] = set()
    min_game_date = games["date"].min()
    max_game_date = games["date"].max()

    for status in ("settled",):
        cursor: str | None = None
        while True:
            params: dict[str, Any] = {
                "series_ticker": NBA_SERIES_TICKER,
                "status": status,
                "limit": MAX_PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor
            payload = kalshi_request_json(EVENTS_URL, params=params)
            events = payload.get("events") or []
            if not isinstance(events, list) or not events:
                break

            for summary in events:
                event_ticker = str(summary.get("event_ticker") or "")
                if not event_ticker or event_ticker in seen_event_tickers:
                    continue
                event_date = parse_event_ticker_date(event_ticker)
                if not pd.isna(max_game_date) and not pd.isna(event_date) and event_date > (max_game_date + pd.Timedelta(days=2)):
                    continue
                if not pd.isna(min_game_date) and not pd.isna(event_date) and event_date < (min_game_date - pd.Timedelta(days=2)):
                    return discovered
                detail = fetch_event_detail(event_ticker)
                teams = extract_event_teams(detail)
                if teams is None:
                    continue
                event_time = event_match_time(detail)
                if not pd.isna(max_game_date) and not pd.isna(event_time) and event_time > (max_game_date + pd.Timedelta(days=2)):
                    continue
                if not pd.isna(min_game_date) and not pd.isna(event_time) and event_time < (min_game_date - pd.Timedelta(days=2)):
                    return discovered
                title = (
                    (detail.get("event") or {}).get("title")
                    or summary.get("title")
                    or event_ticker
                )
                print(f"KALSHI MARKET FOUND: {title}")
                game_row = match_game(detail, teams, games)
                if game_row is None:
                    continue
                print(f"MATCHED GAME: {game_row['game_id']}")
                seen_event_tickers.add(event_ticker)
                discovered.append(
                    {
                        "detail": detail,
                        "game_row": game_row,
                    }
                )
                if len(discovered) >= max_games:
                    return discovered

            cursor = payload.get("cursor")
            if not cursor:
                break
    return discovered


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

    discovered = discover_games(games=games, max_games=args.max_games)

    games_downloaded = 0
    total_rows_collected = 0

    for item in discovered:
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

    print(f"games discovered: {len(discovered)}")
    print(f"games downloaded: {games_downloaded}")
    print(f"total rows collected: {total_rows_collected}")


if __name__ == "__main__":
    main()
