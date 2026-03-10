from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from data_collection import espn_historical_fetch, fetch_game_ids, kalshi_fetch
else:
    from data_collection import espn_historical_fetch, fetch_game_ids, kalshi_fetch

from utils.ingestion_utils import RAW_DIR, ensure_directory, write_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect multi-season ESPN and Kalshi NBA history with the existing schemas.")
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Season start years, for example: 2022 2023 2024 2025")
    parser.add_argument("--game-ids-output", type=Path, default=RAW_DIR / "game_ids.csv")
    parser.add_argument("--espn-output-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--kalshi-output-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def collect_game_ids(seasons: list[int], output_path: Path, overwrite: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for season in seasons:
        window = fetch_game_ids.SeasonWindow(
            season=season,
            start_date=date(season, 10, 1),
            end_date=date(season + 1, 6, 30),
        )
        for game_date in tqdm(fetch_game_ids.iter_dates(window.start_date, window.end_date), desc=f"season {season}"):
            rows.extend(fetch_game_ids.fetch_games_for_date(game_date))

    frame = pd.DataFrame(rows).drop_duplicates(subset=["game_id"]) if rows else pd.DataFrame(columns=["game_id"])
    if output_path.exists() and not overwrite:
        existing = pd.read_csv(output_path, dtype={"game_id": str})
        frame = pd.concat([existing, frame], ignore_index=True).drop_duplicates(subset=["game_id"])
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame = frame.sort_values("date").reset_index(drop=True)
    ensure_directory(output_path.parent)
    frame.to_csv(output_path, index=False)
    return frame


def collect_espn_history(game_ids: list[str], output_dir: Path, overwrite: bool) -> int:
    ensure_directory(output_dir)
    written = 0
    for game_id in tqdm(game_ids, desc="espn games"):
        output_path = output_dir / f"game_{game_id}_espn.parquet"
        if output_path.exists() and not overwrite:
            continue
        payload = espn_historical_fetch.fetch_game(game_id)
        frame = espn_historical_fetch.build_frame(payload, game_id=game_id)
        write_parquet(frame, output_path)
        written += 1
    return written


def collect_kalshi_history(
    game_ids_file: Path,
    kalshi_output_dir: Path,
    espn_output_dir: Path,
    max_games: int | None,
    max_pages: int | None,
    skip_existing: bool,
    overwrite: bool,
) -> dict[str, int]:
    games = kalshi_fetch.load_games(game_ids_file)
    ensure_directory(kalshi_output_dir)
    ensure_directory(espn_output_dir)
    existing_game_ids = {path.stem for path in kalshi_output_dir.glob("*.parquet")}

    discovery = kalshi_fetch.discover_games(
        games=games,
        max_games=max_games,
        max_pages=max_pages,
        existing_game_ids=existing_game_ids,
        skip_existing=(skip_existing and not overwrite),
    )

    games_downloaded = 0
    total_rows_collected = 0
    for item in discovery["queued"]:
        detail = item["detail"]
        game_row = item["game_row"]
        output_path = kalshi_output_dir / f"{game_row['game_id']}.parquet"
        if output_path.exists() and not overwrite:
            continue
        kalshi_fetch.ensure_espn_file(str(game_row["game_id"]), espn_output_dir)
        frame = kalshi_fetch.fetch_market_history(detail=detail, game_row=game_row)
        if frame.empty:
            continue
        write_parquet(frame, output_path)
        games_downloaded += 1
        total_rows_collected += len(frame)

    return {
        "pages_scanned": int(discovery["pages_scanned"]),
        "events_considered": int(discovery["events_considered"]),
        "games_discovered": int(len(discovery["queued"])),
        "games_matched": int(discovery["matched_games"]),
        "games_downloaded": int(games_downloaded),
        "rows_collected": int(total_rows_collected),
    }


def main() -> None:
    args = parse_args()
    game_ids = collect_game_ids(args.seasons, args.game_ids_output, overwrite=args.overwrite)
    espn_written = collect_espn_history(game_ids["game_id"].astype(str).tolist(), args.espn_output_dir, overwrite=args.overwrite)
    kalshi_stats = collect_kalshi_history(
        game_ids_file=args.game_ids_output,
        kalshi_output_dir=args.kalshi_output_dir,
        espn_output_dir=args.espn_output_dir,
        max_games=args.max_games,
        max_pages=args.max_pages,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
    )

    print(f"game ids saved: {len(game_ids)}")
    print(f"espn parquet files written: {espn_written}")
    print(f"kalshi pages scanned: {kalshi_stats['pages_scanned']}")
    print(f"kalshi events considered: {kalshi_stats['events_considered']}")
    print(f"kalshi games discovered: {kalshi_stats['games_discovered']}")
    print(f"kalshi games matched: {kalshi_stats['games_matched']}")
    print(f"kalshi games downloaded: {kalshi_stats['games_downloaded']}")
    print(f"kalshi rows collected: {kalshi_stats['rows_collected']}")


if __name__ == "__main__":
    main()
