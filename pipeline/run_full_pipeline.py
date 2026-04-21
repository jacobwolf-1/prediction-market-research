from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_collection.collect_multi_season_history import (
    collect_espn_history,
    collect_game_ids,
    collect_kalshi_history,
)
from pipeline.workflow import (
    _cleanup_lead_lag_outputs,
    _select_common_game_ids,
    _select_processed_game_ids,
    build_lead_lag_outputs,
    build_merged_dataset_output,
)
from utils.ingestion_utils import DATA_DIR, RAW_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the canonical ESPN/Kalshi matched-sample data build for this repository."
    )
    parser.add_argument("--seasons", nargs="+", type=int, required=True, help="Season start years, for example: 2023 2024 2025 2026")
    parser.add_argument("--game-ids-output", type=Path, default=RAW_DIR / "game_ids.csv")
    parser.add_argument("--espn-output-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--kalshi-output-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--lead-lag-output-dir", type=Path, default=DATA_DIR / "processed" / "lead_lag_dataset")
    parser.add_argument("--merged-output-path", type=Path, default=DATA_DIR / "processed" / "merged_games.parquet")
    parser.add_argument("--kalshi-audit-output", type=Path, default=DATA_DIR / "processed" / "kalshi_discovery_audit.csv")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--max-games", type=int, default=None)
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def select_espn_game_ids(
    *,
    game_ids_frame,
    espn_output_dir: Path,
    max_games: int | None,
    skip_existing: bool,
    overwrite: bool,
) -> list[str]:
    game_ids = game_ids_frame["game_id"].astype(str).tolist()
    if skip_existing and not overwrite:
        existing_game_ids = {
            path.name.removeprefix("game_").removesuffix("_espn.parquet")
            for path in espn_output_dir.glob("game_*_espn.parquet")
        }
        game_ids = [game_id for game_id in game_ids if game_id not in existing_game_ids]
    if max_games is not None:
        game_ids = game_ids[:max_games]
    return game_ids


def main() -> None:
    args = parse_args()

    game_ids = collect_game_ids(args.seasons, args.game_ids_output, overwrite=args.overwrite)
    requested_game_ids = game_ids["game_id"].astype(str).tolist()
    pending_game_ids = list(requested_game_ids)

    if args.skip_existing and not args.overwrite:
        existing_lead_lag_game_ids = {path.stem for path in args.lead_lag_output_dir.glob("*.parquet")}
        pending_game_ids = [game_id for game_id in pending_game_ids if game_id not in existing_lead_lag_game_ids]
    if args.max_games is not None:
        pending_game_ids = pending_game_ids[: args.max_games]

    collection_game_ids = pending_game_ids if not args.overwrite else requested_game_ids[: args.max_games] if args.max_games is not None else requested_game_ids

    if args.overwrite:
        _cleanup_lead_lag_outputs(
            output_dir=args.lead_lag_output_dir,
            keep_game_ids=collection_game_ids,
        )

    espn_written = collect_espn_history(collection_game_ids, args.espn_output_dir, overwrite=args.overwrite)
    kalshi_stats = collect_kalshi_history(
        game_ids_file=args.game_ids_output,
        kalshi_output_dir=args.kalshi_output_dir,
        espn_output_dir=args.espn_output_dir,
        audit_output=args.kalshi_audit_output,
        selected_game_ids=collection_game_ids,
        max_games=args.max_games,
        max_pages=args.max_pages,
        skip_existing=args.skip_existing,
        overwrite=args.overwrite,
    )
    lead_lag_stats = build_lead_lag_outputs(
        espn_dir=args.espn_output_dir,
        market_dir=args.kalshi_output_dir,
        output_dir=args.lead_lag_output_dir,
        candidate_game_ids=collection_game_ids,
        overwrite=args.overwrite,
        max_games=None,
    )
    merged_candidate_game_ids = (
        requested_game_ids
        if not args.overwrite
        else collection_game_ids
    )
    merged_game_ids = _select_processed_game_ids(
        output_dir=args.lead_lag_output_dir,
        candidate_game_ids=_select_common_game_ids(
            espn_dir=args.espn_output_dir,
            market_dir=args.kalshi_output_dir,
            candidate_game_ids=merged_candidate_game_ids,
        ),
    )
    merged_stats = build_merged_dataset_output(
        espn_dir=args.espn_output_dir,
        market_dir=args.kalshi_output_dir,
        output_path=args.merged_output_path,
        freq=args.freq,
        game_ids=merged_game_ids,
    )

    print(f"game ids saved: {len(game_ids)}")
    print(f"espn parquet files written: {espn_written}")
    print(f"kalshi games downloaded: {kalshi_stats['games_downloaded']}")
    print(
        f"kalshi audit {'saved' if kalshi_stats['audit_refreshed'] else 'skipped'}: {kalshi_stats['audit_path']}"
    )
    print(f"lead-lag datasets written: {lead_lag_stats['datasets_written']}")
    print(f"lead-lag datasets empty: {lead_lag_stats['datasets_empty']}")
    print(f"lead-lag games failed: {lead_lag_stats['games_failed']}")
    print(f"merged rows: {merged_stats['rows']}")
    print(f"merged groups: {merged_stats['groups']}")
    print("This command builds a matched-sample research dataset. It does not imply venue-wide coverage or tradable alpha.")


if __name__ == "__main__":
    main()
