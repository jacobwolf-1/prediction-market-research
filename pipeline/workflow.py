from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis import build_lead_lag_dataset, rebuild_merged_dataset
from analysis.raw_time_dataset import load_raw_time_dataset
from utils.ingestion_utils import ensure_directory, write_parquet


def _select_common_game_ids(
    *,
    espn_dir: Path,
    market_dir: Path,
    candidate_game_ids: list[str] | None = None,
    output_dir: Path | None = None,
    overwrite: bool = False,
    max_games: int | None = None,
) -> list[str]:
    espn_files = build_lead_lag_dataset.discover_espn_files(espn_dir)
    market_files = build_lead_lag_dataset.discover_market_files(market_dir)
    available_game_ids = set(espn_files).intersection(market_files)
    common_game_ids = (
        [game_id for game_id in candidate_game_ids if game_id in available_game_ids]
        if candidate_game_ids is not None
        else sorted(available_game_ids)
    )

    if output_dir is not None and not overwrite:
        common_game_ids = [
            game_id for game_id in common_game_ids if not (output_dir / f"{game_id}.parquet").exists()
        ]

    if max_games is not None:
        common_game_ids = common_game_ids[:max_games]

    return common_game_ids


def _cleanup_lead_lag_outputs(
    *,
    output_dir: Path,
    keep_game_ids: list[str],
) -> None:
    ensure_directory(output_dir)
    keep_game_id_set = set(keep_game_ids)
    for path in sorted(output_dir.glob("*.parquet")):
        if path.stem not in keep_game_id_set:
            path.unlink()


def _select_processed_game_ids(
    *,
    output_dir: Path,
    candidate_game_ids: list[str],
) -> list[str]:
    processed_game_ids = {
        path.stem
        for path in sorted(output_dir.glob("*.parquet"))
    }
    return [game_id for game_id in candidate_game_ids if game_id in processed_game_ids]


def build_lead_lag_outputs(
    *,
    espn_dir: Path,
    market_dir: Path,
    output_dir: Path,
    candidate_game_ids: list[str] | None = None,
    overwrite: bool = False,
    max_games: int | None = None,
) -> dict[str, int | list[str]]:
    ensure_directory(output_dir)
    espn_files = build_lead_lag_dataset.discover_espn_files(espn_dir)
    market_files = build_lead_lag_dataset.discover_market_files(market_dir)
    common_game_ids = _select_common_game_ids(
        espn_dir=espn_dir,
        market_dir=market_dir,
        candidate_game_ids=candidate_game_ids,
        output_dir=output_dir,
        overwrite=overwrite,
        max_games=max_games,
    )

    written = 0
    empty = 0
    failed_game_ids: list[str] = []
    for game_id in common_game_ids:
        output_path = output_dir / f"{game_id}.parquet"
        if overwrite and output_path.exists():
            output_path.unlink()
        try:
            frame = build_lead_lag_dataset.build_game_dataset(game_id, espn_files[game_id], market_files[game_id])
        except Exception as exc:
            print(f"{game_id} skipped: {exc}")
            failed_game_ids.append(game_id)
            continue
        if frame.empty:
            empty += 1
            continue
        write_parquet(frame, output_path)
        written += 1

    return {
        "games_considered": len(common_game_ids),
        "datasets_written": written,
        "datasets_empty": empty,
        "games_failed": len(failed_game_ids),
        "failed_game_ids": failed_game_ids,
    }


def build_merged_dataset_output(
    *,
    espn_dir: Path,
    market_dir: Path,
    output_path: Path,
    freq: str = "1s",
    game_ids: list[str] | None = None,
    max_games: int | None = None,
    overwrite: bool = False,
) -> dict[str, int]:
    selected_game_ids = (
        list(game_ids)
        if game_ids is not None
        else _select_common_game_ids(
            espn_dir=espn_dir,
            market_dir=market_dir,
            overwrite=overwrite,
            max_games=max_games,
        )
    )
    selected_game_id_set = set(selected_game_ids)
    dataset = load_raw_time_dataset(espn_dir=espn_dir, market_dir=market_dir)
    frames: list[pd.DataFrame] = []

    for (game_id, team), item in dataset.items():
        if game_id not in selected_game_id_set:
            continue
        espn_path = espn_dir / f"game_{game_id}_espn.parquet"
        market_frame = item["market"][["timestamp", "market_probability"]].copy()
        frame = rebuild_merged_dataset.build_group_frame(game_id, team, espn_path, market_frame, freq)
        if not frame.empty:
            frames.append(frame)

    merged = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "timestamp",
                "game_id",
                "team",
                "espn_probability",
                "market_probability",
                "quarter",
                "seconds_remaining",
                "home_score",
                "away_score",
            ]
        )
    )
    ensure_directory(output_path.parent)
    merged.to_parquet(output_path, index=False)
    return {
        "rows": int(len(merged)),
        "games": int(merged["game_id"].nunique()) if not merged.empty else 0,
        "groups": int(merged[["game_id", "team"]].drop_duplicates().shape[0]) if not merged.empty else 0,
    }
