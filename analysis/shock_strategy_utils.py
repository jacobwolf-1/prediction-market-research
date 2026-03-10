from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from analysis.raw_time_dataset import load_raw_time_dataset
from analysis.raw_time_utils import normalize_frequency_label, resample_series


REGULATION_SECONDS = 48 * 60


def total_seconds_remaining(quarter: object, clock_seconds_remaining: object) -> float:
    if pd.isna(quarter) or pd.isna(clock_seconds_remaining):
        return float("nan")
    try:
        quarter_num = int(quarter)
        clock = float(clock_seconds_remaining)
    except (TypeError, ValueError):
        return float("nan")
    if quarter_num <= 0:
        return float("nan")
    if quarter_num <= 4:
        return float((4 - quarter_num) * 12 * 60 + clock)
    return float(clock)


def classify_game_phase(seconds_remaining: float) -> str | None:
    if pd.isna(seconds_remaining):
        return None
    if seconds_remaining > 32 * 60:
        return "early"
    if seconds_remaining > 16 * 60:
        return "mid"
    return "late"


def compute_market_updates_per_minute(market_frame: pd.DataFrame) -> float:
    if market_frame.empty:
        return float("nan")
    ordered = market_frame.sort_values("timestamp")
    duration_minutes = (ordered["timestamp"].max() - ordered["timestamp"].min()).total_seconds() / 60.0
    if duration_minutes <= 0:
        return float("nan")
    return float(len(ordered) / duration_minutes)


def build_shock_groups(
    espn_dir: Path,
    market_dir: Path,
    freq: str,
) -> tuple[list[dict[str, object]], dict[str, int]]:
    frequency = normalize_frequency_label(freq)
    dataset = load_raw_time_dataset(espn_dir=espn_dir, market_dir=market_dir)
    groups: list[dict[str, object]] = []

    for (game_id, team), item in dataset.items():
        espn = item["espn"].copy()
        market = item["market"].copy()

        raw_espn_path = espn_dir / f"game_{game_id}_espn.parquet"
        if raw_espn_path.exists():
            raw_espn = pd.read_parquet(
                raw_espn_path,
                columns=["timestamp", "team", "quarter", "game_clock_seconds_remaining", "espn_probability"],
            )
            raw_espn = raw_espn.loc[raw_espn["team"] == team].copy()
            raw_espn["timestamp"] = pd.to_datetime(raw_espn["timestamp"], utc=True, errors="coerce")
            raw_espn["total_seconds_remaining"] = raw_espn.apply(
                lambda row: total_seconds_remaining(row["quarter"], row["game_clock_seconds_remaining"]),
                axis=1,
            )
        else:
            raw_espn = espn.assign(total_seconds_remaining=np.nan)

        espn_resampled = resample_series(espn, "timestamp", "espn_probability", freq=frequency).rename(
            columns={"value": "espn_probability", "delta": "delta_espn"}
        )
        market_resampled = resample_series(market, "timestamp", "market_probability", freq=frequency).rename(
            columns={"value": "market_probability", "delta": "delta_market"}
        )
        phase_resampled = resample_series(raw_espn, "timestamp", "total_seconds_remaining", freq=frequency).rename(
            columns={"value": "total_seconds_remaining", "delta": "delta_seconds_remaining"}
        )
        quarter_resampled = resample_series(raw_espn, "timestamp", "quarter", freq=frequency).rename(
            columns={"value": "quarter", "delta": "delta_quarter"}
        )

        merged = espn_resampled.merge(market_resampled, on="timestamp", how="inner")
        merged = merged.merge(phase_resampled[["timestamp", "total_seconds_remaining"]], on="timestamp", how="left")
        merged = merged.merge(quarter_resampled[["timestamp", "quarter"]], on="timestamp", how="left")
        merged = merged.dropna(subset=["delta_espn", "delta_market"]).sort_values("timestamp").reset_index(drop=True)
        if merged.empty:
            continue

        merged["game_id"] = str(game_id)
        merged["team"] = str(team)
        merged["game_phase"] = merged["total_seconds_remaining"].apply(classify_game_phase)

        groups.append(
            {
                "game_id": str(game_id),
                "team": str(team),
                "frame": merged,
                "market_updates_per_minute": compute_market_updates_per_minute(market),
                "raw_market_updates": int(len(market)),
                "raw_market_minutes": float(
                    (market["timestamp"].max() - market["timestamp"].min()).total_seconds() / 60.0
                )
                if not market.empty
                else float("nan"),
            }
        )

    summary = {
        "games": len({game_id for game_id, _ in dataset}),
        "groups": len(groups),
        "espn_rows": sum(len(item["espn"]) for item in dataset.values()),
        "market_rows": sum(len(item["market"]) for item in dataset.values()),
    }
    return groups, summary


def filter_groups_by_liquidity(groups: list[dict[str, object]], min_market_updates_per_minute: float | None) -> list[dict[str, object]]:
    if min_market_updates_per_minute is None:
        return groups
    return [
        group
        for group in groups
        if pd.notna(group["market_updates_per_minute"]) and float(group["market_updates_per_minute"]) >= min_market_updates_per_minute
    ]
