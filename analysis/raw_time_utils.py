from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analysis.raw_time_dataset import dataset_summary, load_raw_time_dataset
from analysis.time_series_utils import fit_simple_regression, normalize_frequency_label


def frequency_to_seconds(frequency: str) -> int:
    return int(pd.Timedelta(normalize_frequency_label(frequency)).total_seconds())


def resample_series(df: pd.DataFrame, time_col: str, value_col: str, freq: str = "1s") -> pd.DataFrame:
    frequency = normalize_frequency_label(freq)
    frame = df[[time_col, value_col]].copy()
    frame[time_col] = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    frame = frame.dropna(subset=[time_col, value_col]).sort_values(time_col)
    frame = frame.drop_duplicates(subset=[time_col], keep="last")
    if frame.empty:
        return pd.DataFrame(columns=["timestamp", "value", "delta"])
    resampled = frame.set_index(time_col)[[value_col]].resample(frequency).ffill()
    resampled["delta"] = resampled[value_col].diff()
    resampled = resampled.dropna(subset=["delta"]).reset_index().rename(columns={time_col: "timestamp", value_col: "value"})
    return resampled[["timestamp", "value", "delta"]]


def build_raw_time_groups(
    espn_dir: Path,
    market_dir: Path,
    freq: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    dataset = load_raw_time_dataset(espn_dir=espn_dir, market_dir=market_dir)
    groups: list[dict[str, Any]] = []
    for (game_id, team), item in dataset.items():
        espn_resampled = resample_series(item["espn"], "timestamp", "espn_probability", freq=freq)
        market_resampled = resample_series(item["market"], "timestamp", "market_probability", freq=freq)
        merged = espn_resampled.merge(
            market_resampled,
            on="timestamp",
            how="inner",
            suffixes=("_espn", "_market"),
        )
        merged = merged.dropna(subset=["delta_espn", "delta_market"]).copy()
        if merged.empty:
            continue
        merged["game_id"] = str(game_id)
        merged["team"] = str(team)
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        groups.append(
            {
                "game_id": str(game_id),
                "team": str(team),
                "frame": merged,
                "d_espn": merged["delta_espn"].to_numpy(dtype=float),
                "d_market": merged["delta_market"].to_numpy(dtype=float),
                "market_probability": merged["value_market"].to_numpy(dtype=float),
                "espn_probability": merged["value_espn"].to_numpy(dtype=float),
                "n_obs": int(len(merged)),
            }
        )
    return groups, dataset_summary(dataset)


def lagged_views(d_espn: np.ndarray, d_market: np.ndarray, lag_periods: int) -> tuple[np.ndarray, np.ndarray]:
    if lag_periods > 0:
        if lag_periods >= len(d_espn):
            return np.array([], dtype=float), np.array([], dtype=float)
        return d_espn[:-lag_periods], d_market[lag_periods:]
    if lag_periods < 0:
        offset = -lag_periods
        if offset >= len(d_espn):
            return np.array([], dtype=float), np.array([], dtype=float)
        return d_espn[offset:], d_market[:-offset]
    return d_espn, d_market


def correlation_from_accumulators(n: int, sum_x: float, sum_y: float, sum_x2: float, sum_y2: float, sum_xy: float) -> float:
    if n < 2:
        return float("nan")
    numerator = (n * sum_xy) - (sum_x * sum_y)
    denominator_left = (n * sum_x2) - (sum_x * sum_x)
    denominator_right = (n * sum_y2) - (sum_y * sum_y)
    denominator = denominator_left * denominator_right
    if denominator <= 0:
        return float("nan")
    return float(numerator / np.sqrt(denominator))


def build_lag_correlation_table(groups: list[dict[str, Any]], max_lag_seconds: int, frequency_seconds: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    max_lag_periods = max_lag_seconds // frequency_seconds
    for lag_periods in range(-max_lag_periods, max_lag_periods + 1):
        lag_seconds = lag_periods * frequency_seconds
        n = 0
        sum_x = 0.0
        sum_y = 0.0
        sum_x2 = 0.0
        sum_y2 = 0.0
        sum_xy = 0.0
        for group in groups:
            x, y = lagged_views(group["d_espn"], group["d_market"], lag_periods)
            if len(x) == 0:
                continue
            n += len(x)
            sum_x += float(x.sum())
            sum_y += float(y.sum())
            sum_x2 += float(np.dot(x, x))
            sum_y2 += float(np.dot(y, y))
            sum_xy += float(np.dot(x, y))
        rows.append(
            {
                "lag_seconds": lag_seconds,
                "lag_periods": lag_periods,
                "correlation": correlation_from_accumulators(n, sum_x, sum_y, sum_x2, sum_y2, sum_xy),
                "num_observations": n,
            }
        )
    return pd.DataFrame(rows)


def build_per_group_peak_summary(groups: list[dict[str, Any]], max_lag_seconds: int, frequency_seconds: int) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for group in groups:
        results = build_lag_correlation_table([group], max_lag_seconds=max_lag_seconds, frequency_seconds=frequency_seconds)
        valid = results.dropna(subset=["correlation"])
        if valid.empty:
            peak_corr = np.nan
            peak_lag_seconds = np.nan
        else:
            peak = valid.loc[valid["correlation"].idxmax()]
            peak_corr = float(peak["correlation"])
            peak_lag_seconds = int(peak["lag_seconds"])
        rows.append(
            {
                "game_id": group["game_id"],
                "team": group["team"],
                "peak_lag_seconds": peak_lag_seconds,
                "peak_correlation": peak_corr,
                "num_observations": group["n_obs"],
            }
        )
    return pd.DataFrame(rows)


def choose_example_games(market_dir: Path) -> list[str]:
    rows = []
    for path in sorted(market_dir.glob("*.parquet")):
        count = len(pd.read_parquet(path, columns=["timestamp"]))
        rows.append({"game_id": path.stem, "raw_market_rows": count})
    counts = pd.DataFrame(rows).sort_values("raw_market_rows").reset_index(drop=True)
    if counts.empty:
        return []
    return [
        str(counts.iloc[-1]["game_id"]),
        str(counts.iloc[len(counts) // 2]["game_id"]),
        str(counts.iloc[0]["game_id"]),
    ]


__all__ = [
    "build_lag_correlation_table",
    "build_per_group_peak_summary",
    "build_raw_time_groups",
    "choose_example_games",
    "fit_simple_regression",
    "frequency_to_seconds",
    "lagged_views",
    "normalize_frequency_label",
    "resample_series",
]
