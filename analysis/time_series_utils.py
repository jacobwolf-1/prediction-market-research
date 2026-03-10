from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def normalize_frequency_label(frequency: str) -> str:
    return str(frequency).strip().lower()


def frequency_to_seconds(frequency: str) -> int:
    normalized = normalize_frequency_label(frequency)
    return int(pd.Timedelta(normalized).total_seconds())


def load_lead_lag_datasets(input_dir: Path) -> pd.DataFrame:
    paths = sorted(input_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame()
    frames = [pd.read_parquet(path) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    required = ["timestamp", "game_id", "team", "espn_probability", "market_probability"]
    return frame[required].dropna(subset=required).sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def build_input_diagnostics(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "num_rows": 0,
            "num_games": 0,
            "num_groups": 0,
            "duplicate_game_team_timestamps": 0,
        }
    return {
        "num_rows": int(len(frame)),
        "num_games": int(frame["game_id"].nunique()),
        "num_groups": int(frame[["game_id", "team"]].drop_duplicates().shape[0]),
        "duplicate_game_team_timestamps": int(frame.duplicated(subset=["game_id", "team", "timestamp"]).sum()),
    }


def resample_group(group: pd.DataFrame, frequency: str) -> pd.DataFrame:
    normalized_frequency = normalize_frequency_label(frequency)
    ordered = group.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").copy()
    if ordered.empty:
        return pd.DataFrame(columns=["timestamp", "game_id", "team", "espn_probability", "market_probability", "d_espn", "d_market"])

    resampled = (
        ordered.set_index("timestamp")[["espn_probability", "market_probability"]]
        .resample(normalized_frequency)
        .ffill()
    )
    resampled["d_espn"] = resampled["espn_probability"].diff()
    resampled["d_market"] = resampled["market_probability"].diff()
    resampled = resampled.dropna(subset=["d_espn", "d_market"]).reset_index()
    resampled["game_id"] = str(group["game_id"].iloc[0])
    resampled["team"] = str(group["team"].iloc[0])
    return resampled[["timestamp", "game_id", "team", "espn_probability", "market_probability", "d_espn", "d_market"]]


def build_resampled_frame(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    groups = [resample_group(group, frequency) for _, group in frame.groupby(["game_id", "team"], sort=True)]
    groups = [group for group in groups if not group.empty]
    if not groups:
        return pd.DataFrame(columns=["timestamp", "game_id", "team", "espn_probability", "market_probability", "d_espn", "d_market"])
    return pd.concat(groups, ignore_index=True).sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True)


def build_resampled_groups(frame: pd.DataFrame, frequency: str) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for (game_id, team), group in frame.groupby(["game_id", "team"], sort=True):
        resampled = resample_group(group, frequency)
        if resampled.empty:
            continue
        groups.append(
            {
                "game_id": str(game_id),
                "team": str(team),
                "d_espn": resampled["d_espn"].to_numpy(dtype=float),
                "d_market": resampled["d_market"].to_numpy(dtype=float),
                "resampled_frame": resampled,
                "n_obs": int(len(resampled)),
            }
        )
    return groups


def build_resampled_diagnostics(groups: list[dict[str, Any]]) -> dict[str, float | int]:
    if not groups:
        return {
            "num_groups": 0,
            "num_games": 0,
            "num_observations": 0,
            "share_zero_d_espn": np.nan,
            "share_zero_d_market": np.nan,
        }
    all_d_espn = np.concatenate([group["d_espn"] for group in groups])
    all_d_market = np.concatenate([group["d_market"] for group in groups])
    return {
        "num_groups": len(groups),
        "num_games": len({group["game_id"] for group in groups}),
        "num_observations": int(sum(group["n_obs"] for group in groups)),
        "share_zero_d_espn": float(np.mean(all_d_espn == 0)),
        "share_zero_d_market": float(np.mean(all_d_market == 0)),
    }


def lagged_views(d_espn: np.ndarray, d_market: np.ndarray, lag: int) -> tuple[np.ndarray, np.ndarray]:
    if lag > 0:
        if lag >= len(d_espn):
            return np.array([], dtype=float), np.array([], dtype=float)
        return d_espn[:-lag], d_market[lag:]
    if lag < 0:
        offset = -lag
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


def build_lag_correlation_table(groups: list[dict[str, Any]], max_lag_seconds: int, frequency_seconds: int = 1) -> pd.DataFrame:
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


def fit_simple_regression(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    valid = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(valid) < 2:
        return np.nan, np.nan, len(valid)
    x_values = valid["x"].to_numpy(dtype=float)
    y_values = valid["y"].to_numpy(dtype=float)
    x_mean = x_values.mean()
    y_mean = y_values.mean()
    x_centered = x_values - x_mean
    y_centered = y_values - y_mean
    denom = float(np.dot(x_centered, x_centered))
    if denom == 0:
        return np.nan, np.nan, len(valid)
    beta = float(np.dot(x_centered, y_centered) / denom)
    intercept = y_mean - beta * x_mean
    fitted = intercept + beta * x_values
    ss_res = float(np.sum((y_values - fitted) ** 2))
    ss_tot = float(np.sum((y_values - y_mean) ** 2))
    r_squared = np.nan if ss_tot == 0 else 1 - (ss_res / ss_tot)
    return beta, r_squared, len(valid)
