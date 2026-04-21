from __future__ import annotations

import pandas as pd
import pytest

from analysis.raw_time_dataset import dataset_summary, load_raw_time_dataset
from analysis.raw_time_utils import lagged_views, resample_series


def test_resample_series_deduplicates_and_forward_fills() -> None:
    frame = pd.DataFrame(
        {
            "ts": [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:02Z",
            ],
            "probability": [0.20, 0.25, 0.40],
        }
    )

    resampled = resample_series(frame, "ts", "probability", freq="1s")

    assert resampled["timestamp"].tolist() == [
        pd.Timestamp("2026-01-01 00:00:01+00:00"),
        pd.Timestamp("2026-01-01 00:00:02+00:00"),
    ]
    assert resampled["value"].tolist() == [0.25, 0.40]
    assert resampled["delta"].tolist() == pytest.approx([0.0, 0.15])


def test_load_raw_time_dataset_keeps_only_shared_game_team_groups(tmp_path) -> None:
    espn_dir = tmp_path / "espn"
    market_dir = tmp_path / "kalshi"
    espn_dir.mkdir()
    market_dir.mkdir()

    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "game_id": "game-1",
                "team": "ATL",
                "espn_probability": 0.45,
            },
            {
                "timestamp": "2026-01-01T00:00:02Z",
                "game_id": "game-1",
                "team": "ATL",
                "espn_probability": 0.47,
            },
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "game_id": "game-1",
                "team": "NYK",
                "espn_probability": 0.55,
            },
        ]
    ).to_parquet(espn_dir / "game_game-1_espn.parquet", index=False)

    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:01Z",
                "game_id": "game-1",
                "team": "ATL",
                "market_probability": 0.44,
                "market_id": "m1",
                "event_id": "e1",
            },
            {
                "timestamp": "2026-01-01T00:00:03Z",
                "game_id": "game-1",
                "team": "ATL",
                "market_probability": 0.48,
                "market_id": "m1",
                "event_id": "e1",
            },
        ]
    ).to_parquet(market_dir / "game-1.parquet", index=False)

    dataset = load_raw_time_dataset(espn_dir=espn_dir, market_dir=market_dir)

    assert list(dataset) == [("game-1", "ATL")]
    assert dataset_summary(dataset) == {
        "games": 1,
        "groups": 1,
        "espn_rows": 2,
        "market_rows": 2,
    }


def test_lagged_views_aligns_positive_and_negative_lags() -> None:
    d_espn = pd.Series([1.0, 2.0, 3.0]).to_numpy()
    d_market = pd.Series([10.0, 20.0, 30.0]).to_numpy()

    x_pos, y_pos = lagged_views(d_espn, d_market, 1)
    x_neg, y_neg = lagged_views(d_espn, d_market, -1)

    assert x_pos.tolist() == [1.0, 2.0]
    assert y_pos.tolist() == [20.0, 30.0]
    assert x_neg.tolist() == [2.0, 3.0]
    assert y_neg.tolist() == [10.0, 20.0]
