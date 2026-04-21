from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from analysis import build_lead_lag_dataset
import data_collection.collect_multi_season_history as collect_multi_season_history
import pipeline.run_full_pipeline as run_full_pipeline
from pipeline.run_smoke_test import materialize_smoke_data, run_smoke_test
from pipeline.workflow import _select_processed_game_ids, build_lead_lag_outputs, build_merged_dataset_output
from utils.ingestion_utils import ensure_directory


def test_build_lead_lag_outputs_from_smoke_fixtures(tmp_path) -> None:
    espn_dir, market_dir = materialize_smoke_data(tmp_path)
    output_dir = tmp_path / "processed" / "lead_lag_dataset"

    stats = build_lead_lag_outputs(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_dir=output_dir,
        overwrite=True,
    )

    assert stats == {
        "games_considered": 1,
        "datasets_written": 1,
        "datasets_empty": 0,
        "games_failed": 0,
        "failed_game_ids": [],
    }

    frame = pd.read_parquet(output_dir / "123456789.parquet")
    assert list(frame.columns) == [
        "timestamp",
        "game_id",
        "team",
        "espn_probability",
        "market_probability",
        "delta_espn",
        "delta_market",
        "delta_espn_lag_1",
        "delta_espn_lag_5",
        "delta_espn_lag_10",
    ]
    assert frame["game_id"].astype(str).unique().tolist() == ["123456789"]
    assert set(frame["team"]) == {"ATL", "NYK"}


def test_build_merged_dataset_output_from_smoke_fixtures(tmp_path) -> None:
    espn_dir, market_dir = materialize_smoke_data(tmp_path)
    output_path = tmp_path / "processed" / "merged_games.parquet"

    stats = build_merged_dataset_output(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_path=output_path,
        freq="1s",
    )

    assert stats == {"rows": 12, "games": 1, "groups": 2}

    merged = pd.read_parquet(output_path)
    assert list(merged.columns) == [
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
    assert merged["timestamp"].is_monotonic_increasing is False
    assert set(merged["team"]) == {"ATL", "NYK"}


def _write_test_game(
    *,
    espn_dir: Path,
    market_dir: Path,
    game_id: str,
    teams: tuple[str, str] = ("ATL", "NYK"),
) -> None:
    ensure_directory(espn_dir)
    ensure_directory(market_dir)

    timestamps = pd.to_datetime(
        [
            "2024-01-01T00:00:00Z",
            "2024-01-01T00:00:01Z",
        ],
        utc=True,
    )
    espn = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[1], timestamps[0], timestamps[1]],
            "game_id": [game_id] * 4,
            "team": [teams[0], teams[0], teams[1], teams[1]],
            "espn_probability": [0.40, 0.45, 0.60, 0.55],
            "quarter": [4, 4, 4, 4],
            "game_clock_seconds_remaining": [120, 119, 120, 119],
            "home_score": [90, 92, 90, 92],
            "away_score": [88, 88, 88, 88],
        }
    )
    market = pd.DataFrame(
        {
            "timestamp": [timestamps[0], timestamps[1], timestamps[0], timestamps[1]],
            "game_id": [game_id] * 4,
            "team": [teams[0], teams[0], teams[1], teams[1]],
            "market_probability": [0.41, 0.46, 0.59, 0.54],
            "market_id": [f"{game_id}-1", f"{game_id}-1", f"{game_id}-2", f"{game_id}-2"],
            "event_id": [f"event-{game_id}"] * 4,
        }
    )

    espn.to_parquet(espn_dir / f"game_{game_id}_espn.parquet", index=False)
    market.to_parquet(market_dir / f"{game_id}.parquet", index=False)


def test_build_lead_lag_outputs_caps_after_filtering_existing_outputs(tmp_path) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    output_dir = tmp_path / "processed" / "lead_lag_dataset"

    for game_id in ("100", "200", "300"):
        _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id=game_id)
    ensure_directory(output_dir)
    pd.DataFrame({"existing": [1]}).to_parquet(output_dir / "100.parquet", index=False)

    stats = build_lead_lag_outputs(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_dir=output_dir,
        overwrite=False,
        max_games=1,
    )

    assert stats == {
        "games_considered": 1,
        "datasets_written": 1,
        "datasets_empty": 0,
        "games_failed": 0,
        "failed_game_ids": [],
    }
    assert pd.read_parquet(output_dir / "100.parquet").columns.tolist() == ["existing"]
    assert (output_dir / "200.parquet").exists()
    assert not (output_dir / "300.parquet").exists()


def test_build_lead_lag_outputs_skips_failed_games_and_reports_them(tmp_path, monkeypatch) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    output_dir = tmp_path / "processed" / "lead_lag_dataset"

    for game_id in ("100", "200"):
        _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id=game_id)

    real_build = build_lead_lag_dataset.build_game_dataset

    def flaky_build(game_id, espn_path, market_path):
        if game_id == "100":
            raise ValueError("broken file")
        return real_build(game_id, espn_path, market_path)

    monkeypatch.setattr(build_lead_lag_dataset, "build_game_dataset", flaky_build)

    stats = build_lead_lag_outputs(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_dir=output_dir,
        overwrite=True,
    )

    assert stats == {
        "games_considered": 2,
        "datasets_written": 1,
        "datasets_empty": 0,
        "games_failed": 1,
        "failed_game_ids": ["100"],
    }
    assert not (output_dir / "100.parquet").exists()
    assert (output_dir / "200.parquet").exists()


def test_build_lead_lag_outputs_overwrite_removes_stale_output_on_failure(tmp_path, monkeypatch) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    output_dir = tmp_path / "processed" / "lead_lag_dataset"

    _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id="100")
    ensure_directory(output_dir)
    pd.DataFrame({"stale": [1]}).to_parquet(output_dir / "100.parquet", index=False)

    def always_fail(game_id, espn_path, market_path):
        raise ValueError("broken file")

    monkeypatch.setattr(build_lead_lag_dataset, "build_game_dataset", always_fail)

    stats = build_lead_lag_outputs(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_dir=output_dir,
        overwrite=True,
    )

    assert stats == {
        "games_considered": 1,
        "datasets_written": 0,
        "datasets_empty": 0,
        "games_failed": 1,
        "failed_game_ids": ["100"],
    }
    assert not (output_dir / "100.parquet").exists()


def test_build_merged_dataset_output_respects_selected_game_subset(tmp_path) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    output_path = tmp_path / "processed" / "merged_games.parquet"

    for game_id in ("100", "200", "300"):
        _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id=game_id)

    stats = build_merged_dataset_output(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_path=output_path,
        freq="1s",
        game_ids=["200"],
    )

    assert stats == {"rows": 4, "games": 1, "groups": 2}

    merged = pd.read_parquet(output_path)
    assert merged["game_id"].astype(str).unique().tolist() == ["200"]


def test_run_full_pipeline_keeps_merged_dataset_cumulative_on_incremental_rerun(tmp_path, monkeypatch) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    lead_lag_dir = tmp_path / "processed" / "lead_lag_dataset"
    merged_output_path = tmp_path / "processed" / "merged_games.parquet"
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"

    for game_id in ("100", "200", "300"):
        _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id=game_id)
    ensure_directory(lead_lag_dir)
    pd.DataFrame({"existing": [1]}).to_parquet(lead_lag_dir / "100.parquet", index=False)

    monkeypatch.setattr(
        run_full_pipeline,
        "parse_args",
        lambda: Namespace(
            seasons=[2024],
            game_ids_output=game_ids_output,
            espn_output_dir=espn_dir,
            kalshi_output_dir=market_dir,
            lead_lag_output_dir=lead_lag_dir,
            merged_output_path=merged_output_path,
            kalshi_audit_output=kalshi_audit_output,
            freq="1s",
            max_games=2,
            max_pages=None,
            overwrite=False,
            skip_existing=True,
        ),
    )
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_game_ids",
        lambda seasons, output_path, overwrite: pd.DataFrame({"game_id": ["100", "200", "300"]}),
    )
    monkeypatch.setattr(run_full_pipeline, "collect_espn_history", lambda game_ids, output_dir, overwrite: 0)
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_kalshi_history",
        lambda **kwargs: {"games_downloaded": 0, "audit_path": str(kalshi_audit_output), "audit_refreshed": False},
    )

    run_full_pipeline.main()

    merged = pd.read_parquet(merged_output_path)
    assert sorted(merged["game_id"].astype(str).unique().tolist()) == ["100", "200", "300"]
    assert pd.read_parquet(lead_lag_dir / "100.parquet").columns.tolist() == ["existing"]
    assert (lead_lag_dir / "200.parquet").exists()
    assert (lead_lag_dir / "300.parquet").exists()


def test_run_full_pipeline_skip_existing_max_games_advances_two_new_games(tmp_path, monkeypatch) -> None:
    lead_lag_dir = tmp_path / "processed" / "lead_lag_dataset"
    merged_output_path = tmp_path / "processed" / "merged_games.parquet"
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"
    ensure_directory(lead_lag_dir)
    pd.DataFrame({"existing": [1]}).to_parquet(lead_lag_dir / "100.parquet", index=False)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        run_full_pipeline,
        "parse_args",
        lambda: Namespace(
            seasons=[2024],
            game_ids_output=game_ids_output,
            espn_output_dir=tmp_path / "raw" / "espn",
            kalshi_output_dir=tmp_path / "raw" / "kalshi",
            lead_lag_output_dir=lead_lag_dir,
            merged_output_path=merged_output_path,
            kalshi_audit_output=kalshi_audit_output,
            freq="1s",
            max_games=2,
            max_pages=None,
            overwrite=False,
            skip_existing=True,
        ),
    )
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_game_ids",
        lambda seasons, output_path, overwrite: pd.DataFrame({"game_id": ["100", "200", "300"]}),
    )

    def fake_collect_espn_history(game_ids, output_dir, overwrite):
        captured["espn_game_ids"] = list(game_ids)
        return 0

    def fake_collect_kalshi_history(**kwargs):
        captured["kalshi_selected_game_ids"] = list(kwargs["selected_game_ids"])
        return {
            "games_downloaded": 0,
            "audit_path": str(kalshi_audit_output),
            "audit_refreshed": False,
            "audit_refreshed": False,
        }

    def fake_build_lead_lag_outputs(**kwargs):
        captured["lead_lag_candidate_game_ids"] = list(kwargs["candidate_game_ids"])
        ensure_directory(lead_lag_dir)
        pd.DataFrame({"built": [1]}).to_parquet(lead_lag_dir / "200.parquet", index=False)
        pd.DataFrame({"built": [1]}).to_parquet(lead_lag_dir / "300.parquet", index=False)
        return {
            "games_considered": 2,
            "datasets_written": 2,
            "datasets_empty": 0,
            "games_failed": 0,
            "failed_game_ids": [],
        }

    monkeypatch.setattr(run_full_pipeline, "collect_espn_history", fake_collect_espn_history)
    monkeypatch.setattr(run_full_pipeline, "collect_kalshi_history", fake_collect_kalshi_history)
    monkeypatch.setattr(run_full_pipeline, "build_lead_lag_outputs", fake_build_lead_lag_outputs)
    monkeypatch.setattr(
        run_full_pipeline,
        "build_merged_dataset_output",
        lambda **kwargs: {"rows": 0, "games": 0, "groups": 0},
    )

    run_full_pipeline.main()

    assert captured["espn_game_ids"] == ["200", "300"]
    assert captured["kalshi_selected_game_ids"] == ["200", "300"]
    assert captured["lead_lag_candidate_game_ids"] == ["200", "300"]


def test_run_full_pipeline_incremental_rerun_with_no_new_games_preserves_merged_dataset(tmp_path, monkeypatch) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    lead_lag_dir = tmp_path / "processed" / "lead_lag_dataset"
    merged_output_path = tmp_path / "processed" / "merged_games.parquet"
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"

    for game_id in ("100", "200"):
        _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id=game_id)
    ensure_directory(lead_lag_dir)
    pd.DataFrame({"existing": [1]}).to_parquet(lead_lag_dir / "100.parquet", index=False)
    pd.DataFrame({"existing": [1]}).to_parquet(lead_lag_dir / "200.parquet", index=False)
    build_merged_dataset_output(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_path=merged_output_path,
        freq="1s",
        game_ids=["100", "200"],
    )

    monkeypatch.setattr(
        run_full_pipeline,
        "parse_args",
        lambda: Namespace(
            seasons=[2024],
            game_ids_output=game_ids_output,
            espn_output_dir=espn_dir,
            kalshi_output_dir=market_dir,
            lead_lag_output_dir=lead_lag_dir,
            merged_output_path=merged_output_path,
            kalshi_audit_output=kalshi_audit_output,
            freq="1s",
            max_games=2,
            max_pages=None,
            overwrite=False,
            skip_existing=True,
        ),
    )
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_game_ids",
        lambda seasons, output_path, overwrite: pd.DataFrame({"game_id": ["100", "200"]}),
    )
    monkeypatch.setattr(run_full_pipeline, "collect_espn_history", lambda game_ids, output_dir, overwrite: 0)
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_kalshi_history",
        lambda **kwargs: {"games_downloaded": 0, "audit_path": str(kalshi_audit_output), "audit_refreshed": False},
    )

    run_full_pipeline.main()

    merged = pd.read_parquet(merged_output_path)
    assert sorted(merged["game_id"].astype(str).unique().tolist()) == ["100", "200"]


def test_run_full_pipeline_overwrite_honors_max_games_despite_old_extra_outputs(tmp_path, monkeypatch) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    lead_lag_dir = tmp_path / "processed" / "lead_lag_dataset"
    merged_output_path = tmp_path / "processed" / "merged_games.parquet"
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"

    for game_id in ("100", "200", "300"):
        _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id=game_id)
    ensure_directory(lead_lag_dir)
    pd.DataFrame({"stale": [1]}).to_parquet(lead_lag_dir / "300.parquet", index=False)

    monkeypatch.setattr(
        run_full_pipeline,
        "parse_args",
        lambda: Namespace(
            seasons=[2024],
            game_ids_output=game_ids_output,
            espn_output_dir=espn_dir,
            kalshi_output_dir=market_dir,
            lead_lag_output_dir=lead_lag_dir,
            merged_output_path=merged_output_path,
            kalshi_audit_output=kalshi_audit_output,
            freq="1s",
            max_games=2,
            max_pages=None,
            overwrite=True,
            skip_existing=False,
        ),
    )
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_game_ids",
        lambda seasons, output_path, overwrite: pd.DataFrame({"game_id": ["100", "200", "300"]}),
    )
    monkeypatch.setattr(run_full_pipeline, "collect_espn_history", lambda game_ids, output_dir, overwrite: 0)
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_kalshi_history",
        lambda **kwargs: {"games_downloaded": 0, "audit_path": str(kalshi_audit_output), "audit_refreshed": False},
    )

    run_full_pipeline.main()

    merged = pd.read_parquet(merged_output_path)
    assert sorted(merged["game_id"].astype(str).unique().tolist()) == ["100", "200"]
    assert _select_processed_game_ids(output_dir=lead_lag_dir, candidate_game_ids=["100", "200"]) == ["100", "200"]
    assert not (lead_lag_dir / "300.parquet").exists()


def test_run_full_pipeline_overwrite_narrower_scope_removes_stale_prior_run_games(tmp_path, monkeypatch) -> None:
    espn_dir = tmp_path / "raw" / "espn"
    market_dir = tmp_path / "raw" / "kalshi"
    lead_lag_dir = tmp_path / "processed" / "lead_lag_dataset"
    merged_output_path = tmp_path / "processed" / "merged_games.parquet"
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"

    for game_id in ("100", "200", "400"):
        _write_test_game(espn_dir=espn_dir, market_dir=market_dir, game_id=game_id)
    ensure_directory(lead_lag_dir)
    pd.DataFrame({"stale": [1]}).to_parquet(lead_lag_dir / "400.parquet", index=False)

    monkeypatch.setattr(
        run_full_pipeline,
        "parse_args",
        lambda: Namespace(
            seasons=[2024],
            game_ids_output=game_ids_output,
            espn_output_dir=espn_dir,
            kalshi_output_dir=market_dir,
            lead_lag_output_dir=lead_lag_dir,
            merged_output_path=merged_output_path,
            kalshi_audit_output=kalshi_audit_output,
            freq="1s",
            max_games=None,
            max_pages=None,
            overwrite=True,
            skip_existing=False,
        ),
    )
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_game_ids",
        lambda seasons, output_path, overwrite: pd.DataFrame({"game_id": ["100", "200"]}),
    )
    monkeypatch.setattr(run_full_pipeline, "collect_espn_history", lambda game_ids, output_dir, overwrite: 0)
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_kalshi_history",
        lambda **kwargs: {"games_downloaded": 0, "audit_path": str(kalshi_audit_output), "audit_refreshed": False},
    )

    run_full_pipeline.main()

    merged = pd.read_parquet(merged_output_path)
    assert sorted(merged["game_id"].astype(str).unique().tolist()) == ["100", "200"]
    assert not (lead_lag_dir / "400.parquet").exists()


def test_run_full_pipeline_uses_one_shared_capped_game_set_for_collection(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}
    lead_lag_dir = tmp_path / "processed" / "lead_lag_dataset"
    merged_output_path = tmp_path / "processed" / "merged_games.parquet"
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"

    monkeypatch.setattr(
        run_full_pipeline,
        "parse_args",
        lambda: Namespace(
            seasons=[2024],
            game_ids_output=game_ids_output,
            espn_output_dir=tmp_path / "raw" / "espn",
            kalshi_output_dir=tmp_path / "raw" / "kalshi",
            lead_lag_output_dir=lead_lag_dir,
            merged_output_path=merged_output_path,
            kalshi_audit_output=kalshi_audit_output,
            freq="1s",
            max_games=2,
            max_pages=None,
            overwrite=False,
            skip_existing=False,
        ),
    )
    monkeypatch.setattr(
        run_full_pipeline,
        "collect_game_ids",
        lambda seasons, output_path, overwrite: pd.DataFrame({"game_id": ["100", "200", "300"]}),
    )

    def fake_collect_espn_history(game_ids, output_dir, overwrite):
        captured["espn_game_ids"] = list(game_ids)
        return 0

    def fake_collect_kalshi_history(**kwargs):
        captured["kalshi_selected_game_ids"] = list(kwargs["selected_game_ids"])
        ensure_directory(lead_lag_dir)
        return {"games_downloaded": 0, "audit_path": str(kalshi_audit_output), "audit_refreshed": False}

    monkeypatch.setattr(run_full_pipeline, "collect_espn_history", fake_collect_espn_history)
    monkeypatch.setattr(run_full_pipeline, "collect_kalshi_history", fake_collect_kalshi_history)
    monkeypatch.setattr(
        run_full_pipeline,
        "build_lead_lag_outputs",
        lambda **kwargs: {
            "games_considered": 0,
            "datasets_written": 0,
            "datasets_empty": 0,
            "games_failed": 0,
            "failed_game_ids": [],
        },
    )
    monkeypatch.setattr(
        run_full_pipeline,
        "build_merged_dataset_output",
        lambda **kwargs: {"rows": 0, "games": 0, "groups": 0},
    )

    run_full_pipeline.main()

    assert captured["espn_game_ids"] == ["100", "200"]
    assert captured["kalshi_selected_game_ids"] == ["100", "200"]


def test_collect_multi_season_history_skip_existing_applies_max_games_after_filter(tmp_path, monkeypatch) -> None:
    kalshi_output_dir = tmp_path / "raw" / "kalshi"
    espn_output_dir = tmp_path / "raw" / "espn"
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"
    ensure_directory(kalshi_output_dir)
    pd.DataFrame({"stale": [1]}).to_parquet(kalshi_output_dir / "100.parquet", index=False)
    pd.DataFrame({"stale": [1]}).to_parquet(kalshi_output_dir / "200.parquet", index=False)

    captured: dict[str, object] = {}

    monkeypatch.setattr(
        collect_multi_season_history,
        "parse_args",
        lambda: Namespace(
            seasons=[2024],
            game_ids_output=game_ids_output,
            espn_output_dir=espn_output_dir,
            kalshi_output_dir=kalshi_output_dir,
            max_games=2,
            max_pages=None,
            overwrite=False,
            skip_existing=True,
            kalshi_audit_output=kalshi_audit_output,
        ),
    )
    monkeypatch.setattr(
        collect_multi_season_history,
        "collect_game_ids",
        lambda seasons, output_path, overwrite: pd.DataFrame({"game_id": ["100", "200", "300", "400"]}),
    )
    monkeypatch.setattr(collect_multi_season_history, "collect_espn_history", lambda game_ids, output_dir, overwrite: 0)

    def fake_collect_kalshi_history(**kwargs):
        captured["selected_game_ids"] = list(kwargs["selected_game_ids"])
        return {
            "pages_scanned": 0,
            "events_considered": 0,
            "games_discovered": 0,
            "games_matched": 0,
            "games_downloaded": 0,
            "rows_collected": 0,
            "audit_path": str(kalshi_audit_output),
            "audit_refreshed": False,
        }

    monkeypatch.setattr(collect_multi_season_history, "collect_kalshi_history", fake_collect_kalshi_history)

    collect_multi_season_history.main()

    assert captured["selected_game_ids"] == ["300", "400"]


def test_collect_kalshi_history_empty_selected_game_ids_is_true_no_op(tmp_path, monkeypatch) -> None:
    game_ids_output = tmp_path / "raw" / "game_ids.csv"
    kalshi_output_dir = tmp_path / "raw" / "kalshi"
    espn_output_dir = tmp_path / "raw" / "espn"
    audit_output = tmp_path / "processed" / "kalshi_discovery_audit.csv"
    ensure_directory(game_ids_output.parent)
    ensure_directory(kalshi_output_dir)
    ensure_directory(espn_output_dir)
    ensure_directory(audit_output.parent)
    pd.DataFrame(
        {
            "game_id": ["100", "200"],
            "date": ["2024-01-01", "2024-01-02"],
            "home_team_abbr": ["ATL", "ATL"],
            "away_team_abbr": ["NYK", "NYK"],
        }
    ).to_csv(game_ids_output, index=False)
    audit_output.write_text("keep-me\n", encoding="utf-8")

    def fail_discovery(*args, **kwargs):
        raise AssertionError("discover_games should not be called")

    monkeypatch.setattr(collect_multi_season_history.kalshi_fetch, "discover_games", fail_discovery)

    stats = collect_multi_season_history.collect_kalshi_history(
        game_ids_file=game_ids_output,
        kalshi_output_dir=kalshi_output_dir,
        espn_output_dir=espn_output_dir,
        audit_output=audit_output,
        selected_game_ids=[],
        max_games=10,
        max_pages=None,
        skip_existing=True,
        overwrite=False,
    )

    assert stats == {
        "pages_scanned": 0,
        "events_considered": 0,
        "games_discovered": 0,
        "games_matched": 0,
        "games_downloaded": 0,
        "rows_collected": 0,
        "audit_path": str(audit_output),
        "audit_refreshed": False,
    }
    assert audit_output.read_text(encoding="utf-8") == "keep-me\n"


def test_public_smoke_test_install_path_includes_parquet_backend() -> None:
    requirements = Path("requirements.txt").read_text(encoding="utf-8")
    ci_workflow = Path(".github/workflows/ci.yml").read_text(encoding="utf-8")

    assert "pyarrow==" in requirements
    assert "pip install -r requirements.txt" in ci_workflow
    assert "python pipeline/run_smoke_test.py" in ci_workflow


def test_run_smoke_test_writes_summary_and_outputs(tmp_path) -> None:
    output_dir = tmp_path / "smoke"

    summary = run_smoke_test(output_dir, freq="1s")

    assert summary["fixture_type"] == "synthetic_smoke_test"
    assert (output_dir / "processed" / "lead_lag_dataset" / "123456789.parquet").exists()
    assert (output_dir / "processed" / "merged_games.parquet").exists()

    summary_path = output_dir / "summary.json"
    saved = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved["lead_lag"]["datasets_written"] == 1
    assert saved["lead_lag"]["games_failed"] == 0
    assert saved["merged_dataset"]["groups"] == 2
