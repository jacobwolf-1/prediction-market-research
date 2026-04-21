from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.ingestion_utils import DATA_DIR, RAW_DIR, ensure_directory, load_game_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Kalshi discovery audit results by reason and season.")
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=DATA_DIR / "processed" / "kalshi_discovery_audit.csv",
    )
    parser.add_argument(
        "--game-ids-file",
        type=Path,
        default=RAW_DIR / "game_ids.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR / "processed",
    )
    parser.add_argument(
        "--examples-per-reason",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=3,
    )
    return parser.parse_args()


def infer_nba_season_from_timestamp(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    season = timestamps.dt.year.astype("Float64")
    season = season.where(timestamps.dt.month >= 7, season - 1)
    return season.astype("Int64")


def prepare_audit_frame(audit_csv: Path, game_ids_file: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv, dtype={"matched_game_id": str, "event_ticker": str})
    if audit.empty:
        return audit

    audit["event_date"] = pd.to_datetime(audit["event_date"], utc=True, errors="coerce")
    audit["event_time"] = pd.to_datetime(audit["event_time"], utc=True, errors="coerce")

    game_ids = load_game_ids(game_ids_file)[["game_id", "season", "date"]].copy()
    game_ids = game_ids.rename(columns={"game_id": "matched_game_id", "season": "matched_season", "date": "matched_game_date"})
    game_ids["matched_game_id"] = game_ids["matched_game_id"].astype(str)
    game_ids["matched_game_date"] = pd.to_datetime(game_ids["matched_game_date"], utc=True, errors="coerce")

    audit = audit.merge(game_ids, on="matched_game_id", how="left")
    audit["inferred_season"] = infer_nba_season_from_timestamp(audit["event_time"].fillna(audit["event_date"]))
    audit["season_label"] = audit["matched_season"].astype("Int64").astype(str)
    missing_match_season = audit["matched_season"].isna()
    audit.loc[missing_match_season, "season_label"] = audit.loc[missing_match_season, "inferred_season"].astype(str)
    audit["season_label"] = audit["season_label"].replace("<NA>", "unknown")
    return audit


def prepare_game_frame(game_ids_file: Path) -> pd.DataFrame:
    games = load_game_ids(game_ids_file)[
        [
            "game_id",
            "date",
            "season",
            "home_team_abbr",
            "away_team_abbr",
            "home_team_display_name",
            "away_team_display_name",
        ]
    ].copy()
    games["game_id"] = games["game_id"].astype(str)
    games["date"] = pd.to_datetime(games["date"], utc=True, errors="coerce")
    return games


def summarize_reason(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["status", "reason", "events"])
    summary = (
        frame.groupby(["status", "reason"], dropna=False)
        .size()
        .reset_index(name="events")
        .sort_values(["events", "status", "reason"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    total = int(summary["events"].sum())
    summary["share"] = summary["events"] / total if total else 0.0
    return summary


def summarize_season_reason(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["season_label", "status", "reason", "events"])
    summary = (
        frame.groupby(["season_label", "status", "reason"], dropna=False)
        .size()
        .reset_index(name="events")
        .sort_values(["season_label", "events", "status", "reason"], ascending=[True, False, True, True])
        .reset_index(drop=True)
    )
    return summary


def summarize_season_totals(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["season_label", "events_scanned", "matched_events", "queued_events", "match_rate", "queue_rate"])
    grouped = frame.groupby("season_label", dropna=False)
    summary = grouped.size().reset_index(name="events_scanned")
    summary["matched_events"] = grouped["status"].apply(lambda s: int(s.isin(["matched_existing", "queued"]).sum())).to_numpy()
    summary["queued_events"] = grouped["status"].apply(lambda s: int((s == "queued").sum())).to_numpy()
    summary["match_rate"] = summary["matched_events"] / summary["events_scanned"]
    summary["queue_rate"] = summary["queued_events"] / summary["events_scanned"]
    return summary.sort_values("season_label").reset_index(drop=True)


def build_reason_examples(frame: pd.DataFrame, examples_per_reason: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "status",
                "reason",
                "season_label",
                "event_ticker",
                "title",
                "team_1",
                "team_2",
                "event_date",
                "event_time",
                "matched_game_id",
                "match_distance_days",
            ]
        )

    ordered = frame.sort_values(["reason", "season_label", "event_time", "event_date", "event_ticker"]).copy()
    examples = (
        ordered.groupby(["status", "reason"], dropna=False, group_keys=False)
        .head(examples_per_reason)
        .reset_index(drop=True)
    )
    columns = [
        "status",
        "reason",
        "season_label",
        "event_ticker",
        "title",
        "team_1",
        "team_2",
        "event_date",
        "event_time",
        "matched_game_id",
        "match_distance_days",
    ]
    return examples[columns]


def build_nearest_candidates(audit: pd.DataFrame, games: pd.DataFrame, candidate_count: int) -> pd.DataFrame:
    candidate_rows: list[dict[str, object]] = []
    failure_rows = audit.loc[~audit["status"].isin(["matched_existing", "queued"])].copy()
    if failure_rows.empty or games.empty:
        return pd.DataFrame(
            columns=[
                "event_ticker",
                "status",
                "reason",
                "season_label",
                "team_1",
                "team_2",
                "candidate_rank",
                "candidate_game_id",
                "candidate_season",
                "candidate_game_date",
                "candidate_home_team_abbr",
                "candidate_away_team_abbr",
                "candidate_home_team_display_name",
                "candidate_away_team_display_name",
                "team_overlap_count",
                "exact_team_match",
                "date_distance_days",
            ]
        )

    for row in failure_rows.itertuples(index=False):
        reference_time = row.event_time if pd.notna(row.event_time) else row.event_date
        if pd.notna(reference_time):
            reference_day = pd.Timestamp(reference_time).normalize()
        else:
            reference_day = pd.NaT

        requested_teams = {team for team in [row.team_1, row.team_2] if isinstance(team, str) and team}
        candidate_pool = games.copy()
        if getattr(row, "season_label", "unknown") != "unknown":
            try:
                candidate_pool = candidate_pool.loc[candidate_pool["season"] == int(row.season_label)].copy()
            except ValueError:
                pass
        if candidate_pool.empty:
            candidate_pool = games.copy()

        if requested_teams:
            candidate_pool["team_overlap_count"] = candidate_pool.apply(
                lambda game_row: len(requested_teams.intersection({game_row["home_team_abbr"], game_row["away_team_abbr"]})),
                axis=1,
            )
        else:
            candidate_pool["team_overlap_count"] = 0
        candidate_pool["exact_team_match"] = candidate_pool["team_overlap_count"] == len(requested_teams)

        if pd.notna(reference_day):
            candidate_pool["date_distance_days"] = (
                candidate_pool["date"].dt.normalize().sub(reference_day).abs().dt.days
            )
        else:
            candidate_pool["date_distance_days"] = pd.NA

        candidate_pool = candidate_pool.sort_values(
            ["team_overlap_count", "exact_team_match", "date_distance_days", "date", "game_id"],
            ascending=[False, False, True, True, True],
            na_position="last",
        ).head(candidate_count)

        for candidate_rank, candidate in enumerate(candidate_pool.itertuples(index=False), start=1):
            candidate_rows.append(
                {
                    "event_ticker": row.event_ticker,
                    "status": row.status,
                    "reason": row.reason,
                    "season_label": row.season_label,
                    "team_1": row.team_1,
                    "team_2": row.team_2,
                    "candidate_rank": candidate_rank,
                    "candidate_game_id": candidate.game_id,
                    "candidate_season": candidate.season,
                    "candidate_game_date": candidate.date,
                    "candidate_home_team_abbr": candidate.home_team_abbr,
                    "candidate_away_team_abbr": candidate.away_team_abbr,
                    "candidate_home_team_display_name": candidate.home_team_display_name,
                    "candidate_away_team_display_name": candidate.away_team_display_name,
                    "team_overlap_count": int(candidate.team_overlap_count),
                    "exact_team_match": bool(candidate.exact_team_match),
                    "date_distance_days": candidate.date_distance_days,
                }
            )

    return pd.DataFrame(candidate_rows)


def print_headline_summary(frame: pd.DataFrame) -> None:
    total = len(frame)
    matched = int(frame["status"].isin(["matched_existing", "queued"]).sum()) if not frame.empty else 0
    queued = int((frame["status"] == "queued").sum()) if not frame.empty else 0
    unmatched = int((frame["status"] == "unmatched").sum()) if not frame.empty else 0
    skipped = int((frame["status"] == "skipped").sum()) if not frame.empty else 0

    print(f"events_scanned: {total}")
    print(f"matched_events: {matched}")
    print(f"queued_events: {queued}")
    print(f"unmatched_events: {unmatched}")
    print(f"skipped_events: {skipped}")
    if total:
        print(f"match_rate: {matched / total:.4f}")
        print(f"queue_rate: {queued / total:.4f}")
    print("")


def main() -> None:
    args = parse_args()
    audit = prepare_audit_frame(args.audit_csv, args.game_ids_file)
    games = prepare_game_frame(args.game_ids_file)
    reason_summary = summarize_reason(audit)
    season_reason_summary = summarize_season_reason(audit)
    season_totals = summarize_season_totals(audit)
    reason_examples = build_reason_examples(audit, args.examples_per_reason)
    nearest_candidates = build_nearest_candidates(audit, games, args.candidate_count)

    ensure_directory(args.output_dir)
    reason_path = args.output_dir / "kalshi_audit_reason_summary.csv"
    season_reason_path = args.output_dir / "kalshi_audit_season_reason_summary.csv"
    season_totals_path = args.output_dir / "kalshi_audit_season_totals.csv"
    reason_examples_path = args.output_dir / "kalshi_audit_reason_examples.csv"
    nearest_candidates_path = args.output_dir / "kalshi_audit_nearest_candidates.csv"
    reason_summary.to_csv(reason_path, index=False)
    season_reason_summary.to_csv(season_reason_path, index=False)
    season_totals.to_csv(season_totals_path, index=False)
    reason_examples.to_csv(reason_examples_path, index=False)
    nearest_candidates.to_csv(nearest_candidates_path, index=False)

    print_headline_summary(audit)
    if not reason_summary.empty:
        print("top_reason_counts")
        print(reason_summary.head(10).to_string(index=False))
        print("")
    if not season_totals.empty:
        print("season_totals")
        print(season_totals.to_string(index=False))
        print("")
    if not reason_examples.empty:
        print("example_rows")
        print(reason_examples.head(min(10, len(reason_examples))).to_string(index=False))
        print("")
    if not nearest_candidates.empty:
        print("nearest_candidates")
        print(nearest_candidates.head(min(10, len(nearest_candidates))).to_string(index=False))
        print("")
    print(f"reason summary saved to: {reason_path}")
    print(f"season-reason summary saved to: {season_reason_path}")
    print(f"season totals saved to: {season_totals_path}")
    print(f"reason examples saved to: {reason_examples_path}")
    print(f"nearest candidates saved to: {nearest_candidates_path}")


if __name__ == "__main__":
    main()
