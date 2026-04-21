from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from utils.ingestion_utils import DATA_DIR, RAW_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print a compact Kalshi coverage summary with before/after deltas.")
    parser.add_argument("--audit-csv", type=Path, default=DATA_DIR / "processed" / "kalshi_discovery_audit.csv")
    parser.add_argument("--game-ids-file", type=Path, default=RAW_DIR / "game_ids.csv")
    parser.add_argument("--kalshi-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--baseline-events-scanned", type=int, default=1378)
    parser.add_argument("--baseline-matched-events", type=int, default=370)
    parser.add_argument("--baseline-unmatched-events", type=int, default=1008)
    parser.add_argument("--baseline-match-rate", type=float, default=0.2685)
    parser.add_argument("--baseline-could-not-extract", type=int, default=951)
    parser.add_argument("--baseline-no-exact-match", type=int, default=57)
    parser.add_argument("--baseline-kalshi-files", type=int, default=81)
    return parser.parse_args()


def delta_text(current: float, baseline: float) -> str:
    delta = current - baseline
    sign = "+" if delta >= 0 else ""
    if isinstance(current, int) and isinstance(baseline, int):
        return f"{sign}{int(delta)}"
    return f"{sign}{delta:.4f}"


def main() -> None:
    args = parse_args()
    audit = pd.read_csv(args.audit_csv)
    kalshi_files = len(list(args.kalshi_dir.glob("*.parquet")))

    events_scanned = int(len(audit))
    matched_events = int(audit["status"].isin(["matched_existing", "queued"]).sum())
    unmatched_events = int((audit["status"] == "unmatched").sum())
    match_rate = float(matched_events / events_scanned) if events_scanned else float("nan")
    could_not_extract = int((audit["reason"] == "could_not_extract_two_teams").sum())
    no_exact_match = int((audit["reason"] == "no_exact_team_and_date_match").sum())

    game_ids = pd.read_csv(args.game_ids_file, dtype={"game_id": str})[["game_id", "season"]]
    matched_ids = {path.stem for path in args.kalshi_dir.glob("*.parquet")}
    matched_seasons = game_ids.loc[game_ids["game_id"].isin(matched_ids), "season"].value_counts().sort_index()

    print("current")
    print(f"events_scanned: {events_scanned}")
    print(f"matched_events: {matched_events}")
    print(f"unmatched_events: {unmatched_events}")
    print(f"match_rate: {match_rate:.4f}")
    print(f"could_not_extract_two_teams: {could_not_extract}")
    print(f"no_exact_team_and_date_match: {no_exact_match}")
    print(f"kalshi_parquet_files: {kalshi_files}")
    print("")
    print("delta_vs_baseline")
    print(f"events_scanned: {delta_text(events_scanned, args.baseline_events_scanned)}")
    print(f"matched_events: {delta_text(matched_events, args.baseline_matched_events)}")
    print(f"unmatched_events: {delta_text(unmatched_events, args.baseline_unmatched_events)}")
    print(f"match_rate: {delta_text(match_rate, args.baseline_match_rate)}")
    print(f"could_not_extract_two_teams: {delta_text(could_not_extract, args.baseline_could_not_extract)}")
    print(f"no_exact_team_and_date_match: {delta_text(no_exact_match, args.baseline_no_exact_match)}")
    print(f"kalshi_parquet_files: {delta_text(kalshi_files, args.baseline_kalshi_files)}")
    print("")
    print("matched_file_seasons")
    if matched_seasons.empty:
        print("none")
    else:
        print(matched_seasons.to_string())


if __name__ == "__main__":
    main()
