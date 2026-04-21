# Work Summary

Date: 2026-04-20

## Goal Of This Round

Focus on unblocking the repo despite incomplete historical data by:

1. making Kalshi discovery failures inspectable
2. adding a validation path that does not depend on missing early-season data
3. improving the Kalshi matcher enough to recover obvious near-matches
4. rerunning the targeted Kalshi pipeline and summarizing the results

## Code Changes Made

### Documentation

- Added [README.md](/Users/jacobwolf/prediction-market-research/README.md:1)
- Updated the README to document:
  - the data pipeline
  - the new Kalshi audit flow
  - the new cross-validation script
  - the new audit-summary script

### New Analysis Scripts

- Added [analysis/shock_strategy_cv.py](/Users/jacobwolf/prediction-market-research/analysis/shock_strategy_cv.py:1)
  - leave-one-game-out cross-validation for the shock strategy
  - works on the current matched sample without requiring 2021-2023 Kalshi coverage

- Added [analysis/kalshi_discovery_audit_summary.py](/Users/jacobwolf/prediction-market-research/analysis/kalshi_discovery_audit_summary.py:1)
  - summarizes the Kalshi discovery audit by reason and season
  - writes example rows per rejection reason
  - writes nearest plausible ESPN game candidates for unmatched rows

### Kalshi Collection / Matching

- Updated [data_collection/kalshi_fetch.py](/Users/jacobwolf/prediction-market-research/data_collection/kalshi_fetch.py:1)
  - writes `data/processed/kalshi_discovery_audit.csv`
  - records discovery status and rejection reason for each scanned Kalshi event
  - records exact-match vs fallback-match outcome in audit reasons
  - matcher is now scored rather than exact-set-only:
    - exact team-set match still preferred
    - alias-based fallback match allowed when overlap is strong and date alignment is close

- Updated [data_collection/collect_multi_season_history.py](/Users/jacobwolf/prediction-market-research/data_collection/collect_multi_season_history.py:1)
  - preserves the Kalshi audit output when using the multi-season wrapper

### Shared Utility Fix

- Updated [utils/ingestion_utils.py](/Users/jacobwolf/prediction-market-research/utils/ingestion_utils.py:143)
  - `normalize_text()` now safely handles `NaN` / non-string inputs
  - this fixed a real runtime failure triggered by the smarter Kalshi matcher

## Commands Run

Key runs during this round:

```bash
python3 -m py_compile data_collection/kalshi_fetch.py data_collection/collect_multi_season_history.py analysis/shock_strategy_cv.py analysis/kalshi_discovery_audit_summary.py utils/ingestion_utils.py
venv/bin/python data_collection/collect_multi_season_history.py --seasons 2023 2024 2025 --skip-existing
venv/bin/python data_collection/kalshi_fetch.py --game-ids-file data/raw/game_ids.csv --skip-existing
venv/bin/python analysis/kalshi_discovery_audit_summary.py --examples-per-reason 5 --candidate-count 3
```

Notes:

- The full multi-season wrapper was too slow for this debugging loop because it spent substantial time backfilling ESPN files.
- The targeted `kalshi_fetch.py` rerun was the useful path for testing the new matcher and refreshing the audit.

## Generated Outputs

### Audit Outputs

- [kalshi_discovery_audit.csv](/Users/jacobwolf/prediction-market-research/data/processed/kalshi_discovery_audit.csv:1)
- [kalshi_audit_reason_summary.csv](/Users/jacobwolf/prediction-market-research/data/processed/kalshi_audit_reason_summary.csv:1)
- [kalshi_audit_season_reason_summary.csv](/Users/jacobwolf/prediction-market-research/data/processed/kalshi_audit_season_reason_summary.csv:1)
- [kalshi_audit_season_totals.csv](/Users/jacobwolf/prediction-market-research/data/processed/kalshi_audit_season_totals.csv:1)
- [kalshi_audit_reason_examples.csv](/Users/jacobwolf/prediction-market-research/data/processed/kalshi_audit_reason_examples.csv:1)
- [kalshi_audit_nearest_candidates.csv](/Users/jacobwolf/prediction-market-research/data/processed/kalshi_audit_nearest_candidates.csv:1)

### Data State At Last Check

- `data/raw/game_ids.csv` updated during this round
- `data/raw/kalshi/*.parquet` count increased from `68` to `81`

## Main Findings

From the refreshed audit summary:

- `events_scanned: 1378`
- `matched_events: 370`
- `queued_events: 370`
- `unmatched_events: 1008`
- `match_rate: 0.2685`

Reason breakdown:

- `could_not_extract_two_teams`: `951`
- `no_exact_team_and_date_match`: `57`
- `exact_team_match_queued_for_download`: `370`

Interpretation:

- The matcher is no longer the main bottleneck.
- The dominant failure mode is still team extraction from Kalshi event structures and titles.
- The audit indicates most successful matches are in the 2026 portion of the scanned coverage.
- The small `no_exact_team_and_date_match` bucket suggests matching logic is now secondary relative to upstream event parsing / coverage.

## Important Runtime Observation

While rerunning the targeted Kalshi fetch, the first version of the smarter matcher failed on this traceback source:

- `normalize_text()` assumed string-like values
- some game metadata fields were `NaN` floats
- fixing that in `utils/ingestion_utils.py` resolved the runtime issue

## Current Open Question

The next likely high-value code change is:

- improve `extract_event_teams()` in [data_collection/kalshi_fetch.py](/Users/jacobwolf/prediction-market-research/data_collection/kalshi_fetch.py:175)

Reason:

- the biggest unresolved bucket is `could_not_extract_two_teams`
- the audit examples show play-in / playoff title formats that the current extraction logic does not parse well

This was intentionally not implemented yet because the next user prompt may redirect the work.

## Current Uncommitted Files

At the last check, these files had local changes:

- [README.md](/Users/jacobwolf/prediction-market-research/README.md:1)
- [data/raw/game_ids.csv](/Users/jacobwolf/prediction-market-research/data/raw/game_ids.csv:1)
- [data_collection/collect_multi_season_history.py](/Users/jacobwolf/prediction-market-research/data_collection/collect_multi_season_history.py:1)
- [data_collection/kalshi_fetch.py](/Users/jacobwolf/prediction-market-research/data_collection/kalshi_fetch.py:1)
- [utils/ingestion_utils.py](/Users/jacobwolf/prediction-market-research/utils/ingestion_utils.py:1)
- [analysis/kalshi_discovery_audit_summary.py](/Users/jacobwolf/prediction-market-research/analysis/kalshi_discovery_audit_summary.py:1)
- [analysis/shock_strategy_cv.py](/Users/jacobwolf/prediction-market-research/analysis/shock_strategy_cv.py:1)

## Short Hand-Off

If continuing from this checkpoint, the repo now has:

- a README
- a Kalshi discovery audit
- season/reason/example/candidate summaries for that audit
- a smarter Kalshi matcher
- a game-level CV script for the shock strategy

The most useful next task is probably to use the audit examples to improve Kalshi team extraction, unless the next prompt changes direction.
