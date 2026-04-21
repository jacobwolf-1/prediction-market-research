from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from pipeline.workflow import build_lead_lag_outputs, build_merged_dataset_output
from utils.ingestion_utils import ensure_directory, write_parquet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMOKE_FIXTURE_DIR = PROJECT_ROOT / "sample_data" / "smoke" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a public smoke test on tiny synthetic ESPN/Kalshi fixtures.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "smoke_test",
        help="Directory for generated smoke-test artifacts.",
    )
    parser.add_argument("--freq", default="1s", help="Resampling frequency for the merged dataset.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Reuse an existing output directory instead of replacing it.",
    )
    return parser.parse_args()


def materialize_smoke_data(output_dir: Path) -> tuple[Path, Path]:
    raw_dir = output_dir / "raw"
    espn_dir = raw_dir / "espn"
    market_dir = raw_dir / "kalshi"
    ensure_directory(espn_dir)
    ensure_directory(market_dir)

    shutil.copy2(SMOKE_FIXTURE_DIR / "game_ids.csv", raw_dir / "game_ids.csv")

    for csv_path in sorted((SMOKE_FIXTURE_DIR / "espn").glob("*.csv")):
        frame = pd.read_csv(csv_path)
        write_parquet(frame, espn_dir / csv_path.name.replace(".csv", ".parquet"))

    for csv_path in sorted((SMOKE_FIXTURE_DIR / "kalshi").glob("*.csv")):
        frame = pd.read_csv(csv_path)
        write_parquet(frame, market_dir / csv_path.name.replace(".csv", ".parquet"))

    return espn_dir, market_dir


def run_smoke_test(output_dir: Path, *, freq: str = "1s", keep_existing: bool = False) -> dict[str, object]:
    if output_dir.exists() and not keep_existing:
        shutil.rmtree(output_dir)
    ensure_directory(output_dir)

    espn_dir, market_dir = materialize_smoke_data(output_dir)
    processed_dir = output_dir / "processed"
    lead_lag_dir = processed_dir / "lead_lag_dataset"
    merged_path = processed_dir / "merged_games.parquet"

    lead_lag_stats = build_lead_lag_outputs(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_dir=lead_lag_dir,
        overwrite=True,
    )
    merged_stats = build_merged_dataset_output(
        espn_dir=espn_dir,
        market_dir=market_dir,
        output_path=merged_path,
        freq=freq,
    )

    summary = {
        "fixture_type": "synthetic_smoke_test",
        "freq": freq,
        "lead_lag": lead_lag_stats,
        "merged_dataset": merged_stats,
        "raw_dir": str(output_dir / "raw"),
        "processed_dir": str(processed_dir),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = run_smoke_test(args.output_dir, freq=args.freq, keep_existing=args.keep_existing)
    print("public smoke test completed")
    print(f"fixture type: {summary['fixture_type']}")
    print(f"lead-lag datasets written: {summary['lead_lag']['datasets_written']}")
    print(f"merged rows: {summary['merged_dataset']['rows']}")
    print(f"merged groups: {summary['merged_dataset']['groups']}")
    print(f"summary saved to: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
