from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mpl-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.raw_time_utils import frequency_to_seconds, normalize_frequency_label
from analysis.shock_strategy_utils import build_shock_groups
from utils.ingestion_utils import PROJECT_ROOT, RAW_DIR, ensure_directory


HORIZONS_SECONDS = [5, 10, 20, 30, 60]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare future market drift after ESPN shocks versus no-shock control timestamps.")
    parser.add_argument("--espn-dir", type=Path, default=RAW_DIR / "espn")
    parser.add_argument("--market-dir", type=Path, default=RAW_DIR / "kalshi")
    parser.add_argument("--freq", default="1s")
    parser.add_argument("--shock-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=PROJECT_ROOT / "visualizations" / "control_drift_curve.png",
    )
    return parser.parse_args()


def prepare_frame(groups: list[dict[str, object]], frequency_seconds: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for group in groups:
        frame = group["frame"].copy()
        frame["shock"] = frame["delta_espn"]
        for horizon in HORIZONS_SECONDS:
            periods = max(1, horizon // frequency_seconds)
            frame[f"future_market_return_{horizon}s"] = frame["market_probability"].shift(-periods) - frame["market_probability"]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True).sort_values(["game_id", "team", "timestamp"]).reset_index(drop=True) if frames else pd.DataFrame()


def build_curve(frame: pd.DataFrame, shock_threshold: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shocks = frame.loc[frame["shock"].abs() > shock_threshold].copy()
    controls = frame.loc[(frame["shock"].abs() <= shock_threshold) & (frame["shock"] != 0)].copy()
    if shocks.empty or controls.empty:
        return pd.DataFrame()

    sample_size = min(len(shocks), len(controls))
    control_idx = rng.choice(controls.index.to_numpy(), size=sample_size, replace=False)
    control_sample = controls.loc[control_idx].copy()
    shock_sample = shocks.sample(n=sample_size, random_state=seed).copy()

    shock_sample["direction"] = np.sign(shock_sample["shock"])
    control_sample["direction"] = np.sign(control_sample["shock"])

    rows: list[dict[str, object]] = []
    for label, sample in (("shock", shock_sample), ("control", control_sample)):
        for horizon in HORIZONS_SECONDS:
            col = f"future_market_return_{horizon}s"
            signed = sample["direction"] * sample[col]
            rows.append(
                {
                    "series": label,
                    "horizon_seconds": horizon,
                    "avg_signed_market_return": float(signed.mean()),
                    "median_signed_market_return": float(signed.median()),
                    "count": int(signed.notna().sum()),
                }
            )
    return pd.DataFrame(rows)


def plot_curve(results: pd.DataFrame, output_path: Path) -> None:
    ensure_directory(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in results.groupby("series", sort=True):
        ax.plot(group["horizon_seconds"], group["avg_signed_market_return"], marker="o", linewidth=1.6, label=label)
    ax.axhline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Horizon (seconds)")
    ax.set_ylabel("Average Signed Market Return")
    ax.set_title("Shock vs Control Drift")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    frequency = normalize_frequency_label(args.freq)
    groups, _ = build_shock_groups(args.espn_dir, args.market_dir, frequency)
    frame = prepare_frame(groups, frequency_to_seconds(frequency))
    results = build_curve(frame, args.shock_threshold, args.seed)
    if results.empty:
        print("Insufficient shock/control rows.")
        return
    plot_curve(results, args.output_plot)
    print(results.to_string(index=False))
    print(f"plot saved to: {args.output_plot}")


if __name__ == "__main__":
    main()
