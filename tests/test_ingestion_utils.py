from __future__ import annotations

import pandas as pd
import pytest

from utils.ingestion_utils import (
    coerce_probability,
    parse_json_field,
    parse_timestamp,
    probability_columns_valid,
    standardize_clock,
    timestamps_monotonic,
)


def test_parse_timestamp_handles_seconds_and_milliseconds() -> None:
    seconds = parse_timestamp(1_700_000_000)
    milliseconds = parse_timestamp(1_700_000_000_000)

    assert seconds == pd.Timestamp("2023-11-14 22:13:20+00:00")
    assert milliseconds == pd.Timestamp("2023-11-14 22:13:20+00:00")


def test_parse_json_field_and_probability_helpers() -> None:
    assert parse_json_field('{"a": 1}') == {"a": 1}
    assert parse_json_field("not json") == "not json"
    assert coerce_probability(55) == pytest.approx(0.55)
    assert coerce_probability(250) == 1.0
    assert coerce_probability("bad") is None


def test_standardize_clock_formats_valid_inputs() -> None:
    assert standardize_clock("5:09") == "05:09"
    assert standardize_clock("75") == "01:15"
    assert standardize_clock("bad") is None


def test_probability_columns_valid_raises_for_out_of_range_values() -> None:
    frame = pd.DataFrame({"market_probability": [0.1, 1.2]})

    with pytest.raises(ValueError, match="outside \\[0, 1\\]"):
        probability_columns_valid(frame, ["market_probability"])


def test_timestamps_monotonic_raises_for_descending_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01T00:00:02Z", "2026-01-01T00:00:01Z"],
                utc=True,
            )
        }
    )

    with pytest.raises(ValueError, match="monotonic increasing"):
        timestamps_monotonic(frame)
