from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from data_collection.kalshi_fetch import extract_event_teams, parse_event_ticker_date


def run_case(name: str, detail: dict, expected: tuple[str, str]) -> None:
    actual = extract_event_teams(detail)
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected}, got {actual}")
    print(f"ok: {name} -> {actual}")


def main() -> None:
    # Standard event-market tickers should continue to work.
    run_case(
        "market ticker suffixes",
        {
            "event": {"event_ticker": "KXNBAGAME-26FEB19ATLPHI"},
            "markets": [
                {"ticker": "KXNBAGAME-26FEB19ATLPHI-ATL", "yes_sub_title": "Atlanta"},
                {"ticker": "KXNBAGAME-26FEB19ATLPHI-PHI", "yes_sub_title": "Philadelphia"},
            ],
        },
        ("ATL", "PHI"),
    )

    # Simple "A vs B" titles were the dominant missing pattern in the audit.
    run_case(
        "simple vs title",
        {
            "event": {"event_ticker": "KXNBAGAME-25APR18DALMEM", "title": "Dallas vs Memphis"},
            "markets": [],
        },
        ("DAL", "MEM"),
    )

    # Play-in / playoff prefixes should be stripped before team parsing.
    run_case(
        "play in title",
        {
            "event": {
                "event_ticker": "KXNBAGAME-25APR15ATLORL",
                "title": "Basketball Play-In: Atlanta vs Orlando",
            },
            "markets": [],
        },
        ("ATL", "ORL"),
    )

    # Trailing descriptors like "Winner?" should not block the second team.
    run_case(
        "winner suffix",
        {
            "event": {
                "event_ticker": "KXNBAGAME-25APR16MIACHI",
                "title": "Basketball Play-In: Miami vs Chicago Winner?",
            },
            "markets": [],
        },
        ("MIA", "CHI"),
    )

    # "@" and "at" variants should map to the same canonical team pair.
    run_case(
        "at title",
        {
            "event": {"event_ticker": "KXNBAGAME-26FEB19ATLPHI", "title": "Atlanta at Philadelphia"},
            "markets": [],
        },
        ("ATL", "PHI"),
    )
    run_case(
        "at-symbol title",
        {
            "event": {"event_ticker": "KXNBAGAME-26FEB19ATLPHI", "title": "Atlanta @ Philadelphia"},
            "markets": [],
        },
        ("ATL", "PHI"),
    )

    # Seeds, records, and extra descriptors should be stripped from each side.
    run_case(
        "seeded playoff title",
        {
            "event": {
                "event_ticker": "KXNBAGAME-25APR20ATLNYK",
                "title": "Game 2: (8) Atlanta (40-42) vs (3) New York (51-31)",
            },
            "markets": [],
        },
        ("ATL", "NYK"),
    )

    # Compact event ticker suffixes remain a last-resort fallback when titles are noisy or truncated.
    run_case(
        "event ticker fallback",
        {
            "event": {"event_ticker": "KXNBAGAME-25APR15MEMGSW", "title": "Play-In Winner?"},
            "markets": [],
        },
        ("MEM", "GSW"),
    )

    parsed_date = parse_event_ticker_date("KXNBAGAME-25APR15ATLORL")
    if str(parsed_date.date()) != "2025-04-15":
        raise AssertionError(f"event ticker date parse failed: {parsed_date}")
    print(f"ok: ticker date -> {parsed_date.date()}")


if __name__ == "__main__":
    main()
