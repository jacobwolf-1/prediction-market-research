from __future__ import annotations

import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_BACKOFF_SECONDS = 1.5


NBA_TEAM_ALIASES: dict[str, set[str]] = {
    "ATL": {"atl", "hawks", "atlanta", "atlanta hawks"},
    "BOS": {"bos", "celtics", "boston", "boston celtics"},
    "BKN": {"bkn", "nets", "brooklyn", "brooklyn nets"},
    "CHA": {"cha", "hornets", "charlotte", "charlotte hornets"},
    "CHI": {"chi", "bulls", "chicago", "chicago bulls"},
    "CLE": {"cle", "cavaliers", "cavs", "cleveland", "cleveland cavaliers"},
    "DAL": {"dal", "mavericks", "mavs", "dallas", "dallas mavericks"},
    "DEN": {"den", "nuggets", "denver", "denver nuggets"},
    "DET": {"det", "pistons", "detroit", "detroit pistons"},
    "GSW": {"gsw", "warriors", "golden state", "golden state warriors"},
    "HOU": {"hou", "rockets", "houston", "houston rockets"},
    "IND": {"ind", "pacers", "indiana", "indiana pacers"},
    "LAC": {"lac", "clippers", "la clippers", "los angeles clippers"},
    "LAL": {"lal", "lakers", "la lakers", "los angeles lakers"},
    "MEM": {"mem", "grizzlies", "memphis", "memphis grizzlies"},
    "MIA": {"mia", "heat", "miami", "miami heat"},
    "MIL": {"mil", "bucks", "milwaukee", "milwaukee bucks"},
    "MIN": {"min", "timberwolves", "wolves", "minnesota", "minnesota timberwolves"},
    "NOP": {"nop", "pelicans", "new orleans", "new orleans pelicans", "no pelicans"},
    "NYK": {"nyk", "knicks", "new york", "new york knicks"},
    "OKC": {"okc", "thunder", "oklahoma city", "oklahoma city thunder"},
    "ORL": {"orl", "magic", "orlando", "orlando magic"},
    "PHI": {"phi", "76ers", "sixers", "philadelphia", "philadelphia 76ers"},
    "PHX": {"phx", "suns", "phoenix", "phoenix suns"},
    "POR": {"por", "trail blazers", "blazers", "portland", "portland trail blazers"},
    "SAC": {"sac", "kings", "sacramento", "sacramento kings"},
    "SAS": {"sas", "spurs", "san antonio", "san antonio spurs"},
    "TOR": {"tor", "raptors", "toronto", "toronto raptors"},
    "UTA": {"uta", "jazz", "utah", "utah jazz"},
    "WAS": {"was", "wizards", "washington", "washington wizards"},
}


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    session: requests.Session | None = None,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> Any:
    http = session or requests.Session()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = http.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(backoff_seconds * attempt)
    raise RuntimeError(f"Request failed for {url}: {last_error}") from last_error


def to_datetime_utc(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def parse_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        magnitude = abs(int(value))
        unit = "ms" if magnitude >= 10**12 else "s"
        return pd.to_datetime(value, utc=True, unit=unit, errors="coerce")
    if isinstance(value, datetime):
        return pd.to_datetime(value, utc=True, errors="coerce")
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return pd.NaT
        magnitude = abs(int(numeric))
        unit = "ms" if magnitude >= 10**12 else "s"
        return pd.to_datetime(numeric, utc=True, unit=unit, errors="coerce")
    return parsed


def coerce_probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(probability):
        return None
    if probability > 1:
        probability /= 100.0
    return max(0.0, min(1.0, probability))


def parse_json_field(value: Any) -> Any:
    if value is None or isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                return value
    return value


def normalize_text(value: Any) -> str:
    if value is None:
        text = ""
    elif isinstance(value, str):
        text = value
    elif pd.isna(value):
        text = ""
    else:
        text = str(value)
    return " ".join(text.lower().replace("'", "").replace(".", "").split())


def team_aliases(*values: str | None) -> set[str]:
    aliases: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        if normalized:
            aliases.add(normalized)
    return aliases


def clock_to_seconds(clock_display: str | None) -> int | None:
    if not clock_display:
        return None
    clock = str(clock_display).strip()
    if ":" in clock:
        minutes, seconds = clock.split(":", 1)
        try:
            return int(minutes) * 60 + int(float(seconds))
        except ValueError:
            return None
    try:
        return int(float(clock))
    except ValueError:
        return None


def standardize_clock(clock_display: str | None) -> str | None:
    seconds = clock_to_seconds(clock_display)
    if seconds is None:
        return None
    minutes, rem_seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{rem_seconds:02d}"


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    ensure_directory(path.parent)
    frame.to_parquet(path, index=False)


def load_game_ids(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"game_id": str})
    if "game_id" not in frame.columns:
        raise ValueError(f"{path} does not contain a game_id column")
    frame["game_id"] = frame["game_id"].astype(str)
    return frame


def probability_columns_valid(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column not in frame.columns:
            continue
        series = frame[column].dropna()
        if ((series < 0) | (series > 1)).any():
            raise ValueError(f"Column {column} contains values outside [0, 1]")


def timestamps_monotonic(frame: pd.DataFrame, timestamp_column: str = "timestamp") -> None:
    if timestamp_column not in frame.columns:
        return
    series = frame[timestamp_column].dropna()
    if not series.is_monotonic_increasing:
        raise ValueError(f"{timestamp_column} must be monotonic increasing")
