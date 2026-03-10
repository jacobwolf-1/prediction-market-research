import requests
import pandas as pd
from datetime import datetime

GAMMA_API = "https://gamma-api.polymarket.com"
TRADE_API = "https://data-api.polymarket.com/trades"

# -----------------------------
# STEP 1: Find NBA Markets
# -----------------------------
def get_nba_markets(limit=50):
    """
    Pulls historical NBA-related markets.
    """
    
    url = f"{GAMMA_API}/events"
    params = {
        "limit": limit,
        "series_slug": "nba"
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()

    events = r.json()

    markets = []

    for event in events:
        event_id = event["id"]
        title = event["title"]
        start = event["startDate"]

        detail = requests.get(f"{GAMMA_API}/events/{event_id}", timeout=30).json()

        for m in detail.get("markets", []):
            markets.append({
                "event_id": event_id,
                "market_id": m["id"],
                "title": title,
                "question": m.get("question"),
                "start": start
            })

    return markets


# -----------------------------
# STEP 2: Fetch Trade History
# -----------------------------
def fetch_trade_history(market_id, limit=5000):
    """
    Pulls raw trades from Polymarket.
    """

    params = {
        "market": market_id,
        "limit": limit
    }

    r = requests.get(TRADE_API, params=params, timeout=30)

    if r.status_code != 200:
        return []

    return r.json()


# -----------------------------
# STEP 3: Convert Trades → Probability Series
# -----------------------------
def process_to_probability_series(trades, interval="10s"):

    if len(trades) == 0:
        return None

    df = pd.DataFrame(trades)

    if "timestamp" not in df:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["market_probability"] = df["price"].astype(float)

    df = df[["timestamp", "market_probability"]]

    df.set_index("timestamp", inplace=True)

    # resample to fixed interval
    series = df.resample(interval).last().ffill()

    return series.reset_index()


# -----------------------------
# STEP 4: Build Research Dataset
# -----------------------------
def build_market_dataset(limit=5):

    markets = get_nba_markets(limit)

    results = []

    for m in markets:

        print(f"Fetching trades for: {m['title']}")

        trades = fetch_trade_history(m["market_id"])

        series = process_to_probability_series(trades)

        if series is None:
            continue

        series["game_title"] = m["title"]
        series["market_id"] = m["market_id"]
        series["source"] = "polymarket"

        results.append(series)

    if not results:
        return None

    df = pd.concat(results)

    return df


# -----------------------------
# RUN TEST
# -----------------------------
if __name__ == "__main__":

    df = build_market_dataset(limit=5)

    if df is None:
        print("No usable data found.")
    else:
        print("\nSample Output:")
        print(df.head(20))

        print("\nDataset Shape:", df.shape)

        df.to_csv("polymarket_sample.csv", index=False)