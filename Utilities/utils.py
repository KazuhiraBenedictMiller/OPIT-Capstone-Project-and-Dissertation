from datetime import datetime, UTC
from urllib.parse import urlencode
from urllib.request import urlopen
from urllib.error import HTTPError, URLError
import json
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator

BASE_URL = "https://api.kraken.com"

def to_datetime(timestamp: int) -> str:
    return datetime.fromtimestamp(timestamp, UTC).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )

# Function to get Raw Data From Kraken.com
def GetOHLC(pair:str ="BTC/USD", interval:int=1440, since=None) -> pd.DataFrame:
    """
    Fetch OHLC candles from Kraken.

    Intervals:
        1      = 1 minute
        5      = 5 minutes
        15     = 15 minutes
        30     = 30 minutes
        60     = 1 hour
        240    = 4 hours
        1440   = 1 day
        10080  = 1 week
        21600  = 15 days
    """

    params = {
        "pair": pair,
        "interval": interval,
    }

    if since is not None:
        params["since"] = since

    url = f"{BASE_URL}/0/public/OHLC?{urlencode(params)}"

    try:
        with urlopen(url) as response:
            payload = json.load(response)

        if payload["error"]:
            raise RuntimeError(payload["error"])

        result = payload["result"]

        # Pair key is dynamic
        pair_key = next(k for k in result.keys() if k != "last")

        candles = []

        for candle in result[pair_key]:
            candles.append({
                "timestamp": candle[0],
                "datetime": to_datetime(candle[0]),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "vwap": float(candle[5]),
                "volume": float(candle[6]),
                "trades": int(candle[7]),
            })

        data = json.dumps(candles, indent=2)

        # create dataframe
        df = pd.DataFrame(json.loads(data))

        # convert datetime column to pandas datetime
        df["datetime"] = pd.to_datetime(df["datetime"])

        return df

    except HTTPError as e:
        raise RuntimeError(f"HTTP error: {e.code}") from e

    except URLError as e:
        raise RuntimeError(f"Connection error: {e.reason}") from e


# Function to Apply Feature Engineering to the Raw DataFrame
def TransformDF(df:pd.DataFrame) -> pd.DataFrame:
    df.drop(["timestamp", "vwap", "trades"], axis = 1, inplace = True)
    df_filtered = df[
        (df["datetime"] >= "2025-01-01") &
        (df["datetime"] <= "2026-04-30")
    ].reset_index(drop=True)
    
    df = df_filtered.copy()

    df["datetime"] = pd.to_datetime(df["datetime"])

    df = df.sort_values("datetime").reset_index(drop=True)

    df["candle_body"] = df["close"] - df["open"]

    df["ret1"] = df["close"].pct_change(1)
    df["ret3"] = df["close"].pct_change(3)
    df["ret7"] = df["close"].pct_change(7)

    df["breakout_20"] = (
        df["close"] /
        df["high"].rolling(20).max()
    )

    df["breakdown_20"] = (
        df["close"] /
        df["low"].rolling(20).min()
    )

    df["rsi"] = RSIIndicator(
        close=df["close"],
        window=14
    ).rsi()

    df["ema20"] = EMAIndicator(
        close=df["close"],
        window=20
    ).ema_indicator()


    df["ema20_dist"] = (
        (df["close"] - df["ema20"]) / df["ema20"]
    )

    df["future_return"] = (
        (df["close"].shift(-1) - df["close"]) / df["close"]
    )

    df["target"] = (
        df["candle_body"] > 0
    ).astype(int)

    df["target"] = df["target"].shift(-1)
    df = df.dropna().reset_index(drop=True)

    features_to_lag = [
        "open",
        "high",
        "low",
        "close",
        "candle_body",
        "volume",
        "rsi",
        "ema20_dist",
    ]

    lags = 5

    for feature in features_to_lag:
        for lag in range(1, lags + 1):    
            df[f"{feature}_lag{lag}"] = (
                df[feature].shift(lag)
            )

    df = df.dropna().reset_index(drop=True)

    return df

# Function that Builds Features and Target and Split the Dataframe
def SplitDF(df:pd.DataFrame):
    excluded_cols = [
        "datetime",
        "future_return",
        "target"
    ]

    feature_cols = [
        col for col in df.columns
        if col not in excluded_cols
    ]

    X = df[feature_cols]

    y = df["target"]

    train_df = df[df["datetime"].dt.year == 2025]

    # test set -> only 2026
    test_df = df[
        (df["datetime"] >= "2026-01-01") &
        (df["datetime"] <= "2026-04-30")
    ].reset_index(drop=True)

    # build X and y
    X_train = train_df[feature_cols]
    y_train = train_df["target"]

    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    return test_df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    daily_btc = get_ohlc(
        pair="BTC/USD",
        interval=1440,   # daily candles
    )

    print(daily_btc.head())