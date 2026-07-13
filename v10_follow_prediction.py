from __future__ import annotations

import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from v3_data import download_daily_data


V8_CANDIDATES_FILE = "v8_today_candidates.csv"
V9_RELATIONS_FILE = "v9_leader_lag_results.csv"
OUTPUT_FILE = "v10_follow_predictions.csv"

MAX_LEADERS = 3
MAX_FOLLOWERS_PER_LEADER = 4
MAX_TOTAL_PREDICTIONS = 8

MIN_RELATIONSHIP_SCORE = 42.0
MIN_TEST_SUCCESS_RATE = 55.0
MIN_TEST_UPLIFT = 15.0
MIN_TEST_EVENTS = 4

MAX_FOLLOWER_RETURN_1D = 5.0
MAX_FOLLOWER_RETURN_5D = 10.0
MAX_FOLLOWER_RETURN_20D = 25.0

MIN_LIVE_VOLUME_RATIO = 0.55
MIN_LIVE_EMA20_DISTANCE = -2.0


OUTPUT_COLUMNS = [
    "rank",
    "leader",
    "follower",
    "lag_days",
    "prediction_score",
    "prediction_classification",
    "live_confirmation_score",
    "live_confirmation_class",
    "live_confirmation_reasons",
    "live_confirmation_risks",
    "train_events",
    "train_success_rate",
    "train_average_return",
    "train_baseline_rate",
    "train_uplift",
    "test_events",
    "test_success_rate",
    "test_average_return",
    "test_median_return",
    "test_baseline_rate",
    "test_uplift",
    "relationship_score",
    "leader_v8_score",
    "leader_smart_money_score",
    "leader_institutional_score",
    "leader_source",
    "follower_price",
    "follower_return_1d",
    "follower_return_5d",
    "follower_return_20d",
    "follower_volume_ratio",
    "follower_rsi",
    "follower_ema20",
    "follower_ema20_distance",
    "follower_ema20_slope_positive",
    "follower_close_position",
    "follower_data_valid",
    "not_extended",
]


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return default

        return bool(value)

    text = str(value).strip().lower()

    if text in {
        "true",
        "1",
        "yes",
        "evet",
        "on",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "hayır",
        "hayir",
        "off",
        "",
        "nan",
        "none",
    }:
        return False

    return default


def empty_output_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )


def get_manual_leaders() -> List[str]:
    """
    Workflow üzerinden manuel test liderleri girilebilir.

    Örnek:
    V10_MANUAL_LEADERS=THYAO,TUPRS

    Boş bırakılırsa V8'in gerçek nihai adayları kullanılır.
    """
    raw_value = os.getenv(
        "V10_MANUAL_LEADERS",
        "",
    ).strip()

    if not raw_value:
        return []

    leaders = [
        item.strip().upper()
        for item in raw_value.split(",")
        if item.strip()
    ]

    return list(dict.fromkeys(leaders))


def load_v8_leaders() -> pd.DataFrame:
    manual_leaders = get_manual_leaders()

    if manual_leaders:
        print(
            "V10 manuel test liderleri:",
            manual_leaders,
        )

        return pd.DataFrame({
            "symbol": manual_leaders,
            "v8_score": [np.nan] * len(manual_leaders),
            "smart_money_score": [np.nan] * len(manual_leaders),
            "institutional_score": [np.nan] * len(manual_leaders),
            "leader_source": ["manual_test"] * len(manual_leaders),
        })

    if not os.path.exists(
        V8_CANDIDATES_FILE
    ):
        print(
            f"{V8_CANDIDATES_FILE} bulunamadı."
        )
        return pd.DataFrame()

    try:
        leaders = pd.read_csv(
            V8_CANDIDATES_FILE
        )

    except Exception as exc:
        print(
            "V8 aday dosyası okunamadı:",
            exc,
        )
        return pd.DataFrame()

    if (
        leaders.empty
        or "symbol" not in leaders.columns
    ):
        print(
            "V8 aday dosyası boş veya symbol kolonu yok."
        )
        return pd.DataFrame()

    leaders = leaders.head(
        MAX_LEADERS
    ).copy()

    leaders["symbol"] = (
        leaders["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    leaders = leaders[
        leaders["symbol"].ne("")
    ].copy()

    leaders["leader_source"] = "v8_final"

    return leaders.reset_index(drop=True)


def load_relationships() -> pd.DataFrame:
    if not os.path.exists(
        V9_RELATIONS_FILE
    ):
        print(
            f"{V9_RELATIONS_FILE} bulunamadı."
        )
        return pd.DataFrame()

    try:
        relationships = pd.read_csv(
            V9_RELATIONS_FILE
        )

    except Exception as exc:
        print(
            "V9 ilişki dosyası okunamadı:",
            exc,
        )
        return pd.DataFrame()

    if relationships.empty:
        print("V9 ilişki dosyası boş.")
        return relationships

    required_columns = [
        "leader",
        "follower",
        "lag_days",
        "test_events",
        "test_success_rate",
        "test_average_return",
        "test_baseline_rate",
        "test_uplift",
        "relationship_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in relationships.columns
    ]

    if missing_columns:
        print(
            "V9 ilişki dosyasında eksik kolonlar:",
            missing_columns,
        )
        return pd.DataFrame()

    relationships["leader"] = (
        relationships["leader"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    relationships["follower"] = (
        relationships["follower"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    numeric_columns = [
        "lag_days",
        "train_events",
        "train_success_rate",
        "train_average_return",
        "train_baseline_rate",
        "train_uplift",
        "test_events",
        "test_success_rate",
        "test_average_return",
        "test_median_return",
        "test_baseline_rate",
        "test_uplift",
        "relationship_score",
    ]

    for column in numeric_columns:
        if column not in relationships.columns:
            relationships[column] = np.nan

        relationships[column] = pd.to_numeric(
            relationships[column],
            errors="coerce",
        )

    relationships = relationships[
        relationships["leader"].ne("")
        & relationships["follower"].ne("")
    ].copy()

    return relationships.reset_index(drop=True)


def calculate_rsi(
    close: pd.Series,
    window: int = 14,
) -> pd.Series:
    """
    Wilder tipi üssel RSI hesaplar.
    """
    delta = close.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    average_gain = gain.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / window,
        adjust=False,
        min_periods=window,
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    rsi = (
        100
        - (
            100
            / (1 + relative_strength)
        )
    )

    return rsi


def calculate_recent_performance(
    symbol: str,
) -> Dict[str, Any]:
    """
    Takipçinin güncel teknik ve hacim teyidini hesaplar.
    """
    try:
        dataframe = download_daily_data(
            symbol=symbol,
            period="6mo",
            interval="1d",
            retries=1,
        )

    except Exception as exc:
        print(
            f"{symbol} canlı veri indirme hatası:",
            exc,
        )

        return {
            "data_valid": False,
            "last_price": np.nan,
            "return_1d": np.nan,
            "return_5d": np.nan,
            "return_20d": np.nan,
            "volume_ratio": np.nan,
            "rsi": np.nan,
            "ema20": np.nan,
            "ema20_distance": np.nan,
            "ema20_slope_positive": False,
            "close_position": np.nan,
        }

    if (
        dataframe is None
        or dataframe.empty
        or len(dataframe) < 60
    ):
        return {
            "data_valid": False,
            "last_price": np.nan,
            "return_1d": np.nan,
            "return_5d": np.nan,
            "return_20d": np.nan,
            "volume_ratio": np.nan,
            "rsi": np.nan,
            "ema20": np.nan,
            "ema20_distance": np.nan,
            "ema20_slope_positive": False,
            "close_position": np.nan,
        }

    required_columns = [
        "Close",
        "High",
        "Low",
        "Volume",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        print(
            f"{symbol} eksik veri kolonları:",
            missing_columns,
        )

        return {
            "data_valid": False,
            "last_price": np.nan,
            "return_1d": np.nan,
            "return_5d": np.nan,
            "return_20d": np.nan,
            "volume_ratio": np.nan,
            "rsi": np.nan,
            "ema20": np.nan,
            "ema20_distance": np.nan,
            "ema20_slope_positive": False,
            "close_position": np.nan,
        }

    close = pd.to_numeric(
        dataframe["Close"],
        errors="coerce",
    )

    high = pd.to_numeric(
        dataframe["High"],
        errors="coerce",
    )

    low = pd.to_numeric(
        dataframe["Low"],
        errors="coerce",
    )

    volume = pd.to_numeric(
        dataframe["Volume"],
        errors="coerce",
    )

    valid_mask = (
        close.notna()
        & high.notna()
        & low.notna()
        & volume.notna()
    )

    close = close[valid_mask]
    high = high[valid_mask]
    low = low[valid_mask]
    volume = volume[valid_mask]

    if len(close) < 60:
        return {
            "data_valid": False,
            "last_price": np.nan,
            "return_1d": np.nan,
            "return_5d": np.nan,
            "return_20d": np.nan,
            "volume_ratio": np.nan,
            "rsi": np.nan,
            "ema20": np.nan,
            "ema20_distance": np.nan,
            "ema20_slope_positive": False,
            "close_position": np.nan,
        }

    ema20 = close.ewm(
        span=20,
        adjust=False,
    ).mean()

    rsi_series = calculate_rsi(
        close=close,
        window=14,
    )

    last_price = safe_float(
        close.iloc[-1],
        np.nan,
    )

    last_ema20 = safe_float(
        ema20.iloc[-1],
        np.nan,
    )

    last_rsi = safe_float(
        rsi_series.iloc[-1],
        np.nan,
    )

    def calculate_return(
        trading_days: int,
    ) -> float:
        if len(close) <= trading_days:
            return np.nan

        old_price = safe_float(
            close.iloc[
                -trading_days - 1
            ],
            np.nan,
        )

        if (
            pd.isna(last_price)
            or pd.isna(old_price)
            or old_price <= 0
        ):
            return np.nan

        return round(
            (
                last_price / old_price
                - 1
            ) * 100,
            2,
        )

    volume_average_20 = safe_float(
        volume.tail(20).mean(),
        0,
    )

    last_volume = safe_float(
        volume.iloc[-1],
        0,
    )

    volume_ratio = (
        last_volume / volume_average_20
        if volume_average_20 > 0
        else np.nan
    )

    ema20_distance = (
        (
            last_price / last_ema20
            - 1
        ) * 100
        if (
            not pd.isna(last_price)
            and not pd.isna(last_ema20)
            and last_ema20 > 0
        )
        else np.nan
    )

    ema20_slope_positive = bool(
        len(ema20) >= 4
        and ema20.iloc[-1]
        >= ema20.iloc[-4]
    )

    last_high = safe_float(
        high.iloc[-1],
        np.nan,
    )

    last_low = safe_float(
        low.iloc[-1],
        np.nan,
    )

    candle_range = (
        last_high - last_low
        if (
            not pd.isna(last_high)
            and not pd.isna(last_low)
        )
        else 0
    )

    close_position = (
        (
            last_price - last_low
        ) / candle_range
        if (
            candle_range > 0
            and not pd.isna(last_price)
        )
        else 0.5
    )

    return {
        "data_valid": True,
        "last_price": round(
            last_price,
            4,
        ),
        "return_1d": calculate_return(1),
        "return_5d": calculate_return(5),
        "return_20d": calculate_return(20),
        "volume_ratio": (
            round(volume_ratio, 2)
            if not pd.isna(volume_ratio)
            else np.nan
        ),
        "rsi": (
            round(last_rsi, 2)
            if not pd.isna(last_rsi)
            else np.nan
        ),
        "ema20": (
            round(last_ema20, 4)
            if not pd.isna(last_ema20)
            else np.nan
        ),
        "ema20_distance": (
            round(ema20_distance, 2)
            if not pd.isna(ema20_distance)
            else np.nan
        ),
        "ema20_slope_positive": (
            ema20_slope_positive
        ),
        "close_position": round(
            close_position,
            2,
        ),
    }


def follower_is_not_extended(
    row: pd.Series,
) -> bool:
    """
    Takipçi zaten çok yükselmişse aday listesinden çıkarılır.
    """
    return_1d = safe_float(
        row.get("follower_return_1d"),
        999,
    )

    return_5d = safe_float(
        row.get("follower_return_5d"),
        999,
    )

    return_20d = safe_float(
        row.get("follower_return_20d"),
        999,
    )

    return (
        return_1d
        <= MAX_FOLLOWER_RETURN_1D
        and return_
