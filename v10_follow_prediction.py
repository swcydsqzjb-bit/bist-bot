from __future__ import annotations

import os
from typing import List

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


def safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def get_manual_leaders() -> List[str]:
    """
    Test amacıyla workflow üzerinden lider girilebilir:

    V10_MANUAL_LEADERS=THYAO,TUPRS

    Boşsa gerçek V8 adayları kullanılır.
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

    if not os.path.exists(V8_CANDIDATES_FILE):
        print(
            f"{V8_CANDIDATES_FILE} bulunamadı."
        )
        return pd.DataFrame()

    try:
        leaders = pd.read_csv(
            V8_CANDIDATES_FILE
        )

    except Exception as exc:
        print("V8 adayları okunamadı:", exc)
        return pd.DataFrame()

    if leaders.empty or "symbol" not in leaders.columns:
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

    leaders["leader_source"] = "v8_final"

    return leaders


def load_relationships() -> pd.DataFrame:
    if not os.path.exists(V9_RELATIONS_FILE):
        print(
            f"{V9_RELATIONS_FILE} bulunamadı."
        )
        return pd.DataFrame()

    try:
        relationships = pd.read_csv(
            V9_RELATIONS_FILE
        )

    except Exception as exc:
        print("V9 ilişkileri okunamadı:", exc)
        return pd.DataFrame()

    if relationships.empty:
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

    missing = [
        column
        for column in required_columns
        if column not in relationships.columns
    ]

    if missing:
        print(
            "V9 ilişki dosyasında eksik kolon:",
            missing,
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

    return relationships


def calculate_recent_performance(
    symbol: str,
) -> dict:
    dataframe = download_daily_data(
        symbol=symbol,
        period="3mo",
        interval="1d",
        retries=1,
    )

    if dataframe.empty or len(dataframe) < 25:
        return {
            "data_valid": False,
            "last_price": np.nan,
            "return_1d": np.nan,
            "return_5d": np.nan,
            "return_20d": np.nan,
            "volume_ratio": np.nan,
        }

    close = pd.to_numeric(
        dataframe["Close"],
        errors="coerce",
    )

    volume = pd.to_numeric(
        dataframe["Volume"],
        errors="coerce",
    )

    last_price = safe_float(
        close.iloc[-1],
        np.nan,
    )

    def calculate_return(days: int) -> float:
        if len(close) <= days:
            return np.nan

        old_price = safe_float(
            close.iloc[-days - 1],
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

    volume_average = safe_float(
        volume.tail(20).mean(),
        0,
    )

    last_volume = safe_float(
        volume.iloc[-1],
        0,
    )

    volume_ratio = (
        last_volume / volume_average
        if volume_average > 0
        else np.nan
    )

    return {
        "data_valid": True,
        "last_price": round(last_price, 4),
        "return_1d": calculate_return(1),
        "return_5d": calculate_return(5),
        "return_20d": calculate_return(20),
        "volume_ratio": round(
            volume_ratio,
            2,
        ) if not pd.isna(volume_ratio) else np.nan,
    }


def follower_is_not_extended(row: pd.Series) -> bool:
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
        return_1d <= MAX_FOLLOWER_RETURN_1D
        and return_5d <= MAX_FOLLOWER_RETURN_5D
        and return_20d <= MAX_FOLLOWER_RETURN_20D
    )


def calculate_prediction_score(
    row: pd.Series,
) -> float:
    relationship_score = safe_float(
        row.get("relationship_score")
    )

    test_success = safe_float(
        row.get("test_success_rate")
    )

    test_uplift = safe_float(
        row.get("test_uplift")
    )

    test_average = safe_float(
        row.get("test_average_return")
    )

    test_events = safe_float(
        row.get("test_events")
    )

    return_1d = safe_float(
        row.get("follower_return_1d")
    )

    return_5d = safe_float(
        row.get("follower_return_5d")
    )

    volume_ratio = safe_float(
        row.get("follower_volume_ratio")
    )

    score = (
        relationship_score * 0.40
        + test_success * 0.25
        + max(0, test_uplift) * 0.25
        + min(max(test_average, 0), 10) * 0.60
        + min(test_events, 12) * 0.30
    )

    # Takipçi henüz fazla gitmemişse küçük bonus.
    if -3 <= return_5d <= 4:
        score += 4

    elif 4 < return_5d <= 8:
        score += 1

    # Hacim yeni canlanıyorsa küçük bonus.
    if 1.10 <= volume_ratio <= 2.20:
        score += 3

    # Aynı gün fazla hareket ettiyse ceza.
    if return_1d > 3:
        score -= 3

    if return_5d > 8:
        score -= 5

    return round(
        max(0, min(100, score)),
        2,
    )


def prediction_classification(
    score: float,
) -> str:
    if score >= 70:
        return "GÜÇLÜ TAKİPÇİ"

    if score >= 58:
        return "ORTA TAKİPÇİ"

    if score >= 48:
        return "İZLEME TAKİPÇİSİ"

    return "ZAYIF"


def build_predictions(
    leaders: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    if leaders.empty or relationships.empty:
        return pd.DataFrame()

    leader_symbols = (
        leaders["symbol"]
        .astype(str)
        .str.upper()
        .tolist()
    )

    candidates = relationships[
        relationships["leader"].isin(
            leader_symbols
        )
        & (
            relationships["relationship_score"]
            >= MIN_RELATIONSHIP_SCORE
        )
        & (
            relationships["test_success_rate"]
            >= MIN_TEST_SUCCESS_RATE
        )
        & (
            relationships["test_uplift"]
            >= MIN_TEST_UPLIFT
        )
        & (
            relationships["test_events"]
            >= MIN_TEST_EVENTS
        )
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    # Liderin kendi V8 bilgilerini ilişkiye ekle.
    leader_columns = [
        column
        for column in [
            "symbol",
            "v8_score",
            "smart_money_score",
            "institutional_score",
            "leader_source",
        ]
        if column in leaders.columns
    ]

    leader_info = leaders[
        leader_columns
    ].copy()

    leader_info = leader_info.rename(
        columns={
            "symbol": "leader",
            "v8_score": "leader_v8_score",
            "smart_money_score": (
                "leader_smart_money_score"
            ),
            "institutional_score": (
                "leader_institutional_score"
            ),
        }
    )

    candidates = candidates.merge(
        leader_info,
        on="leader",
        how="left",
    )

    follower_symbols = (
        candidates["follower"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    performance_rows = []

    for number, follower in enumerate(
        follower_symbols,
        start=1,
    ):
        print(
            f"[{number}/{len(follower_symbols)}] "
            f"V10 takipçi kontrolü: {follower}"
        )

        performance = calculate_recent_performance(
            follower
        )

        performance_rows.append({
            "follower": follower,
            "follower_data_valid": (
                performance["data_valid"]
            ),
            "follower_price": (
                performance["last_price"]
            ),
            "follower_return_1d": (
                performance["return_1d"]
            ),
            "follower_return_5d": (
                performance["return_5d"]
            ),
            "follower_return_20d": (
                performance["return_20d"]
            ),
            "follower_volume_ratio": (
                performance["volume_ratio"]
            ),
        })

    performance_df = pd.DataFrame(
        performance_rows
    )

    candidates = candidates.merge(
        performance_df,
        on="follower",
        how="left",
    )

    candidates = candidates[
        candidates["follower_data_valid"].eq(
            True
        )
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates["not_extended"] = (
        candidates.apply(
            follower_is_not_extended,
            axis=1,
        )
    )

    candidates = candidates[
        candidates["not_extended"].eq(True)
    ].copy()

    if candidates.empty:
        return pd.DataFrame()

    candidates["prediction_score"] = (
        candidates.apply(
            calculate_prediction_score,
            axis=1,
        )
    )

    candidates["prediction_classification"] = (
        candidates["prediction_score"]
        .apply(prediction_classification)
    )

    candidates = candidates.sort_values(
        by=[
            "prediction_score",
            "test_uplift",
            "test_success_rate",
            "relationship_score",
            "test_events",
        ],
        ascending=False,
    )

    # Her lider için en fazla belirlenen sayıda takipçi.
    candidates = (
        candidates.groupby(
            "leader",
            group_keys=False,
        )
        .head(MAX_FOLLOWERS_PER_LEADER)
    )

    # Aynı takipçi birden fazla liderden geliyorsa
    # en yüksek puanlı ilişkiyi tut.
    candidates = candidates.drop_duplicates(
        subset=["follower"],
        keep="first",
    )

    candidates = candidates.head(
        MAX_TOTAL_PREDICTIONS
    ).reset_index(drop=True)

    candidates.insert(
        0,
        "rank",
        range(1, len(candidates) + 1),
    )

    return candidates


def print_summary(
    predictions: pd.DataFrame,
    leaders: pd.DataFrame,
) -> None:
    print("\n====================================")
    print("V10 LEADER FOLLOW PREDICTION")
    print("====================================")

    print(
        "Aktif lider:",
        len(leaders),
    )

    if not leaders.empty:
        print(
            "Liderler:",
            ", ".join(
                leaders["symbol"]
                .astype(str)
                .tolist()
            ),
        )

    print(
        "Takipçi adayı:",
        len(predictions),
    )

    if predictions.empty:
        print(
            "Canlı liderlerle eşleşen ve "
            "filtreleri geçen takipçi bulunamadı."
        )
        return

    display_columns = [
        "rank",
        "leader",
        "follower",
        "lag_days",
        "prediction_score",
        "prediction_classification",
        "test_events",
        "test_success_rate",
        "test_baseline_rate",
        "test_uplift",
        "test_average_return",
        "follower_price",
        "follower_return_1d",
        "follower_return_5d",
        "follower_volume_ratio",
    ]

    existing = [
        column
        for column in display_columns
        if column in predictions.columns
    ]

    print(
        predictions[existing]
        .to_string(index=False)
    )


def main():
    print(
        "V10 takipçi tahmin motoru başladı."
    )

    leaders = load_v8_leaders()
    relationships = load_relationships()

    print(
        "V8 lider sayısı:",
        len(leaders),
    )

    print(
        "V9 ilişki sayısı:",
        len(relationships),
    )

    predictions = build_predictions(
        leaders=leaders,
        relationships=relationships,
    )

    predictions.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        predictions=predictions,
        leaders=leaders,
    )

    print(
        "\nKaydedildi:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
