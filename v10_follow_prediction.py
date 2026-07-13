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


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return default
        return bool(value)

    text = str(value).strip().lower()

    if text in {"true", "1", "yes", "evet", "on"}:
        return True

    if text in {
        "false", "0", "no", "hayÄ±r", "hayir",
        "off", "", "nan", "none",
    }:
        return False

    return default


def empty_output_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def get_manual_leaders() -> List[str]:
    raw_value = os.getenv("V10_MANUAL_LEADERS", "").strip()

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
        print("V10 manuel test liderleri:", manual_leaders)

        return pd.DataFrame({
            "symbol": manual_leaders,
            "v8_score": [np.nan] * len(manual_leaders),
            "smart_money_score": [np.nan] * len(manual_leaders),
            "institutional_score": [np.nan] * len(manual_leaders),
            "leader_source": ["manual_test"] * len(manual_leaders),
        })

    if not os.path.exists(V8_CANDIDATES_FILE):
        print(f"{V8_CANDIDATES_FILE} bulunamadÄ±.")
        return pd.DataFrame()

    try:
        leaders = pd.read_csv(V8_CANDIDATES_FILE)
    except Exception as exc:
        print("V8 aday dosyasÄ± okunamadÄ±:", exc)
        return pd.DataFrame()

    if leaders.empty or "symbol" not in leaders.columns:
        print("V8 aday dosyasÄ± boÅ veya symbol kolonu yok.")
        return pd.DataFrame()

    leaders = leaders.head(MAX_LEADERS).copy()

    leaders["symbol"] = (
        leaders["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    leaders = leaders[leaders["symbol"].ne("")].copy()
    leaders["leader_source"] = "v8_final"

    return leaders.reset_index(drop=True)


def load_relationships() -> pd.DataFrame:
    if not os.path.exists(V9_RELATIONS_FILE):
        print(f"{V9_RELATIONS_FILE} bulunamadÄ±.")
        return pd.DataFrame()

    try:
        relationships = pd.read_csv(V9_RELATIONS_FILE)
    except Exception as exc:
        print("V9 iliÅki dosyasÄ± okunamadÄ±:", exc)
        return pd.DataFrame()

    if relationships.empty:
        print("V9 iliÅki dosyasÄ± boÅ.")
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
        print("V9 iliÅki dosyasÄ±nda eksik kolonlar:", missing_columns)
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


def calculate_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

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

    return 100 - (100 / (1 + relative_strength))


def invalid_performance_result() -> Dict[str, Any]:
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


def calculate_recent_performance(symbol: str) -> Dict[str, Any]:
    try:
        dataframe = download_daily_data(
            symbol=symbol,
            period="6mo",
            interval="1d",
            retries=1,
        )
    except Exception as exc:
        print(f"{symbol} canlÄ± veri indirme hatasÄ±:", exc)
        return invalid_performance_result()

    if dataframe is None or dataframe.empty or len(dataframe) < 60:
        return invalid_performance_result()

    required_columns = ["Close", "High", "Low", "Volume"]

    if not all(column in dataframe.columns for column in required_columns):
        return invalid_performance_result()

    close = pd.to_numeric(dataframe["Close"], errors="coerce")
    high = pd.to_numeric(dataframe["High"], errors="coerce")
    low = pd.to_numeric(dataframe["Low"], errors="coerce")
    volume = pd.to_numeric(dataframe["Volume"], errors="coerce")

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
        return invalid_performance_result()

    ema20 = close.ewm(span=20, adjust=False).mean()
    rsi_series = calculate_rsi(close=close, window=14)

    last_price = safe_float(close.iloc[-1], np.nan)
    last_ema20 = safe_float(ema20.iloc[-1], np.nan)
    last_rsi = safe_float(rsi_series.iloc[-1], np.nan)

    def calculate_return(trading_days: int) -> float:
        if len(close) <= trading_days:
            return np.nan

        old_price = safe_float(
            close.iloc[-trading_days - 1],
            np.nan,
        )

        if (
            pd.isna(last_price)
            or pd.isna(old_price)
            or old_price <= 0
        ):
            return np.nan

        return round(
            ((last_price / old_price) - 1) * 100,
            2,
        )

    volume_average_20 = safe_float(volume.tail(20).mean(), 0)
    last_volume = safe_float(volume.iloc[-1], 0)

    volume_ratio = (
        last_volume / volume_average_20
        if volume_average_20 > 0
        else np.nan
    )

    ema20_distance = (
        ((last_price / last_ema20) - 1) * 100
        if (
            not pd.isna(last_price)
            and not pd.isna(last_ema20)
            and last_ema20 > 0
        )
        else np.nan
    )

    ema20_slope_positive = bool(
        len(ema20) >= 4
        and ema20.iloc[-1] >= ema20.iloc[-4]
    )

    last_high = safe_float(high.iloc[-1], np.nan)
    last_low = safe_float(low.iloc[-1], np.nan)

    candle_range = (
        last_high - last_low
        if (
            not pd.isna(last_high)
            and not pd.isna(last_low)
        )
        else 0
    )

    close_position = (
        (last_price - last_low) / candle_range
        if candle_range > 0 and not pd.isna(last_price)
        else 0.5
    )

    return {
        "data_valid": True,
        "last_price": round(last_price, 4),
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
        "ema20_slope_positive": ema20_slope_positive,
        "close_position": round(close_position, 2),
    }


def follower_is_not_extended(row: pd.Series) -> bool:
    return_1d = safe_float(row.get("follower_return_1d"), 999)
    return_5d = safe_float(row.get("follower_return_5d"), 999)
    return_20d = safe_float(row.get("follower_return_20d"), 999)

    return (
        return_1d <= MAX_FOLLOWER_RETURN_1D
        and return_5d <= MAX_FOLLOWER_RETURN_5D
        and return_20d <= MAX_FOLLOWER_RETURN_20D
    )


def calculate_live_confirmation(row: pd.Series) -> pd.Series:
    score = 0
    reasons: List[str] = []
    risks: List[str] = []

    volume_ratio = safe_float(row.get("follower_volume_ratio"), 0)
    rsi = safe_float(row.get("follower_rsi"), 0)
    ema20_distance = safe_float(
        row.get("follower_ema20_distance"),
        -999,
    )
    close_position = safe_float(
        row.get("follower_close_position"),
        0,
    )
    return_1d = safe_float(row.get("follower_return_1d"))
    return_5d = safe_float(row.get("follower_return_5d"))
    ema_slope_positive = safe_bool(
        row.get("follower_ema20_slope_positive")
    )

    if volume_ratio >= 1.20:
        score += 25
        reasons.append("Hacim teyidi gÃ¼Ã§lÃ¼")
    elif volume_ratio >= 0.80:
        score += 17
        reasons.append("Hacim normal bÃ¶lgede")
    elif volume_ratio >= MIN_LIVE_VOLUME_RATIO:
        score += 8
        reasons.append("Hacim erken aÅamada")
    else:
        risks.append("Hacim teyidi Ã§ok zayÄ±f")

    if ema20_distance >= 0:
        score += 20
        reasons.append("EMA20 Ã¼zerinde")
    elif ema20_distance >= MIN_LIVE_EMA20_DISTANCE:
        score += 10
        reasons.append("EMA20 desteÄine yakÄ±n")
    else:
        risks.append("EMA20 altÄ±nda")

    if ema_slope_positive:
        score += 15
        reasons.append("EMA20 eÄimi pozitif")
    else:
        risks.append("EMA20 eÄimi zayÄ±f")

    if 48 <= rsi <= 66:
        score += 20
        reasons.append("RSI saÄlÄ±klÄ± momentumda")
    elif 42 <= rsi < 48:
        score += 10
        reasons.append("RSI erken toparlanma bÃ¶lgesinde")
    elif 66 < rsi <= 70:
        score += 8
        reasons.append("RSI yÃ¼ksek ama kabul edilebilir")
    else:
        risks.append("RSI uygun bÃ¶lgede deÄil")

    if close_position >= 0.65:
        score += 10
        reasons.append("GÃ¼Ã§lÃ¼ gÃ¼nlÃ¼k kapanÄ±Å")
    elif close_position >= 0.45:
        score += 5
    else:
        risks.append("KapanÄ±Å gÃ¼cÃ¼ zayÄ±f")

    if -2.5 <= return_1d <= 2.5:
        score += 5
        reasons.append("GÃ¼nlÃ¼k hareket henÃ¼z sÄ±nÄ±rlÄ±")

    if -3 <= return_5d <= 5:
        score += 5
        reasons.append("5 gÃ¼nlÃ¼k hareket henÃ¼z sÄ±nÄ±rlÄ±")

    score = int(max(0, min(100, score)))

    if (
        score >= 70
        and volume_ratio >= 0.80
        and ema20_distance >= -1
        and 42 <= rsi <= 70
    ):
        classification = "TEYÄ°TLÄ° TAKÄ°PÃÄ°"

    elif (
        score >= 48
        and volume_ratio >= MIN_LIVE_VOLUME_RATIO
        and ema20_distance >= MIN_LIVE_EMA20_DISTANCE
        and 40 <= rsi <= 72
    ):
        classification = "ERKEN Ä°ZLEME"

    else:
        classification = "TEYÄ°TSÄ°Z"

    return pd.Series({
        "live_confirmation_score": score,
        "live_confirmation_class": classification,
        "live_confirmation_reasons": (
            " | ".join(reasons)
            if reasons
            else "CanlÄ± teyit nedeni yok"
        ),
        "live_confirmation_risks": (
            " | ".join(risks)
            if risks
            else "Belirgin canlÄ± risk yok"
        ),
    })


def live_confirmation_is_acceptable(row: pd.Series) -> bool:
    classification = str(
        row.get("live_confirmation_class", "")
    )

    return classification in {
        "TEYÄ°TLÄ° TAKÄ°PÃÄ°",
        "ERKEN Ä°ZLEME",
    }


def calculate_prediction_score(row: pd.Series) -> float:
    relationship_score = safe_float(row.get("relationship_score"))
    test_success = safe_float(row.get("test_success_rate"))
    test_uplift = safe_float(row.get("test_uplift"))
    test_average = safe_float(row.get("test_average_return"))
    test_events = safe_float(row.get("test_events"))
    live_confirmation = safe_float(
        row.get("live_confirmation_score")
    )
    return_1d = safe_float(row.get("follower_return_1d"))
    return_5d = safe_float(row.get("follower_return_5d"))
    volume_ratio = safe_float(row.get("follower_volume_ratio"))

    historical_score = (
        relationship_score * 0.40
        + test_success * 0.25
        + max(0, test_uplift) * 0.25
        + min(max(test_average, 0), 10) * 0.60
        + min(test_events, 12) * 0.30
    )

    score = (
        historical_score * 0.75
        + live_confirmation * 0.25
    )

    if -3 <= return_5d <= 4:
        score += 4
    elif 4 < return_5d <= 8:
        score += 1

    if 1.10 <= volume_ratio <= 2.20:
        score += 3

    if return_1d > 3:
        score -= 3

    if return_5d > 8:
        score -= 5

    return round(max(0, min(100, score)), 2)


def prediction_classification(row: pd.Series) -> str:
    prediction_score = safe_float(row.get("prediction_score"))
    live_class = str(
        row.get("live_confirmation_class", "")
    )

    if (
        prediction_score >= 70
        and live_class == "TEYÄ°TLÄ° TAKÄ°PÃÄ°"
    ):
        return "GÃÃLÃ TAKÄ°PÃÄ°"

    if prediction_score >= 58:
        return "ORTA TAKÄ°PÃÄ°"

    if prediction_score >= 48:
        return "Ä°ZLEME TAKÄ°PÃÄ°SÄ°"

    return "ZAYIF"


def create_performance_table(
    follower_symbols: List[str],
) -> pd.DataFrame:
    performance_rows = []
    total = len(follower_symbols)

    for number, follower in enumerate(
        follower_symbols,
        start=1,
    ):
        print(
            f"[{number}/{total}] "
            f"V10 takipÃ§i kontrolÃ¼: {follower}"
        )

        performance = calculate_recent_performance(follower)

        performance_rows.append({
            "follower": follower,
            "follower_data_valid": performance["data_valid"],
            "follower_price": performance["last_price"],
            "follower_return_1d": performance["return_1d"],
            "follower_return_5d": performance["return_5d"],
            "follower_return_20d": performance["return_20d"],
            "follower_volume_ratio": performance["volume_ratio"],
            "follower_rsi": performance["rsi"],
            "follower_ema20": performance["ema20"],
            "follower_ema20_distance": performance["ema20_distance"],
            "follower_ema20_slope_positive": (
                performance["ema20_slope_positive"]
            ),
            "follower_close_position": performance["close_position"],
        })

    return pd.DataFrame(performance_rows)


def build_predictions(
    leaders: pd.DataFrame,
    relationships: pd.DataFrame,
) -> pd.DataFrame:
    if leaders.empty or relationships.empty:
        return empty_output_dataframe()

    leader_symbols = (
        leaders["symbol"]
        .astype(str)
        .str.upper()
        .tolist()
    )

    candidates = relationships[
        relationships["leader"].isin(leader_symbols)
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
        print("Aktif liderlerle eÅleÅen tarihsel iliÅki yok.")
        return empty_output_dataframe()

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

    leader_info = leaders[leader_columns].copy()

    leader_info = leader_info.rename(
        columns={
            "symbol": "leader",
            "v8_score": "leader_v8_score",
            "smart_money_score": "leader_smart_money_score",
            "institutional_score": "leader_institutional_score",
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

    performance_df = create_performance_table(
        follower_symbols
    )

    if performance_df.empty:
        return empty_output_dataframe()

    candidates = candidates.merge(
        performance_df,
        on="follower",
        how="left",
    )

    candidates = candidates[
        candidates["follower_data_valid"].eq(True)
    ].copy()

    if candidates.empty:
        print("TakipÃ§ilerin gÃ¼ncel fiyat verisi alÄ±namadÄ±.")
        return empty_output_dataframe()

    candidates["not_extended"] = candidates.apply(
        follower_is_not_extended,
        axis=1,
    )

    candidates = candidates[
        candidates["not_extended"].eq(True)
    ].copy()

    if candidates.empty:
        print("BÃ¼tÃ¼n takipÃ§iler fazla yÃ¼kselmiÅ olduÄu iÃ§in elendi.")
        return empty_output_dataframe()

    confirmation_df = candidates.apply(
        calculate_live_confirmation,
        axis=1,
    )

    candidates = pd.concat(
        [
            candidates.reset_index(drop=True),
            confirmation_df.reset_index(drop=True),
        ],
        axis=1,
    )

    candidates = candidates[
        candidates.apply(
            live_confirmation_is_acceptable,
            axis=1,
        )
    ].copy()

    if candidates.empty:
        print(
            "CanlÄ± hacim, RSI ve EMA20 teyidini geÃ§en "
            "takipÃ§i bulunamadÄ±."
        )
        return empty_output_dataframe()

    candidates["prediction_score"] = candidates.apply(
        calculate_prediction_score,
        axis=1,
    )

    candidates["prediction_classification"] = candidates.apply(
        prediction_classification,
        axis=1,
    )

    candidates = candidates[
        candidates["prediction_score"] >= 48
    ].copy()

    if candidates.empty:
        print("Tahmin puanÄ± eÅiÄini geÃ§en takipÃ§i bulunamadÄ±.")
        return empty_output_dataframe()

    candidates = candidates.sort_values(
        by=[
            "prediction_score",
            "live_confirmation_score",
            "test_uplift",
            "test_success_rate",
            "relationship_score",
            "test_events",
        ],
        ascending=False,
    )

    candidates = (
        candidates.groupby(
            "leader",
            group_keys=False,
        )
        .head(MAX_FOLLOWERS_PER_LEADER)
        .reset_index(drop=True)
    )

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

    for column in OUTPUT_COLUMNS:
        if column not in candidates.columns:
            candidates[column] = np.nan

    return candidates[OUTPUT_COLUMNS]


def print_summary(
    predictions: pd.DataFrame,
    leaders: pd.DataFrame,
) -> None:
    print("\n====================================")
    print("V10 LEADER FOLLOW PREDICTION")
    print("====================================")

    print("Aktif lider:", len(leaders))

    if not leaders.empty:
        print(
            "Liderler:",
            ", ".join(
                leaders["symbol"]
                .astype(str)
                .tolist()
            ),
        )

    print("TakipÃ§i adayÄ±:", len(predictions))

    if predictions.empty:
        print(
            "CanlÄ± liderlerle eÅleÅen, fazla yÃ¼kselmemiÅ "
            "ve canlÄ± teyidi geÃ§en takipÃ§i bulunamadÄ±."
        )
        return

    display_columns = [
        "rank",
        "leader",
        "follower",
        "lag_days",
        "prediction_score",
        "prediction_classification",
        "live_confirmation_score",
        "live_confirmation_class",
        "test_events",
        "test_success_rate",
        "test_baseline_rate",
        "test_uplift",
        "test_average_return",
        "follower_price",
        "follower_return_1d",
        "follower_return_5d",
        "follower_volume_ratio",
        "follower_rsi",
        "follower_ema20_distance",
    ]

    print(
        predictions[display_columns]
        .to_string(index=False)
    )


def main() -> None:
    print("V10 takipÃ§i tahmin motoru baÅladÄ±.")

    leaders = load_v8_leaders()
    relationships = load_relationships()

    print("V8 lider sayÄ±sÄ±:", len(leaders))
    print("V9 iliÅki sayÄ±sÄ±:", len(relationships))

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

    print("\nKaydedildi:", OUTPUT_FILE)


if __name__ == "__main__":
