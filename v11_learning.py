from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

from v3_data import download_daily_data


V8_FILE = "v8_today_candidates.csv"
V10_FILE = "v10_follow_predictions.csv"

HISTORY_FILE = "v11_signal_history.csv"
SUMMARY_FILE = "v11_learning_summary.csv"
WEIGHTS_FILE = "v11_learned_weights.csv"

ISTANBUL = ZoneInfo("Europe/Istanbul")

HORIZONS = {
    "result_1d": 1,
    "result_3d": 3,
    "result_5d": 5,
    "result_10d": 10,
}

MIN_COMPLETED_SIGNALS_FOR_LEARNING = 30

HISTORY_COLUMNS = [
    "signal_id",
    "signal_date",
    "recorded_at",
    "source",
    "symbol",
    "signal_price",
    "rank",
    "classification",
    "v8_score",
    "smart_money_score",
    "institutional_score",
    "historical_support_score",
    "prediction_score",
    "live_confirmation_score",
    "relationship_score",
    "rsi",
    "volume_ratio",
    "ema20_distance",
    "return_1d_at_signal",
    "return_5d_at_signal",
    "result_1d",
    "result_3d",
    "result_5d",
    "result_10d",
    "max_result_5d",
    "min_result_5d",
    "hit_3pct_5d",
    "hit_5pct_10d",
    "status",
]


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    except Exception as exc:
        print(f"{path} okunamadi:", exc)
        return pd.DataFrame()


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def normalize_symbol(value: Any) -> str:
    symbol = str(value).strip().upper()
    return symbol.replace(".IS", "")


def today_str() -> str:
    return datetime.now(ISTANBUL).date().isoformat()


def now_str() -> str:
    return datetime.now(ISTANBUL).isoformat(timespec="seconds")


def make_signal_id(signal_date: str, source: str, symbol: str) -> str:
    return f"{signal_date}|{source}|{symbol}"


def get_first_existing(
    row: pd.Series,
    names: List[str],
    default: Any = np.nan,
) -> Any:
    for name in names:
        if name in row.index and pd.notna(row.get(name)):
            return row.get(name)
    return default


def collect_v8_signals() -> pd.DataFrame:
    df = load_csv(V8_FILE)

    if df.empty or "symbol" not in df.columns:
        return empty_history()

    rows: List[Dict[str, Any]] = []
    signal_date = today_str()
    recorded_at = now_str()

    for _, row in df.iterrows():
        symbol = normalize_symbol(row.get("symbol"))
        if not symbol:
            continue

        price = safe_float(
            get_first_existing(
                row,
                ["close", "price", "signal_price"],
            )
        )

        rows.append({
            "signal_id": make_signal_id(signal_date, "V8", symbol),
            "signal_date": signal_date,
            "recorded_at": recorded_at,
            "source": "V8",
            "symbol": symbol,
            "signal_price": price,
            "rank": safe_float(row.get("rank")),
            "classification": str(
                get_first_existing(
                    row,
                    ["v8_classification", "classification"],
                    "",
                )
            ),
            "v8_score": safe_float(row.get("v8_score")),
            "smart_money_score": safe_float(row.get("smart_money_score")),
            "institutional_score": safe_float(row.get("institutional_score")),
            "historical_support_score": safe_float(
                row.get("historical_support_score")
            ),
            "prediction_score": np.nan,
            "live_confirmation_score": np.nan,
            "relationship_score": np.nan,
            "rsi": safe_float(row.get("rsi")),
            "volume_ratio": safe_float(
                get_first_existing(
                    row,
                    ["volume_ratio", "follower_volume_ratio"],
                )
            ),
            "ema20_distance": safe_float(
                get_first_existing(
                    row,
                    ["ema20_distance", "ema20_dist"],
                )
            ),
            "return_1d_at_signal": safe_float(
                get_first_existing(
                    row,
                    ["return_1d", "return_1d_at_signal"],
                )
            ),
            "return_5d_at_signal": safe_float(
                get_first_existing(
                    row,
                    ["return_5d", "return_5d_at_signal"],
                )
            ),
            "result_1d": np.nan,
            "result_3d": np.nan,
            "result_5d": np.nan,
            "result_10d": np.nan,
            "max_result_5d": np.nan,
            "min_result_5d": np.nan,
            "hit_3pct_5d": np.nan,
            "hit_5pct_10d": np.nan,
            "status": "bekliyor",
        })

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def collect_v10_signals() -> pd.DataFrame:
    df = load_csv(V10_FILE)

    if df.empty or "follower" not in df.columns:
        return empty_history()

    rows: List[Dict[str, Any]] = []
    signal_date = today_str()
    recorded_at = now_str()

    for _, row in df.iterrows():
        symbol = normalize_symbol(row.get("follower"))
        if not symbol:
            continue

        rows.append({
            "signal_id": make_signal_id(signal_date, "V10", symbol),
            "signal_date": signal_date,
            "recorded_at": recorded_at,
            "source": "V10",
            "symbol": symbol,
            "signal_price": safe_float(row.get("follower_price")),
            "rank": safe_float(row.get("rank")),
            "classification": str(
                row.get("prediction_classification", "")
            ),
            "v8_score": safe_float(row.get("leader_v8_score")),
            "smart_money_score": safe_float(
                row.get("leader_smart_money_score")
            ),
            "institutional_score": safe_float(
                row.get("leader_institutional_score")
            ),
            "historical_support_score": np.nan,
            "prediction_score": safe_float(row.get("prediction_score")),
            "live_confirmation_score": safe_float(
                row.get("live_confirmation_score")
            ),
            "relationship_score": safe_float(row.get("relationship_score")),
            "rsi": safe_float(row.get("follower_rsi")),
            "volume_ratio": safe_float(row.get("follower_volume_ratio")),
            "ema20_distance": safe_float(
                row.get("follower_ema20_distance")
            ),
            "return_1d_at_signal": safe_float(
                row.get("follower_return_1d")
            ),
            "return_5d_at_signal": safe_float(
                row.get("follower_return_5d")
            ),
            "result_1d": np.nan,
            "result_3d": np.nan,
            "result_5d": np.nan,
            "result_10d": np.nan,
            "max_result_5d": np.nan,
            "min_result_5d": np.nan,
            "hit_3pct_5d": np.nan,
            "hit_5pct_10d": np.nan,
            "status": "bekliyor",
        })

    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def append_new_signals(
    history: pd.DataFrame,
    new_signals: pd.DataFrame,
) -> pd.DataFrame:
    if history.empty:
        history = empty_history()

    if new_signals.empty:
        return history

    known_ids = set(
        history.get("signal_id", pd.Series(dtype=str)).astype(str)
    )

    new_signals = new_signals[
        ~new_signals["signal_id"].astype(str).isin(known_ids)
    ].copy()

    if new_signals.empty:
        return history

    print("Yeni sinyal kaydi:", len(new_signals))

    combined = pd.concat(
        [history, new_signals],
        ignore_index=True,
    )

    for column in HISTORY_COLUMNS:
        if column not in combined.columns:
            combined[column] = np.nan

    return combined[HISTORY_COLUMNS]


def download_history_for_symbol(symbol: str) -> pd.DataFrame:
    try:
        df = download_daily_data(
            symbol=symbol,
            period="2y",
            interval="1d",
            retries=1,
        )
    except Exception as exc:
        print(f"{symbol} fiyat gecmisi hatasi:", exc)
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return pd.DataFrame()

    df = df.sort_index().copy()

    try:
        df.index = df.index.tz_localize(None)
    except TypeError:
        pass

    return df


def trading_row_after_signal(
    prices: pd.DataFrame,
    signal_date: pd.Timestamp,
) -> int | None:
    valid_positions = np.where(
        prices.index.normalize() >= signal_date.normalize()
    )[0]

    if len(valid_positions) == 0:
        return None

    return int(valid_positions[0])


def calculate_return(
    signal_price: float,
    target_price: float,
) -> float:
    if (
        pd.isna(signal_price)
        or pd.isna(target_price)
        or signal_price <= 0
    ):
        return np.nan

    return round(
        ((target_price / signal_price) - 1) * 100,
        2,
    )


def update_signal_results(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return history

    symbols = (
        history["symbol"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    price_cache: Dict[str, pd.DataFrame] = {}

    for number, symbol in enumerate(symbols, start=1):
        print(
            f"[{number}/{len(symbols)}] "
            f"V11 sonuc guncelleme: {symbol}"
        )
        price_cache[symbol] = download_history_for_symbol(symbol)

    for index, row in history.iterrows():
        symbol = normalize_symbol(row.get("symbol"))
        prices = price_cache.get(symbol, pd.DataFrame())

        if prices.empty or "Close" not in prices.columns:
            continue

        signal_price = safe_float(row.get("signal_price"))

        if pd.isna(signal_price) or signal_price <= 0:
            continue

        try:
            signal_date = pd.Timestamp(row.get("signal_date"))
        except Exception:
            continue

        base_position = trading_row_after_signal(
            prices,
            signal_date,
        )

        if base_position is None:
            continue

        close = pd.to_numeric(prices["Close"], errors="coerce")
        high = pd.to_numeric(
            prices.get("High", prices["Close"]),
            errors="coerce",
        )
        low = pd.to_numeric(
            prices.get("Low", prices["Close"]),
            errors="coerce",
        )

        for column, horizon in HORIZONS.items():
            target_position = base_position + horizon

            if target_position >= len(prices):
                continue

            target_price = safe_float(close.iloc[target_position])
            history.at[index, column] = calculate_return(
                signal_price,
                target_price,
            )

        five_day_end = base_position + 5

        if five_day_end < len(prices):
            max_price = safe_float(
                high.iloc[
                    base_position + 1:
                    five_day_end + 1
                ].max()
            )
            min_price = safe_float(
                low.iloc[
                    base_position + 1:
                    five_day_end + 1
                ].min()
            )

            history.at[index, "max_result_5d"] = calculate_return(
                signal_price,
                max_price,
            )
            history.at[index, "min_result_5d"] = calculate_return(
                signal_price,
                min_price,
            )

            max_result = safe_float(
                history.at[index, "max_result_5d"]
            )

            history.at[index, "hit_3pct_5d"] = int(
                not pd.isna(max_result)
                and max_result >= 3
            )

        ten_day_end = base_position + 10

        if ten_day_end < len(prices):
            max_price_10 = safe_float(
                high.iloc[
                    base_position + 1:
                    ten_day_end + 1
                ].max()
            )
            max_result_10 = calculate_return(
                signal_price,
                max_price_10,
            )

            history.at[index, "hit_5pct_10d"] = int(
                not pd.isna(max_result_10)
                and max_result_10 >= 5
            )

        if pd.notna(history.at[index, "result_10d"]):
            history.at[index, "status"] = "tamamlandi"
        elif pd.notna(history.at[index, "result_5d"]):
            history.at[index, "status"] = "5g_tamamlandi"
        elif pd.notna(history.at[index, "result_1d"]):
            history.at[index, "status"] = "1g_tamamlandi"
        else:
            history.at[index, "status"] = "bekliyor"

    return history


def build_group_summary(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    completed = history[
        history["result_5d"].notna()
    ].copy()

    if completed.empty:
        return pd.DataFrame([
            {
                "group": "GENEL",
                "signal_count": len(history),
                "completed_5d": 0,
                "positive_5d_pct": np.nan,
                "hit_3pct_5d_pct": np.nan,
                "average_result_5d": np.nan,
                "median_result_5d": np.nan,
            }
        ])

    rows: List[Dict[str, Any]] = []
    groups = {"GENEL": completed}

    for source in sorted(
        completed["source"].dropna().astype(str).unique()
    ):
        groups[f"KAYNAK_{source}"] = completed[
            completed["source"] == source
        ]

    for group_name, group_df in groups.items():
        result_5d = pd.to_numeric(
            group_df["result_5d"],
            errors="coerce",
        ).dropna()

        hit_3pct = pd.to_numeric(
            group_df["hit_3pct_5d"],
            errors="coerce",
        ).dropna()

        rows.append({
            "group": group_name,
            "signal_count": len(group_df),
            "completed_5d": len(result_5d),
            "positive_5d_pct": round(
                float((result_5d > 0).mean() * 100),
                2,
            ),
            "hit_3pct_5d_pct": (
                round(float(hit_3pct.mean() * 100), 2)
                if not hit_3pct.empty
                else np.nan
            ),
            "average_result_5d": round(
                float(result_5d.mean()),
                2,
            ),
            "median_result_5d": round(
                float(result_5d.median()),
                2,
            ),
        })

    return pd.DataFrame(rows)


def learn_simple_weights(history: pd.DataFrame) -> pd.DataFrame:
    completed = history[
        history["result_5d"].notna()
    ].copy()

    features = [
        "v8_score",
        "smart_money_score",
        "institutional_score",
        "historical_support_score",
        "prediction_score",
        "live_confirmation_score",
        "relationship_score",
        "rsi",
        "volume_ratio",
        "ema20_distance",
    ]

    if len(completed) < MIN_COMPLETED_SIGNALS_FOR_LEARNING:
        return pd.DataFrame([
            {
                "feature": "VERI_YETERSIZ",
                "sample_count": len(completed),
                "correlation_with_5d": np.nan,
                "normalized_weight": np.nan,
                "message": (
                    "Ogrenme icin en az "
                    f"{MIN_COMPLETED_SIGNALS_FOR_LEARNING} "
                    "tamamlanmis 5 gunluk sinyal gerekli."
                ),
            }
        ])

    rows: List[Dict[str, Any]] = []

    for feature in features:
        if feature not in completed.columns:
            continue

        x = pd.to_numeric(
            completed[feature],
            errors="coerce",
        )
        y = pd.to_numeric(
            completed["result_5d"],
            errors="coerce",
        )

        valid = x.notna() & y.notna()

        if valid.sum() < 8:
            continue

        correlation = float(x[valid].corr(y[valid]))

        if np.isnan(correlation):
            continue

        rows.append({
            "feature": feature,
            "sample_count": int(valid.sum()),
            "correlation_with_5d": round(correlation, 4),
            "absolute_signal": abs(correlation),
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    total_signal = result["absolute_signal"].sum()

    result["normalized_weight"] = (
        result["absolute_signal"] / total_signal * 100
        if total_signal > 0
        else 0
    )

    result["normalized_weight"] = (
        result["normalized_weight"].round(2)
    )

    result["direction"] = np.where(
        result["correlation_with_5d"] >= 0,
        "pozitif",
        "negatif",
    )

    result["message"] = (
        "Ilk asama korelasyon agirligi; "
        "otomatik al-sat karari degildir."
    )

    return result.sort_values(
        by="normalized_weight",
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    print("V11 performans ve ogrenme motoru basladi.")

    history = load_csv(HISTORY_FILE)

    if history.empty:
        history = empty_history()

    history = append_new_signals(
        history,
        collect_v8_signals(),
    )
    history = append_new_signals(
        history,
        collect_v10_signals(),
    )

    history = update_signal_results(history)
    summary = build_group_summary(history)
    weights = learn_simple_weights(history)

    history.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )
    weights.to_csv(
        WEIGHTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n==============================")
    print("V11 OGRENME OZETI")
    print("==============================")
    print("Toplam sinyal:", len(history))
    print(
        "5 gun sonucu tamamlanan:",
        int(history["result_5d"].notna().sum()),
    )

    if not summary.empty:
        print("\nPerformans ozeti:")
        print(summary.to_string(index=False))

    if not weights.empty:
        print("\nOgrenilen ilk agirliklar:")
        print(weights.to_string(index=False))

    print("\nKaydedildi:")
    print("-", HISTORY_FILE)
    print("-", SUMMARY_FILE)
    print("-", WEIGHTS_FILE)


if __name__ == "__main__":
    main()
