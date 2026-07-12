from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")

CANDIDATES_FILE = "v3_today_candidates.csv"
HISTORY_FILE = "v3_signals_history.csv"
SUMMARY_FILE = "v3_performance_summary.csv"

CHECK_DAYS = [1, 3, 5, 10]

FEATURE_COLUMNS = [
    "rank",
    "smart_money_score",
    "selection_score",
    "classification",
    "rsi",
    "atr_pct",
    "volume_ratio",
    "volume_accumulation_ratio",
    "up_down_volume_ratio",
    "range_20_pct",
    "close_position",
    "upper_wick_ratio",
    "distance_to_high_20",
    "ema20_distance",
    "return_1d",
    "return_5d_at_signal",
    "return_10d_at_signal",
    "return_20d_at_signal",
    "trend_score",
    "volume_score",
    "compression_score",
    "candle_score",
    "momentum_score",
    "liquidity_score",
    "risk_penalty",
    "positive_reasons",
    "risk_reasons",
]

RESULT_COLUMNS = [
    "result_1d",
    "result_3d",
    "result_5d",
    "result_10d",
    "max_result_5d",
    "min_result_5d",
    "max_result_10d",
    "min_result_10d",
    "hit_3pct_5d",
    "hit_5pct_10d",
    "status",
]

ALL_COLUMNS = [
    "signal_date",
    "recorded_at",
    "symbol",
    "signal_price",
    *FEATURE_COLUMNS,
    *RESULT_COLUMNS,
]


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_COLUMNS)


def clean_yahoo_dataframe(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe is None or dataframe.empty:
        return pd.DataFrame()

    result = dataframe.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    result = result.loc[
        :,
        ~result.columns.duplicated(),
    ].copy()

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    result = result[
        ~result.index.isna()
    ].sort_index()

    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    return result


def safe_float(
    value,
    default: float = np.nan,
) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        return empty_history()

    try:
        history = pd.read_csv(HISTORY_FILE)

    except Exception as exc:
        print("V3 geçmiş dosyası okunamadı:", exc)
        return empty_history()

    for column in ALL_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan

    return history[ALL_COLUMNS]


def read_today_candidates() -> pd.DataFrame:
    if not os.path.exists(CANDIDATES_FILE):
        print(
            f"{CANDIDATES_FILE} bulunamadı. "
            "Yeni sinyal eklenmeyecek."
        )
        return pd.DataFrame()

    try:
        candidates = pd.read_csv(CANDIDATES_FILE)

    except Exception as exc:
        print("V3 aday dosyası okunamadı:", exc)
        return pd.DataFrame()

    if candidates.empty:
        print("Bugün V3 adayı bulunmuyor.")
        return pd.DataFrame()

    required_columns = [
        "symbol",
        "close",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in candidates.columns
    ]

    if missing_columns:
        print(
            "Aday dosyasında eksik kolonlar:",
            missing_columns,
        )
        return pd.DataFrame()

    return candidates


def download_daily_prices(
    symbol: str,
    period: str = "6mo",
) -> pd.DataFrame:
    ticker = str(symbol).strip().upper()

    if not ticker.endswith(".IS"):
        ticker += ".IS"

    try:
        dataframe = yf.download(
            ticker,
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            actions=False,
            threads=False,
            timeout=25,
        )

    except Exception as exc:
        print(f"{symbol} fiyat indirme hatası:", exc)
        return pd.DataFrame()

    dataframe = clean_yahoo_dataframe(dataframe)

    if (
        dataframe.empty
        or "Close" not in dataframe.columns
    ):
        return pd.DataFrame()

    result = dataframe[["Close"]].copy()

    result["Close"] = pd.to_numeric(
        result["Close"],
        errors="coerce",
    )

    return result.dropna()


def determine_signal_date(
    symbol: str,
) -> str:
    """
    GitHub workflow gece veya hafta sonu çalışsa bile,
    sinyal tarihini son gerçek işlem günü olarak kaydeder.
    """
    prices = download_daily_prices(
        symbol,
        period="10d",
    )

    if prices.empty:
        return datetime.now(
            ISTANBUL_TZ
        ).strftime("%Y-%m-%d")

    return prices.index[-1].strftime("%Y-%m-%d")


def add_new_signals(
    history: pd.DataFrame,
) -> pd.DataFrame:
    candidates = read_today_candidates()

    if candidates.empty:
        return history

    recorded_at = datetime.now(
        ISTANBUL_TZ
    ).strftime("%Y-%m-%d %H:%M:%S")

    new_rows = []

    for _, candidate in candidates.iterrows():
        symbol = str(
            candidate.get("symbol", "")
        ).strip().upper()

        if not symbol:
            continue

        signal_price = safe_float(
            candidate.get("close")
        )

        if (
            pd.isna(signal_price)
            or signal_price <= 0
        ):
            print(
                symbol,
                "için geçerli sinyal fiyatı yok.",
            )
            continue

        signal_date = determine_signal_date(symbol)

        duplicate = (
            history["signal_date"]
            .astype(str)
            .eq(signal_date)
            &
            history["symbol"]
            .astype(str)
            .str.upper()
            .eq(symbol)
        )

        if duplicate.any():
            print(
                f"{symbol} | {signal_date} "
                "zaten kayıtlı."
            )
            continue

        row = {
            "signal_date": signal_date,
            "recorded_at": recorded_at,
            "symbol": symbol,
            "signal_price": round(
                signal_price,
                4,
            ),
            "result_1d": np.nan,
            "result_3d": np.nan,
            "result_5d": np.nan,
            "result_10d": np.nan,
            "max_result_5d": np.nan,
            "min_result_5d": np.nan,
            "max_result_10d": np.nan,
            "min_result_10d": np.nan,
            "hit_3pct_5d": np.nan,
            "hit_5pct_10d": np.nan,
            "status": "bekliyor",
        }

        for feature in FEATURE_COLUMNS:
            if feature == "return_5d_at_signal":
                row[feature] = candidate.get(
                    "return_5d",
                    np.nan,
                )

            elif feature == "return_10d_at_signal":
                row[feature] = candidate.get(
                    "return_10d",
                    np.nan,
                )

            elif feature == "return_20d_at_signal":
                row[feature] = candidate.get(
                    "return_20d",
                    np.nan,
                )

            else:
                row[feature] = candidate.get(
                    feature,
                    np.nan,
                )

        new_rows.append(row)

        print(
            "Yeni V3 sinyali kaydedildi:",
            symbol,
            "| Tarih:",
            signal_date,
            "| Fiyat:",
            round(signal_price, 2),
            "| Smart Money:",
            row.get("smart_money_score"),
        )

    if not new_rows:
        return history

    return pd.concat(
        [
            history,
            pd.DataFrame(new_rows),
        ],
        ignore_index=True,
    )


def calculate_return(
    future_prices: pd.Series,
    signal_price: float,
    trading_day: int,
) -> float:
    if len(future_prices) < trading_day:
        return np.nan

    future_price = safe_float(
        future_prices.iloc[trading_day - 1]
    )

    if (
        pd.isna(future_price)
        or signal_price <= 0
    ):
        return np.nan

    return round(
        (
            future_price / signal_price
            - 1
        ) * 100,
        2,
    )


def calculate_window_extremes(
    future_prices: pd.Series,
    signal_price: float,
    window: int,
) -> tuple[float, float]:
    selected_prices = future_prices.head(window)

    if selected_prices.empty:
        return np.nan, np.nan

    maximum_price = safe_float(
        selected_prices.max()
    )

    minimum_price = safe_float(
        selected_prices.min()
    )

    maximum_return = (
        maximum_price / signal_price
        - 1
    ) * 100

    minimum_return = (
        minimum_price / signal_price
        - 1
    ) * 100

    return (
        round(maximum_return, 2),
        round(minimum_return, 2),
    )


def update_old_signals(
    history: pd.DataFrame,
) -> pd.DataFrame:
    if history.empty:
        return history

    symbols = (
        history["symbol"]
        .dropna()
        .astype(str)
        .str.upper()
        .unique()
        .tolist()
    )

    price_cache = {}

    for symbol in symbols:
        price_cache[symbol] = (
            download_daily_prices(
                symbol,
                period="6mo",
            )
        )

    for index, row in history.iterrows():
        symbol = str(
            row.get("symbol", "")
        ).strip().upper()

        signal_date = pd.to_datetime(
            row.get("signal_date"),
            errors="coerce",
        )

        signal_price = safe_float(
            row.get("signal_price")
        )

        if (
            not symbol
            or pd.isna(signal_date)
            or pd.isna(signal_price)
            or signal_price <= 0
        ):
            continue

        prices = price_cache.get(
            symbol,
            pd.DataFrame(),
        )

        if prices.empty:
            continue

        future_prices = prices.loc[
            prices.index.normalize()
            > signal_date.normalize(),
            "Close",
        ].astype(float)

        if future_prices.empty:
            continue

        for trading_day in CHECK_DAYS:
            column = f"result_{trading_day}d"

            result = calculate_return(
                future_prices,
                signal_price,
                trading_day,
            )

            if not pd.isna(result):
                history.at[
                    index,
                    column,
                ] = result

        (
            max_result_5d,
            min_result_5d,
        ) = calculate_window_extremes(
            future_prices,
            signal_price,
            5,
        )

        (
            max_result_10d,
            min_result_10d,
        ) = calculate_window_extremes(
            future_prices,
            signal_price,
            10,
        )

        history.at[
            index,
            "max_result_5d",
        ] = max_result_5d

        history.at[
            index,
            "min_result_5d",
        ] = min_result_5d

        history.at[
            index,
            "max_result_10d",
        ] = max_result_10d

        history.at[
            index,
            "min_result_10d",
        ] = min_result_10d

        if not pd.isna(max_result_5d):
            history.at[
                index,
                "hit_3pct_5d",
            ] = int(max_result_5d >= 3)

        if not pd.isna(max_result_10d):
            history.at[
                index,
                "hit_5pct_10d",
            ] = int(max_result_10d >= 5)

        available_days = len(future_prices)

        if available_days >= 10:
            status = "tamamlandı"

        elif available_days >= 5:
            status = "5g_güncellendi"

        elif available_days >= 3:
            status = "3g_güncellendi"

        elif available_days >= 1:
            status = "1g_güncellendi"

        else:
            status = "bekliyor"

        history.at[
            index,
            "status",
        ] = status

    return history


def create_performance_summary(
    history: pd.DataFrame,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame()

    working = history.copy()

    numeric_columns = [
        "smart_money_score",
        "selection_score",
        "result_1d",
        "result_3d",
        "result_5d",
        "result_10d",
        "max_result_5d",
        "max_result_10d",
    ]

    for column in numeric_columns:
        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    completed_5d = working[
        working["result_5d"].notna()
    ].copy()

    summary_rows = []

    summary_rows.append({
        "group": "GENEL",
        "signal_count": len(working),
        "completed_5d": len(completed_5d),
        "positive_5d_pct": (
            round(
                (
                    completed_5d["result_5d"] > 0
                ).mean() * 100,
                2,
            )
            if not completed_5d.empty
            else np.nan
        ),
        "hit_3pct_5d_pct": (
            round(
                (
                    completed_5d["max_result_5d"] >= 3
                ).mean() * 100,
                2,
            )
            if not completed_5d.empty
            else np.nan
        ),
        "average_result_5d": (
            round(
                completed_5d["result_5d"].mean(),
                2,
            )
            if not completed_5d.empty
            else np.nan
        ),
        "median_result_5d": (
            round(
                completed_5d["result_5d"].median(),
                2,
            )
            if not completed_5d.empty
            else np.nan
        ),
    })

    score_groups = [
        (
            "SMART_65_74",
            65,
            75,
        ),
        (
            "SMART_75_84",
            75,
            85,
        ),
        (
            "SMART_85_PLUS",
            85,
            101,
        ),
    ]

    for group_name, lower, upper in score_groups:
        group = completed_5d[
            (
                completed_5d["smart_money_score"]
                >= lower
            )
            &
            (
                completed_5d["smart_money_score"]
                < upper
            )
        ]

        if group.empty:
            continue

        summary_rows.append({
            "group": group_name,
            "signal_count": len(group),
            "completed_5d": len(group),
            "positive_5d_pct": round(
                (
                    group["result_5d"] > 0
                ).mean() * 100,
                2,
            ),
            "hit_3pct_5d_pct": round(
                (
                    group["max_result_5d"] >= 3
                ).mean() * 100,
                2,
            ),
            "average_result_5d": round(
                group["result_5d"].mean(),
                2,
            ),
            "median_result_5d": round(
                group["result_5d"].median(),
                2,
            ),
        })

    return pd.DataFrame(summary_rows)


def print_summary(
    history: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    print("\n====================================")
    print("V3 PERFORMANS HAFIZASI")
    print("====================================")
    print("Toplam sinyal:", len(history))

    completed_1d = int(
        history["result_1d"].notna().sum()
    )

    completed_5d = int(
        history["result_5d"].notna().sum()
    )

    completed_10d = int(
        history["result_10d"].notna().sum()
    )

    print("1 günlük sonucu oluşan:", completed_1d)
    print("5 günlük sonucu oluşan:", completed_5d)
    print("10 günlük sonucu oluşan:", completed_10d)

    if summary.empty:
        print("Henüz özet oluşturacak veri yok.")
        return

    print("\nPERFORMANS ÖZETİ")
    print(summary.to_string(index=False))


def main():
    print("V3 performans sistemi başladı.")

    history = load_history()

    history = update_old_signals(history)
    history = add_new_signals(history)

    history = history.drop_duplicates(
        subset=[
            "signal_date",
            "symbol",
        ],
        keep="last",
    )

    history = history.sort_values(
        by=[
            "signal_date",
            "rank",
        ],
        ascending=[
            False,
            True,
        ],
    )

    for column in ALL_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan

    history = history[ALL_COLUMNS]

    history.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = create_performance_summary(
        history
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        history,
        summary,
    )

    print(
        "\nKaydedildi:",
        HISTORY_FILE,
    )

    print(
        "Özet kaydedildi:",
        SUMMARY_FILE,
    )


if __name__ == "__main__":
    main()
