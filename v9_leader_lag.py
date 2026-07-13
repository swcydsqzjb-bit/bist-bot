from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from v3_data import (
    download_daily_data,
    get_bist_symbols,
    symbol_without_suffix,
)


OUTPUT_FILE = "v9_leader_lag_results.csv"

LEADER_EVENT_RETURN = 5.0
FOLLOWER_SUCCESS_RETURN = 3.0

MIN_TRAIN_EVENTS = 8
MIN_TEST_EVENTS = 4

MIN_TEST_SUCCESS_RATE = 45.0
MIN_TEST_UPLIFT = 12.0

MAX_LAG_DAYS = 5
TOP_RESULT_COUNT = 30


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_symbol_limit() -> int:
    """
    V9_SYMBOL_LIMIT=50 -> ilk 50 sembol
    V9_SYMBOL_LIMIT=0  -> bütün semboller
    """
    return max(
        0,
        env_int("V9_SYMBOL_LIMIT", 50),
    )


def safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def download_close_prices(
    symbols: List[str],
) -> pd.DataFrame:
    """
    Sembollerin 3 yıllık günlük kapanışlarını tek tabloda toplar.
    """
    close_series: Dict[str, pd.Series] = {}

    total = len(symbols)

    for number, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{number}/{total}] "
            f"V9 geçmiş veri: {symbol}"
        )

        try:
            dataframe = download_daily_data(
                symbol=symbol,
                period="3y",
                interval="1d",
                retries=2,
            )

            if dataframe.empty or len(dataframe) < 180:
                continue

            clean_symbol = symbol_without_suffix(symbol)

            series = pd.to_numeric(
                dataframe["Close"],
                errors="coerce",
            ).dropna()

            if len(series) < 180:
                continue

            close_series[clean_symbol] = series

        except Exception as exc:
            print(
                f"{symbol} veri hatası: {exc}"
            )

        time.sleep(0.03)

    if not close_series:
        return pd.DataFrame()

    prices = pd.concat(
        close_series,
        axis=1,
        join="outer",
    )

    prices = prices.sort_index()

    # Çok kısa veya aşırı eksik serileri çıkar.
    valid_columns = []

    minimum_required = int(
        len(prices) * 0.65
    )

    for column in prices.columns:
        if prices[column].notna().sum() >= minimum_required:
            valid_columns.append(column)

    prices = prices[valid_columns]

    return prices


def calculate_daily_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    return (
        prices.pct_change(
            fill_method=None
        ) * 100
    )


def split_train_test(
    returns: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Son yaklaşık 1 yılı test dönemi,
    daha eski kısmı eğitim dönemi olarak ayırır.
    """
    if returns.empty:
        return pd.DataFrame(), pd.DataFrame()

    split_position = max(
        1,
        len(returns) - 252,
    )

    train = returns.iloc[
        :split_position
    ].copy()

    test = returns.iloc[
        split_position:
    ].copy()

    return train, test


def calculate_pair_statistics(
    returns: pd.DataFrame,
    leader: str,
    follower: str,
    lag_days: int,
) -> Dict[str, float]:
    """
    Liderin olay günleri sonrasında takipçinin
    belirtilen gecikmedeki performansını hesaplar.
    """
    if leader == follower:
        return {}

    leader_returns = returns[leader]
    follower_returns = returns[follower]

    leader_events = (
        leader_returns >= LEADER_EVENT_RETURN
    )

    # Liderin bugünkü olayı, takipçinin lag_days
    # işlem günü sonraki getirisiyle eşleştirilir.
    future_follower_returns = (
        follower_returns.shift(
            -lag_days
        )
    )

    valid_event_mask = (
        leader_events
        & future_follower_returns.notna()
    )

    event_returns = future_follower_returns[
        valid_event_mask
    ]

    event_count = len(event_returns)

    if event_count == 0:
        return {
            "event_count": 0,
            "success_rate": np.nan,
            "average_return": np.nan,
            "median_return": np.nan,
            "baseline_rate": np.nan,
            "uplift": np.nan,
        }

    success_rate = (
        event_returns >= FOLLOWER_SUCCESS_RETURN
    ).mean() * 100

    # Takipçinin bütün uygun günlerde normal başarı oranı.
    baseline_returns = future_follower_returns.dropna()

    baseline_rate = (
        baseline_returns >= FOLLOWER_SUCCESS_RETURN
    ).mean() * 100

    uplift = success_rate - baseline_rate

    return {
        "event_count": event_count,
        "success_rate": round(
            success_rate,
            2,
        ),
        "average_return": round(
            event_returns.mean(),
            2,
        ),
        "median_return": round(
            event_returns.median(),
            2,
        ),
        "baseline_rate": round(
            baseline_rate,
            2,
        ),
        "uplift": round(
            uplift,
            2,
        ),
    }


def relationship_score(
    train_stats: Dict[str, float],
    test_stats: Dict[str, float],
) -> float:
    """
    Son dönem doğrulamasına daha fazla ağırlık verir.
    """
    train_rate = safe_float(
        train_stats.get("success_rate")
    )

    test_rate = safe_float(
        test_stats.get("success_rate")
    )

    train_uplift = safe_float(
        train_stats.get("uplift")
    )

    test_uplift = safe_float(
        test_stats.get("uplift")
    )

    test_average = safe_float(
        test_stats.get("average_return")
    )

    test_events = safe_float(
        test_stats.get("event_count")
    )

    score = (
        train_rate * 0.20
        + test_rate * 0.35
        + max(0, train_uplift) * 0.15
        + max(0, test_uplift) * 0.20
        + max(0, test_average) * 1.50
        + min(test_events, 12) * 0.50
    )

    return round(
        max(0, min(100, score)),
        2,
    )


def analyze_relationships(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    if returns.empty:
        return pd.DataFrame()

    train_returns, test_returns = (
        split_train_test(returns)
    )

    symbols = returns.columns.tolist()

    rows = []

    total_leaders = len(symbols)

    for leader_number, leader in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{leader_number}/{total_leaders}] "
            f"Lider analiz ediliyor: {leader}"
        )

        for follower in symbols:
            if leader == follower:
                continue

            for lag_days in range(
                1,
                MAX_LAG_DAYS + 1,
            ):
                train_stats = (
                    calculate_pair_statistics(
                        train_returns,
                        leader,
                        follower,
                        lag_days,
                    )
                )

                test_stats = (
                    calculate_pair_statistics(
                        test_returns,
                        leader,
                        follower,
                        lag_days,
                    )
                )

                train_events = int(
                    train_stats.get(
                        "event_count",
                        0,
                    )
                )

                test_events = int(
                    test_stats.get(
                        "event_count",
                        0,
                    )
                )

                if train_events < MIN_TRAIN_EVENTS:
                    continue

                if test_events < MIN_TEST_EVENTS:
                    continue

                test_success = safe_float(
                    test_stats.get(
                        "success_rate"
                    )
                )

                test_uplift = safe_float(
                    test_stats.get(
                        "uplift"
                    )
                )

                if (
                    test_success
                    < MIN_TEST_SUCCESS_RATE
                ):
                    continue

                if test_uplift < MIN_TEST_UPLIFT:
                    continue

                score = relationship_score(
                    train_stats,
                    test_stats,
                )

                rows.append({
                    "leader": leader,
                    "follower": follower,
                    "lag_days": lag_days,

                    "train_events": train_events,
                    "train_success_rate": (
                        train_stats[
                            "success_rate"
                        ]
                    ),
                    "train_average_return": (
                        train_stats[
                            "average_return"
                        ]
                    ),
                    "train_baseline_rate": (
                        train_stats[
                            "baseline_rate"
                        ]
                    ),
                    "train_uplift": (
                        train_stats["uplift"]
                    ),

                    "test_events": test_events,
                    "test_success_rate": (
                        test_stats[
                            "success_rate"
                        ]
                    ),
                    "test_average_return": (
                        test_stats[
                            "average_return"
                        ]
                    ),
                    "test_median_return": (
                        test_stats[
                            "median_return"
                        ]
                    ),
                    "test_baseline_rate": (
                        test_stats[
                            "baseline_rate"
                        ]
                    ),
                    "test_uplift": (
                        test_stats["uplift"]
                    ),

                    "relationship_score": score,
                })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    result = result.sort_values(
        by=[
            "relationship_score",
            "test_uplift",
            "test_success_rate",
            "test_events",
        ],
        ascending=False,
    )

    # Aynı lider-takipçi çifti için en güçlü gecikme kalsın.
    result = result.drop_duplicates(
        subset=[
            "leader",
            "follower",
        ],
        keep="first",
    )

    return result.reset_index(drop=True)


def print_summary(
    result: pd.DataFrame,
    symbol_count: int,
) -> None:
    print("\n====================================")
    print("V9 LİDER–TAKİPÇİ MOTORU")
    print("====================================")
    print("Analiz edilen sembol:", symbol_count)

    if result.empty:
        print(
            "Doğrulama şartlarını geçen "
            "ilişki bulunamadı."
        )
        return

    print(
        "Doğrulanmış ilişki:",
        len(result),
    )

    display_columns = [
        "leader",
        "follower",
        "lag_days",
        "relationship_score",
        "train_events",
        "train_success_rate",
        "test_events",
        "test_success_rate",
        "test_baseline_rate",
        "test_uplift",
        "test_average_return",
    ]

    print("\nEN GÜÇLÜ İLİŞKİLER")

    print(
        result[
            display_columns
        ]
        .head(TOP_RESULT_COUNT)
        .to_string(index=False)
    )


def main():
    print("V9 lider–takipçi analizi başladı.")

    symbols = get_bist_symbols()

    limit = get_symbol_limit()

    if limit > 0:
        symbols = symbols[:limit]

    print("İstenen sembol:", len(symbols))

    prices = download_close_prices(
        symbols
    )

    if prices.empty:
        print("Fiyat matrisi oluşturulamadı.")

        pd.DataFrame().to_csv(
            OUTPUT_FILE,
            index=False,
        )

        return

    print(
        "Yeterli verisi gelen:",
        len(prices.columns),
    )

    returns = calculate_daily_returns(
        prices
    )

    result = analyze_relationships(
        returns
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(
        result,
        symbol_count=len(prices.columns),
    )

    print(
        "\nKaydedildi:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
