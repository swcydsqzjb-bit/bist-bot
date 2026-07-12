from __future__ import annotations

import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from v3_config import (
    MAX_DAILY_CANDIDATES,
    MIN_AVG_TURNOVER_TL,
    MIN_DAILY_BARS,
)
from v3_data import (
    calculate_average_turnover,
    download_daily_data,
    get_bist_symbols,
    symbol_without_suffix,
)
from v3_scanner import (
    calculate_candidate_rank,
    select_daily_candidates,
)
from v3_scoring import score_market


OUTPUT_FILE = "v5_backfill_history.csv"

# Her kaç işlem gününde bir geçmiş tarama yapılacağı.
# 10 yaklaşık iki haftada bir örnek üretir.
DEFAULT_STEP_DAYS = 10


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def get_settings() -> tuple[int, int]:
    """
    V5_SYMBOL_LIMIT=50  -> İlk 50 hisse
    V5_SYMBOL_LIMIT=0   -> Bütün BIST

    V5_STEP_DAYS=10     -> Her 10 işlem gününde bir tarama
    """
    symbol_limit = max(
        0,
        env_int("V5_SYMBOL_LIMIT", 50),
    )

    step_days = max(
        5,
        env_int(
            "V5_STEP_DAYS",
            DEFAULT_STEP_DAYS,
        ),
    )

    return symbol_limit, step_days


def download_market_history(
    symbols: List[str],
) -> Dict[str, pd.DataFrame]:
    market_data: Dict[str, pd.DataFrame] = {}

    for number, symbol in enumerate(
        symbols,
        start=1,
    ):
        print(
            f"[{number}/{len(symbols)}] "
            f"V5 geçmiş veri: {symbol}"
        )

        try:
            df = download_daily_data(
                symbol,
                period="3y",
                interval="1d",
                retries=2,
            )

            if df.empty or len(df) < (
                MIN_DAILY_BARS + 15
            ):
                continue

            market_data[
                symbol_without_suffix(symbol)
            ] = df

        except Exception as exc:
            print(
                f"{symbol} geçmiş veri hatası: {exc}"
            )

        time.sleep(0.03)

    return market_data


def get_reference_dates(
    market_data: Dict[str, pd.DataFrame],
    step_days: int,
) -> List[pd.Timestamp]:
    """
    En uzun veri geçmişine sahip hissenin işlem günlerini
    referans takvim olarak kullanır.
    """
    if not market_data:
        return []

    reference_df = max(
        market_data.values(),
        key=len,
    )

    dates = list(reference_df.index)

    start_position = MIN_DAILY_BARS - 1
    end_position = len(dates) - 11

    if end_position <= start_position:
        return []

    return dates[
        start_position:
        end_position:
        step_days
    ]


def build_historical_snapshot(
    market_data: Dict[str, pd.DataFrame],
    scan_date: pd.Timestamp,
) -> Dict[str, pd.DataFrame]:
    """
    Yalnızca scan_date ve öncesindeki verileri kullanır.
    """
    snapshot: Dict[str, pd.DataFrame] = {}

    for symbol, full_df in market_data.items():
        historical_df = full_df.loc[
            full_df.index <= scan_date
        ].copy()

        if len(historical_df) < MIN_DAILY_BARS:
            continue

        average_turnover = (
            calculate_average_turnover(
                historical_df,
                window=20,
            )
        )

        if average_turnover < MIN_AVG_TURNOVER_TL:
            continue

        snapshot[symbol] = historical_df

    return snapshot


def get_future_prices(
    full_df: pd.DataFrame,
    scan_date: pd.Timestamp,
) -> pd.Series:
    return full_df.loc[
        full_df.index > scan_date,
        "Close",
    ].astype(float)


def calculate_return(
    future_prices: pd.Series,
    signal_price: float,
    trading_day: int,
) -> float:
    if len(future_prices) < trading_day:
        return np.nan

    future_price = float(
        future_prices.iloc[trading_day - 1]
    )

    return round(
        (
            future_price / signal_price
            - 1
        ) * 100,
        2,
    )


def calculate_extremes(
    future_prices: pd.Series,
    signal_price: float,
    window: int,
) -> tuple[float, float]:
    selected = future_prices.head(window)

    if selected.empty:
        return np.nan, np.nan

    maximum_return = (
        float(selected.max()) /
        signal_price - 1
    ) * 100

    minimum_return = (
        float(selected.min()) /
        signal_price - 1
    ) * 100

    return (
        round(maximum_return, 2),
        round(minimum_return, 2),
    )


def candidate_to_history_row(
    candidate: pd.Series,
    full_df: pd.DataFrame,
    scan_date: pd.Timestamp,
) -> dict:
    symbol = str(candidate["symbol"])
    signal_price = float(candidate["close"])

    future_prices = get_future_prices(
        full_df,
        scan_date,
    )

    max_result_5d, min_result_5d = (
        calculate_extremes(
            future_prices,
            signal_price,
            5,
        )
    )

    max_result_10d, min_result_10d = (
        calculate_extremes(
            future_prices,
            signal_price,
            10,
        )
    )

    row = candidate.to_dict()

    row.update({
        "source": "historical_backfill",
        "signal_date": scan_date.strftime(
            "%Y-%m-%d"
        ),
        "symbol": symbol,
        "signal_price": signal_price,

        # V4 öğrenme motoruyla aynı kolon adları.
        "return_5d_at_signal": candidate.get(
            "return_5d",
            np.nan,
        ),
        "return_10d_at_signal": candidate.get(
            "return_10d",
            np.nan,
        ),
        "return_20d_at_signal": candidate.get(
            "return_20d",
            np.nan,
        ),

        "result_1d": calculate_return(
            future_prices,
            signal_price,
            1,
        ),
        "result_3d": calculate_return(
            future_prices,
            signal_price,
            3,
        ),
        "result_5d": calculate_return(
            future_prices,
            signal_price,
            5,
        ),
        "result_10d": calculate_return(
            future_prices,
            signal_price,
            10,
        ),

        "max_result_5d": max_result_5d,
        "min_result_5d": min_result_5d,
        "max_result_10d": max_result_10d,
        "min_result_10d": min_result_10d,

        "hit_3pct_5d": (
            int(max_result_5d >= 3)
            if not pd.isna(max_result_5d)
            else np.nan
        ),
        "hit_5pct_10d": (
            int(max_result_10d >= 5)
            if not pd.isna(max_result_10d)
            else np.nan
        ),
        "status": "tamamlandı",
    })

    return row


def run_backfill() -> pd.DataFrame:
    symbol_limit, step_days = get_settings()

    symbols = get_bist_symbols()

    if symbol_limit > 0:
        symbols = symbols[:symbol_limit]

    print("====================================")
    print("V5 TARİHSEL VERİ ÜRETİCİSİ")
    print("====================================")
    print("Sembol sayısı:", len(symbols))
    print("Tarih aralığı: 3 yıl")
    print("Tarama adımı:", step_days, "işlem günü")

    market_data = download_market_history(
        symbols
    )

    print(
        "Geçmiş verisi alınan:",
        len(market_data),
    )

    reference_dates = get_reference_dates(
        market_data,
        step_days,
    )

    print(
        "Geçmiş tarama tarihi:",
        len(reference_dates),
    )

    all_rows = []

    for date_number, scan_date in enumerate(
        reference_dates,
        start=1,
    ):
        print(
            f"\n[{date_number}/"
            f"{len(reference_dates)}] "
            f"Geçmiş tarama: "
            f"{scan_date.strftime('%Y-%m-%d')}"
        )

        snapshot = build_historical_snapshot(
            market_data,
            scan_date,
        )

        if not snapshot:
            continue

        scored_df = score_market(snapshot)

        if scored_df.empty:
            continue

        ranked_df = calculate_candidate_rank(
            scored_df
        )

        candidates_df = select_daily_candidates(
            ranked_df
        )

        if candidates_df.empty:
            print("Bu tarihte aday bulunamadı.")
            continue

        # Her geçmiş tarihte mevcut canlı sistem gibi
        # en fazla MAX_DAILY_CANDIDATES seçilir.
        candidates_df = candidates_df.head(
            MAX_DAILY_CANDIDATES
        )

        for _, candidate in (
            candidates_df.iterrows()
        ):
            symbol = str(candidate["symbol"])

            full_df = market_data.get(symbol)

            if full_df is None or full_df.empty:
                continue

            row = candidate_to_history_row(
                candidate,
                full_df,
                scan_date,
            )

            all_rows.append(row)

            print(
                f"Kaydedildi: "
                f"{scan_date.date()} | "
                f"{symbol} | "
                f"Skor: "
                f"{candidate['smart_money_score']} | "
                f"5g sonuç: "
                f"%{row['result_5d']}"
            )

    if not all_rows:
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)

    result = result.drop_duplicates(
        subset=[
            "signal_date",
            "symbol",
        ],
        keep="last",
    )

    result = result.sort_values(
        by=[
            "signal_date",
            "rank",
        ],
        ascending=[
            True,
            True,
        ],
    )

    return result.reset_index(drop=True)


def print_summary(result: pd.DataFrame) -> None:
    print("\n====================================")
    print("V5 BACKFILL ÖZETİ")
    print("====================================")

    if result.empty:
        print("Tarihsel sinyal üretilemedi.")
        return

    results_5d = pd.to_numeric(
        result["result_5d"],
        errors="coerce",
    ).dropna()

    max_results_5d = pd.to_numeric(
        result["max_result_5d"],
        errors="coerce",
    ).dropna()

    print("Toplam tarihsel sinyal:", len(result))
    print(
        "Farklı tarih:",
        result["signal_date"].nunique(),
    )

    if results_5d.empty:
        return

    print(
        "5 günde pozitif kapanan:",
        f"%{(results_5d > 0).mean() * 100:.1f}",
    )

    print(
        "5 günde kapanışta en az %3:",
        f"%{(results_5d >= 3).mean() * 100:.1f}",
    )

    print(
        "İlk 5 günde en az %3 gören:",
        f"%{(max_results_5d >= 3).mean() * 100:.1f}",
    )

    print(
        "Ortalama 5 günlük sonuç:",
        f"%{results_5d.mean():.2f}",
    )

    print(
        "Medyan 5 günlük sonuç:",
        f"%{results_5d.median():.2f}",
    )


def main():
    result = run_backfill()

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_summary(result)

    print(
        "\nKaydedildi:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
