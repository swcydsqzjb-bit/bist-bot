from __future__ import annotations

import os
from datetime import datetime
from typing import List, Tuple

import pandas as pd

from v3_config import (
    DAILY_RESULTS_FILE,
    ELITE_SCORE,
    MAX_DAILY_CANDIDATES,
    MIN_SMART_MONEY_SCORE,
    TODAY_CANDIDATES_FILE,
)
from v3_data import (
    download_bist_daily_data,
    get_bist_symbols,
)
from v3_scoring import score_market


def get_symbol_limit() -> int:
    """
    Test amacıyla taranacak hisse sayısını sınırlar.

    V3_SYMBOL_LIMIT=50  -> ilk 50 hisse
    V3_SYMBOL_LIMIT=0   -> bütün BIST
    """
    raw_value = os.getenv("V3_SYMBOL_LIMIT", "0")

    try:
        limit = int(raw_value)
    except (TypeError, ValueError):
        limit = 0

    return max(0, limit)


def select_symbols() -> List[str]:
    symbols = get_bist_symbols()
    limit = get_symbol_limit()

    if limit > 0:
        symbols = symbols[:limit]

    print(f"V3 taranacak sembol sayısı: {len(symbols)}")

    return symbols


def calculate_candidate_rank(
    scored_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Smart Money skoruna ek olarak bileşen puanlarından
    nihai sıralama puanı oluşturur.
    """
    if scored_df.empty:
        return pd.DataFrame()

    result = scored_df.copy()

    required_numeric_columns = [
        "smart_money_score",
        "trend_score",
        "volume_score",
        "compression_score",
        "candle_score",
        "momentum_score",
        "liquidity_score",
        "risk_penalty",
        "volume_ratio",
        "volume_accumulation_ratio",
        "up_down_volume_ratio",
        "return_5d",
        "return_20d",
    ]

    for column in required_numeric_columns:
        if column not in result.columns:
            result[column] = 0

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0)

    # Skorun temel ağırlığı Smart Money skorudur.
    result["selection_score"] = (
        result["smart_money_score"] * 0.55
        + result["volume_score"] * 0.15
        + result["trend_score"] * 0.10
        + result["compression_score"] * 0.10
        + result["momentum_score"] * 0.05
        + result["candle_score"] * 0.05
    )

    # Fiyat çok ilerlemeden oluşan hacim birikimine küçük bonus.
    quiet_accumulation_bonus = (
        (result["volume_accumulation_ratio"] >= 1.15)
        & (result["return_20d"] < 15)
    )

    result.loc[
        quiet_accumulation_bonus,
        "selection_score",
    ] += 3

    # Yükselen gün hacmi belirgin şekilde baskınsa bonus.
    result.loc[
        result["up_down_volume_ratio"] >= 1.50,
        "selection_score",
    ] += 2

    # Son 5 günde fazla hızlanmış hisseleri geriye düşür.
    result.loc[
        result["return_5d"] > 10,
        "selection_score",
    ] -= 4

    result["selection_score"] = (
        result["selection_score"]
        .round(2)
        .clip(lower=0, upper=100)
    )

    return result.sort_values(
        by=[
            "eligible",
            "selection_score",
            "smart_money_score",
            "volume_score",
            "trend_score",
        ],
        ascending=False,
    ).reset_index(drop=True)


def select_daily_candidates(
    ranked_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Güven eşiğini geçen en fazla 3 adayı seçer.
    Eşiği geçen yoksa boş tablo döndürür.
    """
    if ranked_df.empty:
        return pd.DataFrame()

    candidates = ranked_df[
        ranked_df["eligible"].eq(True)
        & (
            ranked_df["smart_money_score"]
            >= MIN_SMART_MONEY_SCORE
        )
    ].copy()

    if candidates.empty:
        return pd.DataFrame(
            columns=ranked_df.columns
        )

    # Elit adaylar ilk sıraya alınır.
    candidates["elite_priority"] = (
        candidates["smart_money_score"] >= ELITE_SCORE
    ).astype(int)

    candidates = candidates.sort_values(
        by=[
            "elite_priority",
            "selection_score",
            "smart_money_score",
            "volume_score",
            "trend_score",
        ],
        ascending=False,
    )

    candidates = candidates.head(
        MAX_DAILY_CANDIDATES
    ).copy()

    candidates.insert(
        0,
        "rank",
        range(1, len(candidates) + 1),
    )

    return candidates.reset_index(drop=True)


def add_scan_metadata(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    now = datetime.now()

    result.insert(
        0,
        "scan_date",
        now.strftime("%Y-%m-%d"),
    )

    result.insert(
        1,
        "scan_time",
        now.strftime("%H:%M:%S"),
    )

    return result


def save_results(
    ranked_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    quality_report: pd.DataFrame,
) -> None:
    """
    Tam puanlama tablosunu ve günlük seçilen adayları kaydeder.
    """
    ranked_with_metadata = add_scan_metadata(
        ranked_df
    )

    candidates_with_metadata = add_scan_metadata(
        candidates_df
    )

    ranked_with_metadata.to_csv(
        DAILY_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    candidates_with_metadata.to_csv(
        TODAY_CANDIDATES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    quality_report.to_csv(
        "v3_data_quality_report.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"Tam sonuç dosyası: {DAILY_RESULTS_FILE}")
    print(
        f"Günlük aday dosyası: "
        f"{TODAY_CANDIDATES_FILE}"
    )
    print(
        "Veri kalite dosyası: "
        "v3_data_quality_report.csv"
    )


def print_scan_summary(
    total_symbols: int,
    quality_report: pd.DataFrame,
    ranked_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
) -> None:
    valid_count = 0

    if (
        not quality_report.empty
        and "valid" in quality_report.columns
    ):
        valid_count = int(
            quality_report["valid"].eq(True).sum()
        )

    eligible_count = 0

    if (
        not ranked_df.empty
        and "eligible" in ranked_df.columns
    ):
        eligible_count = int(
            ranked_df["eligible"].eq(True).sum()
        )

    print("\n====================================")
    print("V3 TARAMA ÖZETİ")
    print("====================================")
    print("Toplam sembol:", total_symbols)
    print("Uygun verisi gelen:", valid_count)
    print("Puanlanan:", len(ranked_df))
    print("Eşiği geçen:", eligible_count)
    print("Seçilen aday:", len(candidates_df))

    if candidates_df.empty:
        print("\nBugün güven eşiğini geçen aday yok.")
        return

    display_columns = [
        "rank",
        "symbol",
        "smart_money_score",
        "selection_score",
        "classification",
        "close",
        "rsi",
        "volume_ratio",
        "volume_accumulation_ratio",
        "range_20_pct",
        "return_5d",
        "return_20d",
        "positive_reasons",
        "risk_reasons",
    ]

    existing_columns = [
        column
        for column in display_columns
        if column in candidates_df.columns
    ]

    print("\nBUGÜNÜN V3 ADAYLARI")

    print(
        candidates_df[existing_columns]
        .to_string(index=False)
    )


def run_scanner() -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print("BIST V3 Smart Money taraması başladı.")

    symbols = select_symbols()

    market_data, quality_report = (
        download_bist_daily_data(
            symbols=symbols,
            sleep_seconds=0.03,
        )
    )

    if not market_data:
        print("Uygun piyasa verisi alınamadı.")

        empty_df = pd.DataFrame()

        save_results(
            empty_df,
            empty_df,
            quality_report,
        )

        return (
            empty_df,
            empty_df,
            quality_report,
        )

    scored_df = score_market(market_data)

    if scored_df.empty:
        print("Puanlama sonucu oluşmadı.")

        empty_df = pd.DataFrame()

        save_results(
            empty_df,
            empty_df,
            quality_report,
        )

        return (
            empty_df,
            empty_df,
            quality_report,
        )

    ranked_df = calculate_candidate_rank(
        scored_df
    )

    candidates_df = select_daily_candidates(
        ranked_df
    )

    save_results(
        ranked_df,
        candidates_df,
        quality_report,
    )

    print_scan_summary(
        total_symbols=len(symbols),
        quality_report=quality_report,
        ranked_df=ranked_df,
        candidates_df=candidates_df,
    )

    return (
        ranked_df,
        candidates_df,
        quality_report,
    )


def main():
    run_scanner()


if __name__ == "__main__":
    main()
