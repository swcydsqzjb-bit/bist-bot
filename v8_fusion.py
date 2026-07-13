from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from v3_config import (
    MAX_DAILY_CANDIDATES,
    MIN_SMART_MONEY_SCORE,
)
from v3_data import (
    download_bist_daily_data,
    get_bist_symbols,
)
from v3_scanner import (
    calculate_candidate_rank,
)
from v3_scoring import score_market
from v6_institutional import (
    calculate_institutional_score,
)
from v7_similarity import (
    analyze_candidates,
    load_historical_memory,
)


FUSION_RESULTS_FILE = "v8_fusion_results.csv"
FUSION_CANDIDATES_FILE = "v8_today_candidates.csv"

# V7 benzerlik hesabı bütün 465 hisseye değil,
# önce teknik filtreyi geçen en güçlü adaylara uygulanır.
SHORTLIST_SIZE = 15


def safe_float(
    value,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def calculate_institutional_market(
    market_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    total = len(market_data)

    for number, (symbol, dataframe) in enumerate(
        market_data.items(),
        start=1,
    ):
        print(
            f"[{number}/{total}] "
            f"V8 Institutional: {symbol}"
        )

        try:
            result = calculate_institutional_score(
                symbol,
                dataframe,
            )

            rows.append(result)

        except Exception as exc:
            print(
                f"{symbol} Institutional hata: {exc}"
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


def prepare_shortlist(
    ranked_df: pd.DataFrame,
) -> pd.DataFrame:
    if ranked_df.empty:
        return pd.DataFrame()

    shortlist = ranked_df[
        ranked_df["eligible"].eq(True)
        & (
            pd.to_numeric(
                ranked_df["smart_money_score"],
                errors="coerce",
            )
            >= MIN_SMART_MONEY_SCORE
        )
    ].copy()

    if shortlist.empty:
        return shortlist

    shortlist = shortlist.sort_values(
        by=[
            "selection_score",
            "smart_money_score",
            "volume_score",
            "trend_score",
        ],
        ascending=False,
    )

    return shortlist.head(
        SHORTLIST_SIZE
    ).reset_index(drop=True)


def similarity_support_score(
    row: pd.Series,
) -> float:
    """
    Tarihsel destek puanı.

    Bu değer yükselme olasılığı değildir.
    Benzer geçmiş örneklerin:
    - pozitif kapanma oranını,
    - %3 başarı oranını,
    - benzerlik seviyesini

    tek karşılaştırmalı puanda birleştirir.
    """
    ready = bool(
        row.get("similarity_ready", False)
    )

    if not ready:
        # Veri yoksa aşırı ödül veya ceza vermiyoruz.
        return 50.0

    positive_rate = safe_float(
        row.get("positive_rate_5d"),
        50.0,
    )

    success_rate = safe_float(
        row.get("success_rate_3pct_5d"),
        0.0,
    )

    average_similarity = safe_float(
        row.get("average_similarity_pct"),
        0.0,
    )

    weighted_result = safe_float(
        row.get("weighted_result_5d"),
        0.0,
    )

    score = (
        positive_rate * 0.45
        + success_rate * 0.30
        + average_similarity * 0.25
    )

    # Benzer örneklerin ağırlıklı getirisi pozitifse küçük bonus.
    if weighted_result >= 3:
        score += 5

    elif weighted_result >= 1:
        score += 2

    elif weighted_result < -2:
        score -= 5

    return round(
        max(0, min(100, score)),
        2,
    )


def calculate_fusion_score(
    row: pd.Series,
) -> float:
    smart_money = safe_float(
        row.get("smart_money_score")
    )

    institutional = safe_float(
        row.get("institutional_score")
    )

    historical_support = safe_float(
        row.get("historical_support_score"),
        50,
    )

    risk_penalty = abs(
        safe_float(
            row.get("risk_penalty")
        )
    )

    institutional_risks = str(
        row.get("institutional_risks", "")
    )

    final_score = (
        smart_money * 0.55
        + institutional * 0.25
        + historical_support * 0.20
    )

    # Smart Money motorundaki risk cezasının bir kısmı
    # birleşik skora da yansıtılır.
    final_score -= risk_penalty * 0.20

    if (
        institutional_risks
        and "Belirgin kurumsal risk yok"
        not in institutional_risks
    ):
        final_score -= 2

    return round(
        max(0, min(100, final_score)),
        2,
    )


def fusion_classification(
    score: float,
) -> str:
    if score >= 82:
        return "V8 ELİT"

    if score >= 72:
        return "V8 GÜÇLÜ"

    if score >= 65:
        return "V8 İZLEME"

    return "V8 ZAYIF"


def merge_analysis(
    shortlist: pd.DataFrame,
    institutional_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
) -> pd.DataFrame:
    if shortlist.empty:
        return pd.DataFrame()

    result = shortlist.copy()

    if not institutional_df.empty:
        result = result.merge(
            institutional_df,
            on="symbol",
            how="left",
        )

    if not similarity_df.empty:
        result = result.merge(
            similarity_df,
            on="symbol",
            how="left",
        )

    if "institutional_score" not in result.columns:
        result["institutional_score"] = 0

    if "similarity_ready" not in result.columns:
        result["similarity_ready"] = False

    result["institutional_score"] = pd.to_numeric(
        result["institutional_score"],
        errors="coerce",
    ).fillna(0)

    result["historical_support_score"] = (
        result.apply(
            similarity_support_score,
            axis=1,
        )
    )

    result["v8_score"] = result.apply(
        calculate_fusion_score,
        axis=1,
    )

    result["v8_classification"] = (
        result["v8_score"]
        .apply(fusion_classification)
    )

    # Çok düşük kurumsal destek varsa adayın sıralaması geriye alınır.
    result["institutional_gate"] = (
        result["institutional_score"] >= 45
    )

    result = result.sort_values(
        by=[
            "institutional_gate",
            "v8_score",
            "smart_money_score",
            "institutional_score",
            "historical_support_score",
        ],
        ascending=False,
    )

    return result.reset_index(drop=True)


def select_final_candidates(
    fusion_df: pd.DataFrame,
) -> pd.DataFrame:
    if fusion_df.empty:
        return pd.DataFrame()

    candidates = fusion_df[
        (
            fusion_df["v8_score"] >= 65
        )
        & (
            fusion_df["smart_money_score"]
            >= MIN_SMART_MONEY_SCORE
        )
        & (
            fusion_df["institutional_score"]
            >= 45
        )
    ].copy()

    if candidates.empty:
        return pd.DataFrame(
            columns=fusion_df.columns
        )

    candidates = candidates.head(
        MAX_DAILY_CANDIDATES
    ).copy()

    if "rank" in candidates.columns:
        candidates = candidates.drop(
            columns=["rank"]
        )

    candidates.insert(
        0,
        "rank",
        range(1, len(candidates) + 1),
    )

    return candidates.reset_index(drop=True)


def run_fusion(
    symbol_limit: int = 0,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    print("====================================")
    print("V8 FUSION ENGINE BAŞLADI")
    print("====================================")

    symbols = get_bist_symbols()

    if symbol_limit > 0:
        symbols = symbols[:symbol_limit]

    print("Taranacak sembol:", len(symbols))

    market_data, quality_report = (
        download_bist_daily_data(
            symbols=symbols,
            sleep_seconds=0.03,
        )
    )

    if not market_data:
        print("Uygun piyasa verisi alınamadı.")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            quality_report,
        )

    # 1. Smart Money
    scored_df = score_market(
        market_data
    )

    if scored_df.empty:
        print("Smart Money sonucu oluşmadı.")
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            quality_report,
        )

    ranked_df = calculate_candidate_rank(
        scored_df
    )

    shortlist = prepare_shortlist(
        ranked_df
    )

    print(
        "Ayrıntılı incelenecek kısa liste:",
        len(shortlist),
    )

    if shortlist.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            quality_report,
        )

    # 2. Institutional Score
    shortlist_symbols = set(
        shortlist["symbol"]
        .astype(str)
        .tolist()
    )

    shortlist_market_data = {
        symbol: dataframe
        for symbol, dataframe in market_data.items()
        if symbol in shortlist_symbols
    }

    institutional_df = (
        calculate_institutional_market(
            shortlist_market_data
        )
    )

    # 3. Tarihsel benzerlik
    historical_memory = (
        load_historical_memory()
    )

    print(
        "V7 tarihsel hafıza:",
        len(historical_memory),
    )

    similarity_df = analyze_candidates(
        historical_memory,
        shortlist,
    )

    # 4. Birleşik karar
    fusion_df = merge_analysis(
        shortlist=shortlist,
        institutional_df=institutional_df,
        similarity_df=similarity_df,
    )

    final_candidates = (
        select_final_candidates(
            fusion_df
        )
    )

    fusion_df.to_csv(
        FUSION_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final_candidates.to_csv(
        FUSION_CANDIDATES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n====================================")
    print("V8 FUSION ÖZETİ")
    print("====================================")
    print("Toplam sembol:", len(symbols))
    print("Uygun veri:", len(market_data))
    print("Kısa liste:", len(shortlist))
    print("Nihai aday:", len(final_candidates))

    if final_candidates.empty:
        print(
            "Bugün birleşik güven eşiğini "
            "geçen aday bulunamadı."
        )

    else:
        display_columns = [
            "rank",
            "symbol",
            "v8_score",
            "v8_classification",
            "smart_money_score",
            "institutional_score",
            "historical_support_score",
            "positive_rate_5d",
            "success_rate_3pct_5d",
            "weighted_result_5d",
            "similarity_confidence",
        ]

        existing_columns = [
            column
            for column in display_columns
            if column in final_candidates.columns
        ]

        print(
            final_candidates[
                existing_columns
            ].to_string(index=False)
        )

    return (
        fusion_df,
        final_candidates,
        quality_report,
    )


if __name__ == "__main__":
    run_fusion()
