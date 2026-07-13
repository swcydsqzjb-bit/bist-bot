from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


BACKFILL_FILE = "v5_backfill_history.csv"
LIVE_HISTORY_FILE = "v3_signals_history.csv"
CANDIDATES_FILE = "v3_today_candidates.csv"
OUTPUT_FILE = "v7_similarity_results.csv"

MIN_HISTORY_ROWS = 30
MIN_SIMILAR_ROWS = 8
MAX_SIMILAR_ROWS = 25

SUCCESS_THRESHOLD_5D = 3.0

FEATURES = [
    "smart_money_score",
    "selection_score",
    "rsi",
    "atr_pct",
    "ema20_distance",
    "volume_ratio",
    "volume_accumulation_ratio",
    "up_down_volume_ratio",
    "range_20_pct",
    "close_position",
    "upper_wick_ratio",
    "distance_to_high_20",
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
]

CANDIDATE_MAP = {
    "return_5d_at_signal": "return_5d",
    "return_10d_at_signal": "return_10d",
    "return_20d_at_signal": "return_20d",
}


def safe_float(
    value: Any,
    default: float = np.nan,
) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def convert_numeric(
    dataframe: pd.DataFrame,
    columns: List[str],
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in columns:
        if column not in result.columns:
            result[column] = np.nan

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    return result


def load_historical_memory() -> pd.DataFrame:
    frames = []

    if os.path.exists(BACKFILL_FILE):
        try:
            backfill = pd.read_csv(BACKFILL_FILE)
            backfill["history_source"] = "backfill"
            frames.append(backfill)

        except Exception as exc:
            print("Backfill geçmişi okunamadı:", exc)

    if os.path.exists(LIVE_HISTORY_FILE):
        try:
            live_history = pd.read_csv(
                LIVE_HISTORY_FILE
            )

            live_history["history_source"] = "live"
            frames.append(live_history)

        except Exception as exc:
            print("Canlı geçmiş okunamadı:", exc)

    if not frames:
        return pd.DataFrame()

    history = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    history = convert_numeric(
        history,
        FEATURES + [
            "result_1d",
            "result_3d",
            "result_5d",
            "result_10d",
            "max_result_5d",
            "min_result_5d",
            "max_result_10d",
            "min_result_10d",
        ],
    )

    history = history[
        history["result_5d"].notna()
    ].copy()

    duplicate_columns = [
        column
        for column in ["signal_date", "symbol"]
        if column in history.columns
    ]

    if duplicate_columns:
        history = history.drop_duplicates(
            subset=duplicate_columns,
            keep="last",
        )

    return history.reset_index(drop=True)


def load_candidates() -> pd.DataFrame:
    if not os.path.exists(CANDIDATES_FILE):
        print(f"{CANDIDATES_FILE} bulunamadı.")
        return pd.DataFrame()

    try:
        candidates = pd.read_csv(
            CANDIDATES_FILE
        )

    except Exception as exc:
        print("Aday dosyası okunamadı:", exc)
        return pd.DataFrame()

    if candidates.empty:
        return candidates

    for history_column, candidate_column in (
        CANDIDATE_MAP.items()
    ):
        if candidate_column in candidates.columns:
            candidates[history_column] = candidates[
                candidate_column
            ]

    candidates = convert_numeric(
        candidates,
        FEATURES,
    )

    return candidates


def determine_features(
    history: pd.DataFrame,
    candidate: pd.Series,
) -> List[str]:
    usable = []

    for feature in FEATURES:
        if feature not in history.columns:
            continue

        candidate_value = safe_float(
            candidate.get(feature)
        )

        if pd.isna(candidate_value):
            continue

        valid_values = history[feature].dropna()

        if len(valid_values) < 20:
            continue

        if valid_values.std(ddof=0) <= 0:
            continue

        usable.append(feature)

    return usable


def standardized_distance(
    history: pd.DataFrame,
    candidate: pd.Series,
    usable_features: List[str],
) -> pd.Series:
    matrix = history[
        usable_features
    ].copy()

    medians = matrix.median(
        numeric_only=True
    )

    matrix = matrix.fillna(medians)

    candidate_vector = pd.Series(
        {
            feature: safe_float(
                candidate.get(feature),
                medians.get(feature, 0),
            )
            for feature in usable_features
        }
    )

    means = matrix.mean()
    deviations = (
        matrix.std(ddof=0)
        .replace(0, 1)
    )

    matrix_z = (
        matrix - means
    ) / deviations

    candidate_z = (
        candidate_vector - means
    ) / deviations

    squared_difference = (
        matrix_z - candidate_z
    ).pow(2)

    return np.sqrt(
        squared_difference.mean(axis=1)
    )


def similarity_percentage(
    distance: float,
) -> float:
    """
    Karşılaştırma kolaylığı için mesafeyi 0-100 ölçeğine
    dönüştürür. Gerçek başarı olasılığı değildir.
    """
    if pd.isna(distance):
        return 0.0

    similarity = 100 / (1 + distance)

    return round(
        max(0, min(100, similarity)),
        2,
    )


def select_similar_history(
    history: pd.DataFrame,
    candidate: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    features = determine_features(
        history,
        candidate,
    )

    if len(features) < 6:
        return pd.DataFrame(), features

    distances = standardized_distance(
        history,
        candidate,
        features,
    )

    similar = history.copy()
    similar["similarity_distance"] = distances
    similar["similarity_pct"] = (
        similar["similarity_distance"]
        .apply(similarity_percentage)
    )

    similar = similar.sort_values(
        by="similarity_distance",
        ascending=True,
    )

    nearest = similar.head(
        MAX_SIMILAR_ROWS
    ).copy()

    if nearest.empty:
        return nearest, features

    median_distance = (
        nearest["similarity_distance"]
        .median()
    )

    maximum_distance = max(
        0.90,
        median_distance * 1.40,
    )

    nearest = nearest[
        nearest["similarity_distance"]
        <= maximum_distance
    ].copy()

    return nearest, features


def weighted_average(
    values: pd.Series,
    weights: pd.Series,
) -> float:
    valid = (
        values.notna()
        & weights.notna()
        & (weights > 0)
    )

    if not valid.any():
        return np.nan

    return float(
        np.average(
            values[valid],
            weights=weights[valid],
        )
    )


def confidence_classification(
    similar_count: int,
    average_similarity: float,
) -> str:
    if similar_count < MIN_SIMILAR_ROWS:
        return "YETERSİZ"

    if (
        similar_count >= 18
        and average_similarity >= 60
    ):
        return "YÜKSEK"

    if (
        similar_count >= 12
        and average_similarity >= 50
    ):
        return "ORTA"

    return "DÜŞÜK"


def build_top_examples(
    similar: pd.DataFrame,
    limit: int = 5,
) -> str:
    if similar.empty:
        return ""

    examples = []

    for _, row in similar.head(limit).iterrows():
        symbol = str(
            row.get("symbol", "")
        ).strip()

        date = str(
            row.get("signal_date", "")
        ).strip()

        result_5d = safe_float(
            row.get("result_5d")
        )

        similarity = safe_float(
            row.get("similarity_pct")
        )

        examples.append(
            f"{symbol} ({date}) "
            f"benzerlik %{similarity:.1f}, "
            f"5g sonuç %{result_5d:.1f}"
        )

    return " | ".join(examples)


def analyze_candidate(
    history: pd.DataFrame,
    candidate: pd.Series,
) -> Dict[str, Any]:
    symbol = str(
        candidate.get("symbol", "")
    ).strip().upper()

    result: Dict[str, Any] = {
        "symbol": symbol,
        "similarity_ready": False,
        "historical_sample_count": len(history),
        "similar_example_count": 0,
        "used_feature_count": 0,
        "average_similarity_pct": np.nan,
        "best_similarity_pct": np.nan,
        "positive_rate_5d": np.nan,
        "success_rate_3pct_5d": np.nan,
        "average_result_5d": np.nan,
        "weighted_result_5d": np.nan,
        "median_result_5d": np.nan,
        "average_max_result_5d": np.nan,
        "average_min_result_5d": np.nan,
        "best_result_5d": np.nan,
        "worst_result_5d": np.nan,
        "similarity_confidence": "YETERSİZ",
        "top_similar_examples": "",
        "similarity_message": "",
    }

    if len(history) < MIN_HISTORY_ROWS:
        result["similarity_message"] = (
            f"En az {MIN_HISTORY_ROWS} tamamlanmış "
            f"tarihsel sinyal gerekli. Mevcut: {len(history)}."
        )
        return result

    similar, features = select_similar_history(
        history,
        candidate,
    )

    result["used_feature_count"] = len(
        features
    )

    result["similar_example_count"] = len(
        similar
    )

    if len(similar) < MIN_SIMILAR_ROWS:
        result["similarity_message"] = (
            f"En az {MIN_SIMILAR_ROWS} anlamlı benzer "
            f"örnek gerekli. Bulunan: {len(similar)}."
        )
        return result

    result_5d = pd.to_numeric(
        similar["result_5d"],
        errors="coerce",
    )

    max_result_5d = pd.to_numeric(
        similar["max_result_5d"],
        errors="coerce",
    )

    min_result_5d = pd.to_numeric(
        similar["min_result_5d"],
        errors="coerce",
    )

    similarities = pd.to_numeric(
        similar["similarity_pct"],
        errors="coerce",
    )

    weights = (
        similarities.clip(lower=1)
        ** 2
    )

    valid_results = result_5d.dropna()

    if len(valid_results) < MIN_SIMILAR_ROWS:
        result["similarity_message"] = (
            "Benzer örneklerin 5 günlük sonuçları yetersiz."
        )
        return result

    average_similarity = float(
        similarities.mean()
    )

    result.update({
        "similarity_ready": True,
        "average_similarity_pct": round(
            average_similarity,
            2,
        ),
        "best_similarity_pct": round(
            similarities.max(),
            2,
        ),
        "positive_rate_5d": round(
            (valid_results > 0).mean() * 100,
            2,
        ),
        "success_rate_3pct_5d": round(
            (
                valid_results
                >= SUCCESS_THRESHOLD_5D
            ).mean() * 100,
            2,
        ),
        "average_result_5d": round(
            valid_results.mean(),
            2,
        ),
        "weighted_result_5d": round(
            weighted_average(
                result_5d,
                weights,
            ),
            2,
        ),
        "median_result_5d": round(
            valid_results.median(),
            2,
        ),
        "average_max_result_5d": round(
            max_result_5d.mean(),
            2,
        ),
        "average_min_result_5d": round(
            min_result_5d.mean(),
            2,
        ),
        "best_result_5d": round(
            valid_results.max(),
            2,
        ),
        "worst_result_5d": round(
            valid_results.min(),
            2,
        ),
        "similarity_confidence": (
            confidence_classification(
                len(similar),
                average_similarity,
            )
        ),
        "top_similar_examples": (
            build_top_examples(similar)
        ),
        "similarity_message": (
            "Geçmiş tarihsel sinyallerle benzerlik "
            "analizi tamamlandı."
        ),
    })

    return result


def analyze_candidates(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, candidate in candidates.iterrows():
        rows.append(
            analyze_candidate(
                history,
                candidate,
            )
        )

    return pd.DataFrame(rows)


def print_results(
    results: pd.DataFrame,
) -> None:
    print("\n====================================")
    print("V7 AI BENZERLİK MOTORU")
    print("====================================")

    if results.empty:
        print("Analiz sonucu oluşmadı.")
        return

    for _, row in results.iterrows():
        print(f"\nHisse: {row['symbol']}")
        print(
            "Tarihsel örnek:",
            row["historical_sample_count"],
        )
        print(
            "Benzer örnek:",
            row["similar_example_count"],
        )
        print(
            "Kullanılan özellik:",
            row["used_feature_count"],
        )

        if not bool(row["similarity_ready"]):
            print("Durum: VERİ YETERSİZ")
            print(
                "Açıklama:",
                row["similarity_message"],
            )
            continue

        print(
            "Ortalama benzerlik:",
            f"%{row['average_similarity_pct']}",
        )
        print(
            "En iyi benzerlik:",
            f"%{row['best_similarity_pct']}",
        )
        print(
            "5 günde pozitif:",
            f"%{row['positive_rate_5d']}",
        )
        print(
            "5 günde en az %3:",
            f"%{row['success_rate_3pct_5d']}",
        )
        print(
            "Ağırlıklı 5g sonuç:",
            f"%{row['weighted_result_5d']}",
        )
        print(
            "Benzerlik güveni:",
            row["similarity_confidence"],
        )
        print(
            "En benzer örnekler:",
            row["top_similar_examples"],
        )


def main():
    print("V7 AI benzerlik motoru başladı.")

    history = load_historical_memory()
    candidates = load_candidates()

    print(
        "Tamamlanmış tarihsel kayıt:",
        len(history),
    )

    print(
        "Bugünkü aday:",
        len(candidates),
    )

    if candidates.empty:
        pd.DataFrame().to_csv(
            OUTPUT_FILE,
            index=False,
        )

        print("Bugünkü aday bulunamadı.")
        return

    results = analyze_candidates(
        history,
        candidates,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print_results(results)

    print(
        "\nKaydedildi:",
        OUTPUT_FILE,
    )


if __name__ == "__main__":
    main()
