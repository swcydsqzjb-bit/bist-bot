from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


HISTORY_FILE = "v3_signals_history.csv"
CANDIDATES_FILE = "v3_today_candidates.csv"
OUTPUT_FILE = "v4_learning_results.csv"

MIN_COMPLETED_SIGNALS = 30
MIN_SIMILAR_EXAMPLES = 10
MAX_SIMILAR_EXAMPLES = 30

SUCCESS_THRESHOLD_5D = 3.0

FEATURES = [
    "smart_money_score",
    "selection_score",
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
]

CANDIDATE_COLUMN_MAP = {
    "return_5d_at_signal": "return_5d",
    "return_10d_at_signal": "return_10d",
    "return_20d_at_signal": "return_20d",
}


def safe_numeric(
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


def load_history() -> pd.DataFrame:
    if not os.path.exists(HISTORY_FILE):
        print(f"{HISTORY_FILE} bulunamadı.")
        return pd.DataFrame()

    try:
        history = pd.read_csv(HISTORY_FILE)
    except Exception as exc:
        print("Geçmiş dosyası okunamadı:", exc)
        return pd.DataFrame()

    history = safe_numeric(
        history,
        FEATURES + [
            "result_5d",
            "max_result_5d",
            "min_result_5d",
        ],
    )

    completed = history[
        history["result_5d"].notna()
    ].copy()

    return completed


def load_candidates() -> pd.DataFrame:
    if not os.path.exists(CANDIDATES_FILE):
        print(f"{CANDIDATES_FILE} bulunamadı.")
        return pd.DataFrame()

    try:
        candidates = pd.read_csv(CANDIDATES_FILE)
    except Exception as exc:
        print("Aday dosyası okunamadı:", exc)
        return pd.DataFrame()

    for learning_column, candidate_column in (
        CANDIDATE_COLUMN_MAP.items()
    ):
        if candidate_column in candidates.columns:
            candidates[learning_column] = candidates[
                candidate_column
            ]

    candidates = safe_numeric(
        candidates,
        FEATURES,
    )

    return candidates


def prepare_feature_matrix(
    history: pd.DataFrame,
    candidate: pd.Series,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    usable_features = []

    for feature in FEATURES:
        if feature not in history.columns:
            continue

        if feature not in candidate.index:
            continue

        candidate_value = candidate.get(feature)

        if pd.isna(candidate_value):
            continue

        valid_count = history[feature].notna().sum()

        if valid_count < 10:
            continue

        if history[feature].std(skipna=True) == 0:
            continue

        usable_features.append(feature)

    if len(usable_features) < 5:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            [],
        )

    matrix = history[usable_features].copy()
    candidate_vector = candidate[usable_features].copy()

    medians = matrix.median(numeric_only=True)
    matrix = matrix.fillna(medians)
    candidate_vector = candidate_vector.fillna(medians)

    standard_deviation = matrix.std(ddof=0)
    standard_deviation = standard_deviation.replace(
        0,
        np.nan,
    )

    valid_features = standard_deviation[
        standard_deviation.notna()
    ].index.tolist()

    if len(valid_features) < 5:
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            [],
        )

    matrix = matrix[valid_features]
    candidate_vector = candidate_vector[valid_features]

    means = matrix.mean()
    standard_deviation = matrix.std(ddof=0).replace(
        0,
        1,
    )

    standardized_matrix = (
        matrix - means
    ) / standard_deviation

    standardized_candidate = (
        candidate_vector - means
    ) / standard_deviation

    return (
        standardized_matrix,
        standardized_candidate,
        valid_features,
    )


def calculate_distances(
    standardized_matrix: pd.DataFrame,
    standardized_candidate: pd.Series,
) -> pd.Series:
    differences = (
        standardized_matrix -
        standardized_candidate
    )

    squared_distances = (
        differences.pow(2).mean(axis=1)
    )

    distances = np.sqrt(squared_distances)

    return distances


def select_similar_examples(
    history: pd.DataFrame,
    candidate: pd.Series,
) -> Tuple[pd.DataFrame, List[str]]:
    (
        standardized_matrix,
        standardized_candidate,
        used_features,
    ) = prepare_feature_matrix(
        history,
        candidate,
    )

    if standardized_matrix.empty:
        return pd.DataFrame(), used_features

    distances = calculate_distances(
        standardized_matrix,
        standardized_candidate,
    )

    similar = history.loc[
        distances.index
    ].copy()

    similar["similarity_distance"] = distances

    similar = similar.sort_values(
        by="similarity_distance",
        ascending=True,
    )

    similar = similar.head(
        MAX_SIMILAR_EXAMPLES
    )

    return similar, used_features


def confidence_label(
    similar_count: int,
    average_distance: float,
) -> str:
    if similar_count < MIN_SIMILAR_EXAMPLES:
        return "YETERSİZ"

    if similar_count >= 20 and average_distance <= 0.80:
        return "YÜKSEK"

    if similar_count >= 15 and average_distance <= 1.10:
        return "ORTA"

    return "DÜŞÜK"


def analyze_candidate(
    history: pd.DataFrame,
    candidate: pd.Series,
) -> Dict:
    symbol = str(
        candidate.get("symbol", "")
    ).strip().upper()

    base_result = {
        "symbol": symbol,
        "learning_ready": False,
        "completed_history_count": len(history),
        "similar_example_count": 0,
        "used_feature_count": 0,
        "success_rate_5d": np.nan,
        "positive_rate_5d": np.nan,
        "average_result_5d": np.nan,
        "median_result_5d": np.nan,
        "average_max_result_5d": np.nan,
        "average_min_result_5d": np.nan,
        "best_result_5d": np.nan,
        "worst_result_5d": np.nan,
        "average_similarity_distance": np.nan,
        "learning_confidence": "YETERSİZ",
        "message": "",
    }

    if len(history) < MIN_COMPLETED_SIGNALS:
        base_result["message"] = (
            f"Öğrenme için en az "
            f"{MIN_COMPLETED_SIGNALS} tamamlanmış sinyal gerekli. "
            f"Mevcut: {len(history)}."
        )
        return base_result

    similar, used_features = select_similar_examples(
        history,
        candidate,
    )

    base_result["used_feature_count"] = len(
        used_features
    )

    if similar.empty:
        base_result["message"] = (
            "Benzerlik hesaplaması için yeterli "
            "ortak özellik bulunamadı."
        )
        return base_result

    # En uzak ve anlamsız örnekleri azaltmak için
    # ilk 30 içinden medyan mesafeye göre süzme.
    distance_median = similar[
        "similarity_distance"
    ].median()

    distance_limit = max(
        0.75,
        distance_median * 1.35,
    )

    similar = similar[
        similar["similarity_distance"]
        <= distance_limit
    ].copy()

    similar_count = len(similar)

    base_result["similar_example_count"] = (
        similar_count
    )

    if similar_count < MIN_SIMILAR_EXAMPLES:
        base_result["message"] = (
            f"En az {MIN_SIMILAR_EXAMPLES} benzer örnek "
            f"gerekli. Bulunan: {similar_count}."
        )
        return base_result

    results_5d = pd.to_numeric(
        similar["result_5d"],
        errors="coerce",
    ).dropna()

    max_results_5d = pd.to_numeric(
        similar["max_result_5d"],
        errors="coerce",
    ).dropna()

    min_results_5d = pd.to_numeric(
        similar["min_result_5d"],
        errors="coerce",
    ).dropna()

    if len(results_5d) < MIN_SIMILAR_EXAMPLES:
        base_result["message"] = (
            "Benzer örneklerin 5 günlük sonuçları "
            "yeterli değil."
        )
        return base_result

    average_distance = float(
        similar["similarity_distance"].mean()
    )

    success_rate = (
        results_5d >= SUCCESS_THRESHOLD_5D
    ).mean() * 100

    positive_rate = (
        results_5d > 0
    ).mean() * 100

    base_result.update({
        "learning_ready": True,
        "success_rate_5d": round(
            success_rate,
            2,
        ),
        "positive_rate_5d": round(
            positive_rate,
            2,
        ),
        "average_result_5d": round(
            results_5d.mean(),
            2,
        ),
        "median_result_5d": round(
            results_5d.median(),
            2,
        ),
        "average_max_result_5d": round(
            max_results_5d.mean(),
            2,
        ) if not max_results_5d.empty else np.nan,
        "average_min_result_5d": round(
            min_results_5d.mean(),
            2,
        ) if not min_results_5d.empty else np.nan,
        "best_result_5d": round(
            results_5d.max(),
            2,
        ),
        "worst_result_5d": round(
            results_5d.min(),
            2,
        ),
        "average_similarity_distance": round(
            average_distance,
            3,
        ),
        "learning_confidence": confidence_label(
            similar_count,
            average_distance,
        ),
        "message": (
            "İstatistik geçmiş tamamlanmış benzer "
            "sinyallerden hesaplandı."
        ),
    })

    return base_result


def build_results(
    history: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, candidate in candidates.iterrows():
        result = analyze_candidate(
            history,
            candidate,
        )

        rows.append(result)

    return pd.DataFrame(rows)


def print_results(results: pd.DataFrame) -> None:
    print("\n====================================")
    print("V4 BENZERLİK / ÖĞRENME MOTORU")
    print("====================================")

    if results.empty:
        print("Analiz edilecek aday bulunamadı.")
        return

    for _, row in results.iterrows():
        print(f"\nHisse: {row['symbol']}")
        print(
            "Tamamlanmış geçmiş:",
            row["completed_history_count"],
        )
        print(
            "Benzer örnek:",
            row["similar_example_count"],
        )
        print(
            "Kullanılan özellik:",
            row["used_feature_count"],
        )

        if not bool(row["learning_ready"]):
            print("Durum: VERİ YETERSİZ")
            print("Açıklama:", row["message"])
            continue

        print(
            "5 günde en az %3 başarı:",
            f"%{row['success_rate_5d']}",
        )
        print(
            "5 günde pozitif kapanma:",
            f"%{row['positive_rate_5d']}",
        )
        print(
            "Ortalama 5g sonuç:",
            f"%{row['average_result_5d']}",
        )
        print(
            "Medyan 5g sonuç:",
            f"%{row['median_result_5d']}",
        )
        print(
            "Benzerlik güveni:",
            row["learning_confidence"],
        )


def main():
    print("V4 öğrenme motoru başladı.")

    history = load_history()
    candidates = load_candidates()

    if candidates.empty:
        print("Bugünkü aday bulunamadı.")
        pd.DataFrame().to_csv(
            OUTPUT_FILE,
            index=False,
        )
        return

    results = build_results(
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
