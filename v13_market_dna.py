from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


CURRENT_FILE = "v8_today_candidates.csv"
MEMORY_FILES = [
    "v5_backfill_history.csv",
    "v11_signal_history.csv",
]

RESULT_FILE = "v13_market_dna_results.csv"
STATUS_FILE = "v13_market_dna_status.json"

NEIGHBOR_COUNT = 25
MIN_HISTORY_ROWS = 30
MIN_FEATURES = 5

FEATURES = [
    "smart_money_score",
    "institutional_score",
    "historical_support_score",
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
    "return_5d",
    "return_10d",
    "return_20d",
]

FEATURE_WEIGHTS = {
    "smart_money_score": 1.20,
    "institutional_score": 1.15,
    "historical_support_score": 0.90,
    "rsi": 0.75,
    "atr_pct": 0.65,
    "ema20_distance": 0.90,
    "volume_ratio": 1.10,
    "volume_accumulation_ratio": 1.10,
    "up_down_volume_ratio": 0.95,
    "range_20_pct": 0.75,
    "close_position": 0.80,
    "upper_wick_ratio": 0.75,
    "distance_to_high_20": 0.75,
    "return_1d": 0.60,
    "return_5d": 0.70,
    "return_10d": 0.70,
    "return_20d": 0.70,
}


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


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def normalize_symbol(value: Any) -> str:
    return str(value).strip().upper().replace(".IS", "")


def standardize_current(current: pd.DataFrame) -> pd.DataFrame:
    if current.empty:
        return current

    result = pd.DataFrame()
    result["symbol"] = current["symbol"].map(normalize_symbol)
    result["close"] = pd.to_numeric(
        current.get("close", current.get("price")),
        errors="coerce",
    )
    result["v8_score"] = pd.to_numeric(
        current.get("v8_score"),
        errors="coerce",
    )

    for feature in FEATURES:
        if feature in current.columns:
            result[feature] = pd.to_numeric(current[feature], errors="coerce")
        else:
            result[feature] = np.nan

    return result


def standardize_memory(df: pd.DataFrame, source: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    symbol_column = "symbol" if "symbol" in df.columns else None
    if symbol_column is None:
        return pd.DataFrame()

    result = pd.DataFrame()
    result["symbol"] = df[symbol_column].map(normalize_symbol)
    result["source"] = source
    result["signal_date"] = df.get(
        "signal_date",
        pd.Series([""] * len(df)),
    ).astype(str)
    result["result_5d"] = pd.to_numeric(df.get("result_5d"), errors="coerce")
    result["max_result_5d"] = pd.to_numeric(df.get("max_result_5d"), errors="coerce")
    result["min_result_5d"] = pd.to_numeric(df.get("min_result_5d"), errors="coerce")

    for feature in FEATURES:
        if feature in df.columns:
            result[feature] = pd.to_numeric(df[feature], errors="coerce")
        else:
            result[feature] = np.nan

    return result[
        result["symbol"].ne("")
        & result["result_5d"].notna()
    ].copy()


def load_memory() -> pd.DataFrame:
    frames: List[pd.DataFrame] = []

    for path in MEMORY_FILES:
        standardized = standardize_memory(load_csv(path), path)
        if not standardized.empty:
            frames.append(standardized)
            print(path, "kullanilabilir satir:", len(standardized))

    if not frames:
        return pd.DataFrame()

    memory = pd.concat(frames, ignore_index=True)
    return memory.drop_duplicates(
        subset=["source", "symbol", "signal_date", "result_5d"],
        keep="last",
    ).reset_index(drop=True)


def robust_scale_params(
    memory: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.Series, pd.Series]:
    median = memory[features].median()
    q1 = memory[features].quantile(0.25)
    q3 = memory[features].quantile(0.75)
    iqr = (q3 - q1).replace(0, np.nan)
    std = memory[features].std().replace(0, np.nan)
    return median, iqr.fillna(std).fillna(1.0)


def available_features(
    current_row: pd.Series,
    memory: pd.DataFrame,
) -> List[str]:
    selected: List[str] = []
    for feature in FEATURES:
        current_value = safe_float(current_row.get(feature))
        history_count = pd.to_numeric(memory.get(feature), errors="coerce").notna().sum()
        if not pd.isna(current_value) and history_count >= MIN_HISTORY_ROWS:
            selected.append(feature)
    return selected


def calculate_distances(
    current_row: pd.Series,
    memory: pd.DataFrame,
    features: List[str],
) -> pd.Series:
    _, scales = robust_scale_params(memory, features)
    squared_sum = pd.Series(0.0, index=memory.index)
    weight_sum = pd.Series(0.0, index=memory.index)

    for feature in features:
        history_values = pd.to_numeric(memory[feature], errors="coerce")
        current_value = safe_float(current_row.get(feature))
        valid = history_values.notna()
        diff = (history_values - current_value) / safe_float(scales[feature], 1.0)
        weight = FEATURE_WEIGHTS.get(feature, 1.0)
        squared_sum.loc[valid] += (diff.loc[valid] ** 2) * weight
        weight_sum.loc[valid] += weight

    return np.sqrt(squared_sum / weight_sum.replace(0, np.nan))


def similarity_from_distance(distance: float) -> float:
    if pd.isna(distance):
        return 0.0
    return round(100.0 / (1.0 + max(0.0, distance)), 2)


def classify_dna(
    positive_rate: float,
    hit_3pct_rate: float,
    avg_result: float,
    confidence: float,
) -> str:
    if confidence >= 70 and positive_rate >= 65 and hit_3pct_rate >= 40 and avg_result >= 2:
        return "GUCLU DNA"
    if confidence >= 55 and positive_rate >= 55 and avg_result >= 0.5:
        return "ORTA DNA"
    if positive_rate >= 50:
        return "KARISIK DNA"
    return "ZAYIF DNA"


def analyze_candidate(
    current_row: pd.Series,
    memory: pd.DataFrame,
) -> Dict[str, Any]:
    symbol = normalize_symbol(current_row.get("symbol"))
    features = available_features(current_row, memory)

    base = {
        "symbol": symbol,
        "used_feature_count": len(features),
        "used_features": " | ".join(features),
        "historical_sample_count": len(memory),
    }

    if len(features) < MIN_FEATURES:
        return {
            **base,
            "dna_ready": False,
            "dna_classification": "VERI YETERSIZ",
            "neighbor_count": 0,
            "average_similarity_pct": np.nan,
            "best_similarity_pct": np.nan,
            "positive_rate_5d": np.nan,
            "hit_3pct_5d_rate": np.nan,
            "average_result_5d": np.nan,
            "median_result_5d": np.nan,
            "average_max_result_5d": np.nan,
            "average_min_result_5d": np.nan,
            "dna_confidence": 0.0,
            "top_examples": "",
            "dna_message": "Yeterli ortak ozellik bulunamadi.",
        }

    working = memory.copy()
    working["distance"] = calculate_distances(current_row, memory, features)
    working = working[working["distance"].notna()].sort_values("distance")
    neighbors = working.head(min(NEIGHBOR_COUNT, len(working))).copy()

    if len(neighbors) < 8:
        return {
            **base,
            "dna_ready": False,
            "dna_classification": "VERI YETERSIZ",
            "neighbor_count": len(neighbors),
            "average_similarity_pct": np.nan,
            "best_similarity_pct": np.nan,
            "positive_rate_5d": np.nan,
            "hit_3pct_5d_rate": np.nan,
            "average_result_5d": np.nan,
            "median_result_5d": np.nan,
            "average_max_result_5d": np.nan,
            "average_min_result_5d": np.nan,
            "dna_confidence": 0.0,
            "top_examples": "",
            "dna_message": "Yeterli yakin tarihsel ornek bulunamadi.",
        }

    neighbors["similarity_pct"] = neighbors["distance"].apply(similarity_from_distance)
    result_5d = pd.to_numeric(neighbors["result_5d"], errors="coerce")
    max_result = pd.to_numeric(neighbors["max_result_5d"], errors="coerce")
    min_result = pd.to_numeric(neighbors["min_result_5d"], errors="coerce")

    positive_rate = round(float((result_5d > 0).mean() * 100), 2)
    hit_3pct_rate = (
        round(float((max_result >= 3).mean() * 100), 2)
        if max_result.notna().any()
        else np.nan
    )
    average_result = round(float(result_5d.mean()), 2)
    median_result = round(float(result_5d.median()), 2)
    average_similarity = round(float(neighbors["similarity_pct"].mean()), 2)
    best_similarity = round(float(neighbors["similarity_pct"].max()), 2)

    sample_factor = min(1.0, len(neighbors) / NEIGHBOR_COUNT)
    feature_factor = min(1.0, len(features) / 10.0)
    consistency = max(0.0, 1.0 - safe_float(result_5d.std(), 10.0) / 20.0)

    dna_confidence = round(
        average_similarity * 0.55
        + best_similarity * 0.15
        + sample_factor * 100 * 0.15
        + feature_factor * 100 * 0.10
        + consistency * 100 * 0.05,
        2,
    )

    classification = classify_dna(
        positive_rate,
        safe_float(hit_3pct_rate, 0.0),
        average_result,
        dna_confidence,
    )

    examples = []
    for _, example in neighbors.head(5).iterrows():
        examples.append(
            f"{example.get('symbol', '')} ({example.get('signal_date', '')}) "
            f"benzerlik %{safe_float(example.get('similarity_pct')):.1f}, "
            f"5g %{safe_float(example.get('result_5d')):.1f}"
        )

    return {
        **base,
        "dna_ready": True,
        "dna_classification": classification,
        "neighbor_count": len(neighbors),
        "average_similarity_pct": average_similarity,
        "best_similarity_pct": best_similarity,
        "positive_rate_5d": positive_rate,
        "hit_3pct_5d_rate": hit_3pct_rate,
        "average_result_5d": average_result,
        "median_result_5d": median_result,
        "average_max_result_5d": (
            round(float(max_result.mean()), 2)
            if max_result.notna().any()
            else np.nan
        ),
        "average_min_result_5d": (
            round(float(min_result.mean()), 2)
            if min_result.notna().any()
            else np.nan
        ),
        "dna_confidence": dna_confidence,
        "top_examples": " | ".join(examples),
        "dna_message": "Market DNA analizi tamamlandi.",
    }


def write_status(
    status: str,
    current_count: int,
    memory_count: int,
    message: str,
) -> None:
    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "status": status,
                "current_candidate_count": current_count,
                "historical_memory_count": memory_count,
                "minimum_history_required": MIN_HISTORY_ROWS,
                "message": message,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    print("V13 Market DNA Engine basladi.")
    current_raw = load_csv(CURRENT_FILE)
    memory = load_memory()

    if current_raw.empty or "symbol" not in current_raw.columns:
        pd.DataFrame().to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
        write_status(
            "current_candidates_missing",
            0,
            len(memory),
            "V8 bugunun aday dosyasi bulunamadi veya bos.",
        )
        print("V8 bugunun aday dosyasi bulunamadi veya bos.")
        return

    current = standardize_current(current_raw)

    if len(memory) < MIN_HISTORY_ROWS:
        pd.DataFrame().to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
        write_status(
            "waiting_for_history",
            len(current),
            len(memory),
            f"Market DNA icin en az {MIN_HISTORY_ROWS} tamamlanmis tarihsel ornek gerekli. Mevcut: {len(memory)}.",
        )
        print("Market DNA icin tarihsel hafiza yetersiz:", len(memory))
        return

    rows = []
    for number, (_, row) in enumerate(current.iterrows(), start=1):
        print(f"[{number}/{len(current)}] V13 DNA: {row.get('symbol')}")
        analysis = analyze_candidate(row, memory)
        analysis["close"] = safe_float(row.get("close"))
        analysis["v8_score"] = safe_float(row.get("v8_score"))
        rows.append(analysis)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            by=["dna_ready", "dna_confidence", "positive_rate_5d", "average_result_5d"],
            ascending=False,
        ).reset_index(drop=True)
        result.insert(0, "rank", range(1, len(result) + 1))

    result.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    ready_count = int(result.get("dna_ready", pd.Series(dtype=bool)).eq(True).sum())
    write_status(
        "dna_ready",
        len(current),
        len(memory),
        f"{ready_count} aday icin Market DNA analizi tamamlandi.",
    )

    print("\n===== V13 MARKET DNA SONUCLARI =====")
    print(result.to_string(index=False))
    print("\nKaydedildi:", RESULT_FILE)


if __name__ == "__main__":
    main()
