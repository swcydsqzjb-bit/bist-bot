from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


HISTORY_FILE = "v11_signal_history.csv"
CURRENT_FILE = "v14_adaptive_decisions.csv"

WEIGHTS_FILE = "v15_model_weights.csv"
RESULT_FILE = "v15_final_decisions.csv"
STATUS_FILE = "v15_status.json"

MIN_COMPLETED_5D = 30
MIN_FEATURE_SAMPLE = 12

FEATURES = [
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

DEFAULT_WEIGHTS = {
    "v8_score": 24.0,
    "smart_money_score": 16.0,
    "institutional_score": 14.0,
    "historical_support_score": 10.0,
    "prediction_score": 8.0,
    "live_confirmation_score": 8.0,
    "relationship_score": 6.0,
    "rsi": 5.0,
    "volume_ratio": 5.0,
    "ema20_distance": 4.0,
}


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()


def normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    cleaned = {}

    for feature in FEATURES:
        value = safe_float(raw.get(feature), 0.0)
        cleaned[feature] = max(1.0, min(30.0, value))

    total = sum(cleaned.values())

    if total <= 0:
        return DEFAULT_WEIGHTS.copy()

    return {
        feature: round(value / total * 100.0, 3)
        for feature, value in cleaned.items()
    }


def learn_weights(history: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, float]]:
    completed = history[
        pd.to_numeric(
            history.get("result_5d"),
            errors="coerce",
        ).notna()
    ].copy()

    y = pd.to_numeric(
        completed["result_5d"],
        errors="coerce",
    )

    rows: List[Dict[str, Any]] = []
    raw_weights: Dict[str, float] = {}

    for feature in FEATURES:
        base_weight = DEFAULT_WEIGHTS[feature]

        if feature not in completed.columns:
            rows.append({
                "feature": feature,
                "sample_count": 0,
                "correlation_5d": np.nan,
                "high_group_avg_5d": np.nan,
                "low_group_avg_5d": np.nan,
                "spread_5d": np.nan,
                "stability": 0.0,
                "learned_multiplier": 1.0,
                "raw_weight": base_weight,
            })
            raw_weights[feature] = base_weight
            continue

        x = pd.to_numeric(
            completed[feature],
            errors="coerce",
        )

        valid = x.notna() & y.notna()
        sample_count = int(valid.sum())

        if sample_count < MIN_FEATURE_SAMPLE:
            rows.append({
                "feature": feature,
                "sample_count": sample_count,
                "correlation_5d": np.nan,
                "high_group_avg_5d": np.nan,
                "low_group_avg_5d": np.nan,
                "spread_5d": np.nan,
                "stability": round(
                    min(1.0, sample_count / MIN_FEATURE_SAMPLE),
                    3,
                ),
                "learned_multiplier": 1.0,
                "raw_weight": base_weight,
            })
            raw_weights[feature] = base_weight
            continue

        xv = x[valid]
        yv = y[valid]

        correlation = safe_float(
            xv.corr(yv),
            0.0,
        )

        median = safe_float(
            xv.median(),
            0.0,
        )

        high = yv[xv >= median]
        low = yv[xv < median]

        high_avg = safe_float(high.mean(), 0.0)
        low_avg = safe_float(low.mean(), 0.0)
        spread = high_avg - low_avg

        stability = min(1.0, sample_count / 100.0)

        direction_score = (
            np.clip(correlation, -0.45, 0.45) * 0.70
            + np.clip(spread / 10.0, -0.35, 0.35) * 0.30
        )

        multiplier = 1.0 + direction_score * stability
        multiplier = float(np.clip(multiplier, 0.65, 1.45))

        raw_weight = base_weight * multiplier
        raw_weights[feature] = raw_weight

        rows.append({
            "feature": feature,
            "sample_count": sample_count,
            "correlation_5d": round(correlation, 4),
            "high_group_avg_5d": round(high_avg, 3),
            "low_group_avg_5d": round(low_avg, 3),
            "spread_5d": round(spread, 3),
            "stability": round(stability, 3),
            "learned_multiplier": round(multiplier, 4),
            "raw_weight": round(raw_weight, 4),
        })

    normalized = normalize_weights(raw_weights)
    report = pd.DataFrame(rows)
    report["normalized_weight"] = report["feature"].map(normalized)

    return report, normalized


def normalize_feature(feature: str, value: Any) -> float:
    number = safe_float(value, 0.0)

    if feature == "rsi":
        # 55-65 araligi daha dengeli kabul edilir.
        distance = abs(number - 60.0)
        return float(np.clip(100.0 - distance * 4.0, 0.0, 100.0))

    if feature == "volume_ratio":
        return float(np.clip(number / 2.5 * 100.0, 0.0, 100.0))

    if feature == "ema20_distance":
        # EMA20'ye yakin pozitif mesafe daha iyi.
        if number < -5:
            return 10.0
        if number <= 8:
            return float(np.clip(100.0 - abs(number - 2.0) * 8.0, 0.0, 100.0))
        return float(np.clip(100.0 - (number - 8.0) * 10.0, 0.0, 100.0))

    return float(np.clip(number, 0.0, 100.0))


def get_current_feature(row: pd.Series, feature: str) -> float:
    if feature in row.index:
        return safe_float(row.get(feature), 0.0)

    aliases = {
        "historical_support_score": [
            "historical_support_score",
            "historical_score",
        ],
        "prediction_score": [
            "prediction_score",
            "v10_score",
        ],
        "live_confirmation_score": [
            "live_confirmation_score",
            "live_score",
        ],
        "relationship_score": [
            "relationship_score",
            "leader_lag_score",
        ],
    }

    for alias in aliases.get(feature, []):
        if alias in row.index:
            return safe_float(row.get(alias), 0.0)

    return 0.0


def classify(score: float, v14_decision: str) -> str:
    if score >= 78:
        return "V15 GÃÃLÃ ONAY"

    if score >= 68:
        return "V15 ONAYLI Ä°ZLEME"

    if score >= 58:
        return "V15 TEMKÄ°NLÄ° Ä°ZLEME"

    if "ONAY" in v14_decision:
        return "V15 GERÄ° ÃEKTÄ°"

    return "V15 ELE"


def build_final_decisions(
    current: pd.DataFrame,
    weights: Dict[str, float],
    model_mode: str,
) -> pd.DataFrame:
    rows = []

    for _, row in current.iterrows():
        learned_component = 0.0

        for feature, weight in weights.items():
            feature_value = get_current_feature(row, feature)
            normalized = normalize_feature(feature, feature_value)
            learned_component += normalized * weight / 100.0

        v14_score = safe_float(row.get("v14_score"), 0.0)
        dna_confidence = safe_float(row.get("dna_confidence"), 0.0)
        positive_rate = safe_float(row.get("positive_rate_5d"), 0.0)
        average_result = safe_float(row.get("average_result_5d"), 0.0)

        if model_mode == "LEARNED":
            final_score = (
                learned_component * 0.55
                + v14_score * 0.30
                + dna_confidence * 0.08
                + positive_rate * 0.05
                + np.clip((average_result + 3.0) / 10.0 * 100.0, 0, 100) * 0.02
            )
        else:
            # Veri yetersizken V14'Ã¼ bozma; sadece izleme katmani ekle.
            final_score = (
                v14_score * 0.85
                + dna_confidence * 0.08
                + positive_rate * 0.05
                + np.clip((average_result + 3.0) / 10.0 * 100.0, 0, 100) * 0.02
            )

        final_score = float(np.clip(final_score, 0.0, 100.0))
        v14_decision = safe_text(row.get("v14_decision"))
        decision = classify(final_score, v14_decision)

        rows.append({
            "symbol": safe_text(row.get("symbol")),
            "close": round(safe_float(row.get("close"), 0.0), 4),
            "model_mode": model_mode,
            "v14_score": round(v14_score, 2),
            "v14_decision": v14_decision,
            "learned_component_score": round(learned_component, 2),
            "dna_confidence": round(dna_confidence, 2),
            "positive_rate_5d": round(positive_rate, 2),
            "average_result_5d": round(average_result, 2),
            "v15_score": round(final_score, 2),
            "v15_decision": decision,
            "v8_score": round(safe_float(row.get("v8_score"), 0.0), 2),
            "smart_money_score": round(
                safe_float(row.get("smart_money_score"), 0.0), 2
            ),
            "institutional_score": round(
                safe_float(row.get("institutional_score"), 0.0), 2
            ),
            "dna_classification": safe_text(
                row.get("dna_classification")
            ),
            "positive_reasons": safe_text(
                row.get("positive_reasons")
            ),
            "risk_reasons": safe_text(
                row.get("risk_reasons")
            ),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    priority = {
        "V15 GÃÃLÃ ONAY": 5,
        "V15 ONAYLI Ä°ZLEME": 4,
        "V15 TEMKÄ°NLÄ° Ä°ZLEME": 3,
        "V15 GERÄ° ÃEKTÄ°": 2,
        "V15 ELE": 1,
    }

    result["_priority"] = (
        result["v15_decision"]
        .map(priority)
        .fillna(0)
    )

    result = (
        result
        .sort_values(
            ["_priority", "v15_score"],
            ascending=False,
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def write_status(payload: Dict[str, Any]) -> None:
    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    print("V15 Ogrenme ve Nihai Karar Motoru basladi.")

    history = load_csv(HISTORY_FILE)
    current = load_csv(CURRENT_FILE)

    if current.empty:
        pd.DataFrame().to_csv(
            RESULT_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame().to_csv(
            WEIGHTS_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        write_status({
            "status": "v14_missing",
            "model_mode": "NONE",
            "completed_5d": 0,
            "message": "V14 karar dosyasi bulunamadi veya bos.",
        })
        return

    completed_count = 0

    if not history.empty and "result_5d" in history.columns:
        completed_count = int(
            pd.to_numeric(
                history["result_5d"],
                errors="coerce",
            ).notna().sum()
        )

    if completed_count >= MIN_COMPLETED_5D:
        weight_report, weights = learn_weights(history)
        model_mode = "LEARNED"
    else:
        weights = DEFAULT_WEIGHTS.copy()
        weight_report = pd.DataFrame([
            {
                "feature": feature,
                "sample_count": completed_count,
                "correlation_5d": np.nan,
                "high_group_avg_5d": np.nan,
                "low_group_avg_5d": np.nan,
                "spread_5d": np.nan,
                "stability": round(
                    min(1.0, completed_count / MIN_COMPLETED_5D),
                    3,
                ),
                "learned_multiplier": 1.0,
                "raw_weight": weight,
                "normalized_weight": weight,
            }
            for feature, weight in DEFAULT_WEIGHTS.items()
        ])
        model_mode = "FALLBACK"

    weight_report.to_csv(
        WEIGHTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    final = build_final_decisions(
        current=current,
        weights=weights,
        model_mode=model_mode,
    )

    final.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    approved_count = int(
        final["v15_decision"].isin(
            ["V15 GÃÃLÃ ONAY", "V15 ONAYLI Ä°ZLEME"]
        ).sum()
    ) if not final.empty else 0

    write_status({
        "status": "ready",
        "model_mode": model_mode,
        "completed_5d": completed_count,
        "minimum_required": MIN_COMPLETED_5D,
        "candidate_count": len(final),
        "approved_count": approved_count,
        "message": (
            "V15 ogrenilmis agirliklari kullaniyor."
            if model_mode == "LEARNED"
            else "V15 veri biriktirirken guvenli FALLBACK modunda."
        ),
    })

    print("\n===== V15 MODEL WEIGHTS =====")
    print(weight_report.to_string(index=False))

    print("\n===== V15 FINAL DECISIONS =====")
    print(final.to_string(index=False))


if __name__ == "__main__":
    main()
