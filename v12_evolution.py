from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import numpy as np
import pandas as pd


HISTORY_FILE = "v11_signal_history.csv"
V11_WEIGHTS_FILE = "v11_learned_weights.csv"

V12_FEATURE_REPORT = "v12_feature_report.csv"
V12_RECOMMENDED_WEIGHTS = "v12_recommended_weights.csv"
V12_STATUS_FILE = "v12_status.json"

MIN_COMPLETED_5D = 30
MIN_FEATURE_SAMPLE = 12
MAX_SINGLE_WEIGHT = 24.0
MIN_SINGLE_WEIGHT = 3.0

BASE_WEIGHTS = {
    "v8_score": 20.0,
    "smart_money_score": 18.0,
    "institutional_score": 16.0,
    "historical_support_score": 12.0,
    "prediction_score": 10.0,
    "live_confirmation_score": 8.0,
    "relationship_score": 6.0,
    "rsi": 4.0,
    "volume_ratio": 4.0,
    "ema20_distance": 2.0,
}

FEATURES = list(BASE_WEIGHTS.keys())


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


def normalize_weights(raw_weights: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}

    for feature in FEATURES:
        value = safe_float(raw_weights.get(feature), 0.0)
        value = max(MIN_SINGLE_WEIGHT, min(MAX_SINGLE_WEIGHT, value))
        cleaned[feature] = value

    total = sum(cleaned.values())

    if total <= 0:
        return BASE_WEIGHTS.copy()

    return {
        feature: round(value / total * 100, 2)
        for feature, value in cleaned.items()
    }


def robust_feature_statistics(
    completed: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    y = pd.to_numeric(
        completed["result_5d"],
        errors="coerce",
    )

    for feature in FEATURES:
        if feature not in completed.columns:
            rows.append({
                "feature": feature,
                "sample_count": 0,
                "correlation": np.nan,
                "positive_group_avg_5d": np.nan,
                "negative_group_avg_5d": np.nan,
                "spread_5d": np.nan,
                "stability_factor": 0.0,
                "signal_strength": 0.0,
                "direction": "veri_yok",
            })
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
                "correlation": np.nan,
                "positive_group_avg_5d": np.nan,
                "negative_group_avg_5d": np.nan,
                "spread_5d": np.nan,
                "stability_factor": round(
                    min(1.0, sample_count / MIN_FEATURE_SAMPLE),
                    3,
                ),
                "signal_strength": 0.0,
                "direction": "veri_yetersiz",
            })
            continue

        x_valid = x[valid]
        y_valid = y[valid]

        correlation = safe_float(
            x_valid.corr(y_valid),
            0.0,
        )

        median_value = safe_float(
            x_valid.median(),
            0.0,
        )

        high_mask = x_valid >= median_value
        low_mask = x_valid < median_value

        high_avg = safe_float(
            y_valid[high_mask].mean(),
            0.0,
        )

        low_avg = safe_float(
            y_valid[low_mask].mean(),
            0.0,
        )

        spread = high_avg - low_avg

        stability_factor = min(
            1.0,
            sample_count / 80.0,
        )

        correlation_component = abs(correlation) * 0.65
        spread_component = min(abs(spread) / 10.0, 1.0) * 0.35

        signal_strength = (
            correlation_component + spread_component
        ) * stability_factor

        direction = (
            "pozitif"
            if correlation >= 0 and spread >= 0
            else "negatif"
            if correlation <= 0 and spread <= 0
            else "karisik"
        )

        rows.append({
            "feature": feature,
            "sample_count": sample_count,
            "correlation": round(correlation, 4),
            "positive_group_avg_5d": round(high_avg, 3),
            "negative_group_avg_5d": round(low_avg, 3),
            "spread_5d": round(spread, 3),
            "stability_factor": round(stability_factor, 3),
            "signal_strength": round(signal_strength, 4),
            "direction": direction,
        })

    return pd.DataFrame(rows)


def build_recommended_weights(
    feature_report: pd.DataFrame,
) -> pd.DataFrame:
    raw_weights: Dict[str, float] = {}

    for _, row in feature_report.iterrows():
        feature = str(row["feature"])
        base = BASE_WEIGHTS[feature]
        strength = safe_float(row["signal_strength"], 0.0)
        direction = str(row["direction"])

        if direction == "pozitif":
            multiplier = 1.0 + min(strength, 0.60)
        elif direction == "negatif":
            multiplier = 1.0 - min(strength, 0.45)
        else:
            multiplier = 1.0

        raw_weights[feature] = base * multiplier

    normalized = normalize_weights(raw_weights)

    rows = []

    for feature in FEATURES:
        base_weight = BASE_WEIGHTS[feature]
        recommended = normalized[feature]
        difference = recommended - base_weight

        feature_row = feature_report[
            feature_report["feature"] == feature
        ].iloc[0]

        rows.append({
            "feature": feature,
            "base_weight": base_weight,
            "recommended_weight": recommended,
            "change": round(difference, 2),
            "direction": feature_row["direction"],
            "sample_count": int(feature_row["sample_count"]),
            "correlation": feature_row["correlation"],
            "spread_5d": feature_row["spread_5d"],
            "signal_strength": feature_row["signal_strength"],
        })

    return pd.DataFrame(rows).sort_values(
        by="recommended_weight",
        ascending=False,
    ).reset_index(drop=True)


def write_status(
    status: str,
    completed_5d: int,
    message: str,
) -> None:
    payload = {
        "status": status,
        "completed_5d": completed_5d,
        "minimum_required": MIN_COMPLETED_5D,
        "message": message,
    }

    with open(
        V12_STATUS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )


def main() -> None:
    print("V12 Evolution Engine basladi.")

    history = load_csv(HISTORY_FILE)

    if history.empty:
        print("V11 sinyal hafizasi bulunamadi veya bos.")
        pd.DataFrame().to_csv(
            V12_FEATURE_REPORT,
            index=False,
            encoding="utf-8-sig",
        )
        pd.DataFrame().to_csv(
            V12_RECOMMENDED_WEIGHTS,
            index=False,
            encoding="utf-8-sig",
        )
        write_status(
            "history_missing",
            0,
            "V11 sinyal hafizasi bulunamadi.",
        )
        return

    completed = history[
        pd.to_numeric(
            history.get("result_5d"),
            errors="coerce",
        ).notna()
    ].copy()

    completed_count = len(completed)

    print("Tamamlanmis 5 gunluk sinyal:", completed_count)

    if completed_count < MIN_COMPLETED_5D:
        feature_report = pd.DataFrame([
            {
                "feature": "VERI_YETERSIZ",
                "sample_count": completed_count,
                "correlation": np.nan,
                "positive_group_avg_5d": np.nan,
                "negative_group_avg_5d": np.nan,
                "spread_5d": np.nan,
                "stability_factor": round(
                    completed_count / MIN_COMPLETED_5D,
                    3,
                ),
                "signal_strength": 0.0,
                "direction": "bekliyor",
            }
        ])

        feature_report.to_csv(
            V12_FEATURE_REPORT,
            index=False,
            encoding="utf-8-sig",
        )

        base_table = pd.DataFrame([
            {
                "feature": feature,
                "base_weight": weight,
                "recommended_weight": weight,
                "change": 0.0,
                "direction": "bekliyor",
                "sample_count": completed_count,
                "correlation": np.nan,
                "spread_5d": np.nan,
                "signal_strength": 0.0,
            }
            for feature, weight in BASE_WEIGHTS.items()
        ])

        base_table.to_csv(
            V12_RECOMMENDED_WEIGHTS,
            index=False,
            encoding="utf-8-sig",
        )

        message = (
            f"Ogrenme icin en az {MIN_COMPLETED_5D} tamamlanmis "
            f"5 gunluk sinyal gerekli. Mevcut: {completed_count}."
        )

        write_status(
            "waiting_for_data",
            completed_count,
            message,
        )

        print(message)
        return

    feature_report = robust_feature_statistics(
        completed
    )

    recommended_weights = build_recommended_weights(
        feature_report
    )

    feature_report.to_csv(
        V12_FEATURE_REPORT,
        index=False,
        encoding="utf-8-sig",
    )

    recommended_weights.to_csv(
        V12_RECOMMENDED_WEIGHTS,
        index=False,
        encoding="utf-8-sig",
    )

    write_status(
        "recommendations_ready",
        completed_count,
        (
            "V12 agirlik tavsiyeleri hazir. "
            "Bu surum V8/V10 agirliklarini otomatik degistirmez."
        ),
    )

    print("\n===== V12 FEATURE REPORT =====")
    print(feature_report.to_string(index=False))

    print("\n===== V12 RECOMMENDED WEIGHTS =====")
    print(recommended_weights.to_string(index=False))

    print("\nKaydedildi:")
    print("-", V12_FEATURE_REPORT)
    print("-", V12_RECOMMENDED_WEIGHTS)
    print("-", V12_STATUS_FILE)


if __name__ == "__main__":
    main()
