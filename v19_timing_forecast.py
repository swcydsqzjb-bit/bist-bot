from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd


CURRENT_FILE = "v18_confidence_decisions.csv"
V15_FILE = "v15_final_decisions.csv"
HISTORY_FILE = "v5_backfill_history.csv"

OUTPUT_FILE = "v19_timing_forecasts.csv"
STATUS_FILE = "v19_timing_status.json"

MIN_NEIGHBORS = 12
MAX_NEIGHBORS = 35


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
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
    except Exception:
        return pd.DataFrame()


def normalize_symbol(value: Any) -> str:
    return clean_text(value).upper().replace(".IS", "")


def numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce")


def build_current() -> pd.DataFrame:
    current = load_csv(CURRENT_FILE)
    v15 = load_csv(V15_FILE)

    if current.empty:
        return current

    current["symbol"] = current["symbol"].map(normalize_symbol)

    if not v15.empty and "symbol" in v15.columns:
        v15["symbol"] = v15["symbol"].map(normalize_symbol)
        extra_columns = [
            column for column in [
                "symbol", "v8_score", "smart_money_score",
                "institutional_score", "historical_support_score",
                "rsi", "volume_ratio", "ema20_distance",
            ]
            if column in v15.columns
        ]
        current = current.merge(
            v15[extra_columns].drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )

    return current


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    result = history.copy()

    aliases = {
        "v8_score": ["v8_score", "selection_score"],
        "smart_money_score": ["smart_money_score"],
        "institutional_score": ["institutional_score"],
        "historical_support_score": ["historical_support_score"],
        "rsi": ["rsi"],
        "volume_ratio": ["volume_ratio"],
        "ema20_distance": ["ema20_distance"],
    }

    for target, options in aliases.items():
        if target in result.columns:
            continue
        for option in options:
            if option in result.columns:
                result[target] = result[option]
                break
        if target not in result.columns:
            result[target] = np.nan

    for horizon in [1, 3, 5, 10]:
        column = f"result_{horizon}d"
        result[column] = numeric(result, column)

    result = result[
        result["result_5d"].notna()
        | result["result_10d"].notna()
    ].copy()

    return result


FEATURES = [
    "v8_score",
    "smart_money_score",
    "institutional_score",
    "historical_support_score",
    "rsi",
    "volume_ratio",
    "ema20_distance",
]


def nearest_neighbors(
    row: pd.Series,
    history: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    usable: list[str] = []

    for feature in FEATURES:
        current_value = safe_float(row.get(feature), np.nan)
        if np.isnan(current_value):
            continue
        if feature not in history.columns:
            continue
        if pd.to_numeric(history[feature], errors="coerce").notna().sum() < MIN_NEIGHBORS:
            continue
        usable.append(feature)

    if len(usable) < 3:
        return pd.DataFrame(), usable

    standardized = []
    for feature in usable:
        values = pd.to_numeric(history[feature], errors="coerce")
        median = values.median()
        std = values.std()
        if not np.isfinite(std) or std == 0:
            std = 1.0
        current_value = safe_float(row.get(feature), median)
        standardized.append(((values.fillna(median) - current_value) / std) ** 2)

    distance = np.sqrt(sum(standardized) / len(standardized))
    neighbors = history.assign(_distance=distance).sort_values("_distance").head(MAX_NEIGHBORS)

    return neighbors, usable


def weighted_stats(neighbors: pd.DataFrame, horizon: int) -> dict[str, float]:
    column = f"result_{horizon}d"
    valid = neighbors.dropna(subset=[column]).copy()

    if len(valid) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "positive_rate": 0.0,
            "hit_3_rate": 0.0,
            "downside": 0.0,
            "upside": 0.0,
            "utility": -999.0,
        }

    weights = 1.0 / (valid["_distance"].astype(float) + 0.15)
    values = valid[column].astype(float)

    weighted_mean = float(np.average(values, weights=weights))
    positive_rate = float(np.average((values > 0).astype(float), weights=weights) * 100)
    hit_3_rate = float(np.average((values >= 3).astype(float), weights=weights) * 100)

    downside = float(values.quantile(0.20))
    upside = float(values.quantile(0.80))

    utility = (
        weighted_mean * 0.45
        + positive_rate / 100 * 2.0
        + hit_3_rate / 100 * 1.5
        + downside * 0.25
    )

    return {
        "count": int(len(valid)),
        "mean": round(weighted_mean, 3),
        "median": round(float(values.median()), 3),
        "positive_rate": round(positive_rate, 2),
        "hit_3_rate": round(hit_3_rate, 2),
        "downside": round(downside, 3),
        "upside": round(upside, 3),
        "utility": round(utility, 4),
    }


def timing_label(best_horizon: int, confidence: float) -> str:
    if confidence < 45:
        return "ZAMANLAMA GÃVENÄ° DÃÅÃK"
    if best_horizon == 1:
        return "ÃOK KISA VADE"
    if best_horizon == 3:
        return "KISA VADE"
    if best_horizon == 5:
        return "ORTA VADE"
    return "SABIRLI Ä°ZLEME"


def main() -> None:
    print("V19 Timing Forecast Engine baÅladÄ±.")

    current = build_current()
    history = prepare_history(load_csv(HISTORY_FILE))

    if current.empty or history.empty:
        pd.DataFrame().to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )
        with open(STATUS_FILE, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "status": "input_missing",
                    "candidate_count": len(current),
                    "history_count": len(history),
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        return

    rows: list[dict[str, Any]] = []

    for _, row in current.iterrows():
        neighbors, used_features = nearest_neighbors(row, history)

        if len(neighbors) < MIN_NEIGHBORS:
            rows.append({
                "symbol": clean_text(row.get("symbol")),
                "close": safe_float(row.get("close")),
                "v18_decision": clean_text(row.get("v18_decision")),
                "confidence_score": safe_float(row.get("confidence_score")),
                "timing_ready": False,
                "neighbor_count": len(neighbors),
                "used_feature_count": len(used_features),
                "used_features": " | ".join(used_features),
                "timing_message": "Yeterli benzer tarihsel Ã¶rnek bulunamadÄ±.",
            })
            continue

        stats = {horizon: weighted_stats(neighbors, horizon) for horizon in [1, 3, 5, 10]}
        best_horizon = max(stats, key=lambda horizon: stats[horizon]["utility"])
        best = stats[best_horizon]

        utility_values = sorted(
            [item["utility"] for item in stats.values()],
            reverse=True,
        )
        separation = utility_values[0] - utility_values[1] if len(utility_values) > 1 else 0.0

        timing_confidence = float(np.clip(
            35
            + min(25, len(neighbors) / MAX_NEIGHBORS * 25)
            + min(20, len(used_features) / len(FEATURES) * 20)
            + min(20, max(0.0, separation) * 8),
            0,
            100,
        ))

        rows.append({
            "symbol": clean_text(row.get("symbol")),
            "close": round(safe_float(row.get("close")), 4),
            "v18_decision": clean_text(row.get("v18_decision")),
            "confidence_score": round(safe_float(row.get("confidence_score")), 2),
            "timing_ready": True,
            "neighbor_count": len(neighbors),
            "used_feature_count": len(used_features),
            "used_features": " | ".join(used_features),
            "best_horizon_days": best_horizon,
            "timing_class": timing_label(best_horizon, timing_confidence),
            "timing_confidence": round(timing_confidence, 2),
            "expected_return": best["mean"],
            "median_return": best["median"],
            "positive_rate": best["positive_rate"],
            "hit_3_rate": best["hit_3_rate"],
            "downside_20pct": best["downside"],
            "upside_80pct": best["upside"],
            "result_1d_mean": stats[1]["mean"],
            "result_3d_mean": stats[3]["mean"],
            "result_5d_mean": stats[5]["mean"],
            "result_10d_mean": stats[10]["mean"],
            "timing_message": (
                f"Benzer Ã¶rneklerde en yÃ¼ksek risk ayarlÄ± sonuÃ§ "
                f"{best_horizon} iÅlem gÃ¼nÃ¼nde oluÅtu."
            ),
        })

    result = pd.DataFrame(rows)

    if "timing_confidence" in result.columns:
        result = result.sort_values(
            ["timing_ready", "timing_confidence", "confidence_score"],
            ascending=False,
        ).reset_index(drop=True)

    result.insert(0, "rank", range(1, len(result) + 1))
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    status = {
        "status": "ready",
        "candidate_count": len(result),
        "timing_ready_count": int(result.get("timing_ready", pd.Series(dtype=bool)).fillna(False).sum()),
        "history_count": len(history),
        "minimum_neighbors": MIN_NEIGHBORS,
    }

    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(status, file, ensure_ascii=False, indent=2)

    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
