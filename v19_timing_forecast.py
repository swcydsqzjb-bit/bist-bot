from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = Path("v18_confidence_decisions.csv")
OUTPUT_FILE = Path("v19_timing_forecasts.csv")
STATUS_FILE = Path("v19_timing_status.json")

MIN_NEIGHBORS = 8
MAX_NEIGHBORS = 35
FEATURES = [
    "v8_score",
    "smart_money_score",
    "institutional_score",
    "historical_support_score",
    "rsi",
    "volume_ratio",
    "ema20_distance",
]


def number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return default if not np.isfinite(result) else result
    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def coalesce(frame: pd.DataFrame, names: list[str]) -> pd.Series:
    result = pd.Series(np.nan, index=frame.index)
    for name in names:
        if name in frame.columns:
            result = result.fillna(pd.to_numeric(frame[name], errors="coerce"))
    return result


def load_history() -> tuple[pd.DataFrame, list[str]]:
    prepared: list[pd.DataFrame] = []
    used_files: list[str] = []

    for filename in [
        "v5_backfill_history.csv",
        "v11_signal_history.csv",
        "signals_history.csv",
        "v3_signals_history.csv",
    ]:
        path = Path(filename)
        frame = load_csv(path)
        if frame.empty:
            continue

        normalized = pd.DataFrame(index=frame.index)
        normalized["symbol"] = frame["symbol"].astype(str) if "symbol" in frame.columns else ""

        normalized["v8_score"] = coalesce(frame, ["v8_score", "selection_score"])
        normalized["smart_money_score"] = coalesce(frame, ["smart_money_score", "selection_score"])
        normalized["institutional_score"] = coalesce(frame, ["institutional_score", "institutional_accumulation_score", "volume_score"])
        normalized["historical_support_score"] = coalesce(frame, ["historical_support_score", "historical_score", "compression_score"])
        normalized["rsi"] = coalesce(frame, ["rsi"])
        normalized["volume_ratio"] = coalesce(frame, ["volume_ratio", "volume_accumulation_ratio"])
        normalized["ema20_distance"] = coalesce(frame, ["ema20_distance", "ema20_distance_pct"])

        for horizon in [1, 3, 5, 10]:
            normalized[f"result_{horizon}d"] = coalesce(
                frame,
                [f"result_{horizon}d", f"return_{horizon}d_at_signal", f"return_{horizon}d"],
            )

        valid = normalized[[f"result_{h}d" for h in [1, 3, 5, 10]]].notna().any(axis=1)
        normalized = normalized[valid].copy()

        if not normalized.empty:
            normalized["_source_file"] = filename
            prepared.append(normalized)
            used_files.append(filename)

    if not prepared:
        return pd.DataFrame(), used_files

    return pd.concat(prepared, ignore_index=True).drop_duplicates(), used_files


def enrich_current(current: pd.DataFrame) -> pd.DataFrame:
    result = current.copy()
    result["symbol"] = result["symbol"].astype(str).str.upper().str.replace(".IS", "", regex=False)

    for filename in [
        "v15_final_decisions.csv",
        "v13_market_dna_results.csv",
        "v8_fusion_results.csv",
    ]:
        frame = load_csv(Path(filename))
        if frame.empty or "symbol" not in frame.columns:
            continue

        frame["symbol"] = frame["symbol"].astype(str).str.upper().str.replace(".IS", "", regex=False)
        available = ["symbol"] + [feature for feature in FEATURES if feature in frame.columns and feature not in result.columns]
        if len(available) > 1:
            result = result.merge(frame[available].drop_duplicates("symbol"), on="symbol", how="left")

    return result


def nearest_neighbors(row: pd.Series, history: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    usable: list[str] = []

    for feature in FEATURES:
        current_value = number(row.get(feature), np.nan)
        if np.isnan(current_value) or feature not in history.columns:
            continue
        if pd.to_numeric(history[feature], errors="coerce").notna().sum() >= MIN_NEIGHBORS:
            usable.append(feature)

    if len(usable) < 2:
        return pd.DataFrame(), usable

    components = []
    for feature in usable:
        values = pd.to_numeric(history[feature], errors="coerce")
        median = values.median()
        std = values.std()
        if not np.isfinite(std) or std == 0:
            std = 1.0
        current_value = number(row.get(feature), median)
        components.append(((values.fillna(median) - current_value) / std) ** 2)

    distance = np.sqrt(sum(components) / len(components))
    return history.assign(_distance=distance).sort_values("_distance").head(MAX_NEIGHBORS), usable


def stats(neighbors: pd.DataFrame, horizon: int) -> dict[str, float]:
    column = f"result_{horizon}d"
    valid = neighbors.dropna(subset=[column]).copy()

    if len(valid) < MIN_NEIGHBORS:
        return {"count": len(valid), "utility": -999.0, "mean": 0.0, "median": 0.0, "positive": 0.0, "hit3": 0.0, "downside": 0.0, "upside": 0.0}

    values = valid[column].astype(float)
    weights = 1.0 / (valid["_distance"].astype(float) + 0.15)

    mean = float(np.average(values, weights=weights))
    positive = float(np.average((values > 0).astype(float), weights=weights) * 100)
    hit3 = float(np.average((values >= 3).astype(float), weights=weights) * 100)
    downside = float(values.quantile(0.20))
    upside = float(values.quantile(0.80))
    utility = mean * 0.45 + positive / 100 * 2 + hit3 / 100 * 1.5 + downside * 0.25

    return {
        "count": len(valid),
        "utility": round(utility, 4),
        "mean": round(mean, 3),
        "median": round(float(values.median()), 3),
        "positive": round(positive, 2),
        "hit3": round(hit3, 2),
        "downside": round(downside, 3),
        "upside": round(upside, 3),
    }


def timing_class(horizon: int, confidence: float) -> str:
    if confidence < 45:
        return "ZAMANLAMA GÃVENÄ° DÃÅÃK"
    return {1: "ÃOK KISA VADE", 3: "KISA VADE", 5: "ORTA VADE", 10: "SABIRLI Ä°ZLEME"}[horizon]


def main() -> None:
    current = load_csv(CURRENT_FILE)
    history, used_files = load_history()

    if not current.empty:
        current = enrich_current(current)

    if current.empty or history.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        STATUS_FILE.write_text(json.dumps({
            "status": "input_missing",
            "candidate_count": len(current),
            "history_count": len(history),
            "history_files": used_files,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        return

    rows: list[dict[str, Any]] = []

    for _, row in current.iterrows():
        neighbors, used_features = nearest_neighbors(row, history)

        if len(neighbors) < MIN_NEIGHBORS:
            rows.append({
                "symbol": str(row.get("symbol", "")).strip(),
                "close": number(row.get("close")),
                "v18_decision": str(row.get("v18_decision", "")).strip(),
                "confidence_score": number(row.get("confidence_score")),
                "timing_ready": False,
                "neighbor_count": len(neighbors),
                "used_feature_count": len(used_features),
                "used_features": " | ".join(used_features),
                "timing_message": "Yeterli benzer tarihsel Ã¶rnek bulunamadÄ±.",
            })
            continue

        horizon_stats = {h: stats(neighbors, h) for h in [1, 3, 5, 10]}
        valid = {h: item for h, item in horizon_stats.items() if item["count"] >= MIN_NEIGHBORS}

        if not valid:
            rows.append({
                "symbol": str(row.get("symbol", "")).strip(),
                "close": number(row.get("close")),
                "v18_decision": str(row.get("v18_decision", "")).strip(),
                "confidence_score": number(row.get("confidence_score")),
                "timing_ready": False,
                "neighbor_count": len(neighbors),
                "used_feature_count": len(used_features),
                "used_features": " | ".join(used_features),
                "timing_message": "KomÅu bulundu fakat sonuÃ§ ufuklarÄ± tamamlanmamÄ±Å.",
            })
            continue

        best_horizon = max(valid, key=lambda h: valid[h]["utility"])
        best = valid[best_horizon]
        utilities = sorted([item["utility"] for item in valid.values()], reverse=True)
        separation = utilities[0] - utilities[1] if len(utilities) > 1 else 0.0

        confidence = float(np.clip(
            35
            + min(25, len(neighbors) / MAX_NEIGHBORS * 25)
            + min(20, len(used_features) / len(FEATURES) * 20)
            + min(20, max(0.0, separation) * 8),
            0,
            100,
        ))

        rows.append({
            "symbol": str(row.get("symbol", "")).strip(),
            "close": round(number(row.get("close")), 4),
            "v18_decision": str(row.get("v18_decision", "")).strip(),
            "confidence_score": round(number(row.get("confidence_score")), 2),
            "timing_ready": True,
            "neighbor_count": len(neighbors),
            "used_feature_count": len(used_features),
            "used_features": " | ".join(used_features),
            "best_horizon_days": best_horizon,
            "timing_class": timing_class(best_horizon, confidence),
            "timing_confidence": round(confidence, 2),
            "expected_return": best["mean"],
            "median_return": best["median"],
            "positive_rate": best["positive"],
            "hit_3_rate": best["hit3"],
            "downside_20pct": best["downside"],
            "upside_80pct": best["upside"],
            "result_1d_mean": horizon_stats[1]["mean"],
            "result_3d_mean": horizon_stats[3]["mean"],
            "result_5d_mean": horizon_stats[5]["mean"],
            "result_10d_mean": horizon_stats[10]["mean"],
            "timing_message": f"En yÃ¼ksek risk ayarlÄ± sonuÃ§ {best_horizon} iÅlem gÃ¼nÃ¼nde oluÅtu.",
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["timing_ready", "timing_confidence", "confidence_score"], ascending=False, na_position="last").reset_index(drop=True)
        result.insert(0, "rank", range(1, len(result) + 1))

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    status = {
        "status": "ready",
        "candidate_count": len(result),
        "timing_ready_count": int(result.get("timing_ready", pd.Series(dtype=bool)).fillna(False).sum()),
        "history_count": len(history),
        "history_files": used_files,
        "minimum_neighbors": MIN_NEIGHBORS,
    }
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
