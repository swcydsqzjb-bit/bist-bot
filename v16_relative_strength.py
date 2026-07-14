from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd

MARKET_FILE = "v8_fusion_results.csv"
V15_FILE = "v15_final_decisions.csv"
RESULT_FILE = "v16_relative_strength.csv"
STATUS_FILE = "v16_status.json"

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
    except (pd.errors.EmptyDataError, OSError):
        return pd.DataFrame()

def first_existing(frame: pd.DataFrame, names: Iterable[str]) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([np.nan] * len(frame), index=frame.index)

def normalize_symbol(value: Any) -> str:
    return clean_text(value).upper().replace(".IS", "")

def percentile_rank(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1:
        return pd.Series([50.0] * len(series), index=series.index)
    return numeric.rank(method="average", pct=True).fillna(0.5) * 100.0

def inverse_percentile_rank(series: pd.Series) -> pd.Series:
    return 100.0 - percentile_rank(series)

def prepare_market(frame: pd.DataFrame) -> pd.DataFrame:
    market = pd.DataFrame(index=frame.index)
    market["symbol"] = first_existing(frame, ["symbol", "ticker"]).map(normalize_symbol)
    market["close"] = pd.to_numeric(first_existing(frame, ["close", "price", "current_price", "signal_price"]), errors="coerce")
    market["v8_score"] = pd.to_numeric(first_existing(frame, ["v8_score", "final_v8_score", "final_score", "fusion_score"]), errors="coerce")
    market["smart_money_score"] = pd.to_numeric(first_existing(frame, ["smart_money_score"]), errors="coerce")
    market["institutional_score"] = pd.to_numeric(first_existing(frame, ["institutional_score", "institutional_accumulation_score"]), errors="coerce")
    market["historical_support_score"] = pd.to_numeric(first_existing(frame, ["historical_support_score", "historical_score"]), errors="coerce")
    market["rsi"] = pd.to_numeric(first_existing(frame, ["rsi"]), errors="coerce")
    market["volume_ratio"] = pd.to_numeric(first_existing(frame, ["volume_ratio", "daily_volume_ratio"]), errors="coerce")
    market["volume_accumulation_ratio"] = pd.to_numeric(first_existing(frame, ["volume_accumulation_ratio", "accumulation_ratio"]), errors="coerce")
    market["ema20_distance"] = pd.to_numeric(first_existing(frame, ["ema20_distance", "ema20_dist", "ema20_distance_pct"]), errors="coerce")
    market["return_1d"] = pd.to_numeric(first_existing(frame, ["return_1d", "change_1d", "daily_return"]), errors="coerce")
    market["return_5d"] = pd.to_numeric(first_existing(frame, ["return_5d", "change_5d"]), errors="coerce")
    market["return_20d"] = pd.to_numeric(first_existing(frame, ["return_20d", "change_20d"]), errors="coerce")
    market["range_20_pct"] = pd.to_numeric(first_existing(frame, ["range_20_pct"]), errors="coerce")
    market["upper_wick_ratio"] = pd.to_numeric(first_existing(frame, ["upper_wick_ratio"]), errors="coerce")
    market["close_position"] = pd.to_numeric(first_existing(frame, ["close_position"]), errors="coerce")
    market = market[market["symbol"].ne("")].drop_duplicates("symbol", keep="first")
    return market.reset_index(drop=True)

def calculate_relative_strength(market: pd.DataFrame) -> pd.DataFrame:
    result = market.copy()
    result["momentum_percentile"] = (
        percentile_rank(result["return_5d"]) * 0.45
        + percentile_rank(result["return_20d"]) * 0.40
        + percentile_rank(result["return_1d"]) * 0.15
    )
    healthy_rsi = (100.0 - (result["rsi"].fillna(50.0) - 60.0).abs() * 3.0).clip(0.0, 100.0)
    ema_health = (100.0 - (result["ema20_distance"].fillna(0.0) - 3.0).abs() * 5.0).clip(0.0, 100.0)
    result["trend_percentile"] = (
        percentile_rank(result["v8_score"]) * 0.35
        + percentile_rank(result["smart_money_score"]) * 0.20
        + percentile_rank(result["institutional_score"]) * 0.20
        + healthy_rsi * 0.10
        + ema_health * 0.15
    )
    result["volume_percentile"] = (
        percentile_rank(result["volume_ratio"]) * 0.45
        + percentile_rank(result["volume_accumulation_ratio"]) * 0.40
        + percentile_rank(result["close_position"]) * 0.15
    )
    result["quality_percentile"] = (
        percentile_rank(result["historical_support_score"]) * 0.35
        + inverse_percentile_rank(result["upper_wick_ratio"]) * 0.20
        + inverse_percentile_rank(result["range_20_pct"]) * 0.15
        + percentile_rank(result["institutional_score"]) * 0.30
    )
    result["relative_strength_score"] = (
        result["momentum_percentile"] * 0.30
        + result["trend_percentile"] * 0.30
        + result["volume_percentile"] * 0.20
        + result["quality_percentile"] * 0.20
    ).clip(0.0, 100.0)
    result["market_percentile"] = percentile_rank(result["relative_strength_score"])
    result["relative_class"] = np.select(
        [
            result["market_percentile"] >= 90,
            result["market_percentile"] >= 75,
            result["market_percentile"] >= 55,
        ],
        ["PÄ°YASA LÄ°DERÄ°", "GÃÃLÃ", "ORTA"],
        default="ZAYIF",
    )
    result = result.sort_values(["market_percentile", "relative_strength_score"], ascending=False).reset_index(drop=True)
    result.insert(0, "market_rank", range(1, len(result) + 1))
    return result

def attach_v15(relative: pd.DataFrame, v15: pd.DataFrame) -> pd.DataFrame:
    if v15.empty or "symbol" not in v15.columns:
        return pd.DataFrame()
    selected = v15.copy()
    selected["symbol"] = selected["symbol"].map(normalize_symbol)
    available = [c for c in ["symbol", "rank", "close", "v15_score", "v15_decision", "v14_score", "v14_decision", "dna_classification", "dna_confidence"] if c in selected.columns]
    selected = selected[available]
    merged = selected.merge(relative, on="symbol", how="left", suffixes=("_v15", ""))
    if "close_v15" in merged.columns:
        merged["close"] = pd.to_numeric(merged["close_v15"], errors="coerce").fillna(pd.to_numeric(merged.get("close"), errors="coerce"))
    final_columns = [
        "rank", "symbol", "close", "v15_score", "v15_decision", "v14_score",
        "v14_decision", "dna_classification", "dna_confidence", "market_rank",
        "market_percentile", "relative_strength_score", "relative_class",
        "momentum_percentile", "trend_percentile", "volume_percentile",
        "quality_percentile", "return_1d", "return_5d", "return_20d",
        "rsi", "volume_ratio", "volume_accumulation_ratio", "ema20_distance",
    ]
    for column in final_columns:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged[final_columns].sort_values(["market_percentile", "v15_score"], ascending=False).reset_index(drop=True)

def main() -> None:
    print("V16 Relative Strength Engine baÅladÄ±.")
    market_raw = load_csv(MARKET_FILE)
    v15 = load_csv(V15_FILE)
    if market_raw.empty:
        pd.DataFrame().to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
        with open(STATUS_FILE, "w", encoding="utf-8") as file:
            json.dump({"status": "market_file_missing", "market_count": 0, "candidate_count": 0}, file, ensure_ascii=False, indent=2)
        return
    market = prepare_market(market_raw)
    relative = calculate_relative_strength(market)
    final = attach_v15(relative, v15)
    final.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "status": "ready",
                "market_count": len(relative),
                "candidate_count": len(final),
                "leader_count": int((relative["market_percentile"] >= 90).sum()),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    print("\n===== V16 TÃM PÄ°YASA Ä°LK 20 =====")
    print(relative[["market_rank", "symbol", "market_percentile", "relative_strength_score", "relative_class"]].head(20).to_string(index=False))
    print("\n===== V16 V15 ADAYLARI =====")
    print(final.to_string(index=False))

if __name__ == "__main__":
    main()
