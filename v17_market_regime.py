from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MARKET_FILE = Path("v16_full_market_snapshot.csv")
CANDIDATE_FILE = Path("v16_relative_strength.csv")
RESULT_FILE = Path("v17_regime_adjusted_decisions.csv")
STATUS_FILE = Path("v17_market_regime_status.json")


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


def positive_ratio(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return 0.0 if values.empty else float((values > 0).mean() * 100)


def detect_regime(market: pd.DataFrame) -> dict[str, Any]:
    for column in [
        "return_1d", "return_5d", "return_20d",
        "ema20_distance", "rsi", "volume_ratio",
    ]:
        if column not in market.columns:
            market[column] = np.nan
        market[column] = pd.to_numeric(market[column], errors="coerce")

    breadth_1d = positive_ratio(market["return_1d"])
    breadth_5d = positive_ratio(market["return_5d"])
    breadth_20d = positive_ratio(market["return_20d"])
    above_ema20 = positive_ratio(market["ema20_distance"])

    median_1d = number(market["return_1d"].median())
    median_5d = number(market["return_5d"].median())
    median_20d = number(market["return_20d"].median())
    median_rsi = number(market["rsi"].median(), 50.0)

    if breadth_1d <= 32 or median_1d <= -2:
        regime = "PANÄ°K"
        confidence = min(100.0, 60 + (50 - breadth_1d) * 1.2 + abs(median_1d) * 6)
    elif breadth_1d >= 68 and breadth_5d >= 60 and median_5d > 1.5:
        regime = "RALLÄ°"
        confidence = min(100.0, 60 + (breadth_1d - 60) + (breadth_5d - 55) * 0.5)
    elif breadth_20d >= 58 and above_ema20 >= 58 and median_20d > 2:
        regime = "TREND"
        confidence = min(100.0, 60 + (breadth_20d - 55) * 0.8 + (above_ema20 - 55) * 0.8)
    else:
        regime = "YATAY"
        confidence = max(50.0, 82 - abs(breadth_1d - 50) * 0.5 - abs(breadth_5d - 50) * 0.4)

    return {
        "regime": regime,
        "regime_confidence": round(confidence, 2),
        "market_count": int(len(market)),
        "breadth_1d_positive_pct": round(breadth_1d, 2),
        "breadth_5d_positive_pct": round(breadth_5d, 2),
        "breadth_20d_positive_pct": round(breadth_20d, 2),
        "above_ema20_pct": round(above_ema20, 2),
        "median_return_1d": round(median_1d, 2),
        "median_return_5d": round(median_5d, 2),
        "median_return_20d": round(median_20d, 2),
        "median_rsi": round(median_rsi, 2),
        "comparison_scope": "FULL_MARKET",
    }


def adjustment(row: pd.Series, regime: str) -> tuple[float, str]:
    momentum = number(row.get("momentum_percentile"), 50)
    trend = number(row.get("trend_percentile"), 50)
    volume = number(row.get("volume_percentile"), 50)
    quality = number(row.get("quality_percentile"), 50)
    market_pct = number(row.get("market_percentile"), 50)

    score = 0.0
    reasons: list[str] = []

    if regime == "RALLÄ°":
        if momentum >= 75:
            score += 5
            reasons.append("Ralli rejiminde gÃ¼Ã§lÃ¼ momentum")
        if volume >= 70:
            score += 3
            reasons.append("Ralli rejiminde hacim desteÄi")
        if market_pct < 55:
            score -= 4
            reasons.append("Ralli rejiminde piyasa gerisinde")
    elif regime == "TREND":
        if trend >= 75:
            score += 5
            reasons.append("Trend rejiminde gÃ¼Ã§lÃ¼ trend")
        if quality >= 70:
            score += 3
            reasons.append("Trend rejiminde kalite desteÄi")
    elif regime == "YATAY":
        if quality >= 70:
            score += 4
            reasons.append("Yatay rejimde kalite avantajÄ±")
        if volume >= 70:
            score += 2
            reasons.append("Yatay rejimde hacim birikimi")
        if momentum >= 90:
            score -= 3
            reasons.append("Yatay rejimde aÅÄ±rÄ± momentum")
    else:
        score -= 8
        reasons.append("Panik rejimi genel risk kesintisi")
        if quality >= 80:
            score += 3
            reasons.append("Panik rejiminde yÃ¼ksek kalite")
        if market_pct >= 90:
            score += 2
            reasons.append("Panik rejiminde piyasa liderliÄi")

    return round(score, 2), " | ".join(reasons)


def classify(score: float, regime: str) -> str:
    thresholds = {
        "PANÄ°K": (86, 77, 64),
        "RALLÄ°": (77, 67, 56),
        "TREND": (78, 68, 57),
        "YATAY": (80, 70, 58),
    }
    strong, approved, cautious = thresholds[regime]
    if score >= strong:
        return "V17 GÃÃLÃ ONAY"
    if score >= approved:
        return "V17 ONAYLI Ä°ZLEME"
    if score >= cautious:
        return "V17 TEMKÄ°NLÄ° Ä°ZLEME"
    return "V17 ELE"


def main() -> None:
    market = load_csv(MARKET_FILE)
    candidates = load_csv(CANDIDATE_FILE)

    if market.empty:
        raise RuntimeError("v16_full_market_snapshot.csv bulunamadÄ± veya boÅ.")

    status = detect_regime(market)

    rows: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        regime_effect, reasons = adjustment(row, status["regime"])
        v15_score = number(row.get("v15_score"))
        relative_score = number(row.get("relative_strength_score"))
        v17_score = float(np.clip(v15_score * 0.72 + relative_score * 0.28 + regime_effect, 0, 100))

        rows.append({
            "symbol": str(row.get("symbol", "")).strip(),
            "close": number(row.get("close")),
            "regime": status["regime"],
            "regime_confidence": status["regime_confidence"],
            "market_rank": int(number(row.get("market_rank"))),
            "market_percentile": number(row.get("market_percentile")),
            "relative_strength_score": relative_score,
            "relative_class": str(row.get("relative_class", "")).strip(),
            "momentum_percentile": number(row.get("momentum_percentile")),
            "trend_percentile": number(row.get("trend_percentile")),
            "volume_percentile": number(row.get("volume_percentile")),
            "quality_percentile": number(row.get("quality_percentile")),
            "v15_score": v15_score,
            "v15_decision": str(row.get("v15_decision", "")).strip(),
            "regime_adjustment": regime_effect,
            "v17_score": round(v17_score, 2),
            "v17_decision": classify(v17_score, status["regime"]),
            "regime_reasons": reasons,
        })

    result = pd.DataFrame(rows)
    if not result.empty:
        order = {
            "V17 GÃÃLÃ ONAY": 4,
            "V17 ONAYLI Ä°ZLEME": 3,
            "V17 TEMKÄ°NLÄ° Ä°ZLEME": 2,
            "V17 ELE": 1,
        }
        result["_order"] = result["v17_decision"].map(order)
        result = result.sort_values(["_order", "v17_score"], ascending=False).drop(columns="_order").reset_index(drop=True)
        result.insert(0, "rank", range(1, len(result) + 1))

    result.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")
    status.update({
        "status": "ready",
        "candidate_count": len(result),
        "approved_count": int(result.get("v17_decision", pd.Series(dtype=str)).isin(["V17 GÃÃLÃ ONAY", "V17 ONAYLI Ä°ZLEME"]).sum()),
    })
    STATUS_FILE.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
