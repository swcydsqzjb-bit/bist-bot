from __future__ import annotations

from typing import Any

import numpy as np


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)
        if not np.isfinite(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def clamp(value: float) -> float:
    return float(np.clip(value, 0.0, 100.0))


def classify_risk(
    risk_score: float,
) -> str:
    if risk_score <= 25:
        return "DÜŞÜK"

    if risk_score <= 55:
        return "ORTA"

    return "YÜKSEK"


def calculate_risk(
    row: Any,
) -> tuple[float, str, list[str]]:
    risk = 40.0
    reasons: list[str] = []

    regime = str(
        row.get("regime", "")
    ).upper()

    confidence = safe_float(
        row.get("confidence_score"),
        50.0,
    )

    consensus = safe_float(
        row.get("consensus_score"),
        50.0,
    )

    relative = safe_float(
        row.get("relative_strength_score"),
        50.0,
    )

    market_percentile = safe_float(
        row.get("market_percentile"),
        50.0,
    )

    downside = safe_float(
        row.get("downside_20pct"),
        0.0,
    )

    timing_confidence = safe_float(
        row.get("timing_confidence"),
        0.0,
    )

    horizon = int(
        safe_float(
            row.get("best_horizon_days"),
            0,
        )
    )

    if regime == "PANİK":
        risk += 22.0
        reasons.append(
            "Piyasa rejimi panik"
        )
    elif regime == "YATAY":
        risk += 6.0
        reasons.append(
            "Piyasa rejimi yatay"
        )
    elif regime == "RALLİ":
        risk -= 8.0
        reasons.append(
            "Piyasa rejimi ralli"
        )
    elif regime == "TREND":
        risk -= 5.0
        reasons.append(
            "Piyasa rejimi trend"
        )

    if confidence >= 80:
        risk -= 12.0
        reasons.append(
            "V18 güveni yüksek"
        )
    elif confidence < 55:
        risk += 12.0
        reasons.append(
            "V18 güveni zayıf"
        )

    if consensus >= 80:
        risk -= 10.0
        reasons.append(
            "Motorlar yüksek uyumda"
        )
    elif consensus < 50:
        risk += 15.0
        reasons.append(
            "Motorlar arasında uyumsuzluk var"
        )

    if relative >= 75:
        risk -= 8.0
        reasons.append(
            "Göreli güç yüksek"
        )
    elif relative < 50:
        risk += 10.0
        reasons.append(
            "Göreli güç zayıf"
        )

    if market_percentile >= 90:
        risk -= 6.0
        reasons.append(
            "Piyasanın üst %10 diliminde"
        )
    elif market_percentile < 45:
        risk += 8.0
        reasons.append(
            "Piyasanın alt yarısında"
        )

    if downside <= -8:
        risk += 18.0
        reasons.append(
            "Tarihsel temkinli senaryo sert"
        )
    elif downside <= -4:
        risk += 9.0
        reasons.append(
            "Tarihsel aşağı yön riski orta"
        )
    elif downside > -2:
        risk -= 4.0
        reasons.append(
            "Tarihsel aşağı yön sınırlı"
        )

    if timing_confidence >= 70:
        risk -= 5.0
        reasons.append(
            "Zamanlama güveni yüksek"
        )
    elif timing_confidence > 0 and timing_confidence < 50:
        risk += 5.0
        reasons.append(
            "Zamanlama güveni düşük"
        )

    if horizon >= 10:
        risk += 4.0
        reasons.append(
            "Önerilen ufuk uzun"
        )

    risk = clamp(risk)

    return (
        round(risk, 2),
        classify_risk(risk),
        reasons,
    )
