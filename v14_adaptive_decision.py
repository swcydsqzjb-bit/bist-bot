from __future__ import annotations

import json
import os
from typing import Any, List

import numpy as np
import pandas as pd


V8_FILE = "v8_today_candidates.csv"
V13_FILE = "v13_market_dna_results.csv"
V12_STATUS_FILE = "v12_status.json"

RESULT_FILE = "v14_adaptive_decisions.csv"
STATUS_FILE = "v14_status.json"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None or pd.isna(value):
        return default

    return str(value).strip().lower() in {
        "true", "1", "yes", "evet", "on"
    }


def clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
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


def load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def normalize_symbol(value: Any) -> str:
    return clean_text(value).upper().replace(".IS", "")


def first_column(
    frame: pd.DataFrame,
    names: List[str],
) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]

    return pd.Series(
        [np.nan] * len(frame),
        index=frame.index,
    )


def normalized_score(
    value: Any,
    minimum: float,
    maximum: float,
) -> float:
    if maximum <= minimum:
        return 0.0

    score = (
        (safe_float(value, minimum) - minimum)
        / (maximum - minimum)
        * 100.0
    )

    return float(np.clip(score, 0.0, 100.0))


def prepare_v8(frame: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=frame.index)

    result["symbol"] = (
        first_column(frame, ["symbol", "ticker"])
        .map(normalize_symbol)
    )

    result["close"] = pd.to_numeric(
        first_column(
            frame,
            ["close", "price", "signal_price", "current_price"],
        ),
        errors="coerce",
    )

    result["v8_score"] = pd.to_numeric(
        first_column(
            frame,
            [
                "v8_score",
                "final_v8_score",
                "final_score",
                "fusion_score",
            ],
        ),
        errors="coerce",
    )

    result["smart_money_score"] = pd.to_numeric(
        first_column(frame, ["smart_money_score"]),
        errors="coerce",
    )

    result["institutional_score"] = pd.to_numeric(
        first_column(
            frame,
            ["institutional_score", "institutional_accumulation_score"],
        ),
        errors="coerce",
    )

    result["historical_support_score"] = pd.to_numeric(
        first_column(
            frame,
            ["historical_support_score", "historical_score"],
        ),
        errors="coerce",
    )

    result["rsi"] = pd.to_numeric(
        first_column(frame, ["rsi"]),
        errors="coerce",
    )

    result["ema20_distance"] = pd.to_numeric(
        first_column(
            frame,
            ["ema20_distance", "ema20_dist", "ema20_distance_pct"],
        ),
        errors="coerce",
    )

    result["classification"] = first_column(
        frame,
        ["v8_classification", "classification"],
    ).map(clean_text)

    return result


def prepare_dna(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["symbol"])

    result = pd.DataFrame(index=frame.index)
    result["symbol"] = frame["symbol"].map(normalize_symbol)

    # V8'den gelen close ve v8_score alanlarini burada almiyoruz.
    # Boylece merge sonrasinda close_x / close_y sorunu olusmaz.
    dna_columns = [
        "dna_ready",
        "dna_classification",
        "dna_confidence",
        "positive_rate_5d",
        "hit_3pct_5d_rate",
        "average_result_5d",
        "average_max_result_5d",
        "average_min_result_5d",
    ]

    for column in dna_columns:
        if column in frame.columns:
            result[column] = frame[column]
        else:
            result[column] = np.nan

    return result


def calculate_bonus(row: pd.Series) -> tuple[float, List[str]]:
    bonus = 0.0
    reasons: List[str] = []

    positive_rate = safe_float(row.get("positive_rate_5d"))
    hit_rate = safe_float(row.get("hit_3pct_5d_rate"))
    average_result = safe_float(row.get("average_result_5d"))

    if positive_rate >= 70:
        bonus += 4.0
        reasons.append("5 gÃ¼nlÃ¼k pozitif oranÄ± gÃ¼Ã§lÃ¼")
    elif positive_rate >= 60:
        bonus += 2.0
        reasons.append("5 gÃ¼nlÃ¼k pozitif oranÄ± olumlu")

    if hit_rate >= 45:
        bonus += 3.0
        reasons.append("%3 hedef geÃ§miÅi gÃ¼Ã§lÃ¼")

    if average_result >= 2:
        bonus += 3.0
        reasons.append("Ortalama 5 gÃ¼nlÃ¼k getiri gÃ¼Ã§lÃ¼")
    elif average_result >= 1:
        bonus += 1.5
        reasons.append("Ortalama 5 gÃ¼nlÃ¼k getiri olumlu")

    return round(bonus, 2), reasons


def calculate_penalty(row: pd.Series) -> tuple[float, List[str]]:
    penalty = 0.0
    reasons: List[str] = []

    dna_ready = safe_bool(row.get("dna_ready"))
    dna_class = clean_text(
        row.get("dna_classification")
    ).upper()

    if not dna_ready:
        penalty += 12.0
        reasons.append("DNA sonucu hazÄ±r deÄil")

    if "ZAYIF" in dna_class:
        penalty += 15.0
        reasons.append("Market DNA zayÄ±f")
    elif "KARIÅIK" in dna_class or "KARISIK" in dna_class:
        penalty += 6.0
        reasons.append("Market DNA karÄ±ÅÄ±k")

    average_result = safe_float(row.get("average_result_5d"))

    if average_result < 0:
        penalty += 6.0
        reasons.append("Benzer Ã¶rneklerin ortalama sonucu negatif")

    rsi = safe_float(row.get("rsi"), 50.0)

    if rsi >= 76:
        penalty += 8.0
        reasons.append("RSI aÅÄ±rÄ± yÃ¼ksek")
    elif rsi >= 70:
        penalty += 4.0
        reasons.append("RSI yÃ¼ksek")

    ema_distance = safe_float(row.get("ema20_distance"))

    if ema_distance >= 15:
        penalty += 8.0
        reasons.append("Fiyat EMA20'den fazla uzak")
    elif ema_distance >= 10:
        penalty += 4.0
        reasons.append("EMA20 mesafesi yÃ¼ksek")

    return round(penalty, 2), reasons


def classify(
    score: float,
    penalty: float,
    dna_ready: bool,
) -> str:
    if score >= 78 and penalty <= 8 and dna_ready:
        return "GÃÃLÃ ONAY"

    if score >= 68 and penalty <= 14 and dna_ready:
        return "ONAYLI Ä°ZLEME"

    if score >= 58:
        return "TEMKÄ°NLÄ° Ä°ZLEME"

    return "ELE"


def main() -> None:
    print("V14 Adaptif Karar Motoru baÅladÄ±.")

    v8_raw = load_csv(V8_FILE)
    dna_raw = load_csv(V13_FILE)

    if v8_raw.empty:
        pd.DataFrame().to_csv(
            RESULT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        with open(STATUS_FILE, "w", encoding="utf-8") as file:
            json.dump(
                {"status": "v8_missing"},
                file,
                ensure_ascii=False,
                indent=2,
            )
        return

    v8 = prepare_v8(v8_raw)
    dna = prepare_dna(dna_raw)

    merged = v8.merge(
        dna,
        on="symbol",
        how="left",
    )

    mode = (
        "V12_ADAPTIVE"
        if load_json(V12_STATUS_FILE).get("status")
        == "recommendations_ready"
        else "BASE"
    )

    rows = []

    for _, row in merged.iterrows():
        raw_score = (
            normalized_score(row.get("v8_score"), 0, 100) * 0.42
            + normalized_score(
                row.get("dna_confidence"), 0, 100
            ) * 0.18
            + normalized_score(
                row.get("positive_rate_5d"), 0, 100
            ) * 0.12
            + normalized_score(
                row.get("hit_3pct_5d_rate"), 0, 100
            ) * 0.08
            + normalized_score(
                row.get("average_result_5d"), -5, 8
            ) * 0.08
            + normalized_score(
                row.get("smart_money_score"), 0, 100
            ) * 0.05
            + normalized_score(
                row.get("institutional_score"), 0, 100
            ) * 0.05
            + normalized_score(
                row.get("historical_support_score"), 0, 100
            ) * 0.02
        )

        bonus, positive_reasons = calculate_bonus(row)
        penalty, risk_reasons = calculate_penalty(row)

        final_score = float(
            np.clip(raw_score + bonus - penalty, 0, 100)
        )

        decision = classify(
            final_score,
            penalty,
            safe_bool(row.get("dna_ready")),
        )

        rows.append({
            "symbol": row.get("symbol"),
            "close": round(safe_float(row.get("close")), 4),
            "classification": clean_text(
                row.get("classification")
            ),
            "v8_score": round(
                safe_float(row.get("v8_score")), 2
            ),
            "smart_money_score": round(
                safe_float(row.get("smart_money_score")), 2
            ),
            "institutional_score": round(
                safe_float(row.get("institutional_score")), 2
            ),
            "dna_classification": clean_text(
                row.get("dna_classification")
            ),
            "dna_confidence": round(
                safe_float(row.get("dna_confidence")), 2
            ),
            "positive_rate_5d": round(
                safe_float(row.get("positive_rate_5d")), 2
            ),
            "hit_3pct_5d_rate": round(
                safe_float(row.get("hit_3pct_5d_rate")), 2
            ),
            "average_result_5d": round(
                safe_float(row.get("average_result_5d")), 2
            ),
            "weight_mode": mode,
            "raw_score": round(raw_score, 2),
            "positive_bonus": bonus,
            "risk_penalty": penalty,
            "v14_score": round(final_score, 2),
            "v14_decision": decision,
            "positive_reasons": " | ".join(positive_reasons),
            "risk_reasons": " | ".join(risk_reasons),
        })

    result = pd.DataFrame(rows)

    priority = {
        "GÃÃLÃ ONAY": 4,
        "ONAYLI Ä°ZLEME": 3,
        "TEMKÄ°NLÄ° Ä°ZLEME": 2,
        "ELE": 1,
    }

    result["_priority"] = (
        result["v14_decision"].map(priority).fillna(0)
    )

    result = (
        result
        .sort_values(
            ["_priority", "v14_score"],
            ascending=False,
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    result.insert(0, "rank", range(1, len(result) + 1))

    result.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(
            {
                "status": "ready",
                "weight_mode": mode,
                "candidate_count": len(result),
                "approved_count": int(
                    result["v14_decision"].isin(
                        ["GÃÃLÃ ONAY", "ONAYLI Ä°ZLEME"]
                    ).sum()
                ),
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
