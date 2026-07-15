from __future__ import annotations

import json
import os
from typing import Any, Iterable

import numpy as np
import pandas as pd


MARKET_FILE = "v8_fusion_results.csv"
V16_FILE = "v16_relative_strength.csv"

RESULT_FILE = "v17_regime_adjusted_decisions.csv"
STATUS_FILE = "v17_market_regime_status.json"


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


def first_existing(
    frame: pd.DataFrame,
    names: Iterable[str],
) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]

    return pd.Series(
        [np.nan] * len(frame),
        index=frame.index,
    )


def normalize_symbol(value: Any) -> str:
    return clean_text(value).upper().replace(".IS", "")


def prepare_market(frame: pd.DataFrame) -> pd.DataFrame:
    market = pd.DataFrame(index=frame.index)

    market["symbol"] = first_existing(
        frame,
        ["symbol", "ticker"],
    ).map(normalize_symbol)

    market["return_1d"] = pd.to_numeric(
        first_existing(
            frame,
            ["return_1d", "change_1d", "daily_return"],
        ),
        errors="coerce",
    )

    market["return_5d"] = pd.to_numeric(
        first_existing(
            frame,
            ["return_5d", "change_5d"],
        ),
        errors="coerce",
    )

    market["return_20d"] = pd.to_numeric(
        first_existing(
            frame,
            ["return_20d", "change_20d"],
        ),
        errors="coerce",
    )

    market["rsi"] = pd.to_numeric(
        first_existing(frame, ["rsi"]),
        errors="coerce",
    )

    market["volume_ratio"] = pd.to_numeric(
        first_existing(
            frame,
            ["volume_ratio", "daily_volume_ratio"],
        ),
        errors="coerce",
    )

    market["ema20_distance"] = pd.to_numeric(
        first_existing(
            frame,
            [
                "ema20_distance",
                "ema20_dist",
                "ema20_distance_pct",
            ],
        ),
        errors="coerce",
    )

    market["smart_money_score"] = pd.to_numeric(
        first_existing(
            frame,
            ["smart_money_score"],
        ),
        errors="coerce",
    )

    market["institutional_score"] = pd.to_numeric(
        first_existing(
            frame,
            [
                "institutional_score",
                "institutional_accumulation_score",
            ],
        ),
        errors="coerce",
    )

    market = (
        market[
            market["symbol"].ne("")
        ]
        .drop_duplicates("symbol", keep="first")
        .reset_index(drop=True)
    )

    return market


def ratio(condition: pd.Series) -> float:
    valid = condition.dropna()

    if len(valid) == 0:
        return 0.0

    return float(valid.mean() * 100.0)


def detect_regime(
    market: pd.DataFrame,
) -> dict:
    return_1d = market["return_1d"]
    return_5d = market["return_5d"]
    return_20d = market["return_20d"]
    rsi = market["rsi"]
    volume_ratio = market["volume_ratio"]
    ema20_distance = market["ema20_distance"]

    breadth_1d = ratio(return_1d > 0)
    breadth_5d = ratio(return_5d > 0)
    breadth_20d = ratio(return_20d > 0)
    above_ema20 = ratio(ema20_distance > 0)
    strong_momentum = ratio(return_5d >= 3)
    weak_momentum = ratio(return_5d <= -3)

    median_1d = safe_float(return_1d.median())
    median_5d = safe_float(return_5d.median())
    median_20d = safe_float(return_20d.median())
    median_rsi = safe_float(rsi.median(), 50.0)
    median_volume = safe_float(volume_ratio.median(), 1.0)

    panic_score = (
        max(0.0, 50.0 - breadth_1d) * 0.7
        + max(0.0, -median_1d) * 8.0
        + max(0.0, weak_momentum - 25.0) * 0.5
    )

    rally_score = (
        max(0.0, breadth_1d - 55.0) * 0.7
        + max(0.0, breadth_5d - 55.0) * 0.5
        + max(0.0, median_5d) * 5.0
        + max(0.0, strong_momentum - 25.0) * 0.4
    )

    trend_score = (
        max(0.0, breadth_20d - 50.0) * 0.5
        + max(0.0, above_ema20 - 50.0) * 0.6
        + max(0.0, median_20d) * 3.0
    )

    if (
        breadth_1d <= 32
        or median_1d <= -2.0
        or panic_score >= 25
    ):
        regime = "PANÄ°K"
        confidence = min(
            100.0,
            55.0 + panic_score,
        )

    elif (
        breadth_1d >= 68
        and breadth_5d >= 60
        and median_5d > 1.5
    ):
        regime = "RALLÄ°"
        confidence = min(
            100.0,
            55.0 + rally_score,
        )

    elif (
        breadth_20d >= 58
        and above_ema20 >= 58
        and median_20d > 2.0
    ):
        regime = "TREND"
        confidence = min(
            100.0,
            55.0 + trend_score,
        )

    else:
        regime = "YATAY"
        balance = (
            abs(breadth_1d - 50.0)
            + abs(breadth_5d - 50.0)
            + abs(above_ema20 - 50.0)
        )
        confidence = max(
            50.0,
            min(85.0, 80.0 - balance * 0.35),
        )

    return {
        "regime": regime,
        "regime_confidence": round(confidence, 2),
        "market_count": int(len(market)),
        "breadth_1d_positive_pct": round(breadth_1d, 2),
        "breadth_5d_positive_pct": round(breadth_5d, 2),
        "breadth_20d_positive_pct": round(breadth_20d, 2),
        "above_ema20_pct": round(above_ema20, 2),
        "strong_momentum_pct": round(strong_momentum, 2),
        "weak_momentum_pct": round(weak_momentum, 2),
        "median_return_1d": round(median_1d, 2),
        "median_return_5d": round(median_5d, 2),
        "median_return_20d": round(median_20d, 2),
        "median_rsi": round(median_rsi, 2),
        "median_volume_ratio": round(median_volume, 2),
    }


def adjustment_for(
    row: pd.Series,
    regime: str,
) -> tuple[float, list[str]]:
    adjustment = 0.0
    reasons: list[str] = []

    momentum = safe_float(
        row.get("momentum_percentile"),
        50.0,
    )
    trend = safe_float(
        row.get("trend_percentile"),
        50.0,
    )
    volume = safe_float(
        row.get("volume_percentile"),
        50.0,
    )
    quality = safe_float(
        row.get("quality_percentile"),
        50.0,
    )
    market_percentile = safe_float(
        row.get("market_percentile"),
        50.0,
    )
    relative_score = safe_float(
        row.get("relative_strength_score"),
        50.0,
    )

    if regime == "RALLÄ°":
        if momentum >= 75:
            adjustment += 5.0
            reasons.append(
                "Ralli rejiminde gÃ¼Ã§lÃ¼ momentum"
            )

        if volume >= 70:
            adjustment += 3.0
            reasons.append(
                "Ralli rejiminde hacim desteÄi"
            )

        if market_percentile < 55:
            adjustment -= 4.0
            reasons.append(
                "Ralli rejiminde piyasa gerisinde"
            )

    elif regime == "TREND":
        if trend >= 75:
            adjustment += 5.0
            reasons.append(
                "Trend rejiminde gÃ¼Ã§lÃ¼ trend"
            )

        if quality >= 70:
            adjustment += 2.5
            reasons.append(
                "Trend rejiminde kalite desteÄi"
            )

        if relative_score < 50:
            adjustment -= 4.0
            reasons.append(
                "Trend rejiminde gÃ¶reli gÃ¼Ã§ zayÄ±f"
            )

    elif regime == "YATAY":
        if quality >= 70:
            adjustment += 4.0
            reasons.append(
                "Yatay rejimde kalite avantajÄ±"
            )

        if volume >= 70:
            adjustment += 2.0
            reasons.append(
                "Yatay rejimde hacim birikimi"
            )

        if momentum >= 90:
            adjustment -= 3.0
            reasons.append(
                "Yatay rejimde aÅÄ±rÄ± kÄ±sa vadeli momentum"
            )

    elif regime == "PANÄ°K":
        adjustment -= 8.0
        reasons.append(
            "Panik rejimi genel risk kesintisi"
        )

        if quality >= 80:
            adjustment += 3.0
            reasons.append(
                "Panik rejiminde yÃ¼ksek kalite"
            )

        if market_percentile >= 90:
            adjustment += 2.0
            reasons.append(
                "Panik rejiminde piyasa liderliÄi"
            )

        if momentum < 50:
            adjustment -= 3.0
            reasons.append(
                "Panik rejiminde momentum zayÄ±f"
            )

    return round(adjustment, 2), reasons


def classify(
    score: float,
    regime: str,
) -> str:
    strong_threshold = 80.0
    approved_threshold = 70.0
    cautious_threshold = 58.0

    if regime == "PANÄ°K":
        strong_threshold = 86.0
        approved_threshold = 77.0
        cautious_threshold = 64.0

    elif regime == "RALLÄ°":
        strong_threshold = 77.0
        approved_threshold = 67.0
        cautious_threshold = 56.0

    elif regime == "TREND":
        strong_threshold = 78.0
        approved_threshold = 68.0
        cautious_threshold = 57.0

    if score >= strong_threshold:
        return "V17 GÃÃLÃ ONAY"

    if score >= approved_threshold:
        return "V17 ONAYLI Ä°ZLEME"

    if score >= cautious_threshold:
        return "V17 TEMKÄ°NLÄ° Ä°ZLEME"

    return "V17 ELE"


def main() -> None:
    print("V17 Market Regime Engine baÅladÄ±.")

    market_raw = load_csv(MARKET_FILE)
    v16 = load_csv(V16_FILE)

    if market_raw.empty:
        pd.DataFrame().to_csv(
            RESULT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        with open(
            STATUS_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                {
                    "status": "market_missing",
                    "regime": "BÄ°LÄ°NMÄ°YOR",
                    "candidate_count": 0,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )
        return

    market = prepare_market(market_raw)
    regime_info = detect_regime(market)

    if v16.empty:
        pd.DataFrame().to_csv(
            RESULT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        regime_info.update({
            "status": "v16_missing",
            "candidate_count": 0,
        })

        with open(
            STATUS_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                regime_info,
                file,
                ensure_ascii=False,
                indent=2,
            )
        return

    rows = []

    for _, row in v16.iterrows():
        adjustment, reasons = adjustment_for(
            row,
            regime_info["regime"],
        )

        v15_score = safe_float(
            row.get("v15_score"),
            0.0,
        )
        relative_score = safe_float(
            row.get("relative_strength_score"),
            0.0,
        )

        base_score = (
            v15_score * 0.72
            + relative_score * 0.28
        )

        final_score = float(
            np.clip(
                base_score + adjustment,
                0.0,
                100.0,
            )
        )

        decision = classify(
            final_score,
            regime_info["regime"],
        )

        rows.append({
            "symbol": clean_text(row.get("symbol")),
            "close": round(
                safe_float(row.get("close")),
                4,
            ),
            "regime": regime_info["regime"],
            "regime_confidence": (
                regime_info["regime_confidence"]
            ),
            "market_rank": int(
                safe_float(row.get("market_rank"))
            ),
            "market_percentile": round(
                safe_float(
                    row.get("market_percentile")
                ),
                2,
            ),
            "relative_class": clean_text(
                row.get("relative_class")
            ),
            "relative_strength_score": round(
                relative_score,
                2,
            ),
            "momentum_percentile": round(
                safe_float(
                    row.get("momentum_percentile")
                ),
                2,
            ),
            "trend_percentile": round(
                safe_float(
                    row.get("trend_percentile")
                ),
                2,
            ),
            "volume_percentile": round(
                safe_float(
                    row.get("volume_percentile")
                ),
                2,
            ),
            "quality_percentile": round(
                safe_float(
                    row.get("quality_percentile")
                ),
                2,
            ),
            "v15_score": round(
                v15_score,
                2,
            ),
            "v15_decision": clean_text(
                row.get("v15_decision")
            ),
            "regime_adjustment": adjustment,
            "v17_score": round(
                final_score,
                2,
            ),
            "v17_decision": decision,
            "regime_reasons": " | ".join(
                reasons
            ),
        })

    result = pd.DataFrame(rows)

    priority = {
        "V17 GÃÃLÃ ONAY": 4,
        "V17 ONAYLI Ä°ZLEME": 3,
        "V17 TEMKÄ°NLÄ° Ä°ZLEME": 2,
        "V17 ELE": 1,
    }

    result["_priority"] = (
        result["v17_decision"]
        .map(priority)
        .fillna(0)
    )

    result = (
        result
        .sort_values(
            ["_priority", "v17_score"],
            ascending=False,
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "rank",
        range(1, len(result) + 1),
    )

    result.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    regime_info.update({
        "status": "ready",
        "candidate_count": len(result),
        "approved_count": int(
            result["v17_decision"].isin(
                [
                    "V17 GÃÃLÃ ONAY",
                    "V17 ONAYLI Ä°ZLEME",
                ]
            ).sum()
        ),
    })

    with open(
        STATUS_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            regime_info,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n===== V17 MARKET REGIME =====")
    print(
        json.dumps(
            regime_info,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n===== V17 KARARLARI =====")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
