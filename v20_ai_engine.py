from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from v20_consensus import calculate_consensus
from v20_risk import calculate_risk


V15_FILE = Path("v15_final_decisions.csv")
V16_FILE = Path("v16_relative_strength.csv")
V17_FILE = Path("v17_regime_adjusted_decisions.csv")
V18_FILE = Path("v18_confidence_decisions.csv")
V19_FILE = Path("v19_timing_forecasts.csv")

OUTPUT_FILE = Path("v20_ai_final_decisions.csv")
STATUS_FILE = Path("v20_ai_status.json")


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


def clean_text(
    value: Any,
) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        return pd.read_csv(
            path,
            encoding="utf-8",
        )


def normalize_symbol(
    value: Any,
) -> str:
    return (
        clean_text(value)
        .upper()
        .replace(".IS", "")
    )


def normalize_timing_score(
    row: pd.Series,
) -> float:
    timing_confidence = safe_float(
        row.get("timing_confidence"),
        0.0,
    )

    expected_return = safe_float(
        row.get("expected_return"),
        0.0,
    )

    positive_rate = safe_float(
        row.get("positive_rate"),
        0.0,
    )

    return_score = float(
        np.clip(
            (expected_return + 5.0)
            / 15.0
            * 100.0,
            0.0,
            100.0,
        )
    )

    return float(
        np.clip(
            timing_confidence * 0.45
            + positive_rate * 0.35
            + return_score * 0.20,
            0.0,
            100.0,
        )
    )


def merge_inputs() -> pd.DataFrame:
    frames = {
        "v15": load_csv(V15_FILE),
        "v16": load_csv(V16_FILE),
        "v17": load_csv(V17_FILE),
        "v18": load_csv(V18_FILE),
        "v19": load_csv(V19_FILE),
    }

    if frames["v18"].empty:
        return pd.DataFrame()

    base = frames["v18"].copy()
    base["symbol"] = base["symbol"].map(
        normalize_symbol
    )

    for key in [
        "v15",
        "v16",
        "v17",
        "v19",
    ]:
        frame = frames[key]

        if frame.empty or "symbol" not in frame.columns:
            continue

        frame = frame.copy()
        frame["symbol"] = frame["symbol"].map(
            normalize_symbol
        )

        duplicate_columns = [
            column
            for column in frame.columns
            if column in base.columns
            and column != "symbol"
        ]

        frame = frame.drop(
            columns=duplicate_columns,
            errors="ignore",
        )

        base = base.merge(
            frame.drop_duplicates("symbol"),
            on="symbol",
            how="left",
        )

    return base


def final_class(
    score: float,
    consensus: float,
    risk_class: str,
) -> str:
    if (
        score >= 85
        and consensus >= 75
        and risk_class == "DÃÅÃK"
    ):
        return "V20 ÃOK GÃÃLÃ ADAY"

    if (
        score >= 75
        and consensus >= 65
        and risk_class != "YÃKSEK"
    ):
        return "V20 GÃÃLÃ ADAY"

    if (
        score >= 62
        and risk_class != "YÃKSEK"
    ):
        return "V20 Ä°ZLEME ADAYI"

    if score >= 52:
        return "V20 TEMKÄ°NLÄ° TAKÄ°P"

    return "V20 ELE"


def explanation(
    row: pd.Series,
) -> list[str]:
    reasons: list[str] = []

    if safe_float(
        row.get("market_percentile")
    ) >= 90:
        reasons.append(
            "GÃ¶reli gÃ¼Ã§te piyasanÄ±n Ã¼st %10 diliminde"
        )

    if safe_float(
        row.get("confidence_score")
    ) >= 75:
        reasons.append(
            "V18 gÃ¼ven puanÄ± gÃ¼Ã§lÃ¼"
        )

    if safe_float(
        row.get("consensus_score")
    ) >= 75:
        reasons.append(
            "Analiz motorlarÄ± aynÄ± yÃ¶nde"
        )

    if safe_float(
        row.get("timing_confidence")
    ) >= 65:
        reasons.append(
            "Zamanlama motoru yeterli gÃ¼ven Ã¼retti"
        )

    regime = clean_text(
        row.get("regime")
    )

    if regime in {
        "RALLÄ°",
        "TREND",
    }:
        reasons.append(
            f"Piyasa rejimi destekliyor: {regime}"
        )

    if safe_float(
        row.get("positive_rate")
    ) >= 65:
        reasons.append(
            "Benzer tarihsel Ã¶rneklerin pozitif oranÄ± yÃ¼ksek"
        )

    if not reasons:
        reasons.append(
            "Katmanlar birlikte deÄerlendirilerek sÄ±ralandÄ±"
        )

    return reasons


def main() -> None:
    print(
        "V20.1 AI Final Decision Engine baÅladÄ±."
    )

    frame = merge_inputs()

    if frame.empty:
        pd.DataFrame().to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        STATUS_FILE.write_text(
            json.dumps(
                {
                    "status": "input_missing",
                    "candidate_count": 0,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    rows: list[dict[str, Any]] = []

    for _, row in frame.iterrows():
        v15_score = safe_float(
            row.get("v15_score"),
            0.0,
        )

        v16_score = safe_float(
            row.get("relative_strength_score"),
            0.0,
        )

        v17_score = safe_float(
            row.get("v17_score"),
            0.0,
        )

        v18_score = safe_float(
            row.get("confidence_score"),
            0.0,
        )

        v19_score = normalize_timing_score(
            row
        )

        consensus_score, dispersion, valid_count = (
            calculate_consensus(
                [
                    v15_score,
                    v16_score,
                    v17_score,
                    v18_score,
                    v19_score,
                ]
            )
        )

        row_for_risk = row.copy()
        row_for_risk["consensus_score"] = (
            consensus_score
        )

        risk_score, risk_class, risk_reasons = (
            calculate_risk(
                row_for_risk
            )
        )

        engine_score = (
            v15_score * 0.25
            + v16_score * 0.15
            + v17_score * 0.15
            + v18_score * 0.25
            + v19_score * 0.20
        )

        ai_score = (
            engine_score * 0.84
            + consensus_score * 0.16
            - risk_score * 0.12
        )

        ai_score = float(
            np.clip(
                ai_score,
                0.0,
                100.0,
            )
        )

        class_name = final_class(
            ai_score,
            consensus_score,
            risk_class,
        )

        reasons = explanation(
            row_for_risk
        )

        rows.append({
            "symbol": clean_text(
                row.get("symbol")
            ),
            "close": round(
                safe_float(
                    row.get("close")
                ),
                4,
            ),
            "v15_score": round(
                v15_score,
                2,
            ),
            "v16_score": round(
                v16_score,
                2,
            ),
            "v17_score": round(
                v17_score,
                2,
            ),
            "v18_score": round(
                v18_score,
                2,
            ),
            "v19_score": round(
                v19_score,
                2,
            ),
            "consensus_score": round(
                consensus_score,
                2,
            ),
            "consensus_dispersion": round(
                dispersion,
                2,
            ),
            "consensus_engine_count": valid_count,
            "risk_score": round(
                risk_score,
                2,
            ),
            "risk_class": risk_class,
            "ai_final_score": round(
                ai_score,
                2,
            ),
            "v20_decision": class_name,
            "regime": clean_text(
                row.get("regime")
            ),
            "market_percentile": round(
                safe_float(
                    row.get("market_percentile")
                ),
                2,
            ),
            "best_horizon_days": int(
                safe_float(
                    row.get("best_horizon_days"),
                    0,
                )
            ),
            "timing_confidence": round(
                safe_float(
                    row.get("timing_confidence")
                ),
                2,
            ),
            "expected_return": round(
                safe_float(
                    row.get("expected_return")
                ),
                2,
            ),
            "positive_rate": round(
                safe_float(
                    row.get("positive_rate")
                ),
                2,
            ),
            "downside_20pct": round(
                safe_float(
                    row.get("downside_20pct")
                ),
                2,
            ),
            "upside_80pct": round(
                safe_float(
                    row.get("upside_80pct")
                ),
                2,
            ),
            "ai_reasons": " | ".join(
                reasons
            ),
            "risk_reasons": " | ".join(
                risk_reasons
            ),
        })

    result = pd.DataFrame(rows)

    priority = {
        "V20 ÃOK GÃÃLÃ ADAY": 5,
        "V20 GÃÃLÃ ADAY": 4,
        "V20 Ä°ZLEME ADAYI": 3,
        "V20 TEMKÄ°NLÄ° TAKÄ°P": 2,
        "V20 ELE": 1,
    }

    result["_priority"] = (
        result["v20_decision"]
        .map(priority)
        .fillna(0)
    )

    result = (
        result
        .sort_values(
            [
                "_priority",
                "ai_final_score",
                "consensus_score",
            ],
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
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": "ready",
        "candidate_count": len(result),
        "approved_count": int(
            result["v20_decision"].isin(
                [
                    "V20 ÃOK GÃÃLÃ ADAY",
                    "V20 GÃÃLÃ ADAY",
                ]
            ).sum()
        ),
        "top_pick": (
            result.iloc[0]["symbol"]
            if len(result) > 0
            else ""
        ),
        "top_pick_score": (
            float(
                result.iloc[0][
                    "ai_final_score"
                ]
            )
            if len(result) > 0
            else 0.0
        ),
        "version": "V20.1",
    }

    STATUS_FILE.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
