from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_PLAN = Path("v20_monitoring_plan.csv")
INPUT_TOP_PICKS = Path("v20_top_picks.csv")
INPUT_V21 = Path("v21_learned_weights.csv")

OUTPUT_FILE = Path("v22_signal_states.csv")
STATUS_FILE = Path("v22_status.json")


def sf(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def tx(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")


def learning_adjustment(weights: pd.DataFrame) -> float:
    if weights.empty:
        return 0.0

    values = pd.to_numeric(
        weights.get("learned_weight"),
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    concentration = float(values.max())
    if concentration >= 25:
        return 2.0
    if concentration >= 20:
        return 1.0
    return 0.0


def signal_state(row: pd.Series) -> str:
    monitoring = tx(row.get("monitoring_state")).upper()
    confirmation = tx(row.get("confirmation_state")).upper()

    top_pick = sf(row.get("top_pick_score"))
    ai_score = sf(row.get("ai_final_score"))
    consensus = sf(row.get("consensus_score"))
    risk_score = sf(row.get("risk_score"))
    expected = sf(row.get("expected_return"))
    timing = sf(row.get("timing_confidence"))
    downside = sf(row.get("downside_20pct"))

    if (
        monitoring == "AKTÄ°F TEYÄ°T"
        and confirmation == "TEYÄ°T GELDÄ°"
        and top_pick >= 78
        and consensus >= 75
        and risk_score <= 30
        and expected > 0
    ):
        return "GÃÃLÃ TEYÄ°T"

    if (
        monitoring in {"AKTÄ°F TEYÄ°T", "TEYÄ°T BEKLE"}
        and confirmation in {"TEYÄ°T GELDÄ°", "Ä°ZLEMEDE TUT"}
        and top_pick >= 68
        and consensus >= 62
        and risk_score <= 45
        and timing >= 65
        and expected > 0
    ):
        return "Ä°ZLEMEYE AL"

    if (
        monitoring in {"TEYÄ°T BEKLE", "SADECE TAKÄ°P", "PASÄ°F"}
        and top_pick >= 55
        and risk_score <= 55
        and downside > -5
    ):
        return "TEYÄ°T BEKLE"

    if risk_score >= 60 or expected <= 0 or downside <= -6:
        return "RÄ°SKLÄ° - ELE"

    if ai_score < 50 or consensus < 50:
        return "ELE"

    return "PASÄ°F Ä°ZLEME"


def signal_score(row: pd.Series, learning_bonus: float) -> float:
    top_pick = sf(row.get("top_pick_score"))
    ai_score = sf(row.get("ai_final_score"))
    consensus = sf(row.get("consensus_score"))
    timing = sf(row.get("timing_confidence"))
    expected = sf(row.get("expected_return"))
    risk_score = sf(row.get("risk_score"))
    market_percentile = sf(row.get("market_percentile"))

    expected_component = float(
        np.clip((expected + 3.0) / 10.0 * 100.0, 0.0, 100.0)
    )

    score = (
        top_pick * 0.30
        + ai_score * 0.22
        + consensus * 0.18
        + timing * 0.10
        + expected_component * 0.08
        + market_percentile * 0.07
        - risk_score * 0.15
        + learning_bonus
    )

    return round(float(np.clip(score, 0.0, 100.0)), 2)


def reason_text(row: pd.Series) -> str:
    reasons: list[str] = []

    if sf(row.get("top_pick_score")) >= 70:
        reasons.append("Top Pick skoru gÃ¼Ã§lÃ¼")

    if sf(row.get("consensus_score")) >= 70:
        reasons.append("Motor uyumu yÃ¼ksek")

    if sf(row.get("timing_confidence")) >= 75:
        reasons.append("Zamanlama gÃ¼veni gÃ¼Ã§lÃ¼")

    if sf(row.get("expected_return")) >= 3:
        reasons.append("Tarihsel beklenen sonuÃ§ gÃ¼Ã§lÃ¼")
    elif sf(row.get("expected_return")) > 0:
        reasons.append("Tarihsel beklenen sonuÃ§ pozitif")

    if sf(row.get("market_percentile")) >= 90:
        reasons.append("Piyasa gÃ¶reli gÃ¼cÃ¼ Ã¼st %10 diliminde")

    if sf(row.get("risk_score")) <= 30:
        reasons.append("Risk puanÄ± dÃ¼ÅÃ¼k")

    if not reasons:
        reasons.append("Katmanlar henÃ¼z ortak gÃ¼Ã§lÃ¼ teyit Ã¼retmedi")

    return " | ".join(reasons)


def risk_text(row: pd.Series) -> str:
    notes: list[str] = []

    if sf(row.get("risk_score")) >= 60:
        notes.append("Risk puanÄ± yÃ¼ksek")
    elif sf(row.get("risk_score")) >= 35:
        notes.append("Risk puanÄ± orta")

    if sf(row.get("consensus_score")) < 55:
        notes.append("Motorlar arasÄ±nda gÃ¶rÃ¼Å ayrÄ±lÄ±ÄÄ± var")

    if sf(row.get("expected_return")) <= 0:
        notes.append("Beklenen istatistiksel sonuÃ§ pozitif deÄil")

    if sf(row.get("downside_20pct")) <= -5:
        notes.append("Temkinli senaryo zayÄ±f")

    conflict = tx(row.get("conflict_note"))
    if conflict and conflict != "Belirgin motor Ã§eliÅkisi yok":
        notes.append(conflict)

    if not notes:
        notes.append("Belirgin ek risk notu yok")

    return " | ".join(notes)


def main() -> None:
    plan = load_csv(INPUT_PLAN)
    top = load_csv(INPUT_TOP_PICKS)
    learned = load_csv(INPUT_V21)

    if plan.empty or top.empty:
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
                    "actionable_count": 0,
                    "version": "V22.0",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    merged = plan.merge(
        top,
        on="symbol",
        how="left",
        suffixes=("_plan", ""),
    )

    bonus = learning_adjustment(learned)

    merged["v22_signal_score"] = merged.apply(
        lambda row: signal_score(row, bonus),
        axis=1,
    )
    merged["v22_signal_state"] = merged.apply(
        signal_state,
        axis=1,
    )
    merged["v22_reasons"] = merged.apply(
        reason_text,
        axis=1,
    )
    merged["v22_risks"] = merged.apply(
        risk_text,
        axis=1,
    )
    merged["learning_bonus"] = bonus

    priority = {
        "GÃÃLÃ TEYÄ°T": 6,
        "Ä°ZLEMEYE AL": 5,
        "TEYÄ°T BEKLE": 4,
        "PASÄ°F Ä°ZLEME": 3,
        "ELE": 2,
        "RÄ°SKLÄ° - ELE": 1,
    }

    merged["_priority"] = (
        merged["v22_signal_state"]
        .map(priority)
        .fillna(0)
    )

    merged = (
        merged.sort_values(
            ["_priority", "v22_signal_score"],
            ascending=False,
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    merged.insert(
        0,
        "v22_rank",
        range(1, len(merged) + 1),
    )

    output_columns = [
        "v22_rank",
        "symbol",
        "v22_signal_state",
        "v22_signal_score",
        "learning_bonus",
        "monitoring_state",
        "confirmation_state",
        "model_weight_pct",
        "close",
        "top_pick_score",
        "ai_final_score",
        "consensus_score",
        "risk_class",
        "risk_score",
        "market_percentile",
        "regime",
        "best_horizon_days",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "v22_reasons",
        "v22_risks",
    ]

    result = merged[
        [column for column in output_columns if column in merged.columns]
    ].copy()

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    actionable_states = {"GÃÃLÃ TEYÄ°T", "Ä°ZLEMEYE AL"}

    status = {
        "status": "ready",
        "candidate_count": int(len(result)),
        "actionable_count": int(
            result["v22_signal_state"].isin(actionable_states).sum()
        ),
        "strong_confirmation_count": int(
            (result["v22_signal_state"] == "GÃÃLÃ TEYÄ°T").sum()
        ),
        "top_symbol": tx(result.iloc[0]["symbol"]) if len(result) else "",
        "top_state": tx(result.iloc[0]["v22_signal_state"]) if len(result) else "",
        "top_score": sf(result.iloc[0]["v22_signal_score"]) if len(result) else 0.0,
        "learning_bonus": bonus,
        "version": "V22.0",
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
