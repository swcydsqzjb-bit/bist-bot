from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

INPUT_FILE = Path("v20_top_picks.csv")
OUTPUT_FILE = Path("v20_portfolio_model.csv")
STATUS_FILE = Path("v20_portfolio_status.json")

MAX_POSITIONS = 5
MAX_SINGLE_WEIGHT = 35.0
MIN_SINGLE_WEIGHT = 8.0
BASE_CASH_RESERVE = 20.0

def sf(v: Any, d: float = 0.0) -> float:
    try:
        x = float(v)
        return d if not np.isfinite(x) else x
    except Exception:
        return d

def tx(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()

def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")

def clamp(v: float, low: float, high: float) -> float:
    return float(np.clip(v, low, high))

def eligible(row: pd.Series) -> bool:
    return (
        tx(row.get("confirmation_state")) != "ELE"
        and tx(row.get("risk_class")).upper() != "YÃKSEK"
        and sf(row.get("top_pick_score")) >= 50
    )

def raw_strength(row: pd.Series) -> float:
    state_bonus = {
        "TEYÄ°T GELDÄ°": 16.0,
        "Ä°ZLEMEDE TUT": 9.0,
        "TEMKÄ°NLÄ° Ä°ZLE": 3.0,
    }.get(tx(row.get("confirmation_state")), 0.0)

    risk_penalty = {
        "DÃÅÃK": 0.0,
        "ORTA": 7.0,
        "YÃKSEK": 18.0,
    }.get(tx(row.get("risk_class")).upper(), 10.0)

    expected = sf(row.get("expected_return"))
    expected_component = clamp((expected + 2.0) * 4.0, 0.0, 25.0)

    return max(
        1.0,
        sf(row.get("top_pick_score")) * 0.42
        + sf(row.get("ai_final_score")) * 0.25
        + sf(row.get("consensus_score")) * 0.15
        + sf(row.get("timing_confidence")) * 0.08
        + expected_component
        + state_bonus
        - risk_penalty
    )

def allocation_label(weight: float) -> str:
    if weight >= 28:
        return "ANA ADAY"
    if weight >= 18:
        return "GÃÃLÃ DESTEK"
    if weight >= 10:
        return "DENGELÄ° PAY"
    return "KÃÃÃK Ä°ZLEME PAYI"

def main() -> None:
    frame = load(INPUT_FILE)

    if frame.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        STATUS_FILE.write_text(
            json.dumps(
                {"status": "input_missing", "position_count": 0, "cash_reserve_pct": 100.0},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    selected = frame[frame.apply(eligible, axis=1)].copy()
    selected = selected.head(MAX_POSITIONS)

    if selected.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        STATUS_FILE.write_text(
            json.dumps(
                {"status": "no_eligible_candidate", "position_count": 0, "cash_reserve_pct": 100.0},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    average_risk = selected["risk_score"].apply(sf).mean()
    average_consensus = selected["consensus_score"].apply(sf).mean()

    cash_reserve = BASE_CASH_RESERVE
    if average_risk >= 45:
        cash_reserve += 15
    elif average_risk >= 30:
        cash_reserve += 8

    if average_consensus < 60:
        cash_reserve += 10
    elif average_consensus >= 80:
        cash_reserve -= 5

    cash_reserve = clamp(cash_reserve, 10.0, 55.0)
    investable = 100.0 - cash_reserve

    selected["raw_strength"] = selected.apply(raw_strength, axis=1)
    total_strength = selected["raw_strength"].sum()

    selected["model_weight_pct"] = (
        selected["raw_strength"] / total_strength * investable
    )

    selected["model_weight_pct"] = selected["model_weight_pct"].clip(
        lower=MIN_SINGLE_WEIGHT,
        upper=MAX_SINGLE_WEIGHT,
    )

    scale = investable / selected["model_weight_pct"].sum()
    selected["model_weight_pct"] = selected["model_weight_pct"] * scale

    selected = selected.sort_values(
        ["model_weight_pct", "top_pick_score"],
        ascending=False,
    ).reset_index(drop=True)

    selected.insert(0, "portfolio_rank", range(1, len(selected) + 1))
    selected["allocation_label"] = selected["model_weight_pct"].apply(allocation_label)

    keep = [
        "portfolio_rank",
        "symbol",
        "model_weight_pct",
        "allocation_label",
        "confirmation_state",
        "top_pick_score",
        "ai_final_score",
        "consensus_score",
        "risk_class",
        "risk_score",
        "close",
        "best_horizon_days",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "why_now",
        "conflict_note",
    ]
    result = selected[[c for c in keep if c in selected.columns]].copy()
    result["model_weight_pct"] = result["model_weight_pct"].round(2)

    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    status = {
        "status": "ready",
        "position_count": int(len(result)),
        "cash_reserve_pct": round(cash_reserve, 2),
        "invested_pct": round(investable, 2),
        "top_symbol": tx(result.iloc[0]["symbol"]) if len(result) else "",
        "top_weight_pct": sf(result.iloc[0]["model_weight_pct"]) if len(result) else 0.0,
        "version": "V20.3",
    }
    STATUS_FILE.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
