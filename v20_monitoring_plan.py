from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

INPUT_FILE = Path("v20_portfolio_model.csv")
OUTPUT_FILE = Path("v20_monitoring_plan.csv")
STATUS_FILE = Path("v20_monitoring_status.json")

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

def entry_state(row: pd.Series) -> str:
    state = tx(row.get("confirmation_state"))
    consensus = sf(row.get("consensus_score"))
    risk = tx(row.get("risk_class")).upper()
    expected = sf(row.get("expected_return"))

    if state == "TEYÄ°T GELDÄ°" and consensus >= 75 and risk == "DÃÅÃK":
        return "AKTÄ°F TEYÄ°T"
    if state in {"TEYÄ°T GELDÄ°", "Ä°ZLEMEDE TUT"} and expected > 0:
        return "TEYÄ°T BEKLE"
    if state == "TEMKÄ°NLÄ° Ä°ZLE":
        return "SADECE TAKÄ°P"
    return "PASÄ°F"

def review_rule(row: pd.Series) -> str:
    horizon = int(sf(row.get("best_horizon_days"), 1))
    if horizon <= 1:
        return "Bir sonraki iÅlem gÃ¼nÃ¼ kapanÄ±ÅÄ±nda yeniden deÄerlendir"
    if horizon <= 3:
        return "3 iÅlem gÃ¼nÃ¼ iÃ§inde momentum teyidini kontrol et"
    if horizon <= 5:
        return "5 iÅlem gÃ¼nÃ¼ iÃ§inde sonuÃ§ ve risk gÃ¶rÃ¼nÃ¼mÃ¼nÃ¼ yenile"
    return f"{horizon} iÅlem gÃ¼nÃ¼ dolmadan ara kontrol yap"

def main() -> None:
    frame = load(INPUT_FILE)

    if frame.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        STATUS_FILE.write_text(
            json.dumps({"status": "input_missing", "plan_count": 0}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    rows = []
    for _, row in frame.iterrows():
        close = sf(row.get("close"))
        downside = sf(row.get("downside_20pct"))
        upside = sf(row.get("upside_80pct"))
        expected = sf(row.get("expected_return"))

        invalidation_pct = min(-1.0, downside)
        first_objective_pct = max(1.0, expected)
        optimistic_objective_pct = max(first_objective_pct, upside)

        invalidation_price = close * (1 + invalidation_pct / 100.0) if close > 0 else 0.0
        first_objective_price = close * (1 + first_objective_pct / 100.0) if close > 0 else 0.0
        optimistic_objective_price = close * (1 + optimistic_objective_pct / 100.0) if close > 0 else 0.0

        rows.append({
            "monitor_rank": int(sf(row.get("portfolio_rank"), len(rows) + 1)),
            "symbol": tx(row.get("symbol")),
            "monitoring_state": entry_state(row),
            "model_weight_pct": round(sf(row.get("model_weight_pct")), 2),
            "close": round(close, 4),
            "review_horizon_days": int(sf(row.get("best_horizon_days"), 1)),
            "review_rule": review_rule(row),
            "statistical_invalidation_pct": round(invalidation_pct, 2),
            "statistical_invalidation_price": round(invalidation_price, 4),
            "first_objective_pct": round(first_objective_pct, 2),
            "first_objective_price": round(first_objective_price, 4),
            "optimistic_objective_pct": round(optimistic_objective_pct, 2),
            "optimistic_objective_price": round(optimistic_objective_price, 4),
            "consensus_score": round(sf(row.get("consensus_score")), 2),
            "risk_class": tx(row.get("risk_class")),
            "risk_score": round(sf(row.get("risk_score")), 2),
            "conflict_note": tx(row.get("conflict_note")),
        })

    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    status = {
        "status": "ready",
        "plan_count": int(len(result)),
        "active_confirmation_count": int((result["monitoring_state"] == "AKTÄ°F TEYÄ°T").sum()),
        "waiting_confirmation_count": int((result["monitoring_state"] == "TEYÄ°T BEKLE").sum()),
        "version": "V20.4",
    }
    STATUS_FILE.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(result.to_string(index=False))

if __name__ == "__main__":
    main()
