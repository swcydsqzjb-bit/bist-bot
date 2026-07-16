from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SIGNALS_FILE = Path("v22_signal_states.csv")
LIVE_FILE = Path("v24_live_confirmations.csv")
PERFORMANCE_FILE = Path("v25_performance_evaluations.csv")
PORTFOLIO_FILE = Path("v26_optimized_portfolio.csv")
OUTPUT_FILE = Path("v27_master_decisions.csv")
STATUS_FILE = Path("v27_status.json")


def sf(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if np.isfinite(number) else default
    except Exception:
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


def decision(row: pd.Series) -> tuple[str, str]:
    master = sf(row.get("v27_master_score"))
    live_state = tx(row.get("v24_state"))
    risk = sf(row.get("risk_score"))
    weight = sf(row.get("optimized_weight_pct"))
    quality = sf(row.get("quality_score"), 50)

    if (
        master >= 78
        and live_state == "CANLI TEYÄ°T GELDÄ°"
        and risk <= 35
        and weight > 0
    ):
        return "ÃST DÃZEY TEYÄ°T", "TÃ¼m ana katmanlar aynÄ± yÃ¶nde gÃ¼Ã§lÃ¼ teyit verdi"

    if (
        master >= 68
        and live_state in {"CANLI TEYÄ°T GELDÄ°", "ERKEN TEYÄ°T"}
        and risk <= 45
        and weight > 0
    ):
        return "AKTÄ°F Ä°ZLEME", "CanlÄ± teyit ve portfÃ¶y uygunluÄu birlikte oluÅtu"

    if master >= 58 and risk <= 55:
        return "TEYÄ°T BEKLE", "Toplam gÃ¶rÃ¼nÃ¼m olumlu fakat giriÅ teyidi tamamlanmadÄ±"

    if quality < 40 or risk >= 65:
        return "ELE", "Performans kalitesi veya risk seviyesi yetersiz"

    return "PASÄ°F Ä°ZLEME", "Katmanlar ortak gÃ¼Ã§lÃ¼ karar Ã¼retmedi"


def main() -> None:
    signals = load_csv(SIGNALS_FILE)
    live = load_csv(LIVE_FILE)
    performance = load_csv(PERFORMANCE_FILE)
    portfolio = load_csv(PORTFOLIO_FILE)

    if signals.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        STATUS_FILE.write_text(
            json.dumps(
                {
                    "status": "input_missing",
                    "candidate_count": 0,
                    "approved_count": 0,
                    "version": "V27.0",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    merged = signals.copy()

    if not live.empty:
        merged = merged.merge(
            live[["symbol", "v24_state", "v24_score"]],
            on="symbol",
            how="left",
        )

    if not performance.empty:
        merged = merged.merge(
            performance[["symbol", "quality_score", "reliability_class"]],
            on="symbol",
            how="left",
        )

    if not portfolio.empty:
        merged = merged.merge(
            portfolio[["symbol", "optimized_weight_pct", "optimizer_score"]],
            on="symbol",
            how="left",
        )

    for column, default in {
        "v24_score": 0.0,
        "quality_score": 50.0,
        "optimized_weight_pct": 0.0,
        "optimizer_score": 0.0,
    }.items():
        if column not in merged.columns:
            merged[column] = default
        merged[column] = pd.to_numeric(
            merged[column], errors="coerce"
        ).fillna(default)

    merged["v27_master_score"] = (
        pd.to_numeric(merged["v22_signal_score"], errors="coerce").fillna(0) * 0.25
        + merged["v24_score"] * 0.22
        + merged["quality_score"] * 0.15
        + merged["optimizer_score"] * 0.18
        + pd.to_numeric(merged["consensus_score"], errors="coerce").fillna(0) * 0.12
        + pd.to_numeric(merged["timing_confidence"], errors="coerce").fillna(0) * 0.08
        - pd.to_numeric(merged["risk_score"], errors="coerce").fillna(50) * 0.18
    ).clip(0, 100)

    decisions = merged.apply(decision, axis=1)
    merged["v27_decision"] = [item[0] for item in decisions]
    merged["v27_reason"] = [item[1] for item in decisions]

    priority = {
        "ÃST DÃZEY TEYÄ°T": 5,
        "AKTÄ°F Ä°ZLEME": 4,
        "TEYÄ°T BEKLE": 3,
        "PASÄ°F Ä°ZLEME": 2,
        "ELE": 1,
    }

    merged["_priority"] = merged["v27_decision"].map(priority).fillna(0)
    merged = merged.sort_values(
        ["_priority", "v27_master_score"],
        ascending=False,
    ).drop(columns="_priority").reset_index(drop=True)
    merged.insert(0, "v27_rank", range(1, len(merged) + 1))

    columns = [
        "v27_rank",
        "symbol",
        "v27_decision",
        "v27_master_score",
        "v27_reason",
        "optimized_weight_pct",
        "v22_signal_state",
        "v22_signal_score",
        "v24_state",
        "v24_score",
        "quality_score",
        "reliability_class",
        "consensus_score",
        "risk_class",
        "risk_score",
        "regime",
        "best_horizon_days",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "close",
    ]

    result = merged[[c for c in columns if c in merged.columns]].copy()
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    approved = {"ÃST DÃZEY TEYÄ°T", "AKTÄ°F Ä°ZLEME"}
    status = {
        "status": "ready",
        "candidate_count": int(len(result)),
        "approved_count": int(result["v27_decision"].isin(approved).sum()),
        "top_level_confirmation_count": int(
            (result["v27_decision"] == "ÃST DÃZEY TEYÄ°T").sum()
        ),
        "top_symbol": tx(result.iloc[0]["symbol"]) if len(result) else "",
        "top_decision": tx(result.iloc[0]["v27_decision"]) if len(result) else "",
        "top_score": sf(result.iloc[0]["v27_master_score"]) if len(result) else 0.0,
        "version": "V27.0",
    }

    STATUS_FILE.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
