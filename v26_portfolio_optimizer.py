from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SIGNALS_FILE = Path("v22_signal_states.csv")
LIVE_FILE = Path("v24_live_confirmations.csv")
PERFORMANCE_FILE = Path("v25_performance_evaluations.csv")
OUTPUT_FILE = Path("v26_optimized_portfolio.csv")
STATUS_FILE = Path("v26_status.json")


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


def main() -> None:
    signals = load_csv(SIGNALS_FILE)
    live = load_csv(LIVE_FILE)
    performance = load_csv(PERFORMANCE_FILE)

    if signals.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        STATUS_FILE.write_text(
            json.dumps(
                {
                    "status": "input_missing",
                    "position_count": 0,
                    "invested_pct": 0.0,
                    "cash_pct": 100.0,
                    "version": "V26.0",
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
    else:
        merged["v24_state"] = ""
        merged["v24_score"] = 0.0

    if not performance.empty:
        merged = merged.merge(
            performance[["symbol", "quality_score", "reliability_class"]],
            on="symbol",
            how="left",
        )
    else:
        merged["quality_score"] = 50.0
        merged["reliability_class"] = "YENÄ°"

    merged["quality_score"] = pd.to_numeric(
        merged["quality_score"], errors="coerce"
    ).fillna(50.0)
    merged["v24_score"] = pd.to_numeric(
        merged["v24_score"], errors="coerce"
    ).fillna(0.0)

    merged["optimizer_score"] = (
        pd.to_numeric(merged["v22_signal_score"], errors="coerce").fillna(0) * 0.35
        + merged["v24_score"] * 0.25
        + pd.to_numeric(merged["consensus_score"], errors="coerce").fillna(0) * 0.15
        + merged["quality_score"] * 0.15
        + pd.to_numeric(merged["market_percentile"], errors="coerce").fillna(0) * 0.10
        - pd.to_numeric(merged["risk_score"], errors="coerce").fillna(50) * 0.20
    ).clip(0, 100)

    eligible = merged[
        (merged["optimizer_score"] >= 55)
        & (pd.to_numeric(merged["risk_score"], errors="coerce").fillna(100) <= 55)
        & (pd.to_numeric(merged["expected_return"], errors="coerce").fillna(0) > 0)
    ].copy()

    eligible = eligible.sort_values("optimizer_score", ascending=False).head(5)

    if eligible.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        status = {
            "status": "no_eligible_candidate",
            "position_count": 0,
            "invested_pct": 0.0,
            "cash_pct": 100.0,
            "version": "V26.0",
        }
        STATUS_FILE.write_text(
            json.dumps(status, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    base = eligible["optimizer_score"].clip(lower=1)
    raw_weights = base / base.sum()

    max_weight = 0.35
    weights = raw_weights.clip(upper=max_weight)
    weights = weights / weights.sum()

    risk_mean = pd.to_numeric(
        eligible["risk_score"], errors="coerce"
    ).fillna(50).mean()

    invested_pct = 80.0 if risk_mean <= 30 else 65.0 if risk_mean <= 45 else 50.0
    weights = weights * invested_pct

    eligible["optimized_weight_pct"] = weights.round(2)
    eligible["portfolio_role"] = [
        "ANA ADAY" if i == 0 else "DESTEK ADAY"
        for i in range(len(eligible))
    ]

    columns = [
        "symbol",
        "portfolio_role",
        "optimized_weight_pct",
        "optimizer_score",
        "v22_signal_state",
        "v22_signal_score",
        "v24_state",
        "v24_score",
        "quality_score",
        "reliability_class",
        "risk_class",
        "risk_score",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "best_horizon_days",
        "close",
    ]

    result = eligible[[c for c in columns if c in eligible.columns]].copy()
    result.insert(0, "v26_rank", range(1, len(result) + 1))
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    total_weight = sf(result["optimized_weight_pct"].sum())
    status = {
        "status": "ready",
        "position_count": int(len(result)),
        "invested_pct": round(total_weight, 2),
        "cash_pct": round(max(0.0, 100.0 - total_weight), 2),
        "top_symbol": tx(result.iloc[0]["symbol"]),
        "top_weight_pct": sf(result.iloc[0]["optimized_weight_pct"]),
        "average_optimizer_score": round(sf(result["optimizer_score"].mean()), 2),
        "version": "V26.0",
    }

    STATUS_FILE.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
