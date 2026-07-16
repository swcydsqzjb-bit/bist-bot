from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


POSITIONS_FILE = Path("v23_positions.csv")
HISTORY_FILE = Path("v23_position_history.csv")
OUTPUT_FILE = Path("v25_performance_evaluations.csv")
MEMORY_FILE = Path("v25_performance_memory.csv")
STATUS_FILE = Path("v25_status.json")


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


def fetch_daily(symbol: str) -> pd.DataFrame:
    ticker = symbol if symbol.endswith(".IS") else f"{symbol}.IS"
    frame = yf.download(
        ticker,
        period="3mo",
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    if frame.empty:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [column[0] for column in frame.columns]
    frame.columns = [str(column).title() for column in frame.columns]
    return frame.dropna(subset=["Close", "High", "Low"])


def evaluate_position(row: pd.Series) -> dict[str, Any]:
    symbol = tx(row.get("symbol"))
    entry = sf(row.get("entry_reference"))
    current = sf(row.get("last_price"), entry)
    frame = fetch_daily(symbol)

    if frame.empty or entry <= 0:
        return {
            "symbol": symbol,
            "evaluation_state": "VERÄ° YETERSÄ°Z",
            "realized_proxy_return_pct": 0.0,
            "max_gain_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "hit_first_objective": False,
            "hit_optimistic_objective": False,
            "hit_invalidation": False,
            "quality_score": 0.0,
            "reliability_class": "VERÄ° YETERSÄ°Z",
        }

    recent = frame.tail(max(2, int(sf(row.get("days_in_position"), 0)) + 2))
    high = sf(recent["High"].max(), current)
    low = sf(recent["Low"].min(), current)

    return_pct = (current / entry - 1) * 100
    max_gain = (high / entry - 1) * 100
    max_drawdown = (low / entry - 1) * 100

    first = sf(row.get("first_objective_price"))
    optimistic = sf(row.get("optimistic_objective_price"))
    invalidation = sf(row.get("statistical_invalidation_price"))

    hit_first = first > 0 and high >= first
    hit_optimistic = optimistic > 0 and high >= optimistic
    hit_invalidation = invalidation > 0 and low <= invalidation

    score = 50.0
    score += np.clip(return_pct * 5, -25, 25)
    score += 12 if hit_first else 0
    score += 12 if hit_optimistic else 0
    score -= 18 if hit_invalidation else 0
    score += 6 if max_drawdown > -3 else -6
    score = float(np.clip(score, 0, 100))

    if score >= 75:
        reliability = "ÃOK Ä°YÄ°"
    elif score >= 60:
        reliability = "Ä°YÄ°"
    elif score >= 45:
        reliability = "ORTA"
    else:
        reliability = "ZAYIF"

    if hit_optimistic:
        state = "OLUMLU SENARYO GERÃEKLEÅTÄ°"
    elif hit_first:
        state = "Ä°LK HEDEF GERÃEKLEÅTÄ°"
    elif hit_invalidation:
        state = "GEÃERSÄ°ZLÄ°K GÃRÃLDÃ"
    elif return_pct > 0:
        state = "POZÄ°TÄ°F Ä°LERLÄ°YOR"
    elif return_pct < 0:
        state = "NEGATÄ°F Ä°LERLÄ°YOR"
    else:
        state = "YATAY"

    return {
        "symbol": symbol,
        "evaluation_state": state,
        "realized_proxy_return_pct": round(return_pct, 2),
        "max_gain_pct": round(max_gain, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "hit_first_objective": bool(hit_first),
        "hit_optimistic_objective": bool(hit_optimistic),
        "hit_invalidation": bool(hit_invalidation),
        "quality_score": round(score, 2),
        "reliability_class": reliability,
    }


def main() -> None:
    positions = load_csv(POSITIONS_FILE)
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if positions.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        STATUS_FILE.write_text(
            json.dumps(
                {
                    "status": "no_positions",
                    "evaluated_count": 0,
                    "average_quality_score": 0.0,
                    "version": "V25.0",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return

    rows = []
    for _, row in positions.iterrows():
        item = evaluate_position(row)
        item["evaluated_at_utc"] = now
        item["position_state"] = tx(row.get("position_state"))
        item["action"] = tx(row.get("action"))
        item["entry_reference"] = sf(row.get("entry_reference"))
        item["last_price"] = sf(row.get("last_price"))
        item["latest_v22_score"] = sf(row.get("latest_v22_score"))
        rows.append(item)

    result = pd.DataFrame(rows).sort_values(
        ["quality_score", "realized_proxy_return_pct"],
        ascending=False,
    )
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    memory = load_csv(MEMORY_FILE)
    combined = result.copy() if memory.empty else pd.concat(
        [memory, result],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(MEMORY_FILE, index=False, encoding="utf-8-sig")

    status = {
        "status": "ready",
        "evaluated_count": int(len(result)),
        "positive_count": int((result["realized_proxy_return_pct"] > 0).sum()),
        "first_objective_count": int(result["hit_first_objective"].sum()),
        "optimistic_objective_count": int(result["hit_optimistic_objective"].sum()),
        "invalidation_count": int(result["hit_invalidation"].sum()),
        "average_quality_score": round(sf(result["quality_score"].mean()), 2),
        "top_symbol": tx(result.iloc[0]["symbol"]) if len(result) else "",
        "top_quality_score": sf(result.iloc[0]["quality_score"]) if len(result) else 0.0,
        "version": "V25.0",
    }

    STATUS_FILE.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(status, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
