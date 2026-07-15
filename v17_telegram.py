from __future__ import annotations

import json
import os
from typing import Any, List

import numpy as np
import pandas as pd
import requests


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

RESULT_FILE = "v17_regime_adjusted_decisions.csv"
STATUS_FILE = "v17_market_regime_status.json"
LIMIT = 3900


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


def load_status() -> dict:
    if not os.path.exists(STATUS_FILE):
        return {}

    try:
        with open(
            STATUS_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)
    except Exception:
        return {}


def split_message(text: str) -> List[str]:
    if len(text) <= LIMIT:
        return [text]

    parts: List[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= LIMIT:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = paragraph

    if current:
        parts.append(current)

    return parts


def send_message(text: str) -> None:
    if not TOKEN or not CHAT_ID:
        print(text)
        return

    for part in split_message(text):
        response = requests.post(
            f"https://api.telegram.org/bot{TOKEN}/sendMessage",
            data={
                "chat_id": CHAT_ID,
                "text": part,
                "disable_web_page_preview": True,
            },
            timeout=30,
        )

        print(
            response.status_code,
            response.text[:300],
        )


def regime_emoji(regime: str) -> str:
    if regime == "RALLÄ°":
        return "ð"

    if regime == "TREND":
        return "ð"

    if regime == "PANÄ°K":
        return "â ï¸"

    return "âï¸"


def decision_emoji(decision: str) -> str:
    if "GÃÃLÃ" in decision:
        return "ð¢"

    if "ONAYLI" in decision:
        return "ðµ"

    if "TEMKÄ°NLÄ°" in decision:
        return "ð¡"

    return "ð´"


def format_reasons(value: Any) -> str:
    items = [
        item.strip()
        for item in clean_text(value).split("|")
        if item.strip()
    ]

    return "\n".join(
        f"â¢ {item}"
        for item in items
    )


def main() -> None:
    try:
        frame = pd.read_csv(
            RESULT_FILE,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        print(
            f"{RESULT_FILE} okunamadÄ±: {exc}"
        )
        return

    status = load_status()

    regime = clean_text(
        status.get("regime")
    )

    if frame.empty:
        send_message(
            "ð§­ LARUS V17 PÄ°YASA REJÄ°M RAPORU\n\n"
            f"Rejim: {regime or 'BÄ°LÄ°NMÄ°YOR'}\n\n"
            "BugÃ¼n rejime gÃ¶re deÄerlendirilecek aday bulunamadÄ±."
        )
        return

    message = (
        "ð§­ LARUS V17 PÄ°YASA REJÄ°M RAPORU\n\n"
        f"{regime_emoji(regime)} Piyasa rejimi: {regime}\n"
        f"Rejim gÃ¼veni: "
        f"{safe_float(status.get('regime_confidence')):.1f}/100\n"
        f"KarÅÄ±laÅtÄ±rÄ±lan hisse: "
        f"{int(safe_float(status.get('market_count')))}\n"
        f"Ä°ncelenen aday: {len(frame)}\n"
        f"Onaylanan: "
        f"{int(safe_float(status.get('approved_count')))}\n\n"
        f"1 gÃ¼nlÃ¼k pozitif geniÅlik: "
        f"%{safe_float(status.get('breadth_1d_positive_pct')):.1f}\n"
        f"5 gÃ¼nlÃ¼k pozitif geniÅlik: "
        f"%{safe_float(status.get('breadth_5d_positive_pct')):.1f}\n"
        f"EMA20 Ã¼zerindeki hisseler: "
        f"%{safe_float(status.get('above_ema20_pct')):.1f}\n\n"
    )

    for _, row in frame.iterrows():
        decision = clean_text(
            row.get("v17_decision")
        )

        message += (
            f"{decision_emoji(decision)} "
            f"{int(safe_float(row.get('rank')))}. "
            f"{clean_text(row.get('symbol'))}\n"
            f"V17 kararÄ±: {decision}\n"
            f"Fiyat: {safe_float(row.get('close')):.2f}\n"
            f"V17 skoru: "
            f"{safe_float(row.get('v17_score')):.1f}/100\n"
            f"Rejim etkisi: "
            f"{safe_float(row.get('regime_adjustment')):+.1f} puan\n"
            f"V15 skoru: "
            f"{safe_float(row.get('v15_score')):.1f}/100\n"
            f"GÃ¶reli gÃ¼Ã§: "
            f"{safe_float(row.get('relative_strength_score')):.1f}/100\n"
            f"Piyasa yÃ¼zdeliÄi: "
            f"%{safe_float(row.get('market_percentile')):.1f}\n"
            f"Momentum: "
            f"{safe_float(row.get('momentum_percentile')):.1f}/100\n"
            f"Trend: "
            f"{safe_float(row.get('trend_percentile')):.1f}/100\n"
            f"Hacim: "
            f"{safe_float(row.get('volume_percentile')):.1f}/100\n"
            f"Kalite: "
            f"{safe_float(row.get('quality_percentile')):.1f}/100\n"
        )

        reasons = format_reasons(
            row.get("regime_reasons")
        )

        if reasons:
            message += (
                "\nRejim deÄerlendirmesi:\n"
                f"{reasons}\n"
            )

        message += "\n--------------------\n\n"

    message += (
        "â ï¸ V17, mevcut sinyalleri piyasa rejimine gÃ¶re "
        "yeniden aÄÄ±rlÄ±klandÄ±rÄ±r. YatÄ±rÄ±m tavsiyesi veya "
        "getiri garantisi deÄildir."
    )

    send_message(message)


if __name__ == "__main__":
    main()
