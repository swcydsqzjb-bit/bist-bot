from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import pandas as pd
import requests


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

RESULT_FILE = "v15_final_decisions.csv"
STATUS_FILE = "v15_status.json"
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


def send(text: str) -> None:
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

        print(response.status_code, response.text[:300])


def emoji(decision: str) -> str:
    if "GÃÃLÃ" in decision:
        return "ð¢"
    if "ONAYLI" in decision:
        return "ðµ"
    if "TEMKÄ°NLÄ°" in decision:
        return "ð¡"
    if "GERÄ°" in decision:
        return "ð "
    return "ð´"


def main() -> None:
    try:
        frame = pd.read_csv(
            RESULT_FILE,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        print(f"{RESULT_FILE} okunamadÄ±: {exc}")
        return

    if frame.empty:
        send(
            "ð¦ LARUS V15 NÄ°HAÄ° KARAR RAPORU\n\n"
            "BugÃ¼n V15'in deÄerlendireceÄi aday bulunamadÄ±."
        )
        return

    mode = clean_text(frame.iloc[0].get("model_mode"))
    approved = int(
        frame["v15_decision"].isin(
            ["V15 GÃÃLÃ ONAY", "V15 ONAYLI Ä°ZLEME"]
        ).sum()
    )

    message = (
        "ð¦ LARUS V15 NÄ°HAÄ° KARAR RAPORU\n\n"
        f"Ä°ncelenen aday: {len(frame)}\n"
        f"Onaylanan: {approved}\n"
        f"Model modu: {mode}\n\n"
    )

    if mode == "FALLBACK":
        message += (
            "V15 henÃ¼z 30 tamamlanmÄ±Å 5 gÃ¼nlÃ¼k sonuca ulaÅmadÄ±ÄÄ± iÃ§in "
            "Ã¶ÄrenilmiÅ aÄÄ±rlÄ±klar yerine gÃ¼venli geÃ§iÅ modunu kullanÄ±yor.\n\n"
        )

    for _, row in frame.iterrows():
        decision = clean_text(row.get("v15_decision"))

        message += (
            f"{emoji(decision)} "
            f"{int(safe_float(row.get('rank')))}. "
            f"{clean_text(row.get('symbol'))}\n"
            f"V15 kararÄ±: {decision}\n"
            f"Fiyat: {safe_float(row.get('close')):.2f}\n"
            f"V15 skoru: {safe_float(row.get('v15_score')):.1f}/100\n"
            f"V14 skoru: {safe_float(row.get('v14_score')):.1f}/100\n"
            f"ÃÄrenme bileÅeni: "
            f"{safe_float(row.get('learned_component_score')):.1f}/100\n"
            f"V8 skoru: {safe_float(row.get('v8_score')):.1f}/100\n"
            f"Smart Money: "
            f"{safe_float(row.get('smart_money_score')):.1f}/100\n"
            f"Kurumsal: "
            f"{safe_float(row.get('institutional_score')):.1f}/100\n"
            f"DNA: {clean_text(row.get('dna_classification'))} | "
            f"{safe_float(row.get('dna_confidence')):.1f}/100\n"
            f"5 gÃ¼nde pozitif: "
            f"%{safe_float(row.get('positive_rate_5d')):.1f}\n"
            f"Ortalama 5 gÃ¼nlÃ¼k sonuÃ§: "
            f"{safe_float(row.get('average_result_5d')):+.2f}%\n"
            f"Ãnceki V14 kararÄ±: "
            f"{clean_text(row.get('v14_decision'))}\n"
            "\n--------------------\n\n"
        )

    message += (
        "â ï¸ V15 geÃ§miÅ sinyallerden istatistiksel aÄÄ±rlÄ±k Ã¶Ärenir. "
        "Bu sistem yatÄ±rÄ±m tavsiyesi veya getiri garantisi deÄildir."
    )

    send(message)


if __name__ == "__main__":
    main()
