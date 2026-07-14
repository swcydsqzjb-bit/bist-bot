from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import pandas as pd
import requests


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RESULT_FILE = "v14_adaptive_decisions.csv"
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
    if value is None or pd.isna(value):
        return ""
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


def reason_lines(value: Any, marker: str) -> str:
    text = clean_text(value)

    if not text:
        return ""

    lines = [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]

    return "\n".join(
        f"{marker} {item}"
        for item in lines
    )


def emoji_for(decision: str) -> str:
    if decision == "GÃÃLÃ ONAY":
        return "ð¢"
    if decision == "ONAYLI Ä°ZLEME":
        return "ðµ"
    if decision == "TEMKÄ°NLÄ° Ä°ZLEME":
        return "ð¡"
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
            "ð§  LARUS V14 ADAPTÄ°F KARAR RAPORU\n\n"
            "BugÃ¼n incelenecek aday bulunamadÄ±."
        )
        return

    approved = int(
        frame["v14_decision"].isin(
            ["GÃÃLÃ ONAY", "ONAYLI Ä°ZLEME"]
        ).sum()
    )

    message = (
        "ð§  LARUS V14 ADAPTÄ°F KARAR RAPORU\n\n"
        f"Ä°ncelenen aday: {len(frame)}\n"
        f"Onaylanan: {approved}\n"
        f"AÄÄ±rlÄ±k modu: {clean_text(frame.iloc[0].get('weight_mode'))}\n\n"
    )

    for _, row in frame.iterrows():
        decision = clean_text(row.get("v14_decision"))

        message += (
            f"{emoji_for(decision)} "
            f"{int(safe_float(row.get('rank')))}. "
            f"{clean_text(row.get('symbol'))}\n"
            f"Karar: {decision}\n"
            f"Fiyat: {safe_float(row.get('close')):.2f}\n"
            f"V14 skoru: {safe_float(row.get('v14_score')):.1f}/100\n"
            f"V8 skoru: {safe_float(row.get('v8_score')):.1f}/100\n"
            f"Smart Money: {safe_float(row.get('smart_money_score')):.1f}/100\n"
            f"Kurumsal: {safe_float(row.get('institutional_score')):.1f}/100\n"
            f"DNA: {clean_text(row.get('dna_classification'))} | "
            f"{safe_float(row.get('dna_confidence')):.1f}/100\n"
            f"5 gÃ¼nde pozitif: %{safe_float(row.get('positive_rate_5d')):.1f}\n"
            f"5 gÃ¼nde en az %3: %{safe_float(row.get('hit_3pct_5d_rate')):.1f}\n"
            f"Ortalama 5 gÃ¼nlÃ¼k sonuÃ§: "
            f"{safe_float(row.get('average_result_5d')):+.2f}%\n"
            f"Pozitif bonus: +{safe_float(row.get('positive_bonus')):.1f}\n"
            f"Risk kesintisi: -{safe_float(row.get('risk_penalty')):.1f}\n"
        )

        positive = reason_lines(
            row.get("positive_reasons"),
            "â",
        )

        risks = reason_lines(
            row.get("risk_reasons"),
            "â¢",
        )

        if positive:
            message += f"\nOnay nedenleri:\n{positive}\n"

        if risks:
            message += f"\nRiskler:\n{risks}\n"

        message += "\n--------------------\n\n"

    message += (
        "â ï¸ V14 istatistiksel bir karar katmanÄ±dÄ±r. "
        "YatÄ±rÄ±m tavsiyesi veya getiri garantisi deÄildir."
    )

    send(message)


if __name__ == "__main__":
    main()
