from __future__ import annotations

import json
import os
from typing import Any, List

import numpy as np
import pandas as pd
import requests

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RESULT_FILE = "v16_relative_strength.csv"
STATUS_FILE = "v16_status.json"
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
        with open(STATUS_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}

def split_message(text: str) -> List[str]:
    if len(text) <= LIMIT:
        return [text]
    parts, current = [], ""
    for paragraph in text.split("\n\n"):
        candidate = paragraph if not current else current + "\n\n" + paragraph
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
            data={"chat_id": CHAT_ID, "text": part, "disable_web_page_preview": True},
            timeout=30,
        )
        print(response.status_code, response.text[:300])

def class_emoji(relative_class: str) -> str:
    if relative_class == "PÄ°YASA LÄ°DERÄ°":
        return "ð¢"
    if relative_class == "GÃÃLÃ":
        return "ðµ"
    if relative_class == "ORTA":
        return "ð¡"
    return "ð´"

def main() -> None:
    try:
        frame = pd.read_csv(RESULT_FILE, encoding="utf-8-sig")
    except Exception as exc:
        print(f"{RESULT_FILE} okunamadÄ±: {exc}")
        return
    status = load_status()
    if frame.empty:
        send_message("ð LARUS V16 GÃRELÄ° GÃÃ RAPORU\n\nBugÃ¼n karÅÄ±laÅtÄ±rÄ±lacak V15 adayÄ± bulunamadÄ±.")
        return
    market_count = int(safe_float(status.get("market_count"), 0))
    message = (
        "ð LARUS V16 GÃRELÄ° GÃÃ RAPORU\n\n"
        f"KarÅÄ±laÅtÄ±rÄ±lan piyasa: {market_count} hisse\n"
        f"Ä°ncelenen V15 adayÄ±: {len(frame)}\n\n"
        "V15 adaylarÄ±; momentum, trend, hacim ve kalite aÃ§Ä±sÄ±ndan tÃ¼m BIST ile karÅÄ±laÅtÄ±rÄ±ldÄ±.\n\n"
    )
    for index, row in frame.iterrows():
        relative_class = clean_text(row.get("relative_class"))
        message += (
            f"{class_emoji(relative_class)} {index + 1}. {clean_text(row.get('symbol'))}\n"
            f"V16 sÄ±nÄ±fÄ±: {relative_class}\n"
            f"Piyasa sÄ±rasÄ±: {int(safe_float(row.get('market_rank')))}/{market_count}\n"
            f"Piyasa yÃ¼zdeliÄi: %{safe_float(row.get('market_percentile')):.1f}\n"
            f"GÃ¶reli gÃ¼Ã§ skoru: {safe_float(row.get('relative_strength_score')):.1f}/100\n"
            f"Momentum: {safe_float(row.get('momentum_percentile')):.1f}/100\n"
            f"Trend: {safe_float(row.get('trend_percentile')):.1f}/100\n"
            f"Hacim: {safe_float(row.get('volume_percentile')):.1f}/100\n"
            f"Kalite: {safe_float(row.get('quality_percentile')):.1f}/100\n"
            f"V15 kararÄ±: {clean_text(row.get('v15_decision'))}\n"
            f"V15 skoru: {safe_float(row.get('v15_score')):.1f}/100\n"
            f"1 gÃ¼nlÃ¼k deÄiÅim: {safe_float(row.get('return_1d')):+.1f}%\n"
            f"5 gÃ¼nlÃ¼k deÄiÅim: {safe_float(row.get('return_5d')):+.1f}%\n"
            f"20 gÃ¼nlÃ¼k deÄiÅim: {safe_float(row.get('return_20d')):+.1f}%\n"
            "\n--------------------\n\n"
        )
    message += "â ï¸ GÃ¶reli gÃ¼Ã§, hissenin piyasadaki diÄer hisselere gÃ¶re istatistiksel konumudur. YatÄ±rÄ±m tavsiyesi deÄildir."
    send_message(message)

if __name__ == "__main__":
    main()
