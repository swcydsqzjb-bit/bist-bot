from __future__ import annotations

import json
import os
from typing import Any, List

import numpy as np
import pandas as pd
import requests


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RESULT_FILE = "v19_timing_forecasts.csv"
STATUS_FILE = "v19_timing_status.json"
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
            data={
                "chat_id": CHAT_ID,
                "text": part,
                "disable_web_page_preview": True,
            },
            timeout=30,
        )
        print(response.status_code, response.text[:300])


def main() -> None:
    try:
        frame = pd.read_csv(RESULT_FILE, encoding="utf-8-sig")
    except Exception as exc:
        print(f"{RESULT_FILE} okunamadÄ±: {exc}")
        return

    status = load_status()

    if frame.empty:
        send_message(
            "â±ï¸ LARUS V19 ZAMANLAMA RAPORU\n\n"
            "BugÃ¼n zamanlama tahmini yapÄ±lacak aday bulunamadÄ±."
        )
        return

    message = (
        "â±ï¸ LARUS V19 ZAMANLAMA RAPORU\n\n"
        f"Ä°ncelenen aday: {len(frame)}\n"
        f"ZamanlamasÄ± hesaplanan: {int(safe_float(status.get('timing_ready_count')))}\n"
        f"Tarihsel hafÄ±za: {int(safe_float(status.get('history_count')))} Ã¶rnek\n\n"
        "V19, geÃ§miÅteki benzer sinyallerin 1, 3, 5 ve 10 iÅlem gÃ¼nlÃ¼k "
        "sonuÃ§larÄ±nÄ± karÅÄ±laÅtÄ±rarak en uygun izleme ufkunu tahmin eder.\n\n"
    )

    for _, row in frame.iterrows():
        symbol = clean_text(row.get("symbol"))
        ready = str(row.get("timing_ready")).lower() in {"true", "1"}

        message += f"ð¯ {int(safe_float(row.get('rank')))}. {symbol}\n"

        if not ready:
            message += (
                "Durum: VERÄ° YETERSÄ°Z\n"
                f"Benzer Ã¶rnek: {int(safe_float(row.get('neighbor_count')))}\n"
                f"AÃ§Ä±klama: {clean_text(row.get('timing_message'))}\n"
                "\n--------------------\n\n"
            )
            continue

        message += (
            f"Zamanlama sÄ±nÄ±fÄ±: {clean_text(row.get('timing_class'))}\n"
            f"Ãnerilen izleme ufku: {int(safe_float(row.get('best_horizon_days')))} iÅlem gÃ¼nÃ¼\n"
            f"Zamanlama gÃ¼veni: {safe_float(row.get('timing_confidence')):.1f}/100\n"
            f"Benzer tarihsel Ã¶rnek: {int(safe_float(row.get('neighbor_count')))}\n"
            f"Beklenen ortalama sonuÃ§: {safe_float(row.get('expected_return')):+.2f}%\n"
            f"Medyan sonuÃ§: {safe_float(row.get('median_return')):+.2f}%\n"
            f"Pozitif sonuÃ§ oranÄ±: %{safe_float(row.get('positive_rate')):.1f}\n"
            f"En az %3 oranÄ±: %{safe_float(row.get('hit_3_rate')):.1f}\n"
            f"Temkinli senaryo: {safe_float(row.get('downside_20pct')):+.2f}%\n"
            f"Olumlu senaryo: {safe_float(row.get('upside_80pct')):+.2f}%\n\n"
            "Ufuk karÅÄ±laÅtÄ±rmasÄ±:\n"
            f"â¢ 1 gÃ¼n: {safe_float(row.get('result_1d_mean')):+.2f}%\n"
            f"â¢ 3 gÃ¼n: {safe_float(row.get('result_3d_mean')):+.2f}%\n"
            f"â¢ 5 gÃ¼n: {safe_float(row.get('result_5d_mean')):+.2f}%\n"
            f"â¢ 10 gÃ¼n: {safe_float(row.get('result_10d_mean')):+.2f}%\n"
            "\n--------------------\n\n"
        )

    message += (
        "â ï¸ V19 zamanlama Ã§Ä±ktÄ±sÄ± geÃ§miÅ benzer Ã¶rneklerin istatistiksel "
        "Ã¶zetidir; alÄ±m-satÄ±m talimatÄ± veya getiri garantisi deÄildir."
    )

    send_message(message)


if __name__ == "__main__":
    main()
