from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

TELEGRAM_LIMIT = 3900


def safe_float(
    value,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def split_message(
    text: str,
    limit: int = TELEGRAM_LIMIT,
) -> list[str]:
    if len(text) <= limit:
        return [text]

    parts = []
    current = ""

    for paragraph in text.split("\n\n"):
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                parts.append(current)

            current = paragraph

    if current:
        parts.append(current)

    return parts


def send_message(
    text: str,
) -> bool:
    if not TOKEN or not CHAT_ID:
        print("TOKEN veya CHAT_ID bulunamadı.")
        print(text)
        return False

    success = True

    for part in split_message(text):
        try:
            response = requests.post(
                (
                    "https://api.telegram.org/"
                    f"bot{TOKEN}/sendMessage"
                ),
                data={
                    "chat_id": CHAT_ID,
                    "text": part,
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )

            print(
                "Telegram:",
                response.status_code,
                response.text[:250],
            )

            if response.status_code != 200:
                success = False

        except Exception as exc:
            print("Telegram hatası:", exc)
            success = False

    return success


def short_reasons(
    value,
    limit: int = 5,
) -> list[str]:
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ][:limit]


def build_candidate(
    row: pd.Series,
) -> str:
    rank = int(
        safe_float(row.get("rank"), 0)
    )

    symbol = str(
        row.get("symbol", "")
    )

    price = safe_float(
        row.get("close")
    )

    v8_score = safe_float(
        row.get("v8_score")
    )

    smart_money = safe_float(
        row.get("smart_money_score")
    )

    institutional = safe_float(
        row.get("institutional_score")
    )

    historical_support = safe_float(
        row.get("historical_support_score")
    )

    positive_rate = safe_float(
        row.get("positive_rate_5d")
    )

    success_rate = safe_float(
        row.get("success_rate_3pct_5d")
    )

    weighted_result = safe_float(
        row.get("weighted_result_5d")
    )

    similar_count = int(
        safe_float(
            row.get("similar_example_count"),
            0,
        )
    )

    average_similarity = safe_float(
        row.get("average_similarity_pct")
    )

    similarity_confidence = str(
        row.get(
            "similarity_confidence",
            "YETERSİZ",
        )
    )

    institutional_class = str(
        row.get(
            "institutional_classification",
            "Bilinmiyor",
        )
    )

    text = (
        f"🏆 {rank}. {symbol}\n"
        f"Fiyat: {price:.2f}\n"
        f"Nihai V8 skor: {v8_score:.1f}/100\n"
        f"Sınıf: {row.get('v8_classification', '')}\n\n"
        f"Smart Money: {smart_money:.0f}/100\n"
        f"Kurumsal birikim: {institutional:.0f}/100\n"
        f"Kurumsal sınıf: {institutional_class}\n"
        f"Tarihsel destek: {historical_support:.1f}/100\n\n"
    )

    if bool(row.get("similarity_ready", False)):
        text += (
            "Tarihsel benzerlik:\n"
            f"• Benzer örnek: {similar_count}\n"
            f"• Ortalama benzerlik: %{average_similarity:.1f}\n"
            f"• 5 günde pozitif: %{positive_rate:.1f}\n"
            f"• 5 günde en az %3: %{success_rate:.1f}\n"
            f"• Ağırlıklı 5g sonuç: %{weighted_result:.2f}\n"
            f"• Benzerlik güveni: {similarity_confidence}\n\n"
        )
    else:
        text += (
            "Tarihsel benzerlik: "
            "Yeterli yakın örnek bulunamadı.\n\n"
        )

    reasons = short_reasons(
        row.get("positive_reasons", "")
    )

    if reasons:
        text += "Smart Money nedenleri:\n"

        for reason in reasons:
            text += f"✓ {reason}\n"

    institutional_reasons = short_reasons(
        row.get(
            "institutional_reasons",
            "",
        ),
        limit=4,
    )

    if institutional_reasons:
        text += "\nKurumsal davranış nedenleri:\n"

        for reason in institutional_reasons:
            text += f"✓ {reason}\n"

    institutional_risks = str(
        row.get(
            "institutional_risks",
            "Belirgin kurumsal risk yok",
        )
    )

    text += (
        "\nRisk notları:\n"
        f"• Teknik: {row.get('risk_reasons', 'Yok')}\n"
        f"• Kurumsal: {institutional_risks}\n"
    )

    return text


def build_report(
    candidates: pd.DataFrame,
    total_symbols: int,
    valid_symbols: int,
    shortlist_count: int,
) -> str:
    message = "🦅 LARUS V8 FUSION RAPORU\n\n"

    message += (
        f"Taranan hisse: {total_symbols}\n"
        f"Uygun verisi gelen: {valid_symbols}\n"
        f"Ayrıntılı incelenen: {shortlist_count}\n"
        f"Nihai aday: {len(candidates)}\n\n"
    )

    if candidates.empty:
        message += (
            "Bugün birleşik güven eşiğini geçen "
            "aday bulunamadı.\n\n"
            "Sistem zorla hisse seçmedi."
        )

        return message

    message += (
        "Smart Money + Kurumsal Davranış + "
        "Tarihsel Benzerlik\n\n"
    )

    for _, row in candidates.iterrows():
        message += build_candidate(row)
        message += "\n────────────────────\n\n"

    message += (
        "⚠️ Tarihsel oranlar gelecekteki getiriyi garanti etmez. "
        "Kurumsal birikim skoru gerçek takas verisi değil, "
        "fiyat-hacim davranışından üretilen göstergedir. "
        "Bu rapor yatırım tavsiyesi değildir."
    )

    return message


def send_v8_report(
    candidates: pd.DataFrame,
    total_symbols: int,
    valid_symbols: int,
    shortlist_count: int,
) -> bool:
    report = build_report(
        candidates=candidates,
        total_symbols=total_symbols,
        valid_symbols=valid_symbols,
        shortlist_count=shortlist_count,
    )

    print(report)

    return send_message(report)
