from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import requests

from v3_config import CHAT_ID, TOKEN


TELEGRAM_MESSAGE_LIMIT = 3900


def safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(round(float(value)))

    except (TypeError, ValueError):
        return default


def send_telegram_message(
    text: str,
    token: str | None = None,
    chat_id: str | None = None,
) -> bool:
    """
    Uzun Telegram mesajlarını güvenli biçimde parçalara ayırır.
    """
    bot_token = token or TOKEN or os.getenv("TOKEN")
    target_chat = chat_id or CHAT_ID or os.getenv("CHAT_ID")

    if not bot_token or not target_chat:
        print("Telegram TOKEN veya CHAT_ID bulunamadı.")
        print(text)
        return False

    message_parts = split_message(
        text,
        TELEGRAM_MESSAGE_LIMIT,
    )

    all_successful = True

    for part_number, part in enumerate(
        message_parts,
        start=1,
    ):
        try:
            response = requests.post(
                (
                    f"https://api.telegram.org/"
                    f"bot{bot_token}/sendMessage"
                ),
                data={
                    "chat_id": target_chat,
                    "text": part,
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )

            print(
                f"Telegram parça "
                f"{part_number}/{len(message_parts)}:",
                response.status_code,
                response.text[:300],
            )

            if response.status_code != 200:
                all_successful = False

        except Exception as exc:
            all_successful = False
            print("Telegram gönderim hatası:", exc)

    return all_successful


def split_message(
    text: str,
    maximum_length: int,
) -> List[str]:
    """
    Mesajı mümkün olduğunca paragraf sınırlarından böler.
    """
    if len(text) <= maximum_length:
        return [text]

    paragraphs = text.split("\n\n")
    parts: List[str] = []
    current_part = ""

    for paragraph in paragraphs:
        candidate = (
            paragraph
            if not current_part
            else current_part + "\n\n" + paragraph
        )

        if len(candidate) <= maximum_length:
            current_part = candidate
            continue

        if current_part:
            parts.append(current_part)

        if len(paragraph) <= maximum_length:
            current_part = paragraph
            continue

        # Tek paragraf çok uzunsa satır bazında böl.
        lines = paragraph.splitlines()
        current_part = ""

        for line in lines:
            candidate_line = (
                line
                if not current_part
                else current_part + "\n" + line
            )

            if len(candidate_line) <= maximum_length:
                current_part = candidate_line
            else:
                if current_part:
                    parts.append(current_part)

                current_part = line[:maximum_length]

    if current_part:
        parts.append(current_part)

    return parts


def classification_icon(classification: str) -> str:
    classification = str(classification).upper()

    if "ELİT" in classification:
        return "🏆"

    if "GÜÇLÜ" in classification:
        return "🔥"

    if "İZLEME" in classification:
        return "👀"

    return "⚪"


def risk_level(row: pd.Series) -> str:
    penalty = abs(
        safe_float(
            row.get("risk_penalty", 0)
        )
    )

    risk_reasons = str(
        row.get("risk_reasons", "")
    ).strip()

    if penalty >= 20:
        return "Yüksek"

    if penalty >= 8:
        return "Orta"

    if (
        risk_reasons
        and "Belirgin risk yok" not in risk_reasons
    ):
        return "Orta"

    return "Düşük"


def confidence_level(row: pd.Series) -> int:
    """
    Bu değer gerçek başarı ihtimali değildir.

    Smart Money skoru, seçim skoru ve risk cezasından
    oluşturulan karşılaştırmalı güven puanıdır.
    """
    smart_score = safe_float(
        row.get("smart_money_score", 0)
    )

    selection_score = safe_float(
        row.get("selection_score", smart_score)
    )

    risk_penalty = abs(
        safe_float(
            row.get("risk_penalty", 0)
        )
    )

    confidence = (
        smart_score * 0.60
        + selection_score * 0.40
        - risk_penalty * 0.25
    )

    return safe_int(
        max(0, min(100, confidence))
    )


def candidate_time_horizon(row: pd.Series) -> str:
    """
    Teknik yapıya göre yaklaşık izleme süresi.
    Kesin tahmin değildir.
    """
    range_20 = safe_float(
        row.get("range_20_pct", 0)
    )

    volume_ratio = safe_float(
        row.get("volume_ratio", 0)
    )

    distance_to_high = safe_float(
        row.get("distance_to_high_20", 100)
    )

    if (
        distance_to_high <= 3
        and volume_ratio >= 1.30
    ):
        return "1–3 işlem günü"

    if range_20 <= 15:
        return "2–7 işlem günü"

    return "3–10 işlem günü"


def calculate_price_levels(
    row: pd.Series,
) -> dict:
    """
    ATR kullanılarak örnek takip seviyeleri oluşturur.

    Bunlar garanti hedef veya yatırım tavsiyesi değildir.
    """
    close = safe_float(
        row.get("close", 0)
    )

    atr_pct = safe_float(
        row.get("atr_pct", 0)
    )

    if close <= 0:
        return {
            "stop": 0,
            "target_1": 0,
            "target_2": 0,
        }

    # ATR verisi bozuksa temkinli varsayılan kullan.
    if atr_pct <= 0:
        atr_pct = 3.0

    atr_amount = close * atr_pct / 100

    stop = close - atr_amount * 1.15
    target_1 = close + atr_amount * 1.50
    target_2 = close + atr_amount * 2.40

    return {
        "stop": round(max(0, stop), 2),
        "target_1": round(target_1, 2),
        "target_2": round(target_2, 2),
    }


def shorten_reasons(
    text: str,
    maximum_items: int = 6,
) -> List[str]:
    reasons = [
        reason.strip()
        for reason in str(text).split("|")
        if reason.strip()
    ]

    return reasons[:maximum_items]


def build_candidate_section(
    row: pd.Series,
) -> str:
    rank = safe_int(
        row.get("rank", 0)
    )

    symbol = str(
        row.get("symbol", "")
    ).strip()

    classification = str(
        row.get("classification", "ADAY")
    )

    icon = classification_icon(classification)

    smart_score = safe_int(
        row.get("smart_money_score", 0)
    )

    selection_score = safe_float(
        row.get("selection_score", 0)
    )

    close = safe_float(
        row.get("close", 0)
    )

    rsi = safe_float(
        row.get("rsi", 0)
    )

    volume_ratio = safe_float(
        row.get("volume_ratio", 0)
    )

    accumulation_ratio = safe_float(
        row.get(
            "volume_accumulation_ratio",
            0,
        )
    )

    range_20 = safe_float(
        row.get("range_20_pct", 0)
    )

    return_5d = safe_float(
        row.get("return_5d", 0)
    )

    return_20d = safe_float(
        row.get("return_20d", 0)
    )

    confidence = confidence_level(row)
    risk = risk_level(row)
    horizon = candidate_time_horizon(row)
    levels = calculate_price_levels(row)

    reasons = shorten_reasons(
        row.get("positive_reasons", "")
    )

    risk_reasons = str(
        row.get(
            "risk_reasons",
            "Belirgin risk yok",
        )
    )

    text = (
        f"{icon} {rank}. {symbol}\n"
        f"Tür: {classification}\n"
        f"Fiyat: {close:.2f}\n"
        f"Smart Money: {smart_score}/100\n"
        f"Seçim skoru: {selection_score:.1f}/100\n"
        f"Karşılaştırmalı güven: {confidence}/100\n"
        f"Risk seviyesi: {risk}\n"
        f"İzleme süresi: {horizon}\n\n"
        f"RSI: {rsi:.1f}\n"
        f"Günlük hacim: {volume_ratio:.2f}x\n"
        f"Hacim birikimi: {accumulation_ratio:.2f}x\n"
        f"20 günlük bant: %{range_20:.1f}\n"
        f"5 günlük değişim: %{return_5d:.1f}\n"
        f"20 günlük değişim: %{return_20d:.1f}\n\n"
    )

    if reasons:
        text += "Güçlü nedenler:\n"

        for reason in reasons:
            text += f"✓ {reason}\n"

    text += (
        "\nTakip seviyeleri:\n"
        f"• Geçersizlik/stop bölgesi: {levels['stop']:.2f}\n"
        f"• İlk izleme hedefi: {levels['target_1']:.2f}\n"
        f"• İkinci izleme hedefi: {levels['target_2']:.2f}\n"
        f"• Risk notu: {risk_reasons}\n"
    )

    return text


def build_v3_report(
    candidates_df: pd.DataFrame,
    total_symbols: int,
    valid_symbols: int,
    eligible_count: int,
) -> str:
    message = "🧠 BIST V3 SMART MONEY RAPORU\n\n"

    message += (
        f"Taranan hisse: {total_symbols}\n"
        f"Uygun verisi gelen: {valid_symbols}\n"
        f"Güven eşiğini geçen: {eligible_count}\n"
        f"Seçilen aday: {len(candidates_df)}\n\n"
    )

    if candidates_df.empty:
        message += (
            "Bugün yeterli güven seviyesine ulaşan "
            "aday bulunamadı.\n\n"
            "Sistem zorla hisse seçmedi."
        )

        return message

    message += "🏅 Bugünün seçilmiş adayları\n\n"

    for _, row in candidates_df.iterrows():
        message += build_candidate_section(row)
        message += "\n────────────────────\n\n"

    message += (
        "⚠️ Not: Güven puanı gerçek yükselme olasılığı değildir. "
        "Fiyat ve hacim ölçütlerinin karşılaştırmalı skorudur. "
        "Takip seviyeleri ATR tabanlı örnek risk bölgeleridir; "
        "yatırım tavsiyesi veya getiri garantisi değildir."
    )

    return message


def send_v3_report(
    candidates_df: pd.DataFrame,
    total_symbols: int,
    valid_symbols: int,
    eligible_count: int,
) -> bool:
    report = build_v3_report(
        candidates_df=candidates_df,
        total_symbols=total_symbols,
        valid_symbols=valid_symbols,
        eligible_count=eligible_count,
    )

    print("\nTELEGRAM RAPORU\n")
    print(report)

    return send_telegram_message(report)


def main():
    """
    Mevcut aday CSV dosyasını kullanarak Telegram görünümünü test eder.
    """
    candidate_file = "v3_today_candidates.csv"

    if not os.path.exists(candidate_file):
        print(
            "v3_today_candidates.csv bulunamadı. "
            "Önce v3_scanner.py çalıştırılmalı."
        )
        return

    candidates = pd.read_csv(candidate_file)

    send_v3_report(
        candidates_df=candidates,
        total_symbols=0,
        valid_symbols=0,
        eligible_count=len(candidates),
    )


if __name__ == "__main__":
    main()
