from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import pandas as pd
import requests

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
RESULT_FILE = "v13_market_dna_results.csv"
MESSAGE_LIMIT = 3900


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "evet", "on"}:
        return True
    if text in {"false", "0", "no", "hayir", "off", "", "nan", "none"}:
        return False
    return default


def repair_text(value: Any) -> str:
    text = "" if value is None else str(value)
    suspicious_codes = {194, 195, 196, 197, 226, 240}
    if not any(ord(character) in suspicious_codes for character in text):
        return text
    for encoding in ("latin1", "cp1252"):
        try:
            repaired = text.encode(encoding).decode("utf-8")
            if repaired != text:
                return repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return text


def split_message(text: str, limit: int = MESSAGE_LIMIT) -> List[str]:
    text = repair_text(text)
    if len(text) <= limit:
        return [text]
    parts: List[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        candidate = paragraph if not current else current + "\n\n" + paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            parts.append(current)
        current = paragraph
    if current:
        parts.append(current)
    return parts


def send_message(text: str) -> bool:
    if not TOKEN or not CHAT_ID:
        print("Telegram TOKEN veya CHAT_ID bulunamadi.")
        print(text)
        return False
    success = True
    parts = split_message(text)
    for index, part in enumerate(parts, start=1):
        try:
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
                f"Telegram parca {index}/{len(parts)}:",
                response.status_code,
                response.text[:250],
            )
            if response.status_code != 200:
                success = False
        except Exception as exc:
            print("Telegram gonderim hatasi:", exc)
            success = False
    return success


def format_signed_percent(value: Any, digits: int = 2) -> str:
    number = safe_float(value)
    return f"{number:+.{digits}f}%"


def classification_emoji(classification: str) -> str:
    text = repair_text(classification).upper()
    if "GUCLU" in text or "G\u00dc\u00c7L\u00dc" in text:
        return "\U0001f7e2"
    if "ORTA" in text:
        return "\U0001f7e1"
    if "KARISIK" in text or "KARI\u015eIK" in text:
        return "\U0001f7e0"
    if "ZAYIF" in text:
        return "\U0001f534"
    return "\u26aa"


def build_candidate_section(row: pd.Series) -> str:
    rank = safe_int(row.get("rank"), 0)
    symbol = repair_text(row.get("symbol", "")).strip()
    classification = repair_text(row.get("dna_classification", "BILINMIYOR"))
    close = safe_float(row.get("close"))
    v8_score = safe_float(row.get("v8_score"))
    dna_confidence = safe_float(row.get("dna_confidence"))
    neighbor_count = safe_int(row.get("neighbor_count"), 0)
    historical_count = safe_int(row.get("historical_sample_count"), 0)
    average_similarity = safe_float(row.get("average_similarity_pct"))
    best_similarity = safe_float(row.get("best_similarity_pct"))
    positive_rate = safe_float(row.get("positive_rate_5d"))
    hit_3pct_rate = safe_float(row.get("hit_3pct_5d_rate"))
    average_result = safe_float(row.get("average_result_5d"))
    median_result = safe_float(row.get("median_result_5d"))
    average_max = safe_float(row.get("average_max_result_5d"))
    average_min = safe_float(row.get("average_min_result_5d"))
    used_feature_count = safe_int(row.get("used_feature_count"), 0)
    emoji = classification_emoji(classification)
    return (
        f"{emoji} {rank}. {symbol}\n"
        f"DNA sinifi: {classification}\n"
        f"Fiyat: {close:.2f}\n"
        f"V8 skoru: {v8_score:.1f}/100\n"
        f"DNA guveni: {dna_confidence:.1f}/100\n\n"
        f"Tarihsel DNA:\n"
        f"\u2022 Toplam hafiza: {historical_count}\n"
        f"\u2022 En yakin ornek: {neighbor_count}\n"
        f"\u2022 Kullanilan ozellik: {used_feature_count}\n"
        f"\u2022 Ortalama benzerlik: %{average_similarity:.1f}\n"
        f"\u2022 En iyi benzerlik: %{best_similarity:.1f}\n"
        f"\u2022 5 gunde pozitif: %{positive_rate:.1f}\n"
        f"\u2022 5 gunde en az %3: %{hit_3pct_rate:.1f}\n"
        f"\u2022 Ortalama 5g sonuc: {format_signed_percent(average_result)}\n"
        f"\u2022 Medyan 5g sonuc: {format_signed_percent(median_result)}\n"
        f"\u2022 Ortalama en yuksek hareket: {format_signed_percent(average_max)}\n"
        f"\u2022 Ortalama en dusuk hareket: {format_signed_percent(average_min)}"
    )


def build_report(results: pd.DataFrame) -> str:
    message = "\U0001f9ec LARUS V13 MARKET DNA RAPORU\n\n"
    if results.empty:
        return message + (
            "Bugunku V8 adaylari icin Market DNA sonucu bulunamadi.\n\n"
            "Bu durum aday dosyasinin bos olmasindan veya tarihsel "
            "hafizanin yetersizliginden kaynaklanabilir."
        )

    ready_mask = results.get(
        "dna_ready",
        pd.Series([False] * len(results)),
    ).apply(safe_bool)
    ready_results = results[ready_mask].copy()

    message += (
        f"Analiz edilen aday: {len(results)}\n"
        f"DNA sonucu hazir: {len(ready_results)}\n\n"
        "V8 adaylari, gecmisteki benzer fiyat-hacim yapilarinin "
        "5 gunluk sonuclariyla karsilastirildi.\n\n"
    )

    if ready_results.empty:
        return message + (
            "Yeterli ortak ozellik veya yakin tarihsel ornek bulunan aday cikmadi."
        )

    ready_results = ready_results.sort_values(
        by=["dna_confidence", "positive_rate_5d", "average_result_5d"],
        ascending=False,
    )

    for _, row in ready_results.iterrows():
        message += build_candidate_section(row)
        message += "\n\n--------------------\n\n"

    message += (
        "\u26a0\ufe0f Market DNA, gecmisteki benzer orneklerin istatistiksel "
        "sonucudur. Gelecekteki getiriyi garanti etmez ve yatirim tavsiyesi degildir."
    )
    return message


def main() -> None:
    if not os.path.exists(RESULT_FILE):
        print(f"{RESULT_FILE} bulunamadi.")
        return
    try:
        results = pd.read_csv(RESULT_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        results = pd.read_csv(RESULT_FILE, encoding="utf-8")
    except pd.errors.EmptyDataError:
        results = pd.DataFrame()
    except Exception as exc:
        print("V13 sonuc dosyasi okunamadi:", exc)
        return

    report = build_report(results)
    print(report)
    send_message(report)


if __name__ == "__main__":
    main()
