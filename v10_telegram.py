from __future__ import annotations

import os

import numpy as np
import pandas as pd
import requests


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

PREDICTIONS_FILE = "v10_follow_predictions.csv"
MESSAGE_LIMIT = 3900


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
) -> list[str]:
    if len(text) <= MESSAGE_LIMIT:
        return [text]

    parts = []
    current = ""

    for paragraph in text.split("\n\n"):
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= MESSAGE_LIMIT:
            current = candidate
        else:
            if current:
                parts.append(current)

            current = paragraph

    if current:
        parts.append(current)

    return parts


def send_message(text: str) -> bool:
    if not TOKEN or not CHAT_ID:
        print(
            "Telegram TOKEN veya CHAT_ID yok."
        )
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
                response.text[:300],
            )

            if response.status_code != 200:
                success = False

        except Exception as exc:
            print(
                "Telegram gönderim hatası:",
                exc,
            )
            success = False

    return success


def build_prediction_section(
    row: pd.Series,
) -> str:
    rank = int(
        safe_float(row.get("rank"))
    )

    leader = str(
        row.get("leader", "")
    )

    follower = str(
        row.get("follower", "")
    )

    lag_days = int(
        safe_float(row.get("lag_days"))
    )

    score = safe_float(
        row.get("prediction_score")
    )

    classification = str(
        row.get(
            "prediction_classification",
            "",
        )
    )

    test_events = int(
        safe_float(row.get("test_events"))
    )

    success_rate = safe_float(
        row.get("test_success_rate")
    )

    baseline_rate = safe_float(
        row.get("test_baseline_rate")
    )

    uplift = safe_float(
        row.get("test_uplift")
    )

    average_return = safe_float(
        row.get("test_average_return")
    )

    price = safe_float(
        row.get("follower_price")
    )

    return_1d = safe_float(
        row.get("follower_return_1d")
    )

    return_5d = safe_float(
        row.get("follower_return_5d")
    )

    volume_ratio = safe_float(
        row.get("follower_volume_ratio")
    )

    return (
        f"🎯 {rank}. {follower}\n"
        f"Lider: {leader}\n"
        f"Tarihsel gecikme: {lag_days} işlem günü\n"
        f"Takipçi skoru: {score:.1f}/100\n"
        f"Sınıf: {classification}\n\n"
        f"Son dönem ilişki örneği: {test_events}\n"
        f"Lider sonrası başarı: %{success_rate:.1f}\n"
        f"Normal başarı: %{baseline_rate:.1f}\n"
        f"Ek tarihsel avantaj: +{uplift:.1f} puan\n"
        f"Lider sonrası ort. getiri: %{average_return:.2f}\n\n"
        f"Güncel takipçi fiyatı: {price:.2f}\n"
        f"Bugünkü değişim: %{return_1d:.1f}\n"
        f"5 günlük değişim: %{return_5d:.1f}\n"
        f"Günlük hacim: {volume_ratio:.2f}x\n"
    )


def build_report(
    predictions: pd.DataFrame,
) -> str:
    message = (
        "🦅 LARUS V10 LİDER–TAKİPÇİ RAPORU\n\n"
    )

    if predictions.empty:
        message += (
            "Bugünkü güçlü liderlerle eşleşen, "
            "henüz fazla yükselmemiş ve doğrulama "
            "şartlarını geçen takipçi bulunamadı.\n\n"
            "Sistem zorla takipçi seçmedi."
        )

        return message

    leaders = (
        predictions["leader"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    message += (
        "Aktif liderler: "
        + ", ".join(leaders)
        + "\n"
    )

    message += (
        f"Takipçi adayı: {len(predictions)}\n\n"
        "Aşağıdaki hisseler geçmiş zamanlama "
        "ilişkilerine göre izleme adayıdır.\n\n"
    )

    for _, row in predictions.iterrows():
        message += build_prediction_section(row)
        message += "\n────────────────────\n\n"

    message += (
        "⚠️ Lider–takipçi ilişkisi nedensellik veya "
        "gelecekte yükseliş garantisi değildir. Oranlar, "
        "geçmişte gözlenen zamanlama ilişkileridir. "
        "Haber, sektör ve piyasa koşulları ayrıca incelenmelidir."
    )

    return message


def main():
    if not os.path.exists(PREDICTIONS_FILE):
        print(
            f"{PREDICTIONS_FILE} bulunamadı."
        )
        return

    predictions = pd.read_csv(
        PREDICTIONS_FILE
    )

    report = build_report(predictions)

    print(report)

    send_message(report)


if __name__ == "__main__":
    main()
