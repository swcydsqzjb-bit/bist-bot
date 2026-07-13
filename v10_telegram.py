from __future__ import annotations

import os
from typing import Any, List

import numpy as np
import pandas as pd
import requests


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

PREDICTIONS_FILE = "v10_follow_predictions.csv"
MESSAGE_LIMIT = 3900


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    if isinstance(value, (int, float)):
        if pd.isna(value):
            return default
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "evet",
        "on",
    }


def repair_text(value: Any) -> str:
    text = "" if value is None else str(value)

    # Mojibake ihtimali yoksa metni oldugu gibi birak.
    if not any(ord(ch) in {195, 196, 197, 226, 240, 194} for ch in text):
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
    text = repair_text(text)

    if not TOKEN or not CHAT_ID:
        print("Telegram TOKEN veya CHAT_ID bulunamadi.")
        print(text)
        return False

    success = True

    for index, part in enumerate(split_message(text), start=1):
        try:
            response = requests.post(
                "https://api.telegram.org/bot"
                + TOKEN
                + "/sendMessage",
                data={
                    "chat_id": CHAT_ID,
                    "text": part,
                    "disable_web_page_preview": True,
                },
                timeout=30,
            )

            print(
                "Telegram parca "
                + str(index)
                + ": "
                + str(response.status_code)
            )

            if response.status_code != 200:
                print(response.text[:500])
                success = False

        except Exception as exc:
            print("Telegram gonderim hatasi:", exc)
            success = False

    return success


def parse_items(value: Any, limit: int = 5) -> List[str]:
    return [
        repair_text(item.strip())
        for item in repair_text(value).split("|")
        if item.strip()
    ][:limit]


def format_signed_percentage(value: Any, digits: int = 1) -> str:
    number = safe_float(value)
    return f"{number:+.{digits}f}%"


def build_prediction_section(row: pd.Series) -> str:
    rank = int(safe_float(row.get("rank"), 0))
    leader = repair_text(row.get("leader", "")).strip()
    follower = repair_text(row.get("follower", "")).strip()
    lag_days = int(safe_float(row.get("lag_days"), 0))

    prediction_score = safe_float(row.get("prediction_score"))
    prediction_class = repair_text(
        row.get("prediction_classification", "")
    )

    live_score = safe_float(row.get("live_confirmation_score"))
    live_class = repair_text(
        row.get("live_confirmation_class", "Bilinmiyor")
    )

    test_events = int(safe_float(row.get("test_events"), 0))
    test_success = safe_float(row.get("test_success_rate"))
    baseline = safe_float(row.get("test_baseline_rate"))
    uplift = safe_float(row.get("test_uplift"))
    average_return = safe_float(row.get("test_average_return"))
    relationship_score = safe_float(row.get("relationship_score"))

    price = safe_float(row.get("follower_price"))
    return_1d = safe_float(row.get("follower_return_1d"))
    return_5d = safe_float(row.get("follower_return_5d"))
    return_20d = safe_float(row.get("follower_return_20d"))
    volume_ratio = safe_float(row.get("follower_volume_ratio"))
    rsi = safe_float(row.get("follower_rsi"))
    ema20_distance = safe_float(row.get("follower_ema20_distance"))
    ema_slope_positive = safe_bool(
        row.get("follower_ema20_slope_positive")
    )
    close_position = safe_float(row.get("follower_close_position"))

    live_reasons = parse_items(
        row.get("live_confirmation_reasons", ""),
        limit=6,
    )

    live_risks = parse_items(
        row.get("live_confirmation_risks", ""),
        limit=5,
    )

    text = (
        "\U0001f3af "
        + str(rank)
        + ". "
        + follower
        + "\nLider: "
        + leader
        + "\nTarihsel gecikme: "
        + str(lag_days)
        + " i\u015flem g\u00fcn\u00fc"
        + "\nTakip\u00e7i skoru: "
        + f"{prediction_score:.1f}/100"
        + "\nS\u0131n\u0131f: "
        + prediction_class
        + "\nCanl\u0131 teyit: "
        + f"{live_score:.0f}/100"
        + "\nCanl\u0131 durum: "
        + live_class
        + "\n\nTarihsel ili\u015fki:"
        + "\n\u2022 \u0130li\u015fki skoru: "
        + f"{relationship_score:.1f}/100"
        + "\n\u2022 Son d\u00f6nem \u00f6rnek: "
        + str(test_events)
        + "\n\u2022 Lider sonras\u0131 ba\u015far\u0131: %"
        + f"{test_success:.1f}"
        + "\n\u2022 Normal ba\u015far\u0131: %"
        + f"{baseline:.1f}"
        + "\n\u2022 Ek tarihsel avantaj: +"
        + f"{uplift:.1f}"
        + " puan"
        + "\n\u2022 Lider sonras\u0131 ort. getiri: %"
        + f"{average_return:.2f}"
        + "\n\nG\u00fcncel takip\u00e7i g\u00f6r\u00fcn\u00fcm\u00fc:"
        + "\n\u2022 Fiyat: "
        + f"{price:.2f}"
        + "\n\u2022 1 g\u00fcnl\u00fck de\u011fi\u015fim: "
        + format_signed_percentage(return_1d)
        + "\n\u2022 5 g\u00fcnl\u00fck de\u011fi\u015fim: "
        + format_signed_percentage(return_5d)
        + "\n\u2022 20 g\u00fcnl\u00fck de\u011fi\u015fim: "
        + format_signed_percentage(return_20d)
        + "\n\u2022 Hacim: "
        + f"{volume_ratio:.2f}x"
        + "\n\u2022 RSI: "
        + f"{rsi:.1f}"
        + "\n\u2022 EMA20 fark\u0131: "
        + f"{ema20_distance:+.1f}%"
        + "\n\u2022 EMA20 e\u011fimi: "
        + ("Pozitif" if ema_slope_positive else "Zay\u0131f")
        + "\n\u2022 Kapan\u0131\u015f g\u00fcc\u00fc: "
        + f"{close_position:.2f}"
        + "\n"
    )

    if live_reasons:
        text += "\nCanl\u0131 teyit nedenleri:\n"
        for reason in live_reasons:
            text += "\u2713 " + reason + "\n"

    if live_risks:
        text += "\nCanl\u0131 riskler:\n"
        for risk in live_risks:
            text += "\u2022 " + risk + "\n"

    return repair_text(text.rstrip())


def build_report(predictions: pd.DataFrame) -> str:
    message = (
        "\U0001f985 LARUS V10 L\u0130DER\u2013TAK\u0130P\u00c7\u0130 RAPORU\n\n"
    )

    if predictions.empty:
        message += (
            "Bug\u00fcnk\u00fc g\u00fc\u00e7l\u00fc liderlerle e\u015fle\u015fen, "
            "hen\u00fcz fazla y\u00fckselmemi\u015f ve canl\u0131 hacim/RSI/"
            "EMA20 teyidini ge\u00e7en takip\u00e7i bulunamad\u0131.\n\n"
            "Sistem yaln\u0131zca tarihsel ili\u015fkiye g\u00fcvenerek "
            "zorla takip\u00e7i se\u00e7medi."
        )

        return message

    leaders = (
        predictions["leader"]
        .dropna()
        .astype(str)
        .map(repair_text)
        .unique()
        .tolist()
    )

    message += "Aktif liderler: " + ", ".join(leaders) + "\n"
    message += (
        "Canl\u0131 teyidi ge\u00e7en takip\u00e7i: "
        + str(len(predictions))
        + "\n\n"
        + "A\u015fa\u011f\u0131daki hisseler; ge\u00e7mi\u015f zamanlama ili\u015fkisi, "
        + "g\u00fcncel hacim, RSI ve EMA20 g\u00f6r\u00fcn\u00fcm\u00fc birlikte "
        + "de\u011ferlendirilerek se\u00e7ilmi\u015ftir.\n\n"
    )

    for _, row in predictions.iterrows():
        message += build_prediction_section(row)
        message += "\n\n--------------------\n\n"

    message += (
        "\u26a0\ufe0f Lider\u2013takip\u00e7i ili\u015fkisi nedensellik veya "
        "gelecekte y\u00fckseli\u015f garantisi de\u011fildir. Oranlar "
        "ge\u00e7mi\u015fte g\u00f6zlenen zamanlama ili\u015fkileridir. "
        "Canl\u0131 teyit de yaln\u0131zca fiyat-hacim g\u00f6stergelerinden "
        "olu\u015fur. Haber, sekt\u00f6r ve piyasa ko\u015fullar\u0131 ayr\u0131ca "
        "incelenmelidir; bu rapor yat\u0131r\u0131m tavsiyesi de\u011fildir."
    )

    return repair_text(message)


def main() -> None:
    if not os.path.exists(PREDICTIONS_FILE):
        print(PREDICTIONS_FILE + " bulunamadi.")
        return

    try:
        predictions = pd.read_csv(
            PREDICTIONS_FILE,
            encoding="utf-8-sig",
        )
    except UnicodeDecodeError:
        predictions = pd.read_csv(
            PREDICTIONS_FILE,
            encoding="utf-8",
        )
    except pd.errors.EmptyDataError:
        predictions = pd.DataFrame()
    except Exception as exc:
        print("V10 tahmin dosyasi okunamadi:", exc)
        return

    report = build_report(predictions)

    print(report)
    send_message(report)


if __name__ == "__main__":
    main()
