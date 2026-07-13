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


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
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


def split_message(
    text: str,
    limit: int = MESSAGE_LIMIT,
) -> List[str]:
    if len(text) <= limit:
        return [text]

    parts: List[str] = []
    current = ""

    for paragraph in text.split("\n\n"):
        candidate = (
            paragraph
            if not current
            else current + "\n\n" + paragraph
        )

        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            parts.append(current)

        if len(paragraph) <= limit:
            current = paragraph
            continue

        current = ""

        for line in paragraph.splitlines():
            candidate_line = (
                line
                if not current
                else current + "\n" + line
            )

            if len(candidate_line) <= limit:
                current = candidate_line
            else:
                if current:
                    parts.append(current)

                current = line[:limit]

    if current:
        parts.append(current)

    return parts


def send_message(text: str) -> bool:
    if not TOKEN or not CHAT_ID:
        print("Telegram TOKEN veya CHAT_ID bulunamadÄ±.")
        print(text)
        return False

    success = True
    parts = split_message(text)

    for index, part in enumerate(parts, start=1):
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
                f"Telegram parÃ§a {index}/{len(parts)}:",
                response.status_code,
                response.text[:300],
            )

            if response.status_code != 200:
                success = False

        except Exception as exc:
            print("Telegram gÃ¶nderim hatasÄ±:", exc)
            success = False

    return success


def parse_items(
    value: Any,
    limit: int = 5,
) -> List[str]:
    return [
        item.strip()
        for item in str(value).split("|")
        if item.strip()
    ][:limit]


def format_signed_percentage(
    value: Any,
    digits: int = 1,
) -> str:
    number = safe_float(value)
    return f"{number:+.{digits}f}%"


def build_prediction_section(
    row: pd.Series,
) -> str:
    rank = int(safe_float(row.get("rank"), 0))
    leader = str(row.get("leader", "")).strip()
    follower = str(row.get("follower", "")).strip()
    lag_days = int(safe_float(row.get("lag_days"), 0))

    prediction_score = safe_float(
        row.get("prediction_score")
    )

    prediction_class = str(
        row.get(
            "prediction_classification",
            "",
        )
    )

    live_score = safe_float(
        row.get("live_confirmation_score")
    )

    live_class = str(
        row.get(
            "live_confirmation_class",
            "Bilinmiyor",
        )
    )

    test_events = int(
        safe_float(row.get("test_events"), 0)
    )

    test_success = safe_float(
        row.get("test_success_rate")
    )

    baseline = safe_float(
        row.get("test_baseline_rate")
    )

    uplift = safe_float(
        row.get("test_uplift")
    )

    average_return = safe_float(
        row.get("test_average_return")
    )

    relationship_score = safe_float(
        row.get("relationship_score")
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

    return_20d = safe_float(
        row.get("follower_return_20d")
    )

    volume_ratio = safe_float(
        row.get("follower_volume_ratio")
    )

    rsi = safe_float(
        row.get("follower_rsi")
    )

    ema20_distance = safe_float(
        row.get("follower_ema20_distance")
    )

    ema_slope_positive = safe_bool(
        row.get("follower_ema20_slope_positive")
    )

    close_position = safe_float(
        row.get("follower_close_position")
    )

    live_reasons = parse_items(
        row.get(
            "live_confirmation_reasons",
            "",
        ),
        limit=6,
    )

    live_risks = parse_items(
        row.get(
            "live_confirmation_risks",
            "",
        ),
        limit=5,
    )

    text = (
        f"ð¯ {rank}. {follower}\n"
        f"Lider: {leader}\n"
        f"Tarihsel gecikme: {lag_days} iÅlem gÃ¼nÃ¼\n"
        f"TakipÃ§i skoru: {prediction_score:.1f}/100\n"
        f"SÄ±nÄ±f: {prediction_class}\n"
        f"CanlÄ± teyit: {live_score:.0f}/100\n"
        f"CanlÄ± durum: {live_class}\n\n"
        f"Tarihsel iliÅki:\n"
        f"â¢ Ä°liÅki skoru: {relationship_score:.1f}/100\n"
        f"â¢ Son dÃ¶nem Ã¶rnek: {test_events}\n"
        f"â¢ Lider sonrasÄ± baÅarÄ±: %{test_success:.1f}\n"
        f"â¢ Normal baÅarÄ±: %{baseline:.1f}\n"
        f"â¢ Ek tarihsel avantaj: +{uplift:.1f} puan\n"
        f"â¢ Lider sonrasÄ± ort. getiri: %{average_return:.2f}\n\n"
        f"GÃ¼ncel takipÃ§i gÃ¶rÃ¼nÃ¼mÃ¼:\n"
        f"â¢ Fiyat: {price:.2f}\n"
        f"â¢ 1 gÃ¼nlÃ¼k deÄiÅim: {format_signed_percentage(return_1d)}\n"
        f"â¢ 5 gÃ¼nlÃ¼k deÄiÅim: {format_signed_percentage(return_5d)}\n"
        f"â¢ 20 gÃ¼nlÃ¼k deÄiÅim: {format_signed_percentage(return_20d)}\n"
        f"â¢ Hacim: {volume_ratio:.2f}x\n"
        f"â¢ RSI: {rsi:.1f}\n"
        f"â¢ EMA20 farkÄ±: {ema20_distance:+.1f}%\n"
        f"â¢ EMA20 eÄimi: "
        f"{'Pozitif' if ema_slope_positive else 'ZayÄ±f'}\n"
        f"â¢ KapanÄ±Å gÃ¼cÃ¼: {close_position:.2f}\n"
    )

    if live_reasons:
        text += "\nCanlÄ± teyit nedenleri:\n"

        for reason in live_reasons:
            text += f"â {reason}\n"

    if live_risks:
        text += "\nCanlÄ± riskler:\n"

        for risk in live_risks:
            text += f"â¢ {risk}\n"

    return text.rstrip()


def build_report(
    predictions: pd.DataFrame,
) -> str:
    message = "ð¦ LARUS V10 LÄ°DERâTAKÄ°PÃÄ° RAPORU\n\n"

    if predictions.empty:
        message += (
            "BugÃ¼nkÃ¼ gÃ¼Ã§lÃ¼ liderlerle eÅleÅen, "
            "henÃ¼z fazla yÃ¼kselmemiÅ ve canlÄ± hacim/RSI/"
            "EMA20 teyidini geÃ§en takipÃ§i bulunamadÄ±.\n\n"
            "Sistem yalnÄ±zca tarihsel iliÅkiye gÃ¼venerek "
            "zorla takipÃ§i seÃ§medi."
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
        f"CanlÄ± teyidi geÃ§en takipÃ§i: "
        f"{len(predictions)}\n\n"
        "AÅaÄÄ±daki hisseler; geÃ§miÅ zamanlama iliÅkisi, "
        "gÃ¼ncel hacim, RSI ve EMA20 gÃ¶rÃ¼nÃ¼mÃ¼ birlikte "
        "deÄerlendirilerek seÃ§ilmiÅtir.\n\n"
    )

    for _, row in predictions.iterrows():
        message += build_prediction_section(row)
        message += "\n\nââââââââââââââââââââ\n\n"

    message += (
        "â ï¸ LiderâtakipÃ§i iliÅkisi nedensellik veya "
        "gelecekte yÃ¼kseliÅ garantisi deÄildir. Oranlar "
        "geÃ§miÅte gÃ¶zlenen zamanlama iliÅkileridir. "
        "CanlÄ± teyit de yalnÄ±zca fiyat-hacim gÃ¶stergelerinden "
        "oluÅur. Haber, sektÃ¶r ve piyasa koÅullarÄ± ayrÄ±ca "
        "incelenmelidir; bu rapor yatÄ±rÄ±m tavsiyesi deÄildir."
    )

    return message


def main() -> None:
    if not os.path.exists(PREDICTIONS_FILE):
        print(f"{PREDICTIONS_FILE} bulunamadÄ±.")
        return

    try:
        predictions = pd.read_csv(PREDICTIONS_FILE)
    except pd.errors.EmptyDataError:
        predictions = pd.DataFrame()
    except Exception as exc:
        print("V10 tahmin dosyasÄ± okunamadÄ±:", exc)
        return

    report = build_report(predictions)

    print(report)
    send_message(report)


if __name__ == "__main__":
    main()
