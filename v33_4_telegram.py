from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


# ============================================================
# V33.4 - TELEGRAM GERCEK PERFORMANS RAPORU
# ============================================================

VERSION = "V33.4"

TRACKING_FILE = Path("v33_4_candidate_history.csv")
STATUS_FILE = Path("v33_4_status.json")

TOKEN = os.getenv("TOKEN", "").strip()
CHAT_ID = os.getenv("CHAT_ID", "").strip()

RSI_USAGE = "DISABLED"

MAX_DETAIL_ROWS = 10


# ============================================================
# YARDIMCILAR
# ============================================================

def text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result

        return default

    except (TypeError, ValueError):
        return default


def optional_number(
    value: Any,
) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return np.nan


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            return pd.DataFrame()
    except OSError:
        return pd.DataFrame()

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        except UnicodeDecodeError:
            continue

        except Exception as exc:
            print(
                f"UYARI: {path} okunamadi: {exc}"
            )
            return pd.DataFrame()

    return pd.DataFrame()


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        raw = path.read_text(
            encoding="utf-8"
        )

        if not raw.strip():
            return {}

        data = json.loads(raw)

        if isinstance(data, dict):
            return data

    except Exception as exc:
        print(
            f"UYARI: {path} okunamadi: {exc}"
        )

    return {}


def format_return(
    value: Any,
) -> str:
    result = optional_number(
        value
    )

    if not np.isfinite(result):
        return "—"

    sign = "+" if result > 0 else ""

    return f"{sign}{result:.2f}%"


def format_number(
    value: Any,
    decimals: int = 2,
) -> str:
    result = optional_number(
        value
    )

    if not np.isfinite(result):
        return "—"

    return f"{result:.{decimals}f}"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    message: str,
) -> None:
    if not TOKEN or not CHAT_ID:
        print(
            "Telegram TOKEN veya CHAT_ID eksik."
        )
        print(message)
        return

    url = (
        f"https://api.telegram.org/"
        f"bot{TOKEN}/sendMessage"
    )

    payload = {
        "chat_id": CHAT_ID,
        "text": message,
        "disable_web_page_preview": True,
    }

    try:
        response = requests.post(
            url,
            data=payload,
            timeout=30,
        )

        response.raise_for_status()

        print(
            "V33.4 Telegram mesaji gonderildi."
        )

    except Exception as exc:
        print(
            f"V33.4 Telegram hatasi: {exc}"
        )


# ============================================================
# SON DURUM RAPORU
# ============================================================

def build_summary_message(
    status: dict[str, Any],
    history: pd.DataFrame,
) -> str:

    tracking_date = text(
        status.get("tracking_date")
    )

    new_count = int(
        number(
            status.get(
                "new_candidate_count"
            )
        )
    )

    total_count = int(
        number(
            status.get(
                "total_tracking_count"
            )
        )
    )

    completed_count = int(
        number(
            status.get(
                "completed_count"
            )
        )
    )

    active_count = int(
        number(
            status.get(
                "active_count"
            )
        )
    )

    price_update_count = int(
        number(
            status.get(
                "price_update_count"
            )
        )
    )

    hit_3 = int(
        number(
            status.get(
                "hit_3pct_count"
            )
        )
    )

    hit_5 = int(
        number(
            status.get(
                "hit_5pct_count"
            )
        )
    )

    hit_7 = int(
        number(
            status.get(
                "hit_7pct_count"
            )
        )
    )

    hit_9 = int(
        number(
            status.get(
                "hit_9pct_count"
            )
        )
    )

    average_d1 = format_return(
        status.get(
            "average_return_d1"
        )
    )

    average_d3 = format_return(
        status.get(
            "average_return_d3"
        )
    )

    average_d5 = format_return(
        status.get(
            "average_return_d5"
        )
    )

    message = (
        "📊 LARUS V33.4 GERÇEK PERFORMANS TAKİBİ\n\n"
        f"Takip tarihi: {tracking_date or '—'}\n"
        f"Yeni aday: {new_count}\n"
        f"Toplam takip kaydı: {total_count}\n"
        f"Aktif takip: {active_count}\n"
        f"Tamamlanan takip: {completed_count}\n"
        f"Bu çalıştırmadaki fiyat güncellemesi: {price_update_count}\n\n"

        "📈 ORTALAMA GERÇEK PERFORMANS\n"
        f"• D+1: {average_d1}\n"
        f"• D+3: {average_d3}\n"
        f"• D+5: {average_d5}\n\n"

        "🎯 HEDEF İSTATİSTİĞİ\n"
        f"• +%3 gören: {hit_3}\n"
        f"• +%5 gören: {hit_5}\n"
        f"• +%7 gören: {hit_7}\n"
        f"• +%9 gören: {hit_9}\n"
    )

    if history.empty:
        message += (
            "\nHenüz ayrıntılı takip kaydı yok."
        )

        return message

    # --------------------------------------------------------
    # BUGUN KAYDEDILEN ADAYLAR
    # --------------------------------------------------------

    if (
        tracking_date
        and "tracking_date" in history.columns
    ):

        today_rows = history[
            history[
                "tracking_date"
            ].astype(str)
            == tracking_date
        ].copy()

    else:
        today_rows = pd.DataFrame()

    if not today_rows.empty:

        if "v33_3_rank" in today_rows.columns:
            today_rows[
                "v33_3_rank"
            ] = pd.to_numeric(
                today_rows[
                    "v33_3_rank"
                ],
                errors="coerce",
            )

            today_rows = today_rows.sort_values(
                "v33_3_rank"
            )

        message += (
            "\n\n🆕 BUGÜN TAKİBE ALINANLAR\n"
        )

        for _, row in today_rows.head(
            MAX_DETAIL_ROWS
        ).iterrows():

            symbol = text(
                row.get("symbol")
            )

            decision = text(
                row.get(
                    "v33_3_decision"
                )
            )

            score = format_number(
                row.get(
                    "v33_3_score"
                ),
                1,
            )

            reference_price = format_number(
                row.get(
                    "reference_price"
                ),
                2,
            )

            message += (
                f"• {symbol} | "
                f"{decision} | "
                f"Skor {score} | "
                f"Ref {reference_price}\n"
            )

    # --------------------------------------------------------
    # PERFORMANSI BASLAMIS KAYITLAR
    # --------------------------------------------------------

    performance_mask = pd.Series(
        False,
        index=history.index,
    )

    for column in (
        "return_d1",
        "return_d2",
        "return_d3",
        "return_d5",
    ):

        if column in history.columns:

            values = pd.to_numeric(
                history[column],
                errors="coerce",
            )

            performance_mask = (
                performance_mask
                | values.notna()
            )

    performance_rows = history[
        performance_mask
    ].copy()

    if not performance_rows.empty:

        performance_rows = (
            performance_rows
            .sort_values(
                [
                    "tracking_date",
                    "v33_3_rank",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .head(
                MAX_DETAIL_ROWS
            )
        )

        message += (
            "\n\n📌 PERFORMANSI BAŞLAYAN KAYITLAR\n"
        )

        for _, row in performance_rows.iterrows():

            symbol = text(
                row.get("symbol")
            )

            date_value = text(
                row.get(
                    "tracking_date"
                )
            )

            d1 = format_return(
                row.get(
                    "return_d1"
                )
            )

            d2 = format_return(
                row.get(
                    "return_d2"
                )
            )

            d3 = format_return(
                row.get(
                    "return_d3"
                )
            )

            d5 = format_return(
                row.get(
                    "return_d5"
                )
            )

            result_class = text(
                row.get(
                    "result_class"
                )
            )

            message += (
                f"• {symbol} ({date_value})\n"
                f"  D+1 {d1} | "
                f"D+2 {d2} | "
                f"D+3 {d3} | "
                f"D+5 {d5}\n"
                f"  Sonuç: {result_class or 'BEKLİYOR'}\n"
            )

    message += (
        "\n\n🧠 RSI kullanımı: DEVRE DIŞI"
        "\n⚠️ Bu rapor performans ölçümü ve model doğrulaması içindir; yatırım tavsiyesi değildir."
    )

    return message


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    status = load_json(
        STATUS_FILE
    )

    history = load_csv(
        TRACKING_FILE
    )

    if not status:

        message = (
            "📊 LARUS V33.4 GERÇEK PERFORMANS TAKİBİ\n\n"
            "V33.4 durum dosyası bulunamadı veya okunamadı."
        )

        send_telegram(
            message
        )

        return

    message = build_summary_message(
        status,
        history,
    )

    print(
        "===== V33.4 TELEGRAM RAPORU ====="
    )

    print(message)

    send_telegram(
        message
    )


if __name__ == "__main__":
    main()
