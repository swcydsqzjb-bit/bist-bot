from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


RESULT_FILE = Path("v33_3_confirmed_candidates.csv")
STATUS_FILE = Path("v33_3_status.json")

VERSION = "V33.3"


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


def integer(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


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
            encoding="utf-8",
        )

        if not raw.strip():
            return {}

        result = json.loads(raw)

        if isinstance(result, dict):
            return result

        return {}

    except Exception as exc:
        print(
            f"UYARI: {path} okunamadi: {exc}"
        )
        return {}


def decision_icon(
    decision: str,
) -> str:
    icons = {
        "ÜST DÜZEY TEYİT": "🟢",
        "AKTİF İZLEME": "🔵",
        "TEYİT BEKLE": "🟡",
        "PASİF İZLEME": "⚪",
        "ELE": "🔴",
    }

    return icons.get(
        text(decision),
        "⚪",
    )


def split_message(
    message: str,
    limit: int = 3900,
) -> list[str]:
    if len(message) <= limit:
        return [message]

    parts: list[str] = []
    current = ""

    for block in message.split("\n\n"):
        candidate = (
            block
            if not current
            else current + "\n\n" + block
        )

        if len(candidate) <= limit:
            current = candidate
            continue

        if current:
            parts.append(current)

        if len(block) <= limit:
            current = block
        else:
            for start in range(
                0,
                len(block),
                limit,
            ):
                parts.append(
                    block[
                        start:
                        start + limit
                    ]
                )

            current = ""

    if current:
        parts.append(current)

    return parts


def send_telegram(
    message: str,
) -> None:
    token = text(
        os.getenv("TOKEN")
    )

    chat_id = text(
        os.getenv("CHAT_ID")
    )

    if not token:
        raise RuntimeError(
            "TOKEN bulunamadi."
        )

    if not chat_id:
        raise RuntimeError(
            "CHAT_ID bulunamadi."
        )

    url = (
        "https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    for part in split_message(
        message
    ):
        response = requests.post(
            url,
            json={
                "chat_id": chat_id,
                "text": part,
                "disable_web_page_preview": True,
            },
            timeout=30,
        )

        if not response.ok:
            raise RuntimeError(
                "Telegram mesaji gonderilemedi: "
                f"{response.status_code} "
                f"{response.text}"
            )


def candidate_block(
    row: pd.Series,
    index: int,
) -> str:
    decision = text(
        row.get("v33_3_decision")
    )

    icon = decision_icon(
        decision
    )

    symbol = text(
        row.get("symbol")
    )

    final_score = number(
        row.get("v33_3_score")
    )

    confidence = number(
        row.get("v33_3_confidence")
    )

    v33_score = number(
        row.get("v33_score")
    )

    prescan_score = number(
        row.get("prescan_score")
    )

    positive_5d = number(
        row.get("positive_5d_rate")
    )

    average_5d = number(
        row.get("average_return_5d")
    )

    median_5d = number(
        row.get("median_return_5d")
    )

    best_5d = number(
        row.get("best_return_5d")
    )

    worst_5d = number(
        row.get("worst_return_5d")
    )

    consistency = number(
        row.get("consistency_score")
    )

    technical = number(
        row.get("current_technical_score")
    )

    historical = number(
        row.get("historical_quality_score")
    )

    range_5d = number(
        row.get("return_range_5d")
    )

    downside_penalty = number(
        row.get("downside_penalty")
    )

    risk_score = number(
        row.get("risk_score"),
        50.0,
    )

    risk_class = (
        text(
            row.get("risk_class")
        )
        or "BILINMIYOR"
    )

    regime = (
        text(
            row.get("regime")
        )
        or "BILINMIYOR"
    )

    market_pct = number(
        row.get("market_percentile")
    )

    timing = number(
        row.get("timing_confidence")
    )

    close = number(
        row.get("close")
    )

    return_1d = number(
        row.get("return_1d")
    )

    return_5d = number(
        row.get("return_5d")
    )

    return_20d = number(
        row.get("return_20d")
    )

    ema20 = number(
        row.get("ema20_distance")
    )

    volume = number(
        row.get("volume_ratio")
    )

    reason = (
        text(
            row.get("v33_3_reason")
        )
        or "Karar aciklamasi yok"
    )

    supporting = (
        text(
            row.get("supporting_factors")
        )
        or "-"
    )

    risks = (
        text(
            row.get("risk_notes")
        )
        or "-"
    )

    return (
        f"{icon} {index}. {symbol}\n"
        f"V33.3 karari: {decision}\n"
        f"Final skor: {final_score:.1f}/100\n"
        f"Guven: {confidence:.1f}/100\n"
        f"V33 benzer gun skoru: {v33_score:.1f}/100\n"
        f"Prescan skoru: {prescan_score:.1f}/100\n"
        f"Referans fiyat: {close:.2f}\n\n"

        f"📊 Gecmis kalite\n"
        f"• 5G pozitif oran: %{positive_5d:.1f}\n"
        f"• 5G ortalama: {average_5d:+.2f}%\n"
        f"• 5G medyan: {median_5d:+.2f}%\n"
        f"• En iyi 5G: {best_5d:+.2f}%\n"
        f"• En kotu 5G: {worst_5d:+.2f}%\n"
        f"• Dagilim araligi: {range_5d:.2f} puan\n"
        f"• Tutarlilik: {consistency:.1f}/100\n"
        f"• Tarihsel kalite: {historical:.1f}/100\n\n"

        f"📈 Bugunku teknik yapi\n"
        f"• Teknik skor: {technical:.1f}/100\n"
        f"• 1G: {return_1d:+.2f}%\n"
        f"• 5G: {return_5d:+.2f}%\n"
        f"• 20G: {return_20d:+.2f}%\n"
        f"• EMA20 mesafesi: {ema20:+.2f}%\n"
        f"• Hacim orani: {volume:.2f}x\n"
        f"• Piyasa goreli yuzdelik: {market_pct:.1f}\n"
        f"• Zamanlama guveni: {timing:.1f}/100\n\n"

        f"🛡 Risk\n"
        f"• Risk: {risk_class} | {risk_score:.1f}/100\n"
        f"• Rejim: {regime}\n"
        f"• Tarihsel downside cezasi: {downside_penalty:.1f}\n\n"

        f"✅ Destekleyenler\n"
        f"• {supporting}\n\n"

        f"⚠️ Risk notlari\n"
        f"• {risks}\n\n"

        f"Karar aciklamasi:\n"
        f"• {reason}"
    )


def build_report(
    frame: pd.DataFrame,
    status: dict[str, Any],
) -> str:
    strong = integer(
        status.get(
            "strong_confirmation_count"
        )
    )

    active = integer(
        status.get(
            "active_tracking_count"
        )
    )

    waiting = integer(
        status.get(
            "waiting_count"
        )
    )

    passive = integer(
        status.get(
            "passive_count"
        )
    )

    eliminated = integer(
        status.get(
            "eliminated_count"
        )
    )

    approved = integer(
        status.get(
            "approved_count"
        )
    )

    candidate_count = integer(
        status.get(
            "candidate_count"
        )
    )

    top_symbol = (
        text(
            status.get("top_symbol")
        )
        or "-"
    )

    top_decision = (
        text(
            status.get("top_decision")
        )
        or "-"
    )

    top_score = number(
        status.get("top_score")
    )

    header = (
        "🧠 LARUS V33.3 IKINCI DOGRULAMA RAPORU\n\n"
        f"Incelenen final aday: {candidate_count}\n"
        f"🟢 Ust duzey teyit: {strong}\n"
        f"🔵 Aktif izleme: {active}\n"
        f"🟡 Teyit bekle: {waiting}\n"
        f"⚪ Pasif izleme: {passive}\n"
        f"🔴 Ele: {eliminated}\n"
        f"Toplam onay: {approved}\n\n"
        f"Birinci aday: {top_symbol}\n"
        f"Birinci karar: {top_decision}\n"
        f"Birinci skor: {top_score:.1f}/100\n\n"
        "RSI kullanimi: DEVRE DISI"
    )

    if frame.empty:
        return (
            header
            + "\n\n"
            + "V33.3 icin aday sonucu olusmadi."
        )

    blocks = [
        header
    ]

    max_rows = min(
        len(frame),
        10,
    )

    for index in range(
        max_rows
    ):
        blocks.append(
            candidate_block(
                frame.iloc[index],
                index + 1,
            )
        )

    blocks.append(
        (
            "📌 V33.3 NE YAPIYOR?\n\n"
            "V33.3; tam piyasa on tarama sonucunu, "
            "benzer piyasa gunlerini, bugunku teknik yapinin kalitesini, "
            "tarihsel tutarliligi ve kotu senaryo riskini birlikte degerlendirir.\n\n"
            "Yuksek ortalama getiri tek basina yeterli degildir. "
            "Uclarda olusan cok buyuk kazanc veya kayiplar ayrica cezalandirilir."
        )
    )

    blocks.append(
        (
            "🟣 GOLGE MODU\n\n"
            "Bu katman su anda onceki motorlarin kararlarini otomatik degistirmez. "
            "Gercek sonuclar biriktikce performansi olculecek."
        )
    )

    blocks.append(
        (
            "⚠️ Bu rapor otomatik alim-satim emri degildir. "
            "Yatirim tavsiyesi veya getiri garantisi degildir."
        )
    )

    return (
        "\n\n--------------------\n\n"
        .join(blocks)
    )


def main() -> None:
    frame = load_csv(
        RESULT_FILE
    )

    status = load_json(
        STATUS_FILE
    )

    report = build_report(
        frame=frame,
        status=status,
    )

    send_telegram(
        report
    )

    print(
        "V33.3 Telegram raporu gonderildi."
    )


if __name__ == "__main__":
    main()
