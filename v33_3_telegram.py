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


def integer(
    value: Any,
    default: int = 0,
) -> int:
    try:
        result = float(value)

        if np.isfinite(result):
            return int(result)

    except (TypeError, ValueError):
        pass

    return default


def boolean(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    normalized = str(value).strip().lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "evet",
        "var",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "hayir",
        "hayır",
        "yok",
        "",
    }:
        return False

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

    except Exception as exc:
        print(
            f"UYARI: {path} okunamadi: {exc}"
        )

    return {}


# ============================================================
# FORMATLAMA
# ============================================================

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


def format_optional_number(
    value: Any,
    decimals: int = 1,
) -> str:
    result = optional_number(
        value
    )

    if not np.isfinite(result):
        return "VERİ YOK"

    return f"{result:.{decimals}f}"


def format_optional_percent(
    value: Any,
    decimals: int = 1,
) -> str:
    result = optional_number(
        value
    )

    if not np.isfinite(result):
        return "VERİ YOK"

    return f"{result:.{decimals}f}%"


def format_risk(
    row: pd.Series,
) -> str:
    available = boolean(
        row.get("risk_available")
    )

    risk_score = optional_number(
        row.get("risk_score")
    )

    risk_class = (
        text(
            row.get("risk_class")
        )
        or "VERİ YOK"
    )

    source = (
        text(
            row.get("risk_source")
        )
        or "VERI_YOK"
    )

    if (
        not available
        or not np.isfinite(risk_score)
    ):
        return (
            "VERİ YOK "
            f"| Kaynak: {source}"
        )

    return (
        f"{risk_class} | "
        f"{risk_score:.1f}/100 "
        f"| Kaynak: {source}"
    )


def format_timing(
    row: pd.Series,
) -> str:
    available = boolean(
        row.get("timing_available")
    )

    timing = optional_number(
        row.get("timing_confidence")
    )

    source = (
        text(
            row.get("timing_source")
        )
        or "VERI_YOK"
    )

    if (
        not available
        or not np.isfinite(timing)
    ):
        return (
            "VERİ YOK "
            f"| Kaynak: {source}"
        )

    return (
        f"{timing:.1f}/100 "
        f"| Kaynak: {source}"
    )


def format_regime(
    row: pd.Series,
) -> str:
    regime = (
        text(
            row.get("regime")
        )
        or "BİLİNMİYOR"
    )

    confidence = optional_number(
        row.get("regime_confidence")
    )

    if np.isfinite(confidence):
        return (
            f"{regime} "
            f"| Güven: {confidence:.1f}/100"
        )

    return regime


# ============================================================
# TELEGRAM
# ============================================================

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
            lines = block.splitlines()
            current = ""

            for line in lines:
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
            "TOKEN secret bulunamadi."
        )

    if not chat_id:
        raise RuntimeError(
            "CHAT_ID secret bulunamadi."
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


# ============================================================
# ADAY BLOĞU
# ============================================================

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
        row.get("volume_ratio"),
        1.0,
    )

    market_pct = number(
        row.get("market_percentile")
    )

    risk_text = format_risk(
        row
    )

    timing_text = format_timing(
        row
    )

    regime_text = format_regime(
        row
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
        f"V33.3 kararı: {decision}\n"
        f"Final skor: {final_score:.1f}/100\n"
        f"Güven: {confidence:.1f}/100\n"
        f"V33 benzer gün skoru: {v33_score:.1f}/100\n"
        f"Prescan skoru: {prescan_score:.1f}/100\n"
        f"Referans fiyat: {close:.2f}\n\n"

        f"📊 GEÇMİŞ KALİTE\n"
        f"• 5G pozitif oran: %{positive_5d:.1f}\n"
        f"• 5G ortalama: {average_5d:+.2f}%\n"
        f"• 5G medyan: {median_5d:+.2f}%\n"
        f"• En iyi 5G: {best_5d:+.2f}%\n"
        f"• En kötü 5G: {worst_5d:+.2f}%\n"
        f"• Dağılım aralığı: {range_5d:.2f} puan\n"
        f"• Tutarlılık: {consistency:.1f}/100\n"
        f"• Tarihsel kalite: {historical:.1f}/100\n\n"

        f"📈 BUGÜNKÜ TEKNİK YAPI\n"
        f"• Teknik skor: {technical:.1f}/100\n"
        f"• 1G: {return_1d:+.2f}%\n"
        f"• 5G: {return_5d:+.2f}%\n"
        f"• 20G: {return_20d:+.2f}%\n"
        f"• EMA20 mesafesi: {ema20:+.2f}%\n"
        f"• Hacim oranı: {volume:.2f}x\n"
        f"• Piyasa göreli yüzdelik: {market_pct:.1f}\n"
        f"• Zamanlama: {timing_text}\n\n"

        f"🛡 RİSK VE REJİM\n"
        f"• Risk: {risk_text}\n"
        f"• Rejim: {regime_text}\n"
        f"• Tarihsel downside cezası: {downside_penalty:.1f}\n\n"

        f"✅ DESTEKLEYENLER\n"
        f"• {supporting}\n\n"

        f"⚠️ RİSK NOTLARI\n"
        f"• {risks}\n\n"

        f"Karar açıklaması:\n"
        f"• {reason}"
    )


# ============================================================
# RAPOR
# ============================================================

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

    risk_available = integer(
        status.get(
            "risk_available_count"
        )
    )

    risk_missing = integer(
        status.get(
            "risk_missing_count"
        )
    )

    timing_available = integer(
        status.get(
            "timing_available_count"
        )
    )

    timing_missing = integer(
        status.get(
            "timing_missing_count"
        )
    )

    regime = (
        text(
            status.get("regime")
        )
        or "BİLİNMİYOR"
    )

    regime_confidence = optional_number(
        status.get(
            "regime_confidence"
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

    if np.isfinite(
        regime_confidence
    ):
        regime_line = (
            f"{regime} | "
            f"Güven: {regime_confidence:.1f}/100"
        )
    else:
        regime_line = regime

    header = (
        "🧠 LARUS V33.3 İKİNCİ DOĞRULAMA RAPORU\n\n"
        f"İncelenen final aday: {candidate_count}\n"
        f"🟢 Üst düzey teyit: {strong}\n"
        f"🔵 Aktif izleme: {active}\n"
        f"🟡 Teyit bekle: {waiting}\n"
        f"⚪ Pasif izleme: {passive}\n"
        f"🔴 Ele: {eliminated}\n"
        f"Toplam onay: {approved}\n\n"

        f"Veri doğrulama:\n"
        f"• Gerçek risk verisi olan: {risk_available}\n"
        f"• Risk verisi olmayan: {risk_missing}\n"
        f"• Zamanlama verisi olan: {timing_available}\n"
        f"• Zamanlama verisi olmayan: {timing_missing}\n"
        f"• Piyasa rejimi: {regime_line}\n\n"

        f"Birinci aday: {top_symbol}\n"
        f"Birinci karar: {top_decision}\n"
        f"Birinci skor: {top_score:.1f}/100\n\n"

        "RSI kullanımı: DEVRE DIŞI"
    )

    if frame.empty:
        return (
            header
            + "\n\n"
            + "V33.3 için aday sonucu oluşmadı."
        )

    blocks: list[str] = [
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
            "📌 V33.3 VERİ GÜVENLİĞİ\n\n"
            "Risk verisi olmayan bir hisseye artık otomatik 50 risk puanı verilmez.\n"
            "Zamanlama verisi olmayan bir hisseye otomatik 0 zamanlama puanı verilmez.\n"
            "Eksik alanlar açıkça VERİ YOK olarak gösterilir.\n\n"
            "Gerçek risk verisi olmayan bir aday AKTİF İZLEME olabilir, "
            "ancak ÜST DÜZEY TEYİT alamaz."
        )
    )

    blocks.append(
        (
            "🟣 GÖLGE MODU\n\n"
            "V33.3 hâlâ doğrulama katmanıdır. "
            "Önceki motorların kararlarını otomatik olarak değiştirmez."
        )
    )

    blocks.append(
        (
            "⚠️ Bu rapor otomatik alım-satım emri değildir. "
            "Yatırım tavsiyesi veya getiri garantisi değildir."
        )
    )

    return (
        "\n\n--------------------\n\n"
        .join(blocks)
    )


# ============================================================
# MAIN
# ============================================================

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
