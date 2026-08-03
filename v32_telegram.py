from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


# ============================================================
# DOSYA YOLLARI
# ============================================================

INPUT_FILE = Path("v32_adaptive_decisions.csv")
STATUS_FILE = Path("v32_status.json")

VERSION = "V32.0"


# ============================================================
# TEMEL YARDIMCILAR
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


def integer(
    value: Any,
    default: int = 0,
) -> int:
    try:
        result = float(value)

        if np.isfinite(result):
            return int(result)

        return default

    except (TypeError, ValueError):
        return default


def boolean_value(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = text(value).lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "evet",
        "active",
        "aktif",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "hayır",
        "hayir",
        "inactive",
        "pasif",
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
                f"UYARI: {path} okunamadı: {exc}"
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

        data = json.loads(raw)

        if isinstance(data, dict):
            return data

        return {}

    except Exception as exc:
        print(
            f"UYARI: {path} okunamadı: {exc}"
        )
        return {}


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
            continue

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
            "TOKEN secret bulunamadı."
        )

    if not chat_id:
        raise RuntimeError(
            "CHAT_ID secret bulunamadı."
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
                "Telegram mesajı gönderilemedi: "
                f"{response.status_code} "
                f"{response.text}"
            )


# ============================================================
# RAPOR YARDIMCILARI
# ============================================================

def decision_icon(
    decision: str,
) -> str:
    icons = {
        "ADAPTİF GÜÇLÜ TEYİT": "🟢",
        "ADAPTİF AKTİF İZLEME": "🔵",
        "GÖLGE İZLEME": "🟣",
        "TEYİT BEKLE": "🟡",
        "PASİF İZLEME": "⚪",
        "ELE": "🔴",
    }

    return icons.get(
        text(decision),
        "⚪",
    )


def mode_text(
    mode: str,
) -> str:
    normalized = text(mode).upper()

    if normalized == "ACTIVE":
        return "AKTİF"

    return "GÖLGE"


def yes_no(
    value: Any,
) -> str:
    return (
        "EVET"
        if boolean_value(value)
        else "HAYIR"
    )


def clean_note(
    value: Any,
    fallback: str,
) -> str:
    result = text(value)

    if not result:
        return fallback

    if result.lower() == "nan":
        return fallback

    return result


def build_candidate_block(
    row: pd.Series,
    index: int,
) -> str:
    symbol = text(
        row.get("symbol")
    )

    decision = text(
        row.get("v32_decision")
    )

    icon = decision_icon(
        decision
    )

    score = number(
        row.get("v32_score")
    )

    confidence = number(
        row.get("v32_confidence")
    )

    adjustment = number(
        row.get("v32_ai_adjustment")
    )

    mode = mode_text(
        row.get("v32_mode")
    )

    reason = clean_note(
        row.get("v32_reason"),
        "Karar açıklaması bulunamadı",
    )

    supports = clean_note(
        row.get("v32_supports"),
        "Belirgin destekleyici unsur yok",
    )

    risks = clean_note(
        row.get("v32_risks"),
        "Belirgin ek risk notu yok",
    )

    v31_decision = text(
        row.get("v31_decision")
    ) or "-"

    v31_score = number(
        row.get("v31_score")
    )

    learning_bonus = number(
        row.get("v31_learning_bonus")
    )

    matched_patterns = integer(
        row.get("matched_pattern_count")
    )

    v27_decision = text(
        row.get("v27_decision")
    ) or "-"

    v27_score = number(
        row.get("v27_master_score")
    )

    risk_class = text(
        row.get("risk_class")
    ) or "BİLİNMİYOR"

    risk_score = number(
        row.get("risk_score")
    )

    regime = text(
        row.get("regime")
    ) or "BİLİNMİYOR"

    expected_return = number(
        row.get("expected_return")
    )

    downside = number(
        row.get("downside_20pct")
    )

    upside = number(
        row.get("upside_80pct")
    )

    close = number(
        row.get("close")
    )

    consensus = number(
        row.get("consensus_score")
    )

    timing = number(
        row.get("timing_confidence")
    )

    market_percentile = number(
        row.get("market_percentile")
    )

    v24_state = text(
        row.get("v24_state")
    ) or "-"

    return (
        f"{icon} {index}. {symbol}\n"
        f"V32 kararı: {decision}\n"
        f"V32 skoru: {score:.1f}/100\n"
        f"V32 güveni: {confidence:.1f}/100\n"
        f"AI puan etkisi: {adjustment:+.2f}\n"
        f"Çalışma modu: {mode}\n"
        f"Referans fiyat: {close:.2f}\n\n"
        f"Önceki katmanlar:\n"
        f"• V27: {v27_decision} | {v27_score:.1f}/100\n"
        f"• V31: {v31_decision} | {v31_score:.1f}/100\n"
        f"• V31 öğrenme bonusu: {learning_bonus:+.2f}\n"
        f"• Eşleşen örüntü: {matched_patterns}\n"
        f"• V24 canlı durum: {v24_state}\n\n"
        f"Teknik ve istatistiksel görünüm:\n"
        f"• Risk: {risk_class} | {risk_score:.1f}/100\n"
        f"• Rejim: {regime}\n"
        f"• Consensus: {consensus:.1f}/100\n"
        f"• Zamanlama güveni: {timing:.1f}/100\n"
        f"• Piyasa göreli yüzdelik: {market_percentile:.1f}\n"
        f"• Beklenen ortalama: {expected_return:+.2f}%\n"
        f"• Temkinli senaryo: {downside:+.2f}%\n"
        f"• Olumlu senaryo: {upside:+.2f}%\n\n"
        f"Durumu destekleyenler:\n"
        f"• {supports}\n\n"
        f"Risk ve çelişkiler:\n"
        f"• {risks}\n\n"
        f"Karar açıklaması:\n"
        f"• {reason}"
    )


# ============================================================
# RAPOR OLUŞTURMA
# ============================================================

def build_empty_report(
    status: dict[str, Any],
) -> str:
    return (
        "🤖 LARUS V32 ADAPTİF AI RAPORU\n\n"
        "Değerlendirilebilecek aday bulunamadı.\n\n"
        f"Durum: {text(status.get('status')) or 'veri yok'}\n"
        f"Açıklama: "
        f"{text(status.get('message')) or 'Girdi dosyası boş veya eksik.'}\n"
        f"Çalışma modu: "
        f"{mode_text(status.get('mode'))}\n"
        f"Öğrenme hazır: "
        f"{yes_no(status.get('learning_ready'))}\n"
        "RSI kullanımı: DEVRE DIŞI\n\n"
        "⚠️ V32 istatistiksel bir analiz katmanıdır. "
        "Otomatik emir üretmez ve yatırım tavsiyesi değildir."
    )


def build_report(
    frame: pd.DataFrame,
    status: dict[str, Any],
) -> str:
    if frame.empty:
        return build_empty_report(
            status
        )

    candidate_count = integer(
        status.get(
            "candidate_count",
            len(frame),
        )
    )

    approved_count = integer(
        status.get("approved_count")
    )

    strong_count = integer(
        status.get(
            "strong_confirmation_count"
        )
    )

    active_count = integer(
        status.get(
            "active_tracking_count"
        )
    )

    shadow_count = integer(
        status.get(
            "shadow_tracking_count"
        )
    )

    waiting_count = integer(
        status.get("waiting_count")
    )

    passive_count = integer(
        status.get("passive_count")
    )

    eliminated_count = integer(
        status.get("eliminated_count")
    )

    completed = integer(
        status.get(
            "completed_observation_count"
        )
    )

    minimum_completed = integer(
        status.get(
            "minimum_completed_required"
        )
    )

    usable_patterns = integer(
        status.get(
            "usable_pattern_count"
        )
    )

    minimum_patterns = integer(
        status.get(
            "minimum_pattern_required"
        )
    )

    matched_candidates = integer(
        status.get(
            "matched_candidate_count"
        )
    )

    learning_ready = boolean_value(
        status.get("learning_ready")
    )

    mode = mode_text(
        status.get("mode")
    )

    top_symbol = text(
        status.get("top_symbol")
    ) or "-"

    top_decision = text(
        status.get("top_decision")
    ) or "-"

    top_score = number(
        status.get("top_score")
    )

    top_confidence = number(
        status.get("top_confidence")
    )

    header = (
        "🤖 LARUS V32 ADAPTİF AI RAPORU\n\n"
        f"Çalışma modu: {mode}\n"
        f"Öğrenme hazır: "
        f"{'EVET' if learning_ready else 'HAYIR'}\n"
        f"İncelenen aday: {candidate_count}\n"
        f"Adaptif onay alan: {approved_count}\n"
        f"Güçlü teyit: {strong_count}\n"
        f"Aktif izleme: {active_count}\n"
        f"Gölge izleme: {shadow_count}\n"
        f"Teyit bekleyen: {waiting_count}\n"
        f"Pasif izleme: {passive_count}\n"
        f"Elenen: {eliminated_count}\n\n"
        f"Öğrenme altyapısı:\n"
        f"• Tamamlanan gözlem: "
        f"{completed}/{minimum_completed}\n"
        f"• Kullanılabilir örüntü: "
        f"{usable_patterns}/{minimum_patterns}\n"
        f"• Örüntü eşleşen aday: "
        f"{matched_candidates}\n\n"
        f"İlk aday: {top_symbol}\n"
        f"İlk karar: {top_decision}\n"
        f"İlk skor: {top_score:.1f}/100\n"
        f"İlk güven: {top_confidence:.1f}/100\n"
        "RSI kullanımı: DEVRE DIŞI"
    )

    blocks: list[str] = [
        header
    ]

    max_rows = min(
        len(frame),
        5,
    )

    for index in range(
        max_rows
    ):
        blocks.append(
            build_candidate_block(
                frame.iloc[index],
                index + 1,
            )
        )

    if len(frame) > max_rows:
        blocks.append(
            (
                f"ℹ️ Toplam {len(frame)} adaydan "
                f"ilk {max_rows} aday gösterildi."
            )
        )

    if not learning_ready:
        blocks.append(
            (
                "🟣 GÖLGE MODU AÇIK\n\n"
                "V32 mevcut adayları değerlendiriyor ancak "
                "öğrenilmiş veri yeterli seviyeye ulaşmadığı için "
                "adaptif öğrenme etkisini nihai karar gibi kullanmıyor.\n\n"
                "Tamamlanan gözlem ve güvenilir örüntü sayısı "
                "artınca sistem otomatik olarak AKTİF moda geçecek."
            )
        )

    blocks.append(
        (
            "📌 V32; V27 temel kararını, V31 öğrenilmiş "
            "örüntülerini, canlı teyidi, piyasa rejimini, "
            "risk ve zamanlama verilerini birleştirir.\n"
            "RSI ve RSI tabanlı örüntüler kullanılmaz.\n\n"
            "⚠️ Bu rapor otomatik alım-satım emri değildir. "
            "Yatırım tavsiyesi veya getiri garantisi değildir."
        )
    )

    return "\n\n--------------------\n\n".join(
        blocks
    )


# ============================================================
# ANA FONKSİYON
# ============================================================

def main() -> None:
    frame = load_csv(
        INPUT_FILE
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
        "V32 Telegram raporu gönderildi."
    )


if __name__ == "__main__":
    main()
