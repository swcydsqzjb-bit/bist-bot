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

INPUT_FILE = Path("v31_learned_decisions.csv")
STATUS_FILE = Path("v31_status.json")

VERSION = "V31.0"


# ============================================================
# YARDIMCI FONKSİYONLAR
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


def load_csv(path: Path) -> pd.DataFrame:
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
            line_candidate = (
                line
                if not current
                else current + "\n" + line
            )

            if len(line_candidate) <= limit:
                current = line_candidate
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
        f"https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    for part in split_message(message):
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
        "ÖĞRENİLMİŞ GÜÇLÜ TEYİT": "🟢",
        "ÖĞRENİLMİŞ AKTİF İZLEME": "🔵",
        "TEYİT BEKLE": "🟡",
        "PASİF İZLEME": "⚪",
        "ELE": "🔴",
    }

    return icons.get(
        decision,
        "⚪",
    )


def format_percent(
    value: Any,
) -> str:
    return f"{number(value):+.2f}%"


def clean_patterns(
    value: Any,
) -> str:
    raw = text(value)

    if not raw:
        return "Eşleşen güvenilir örüntü yok"

    if raw.lower() == "nan":
        return "Eşleşen güvenilir örüntü yok"

    return raw


def build_candidate_block(
    row: pd.Series,
    index: int,
) -> str:
    symbol = text(
        row.get("symbol")
    )

    decision = text(
        row.get("v31_decision")
    )

    icon = decision_icon(
        decision
    )

    score = number(
        row.get("v31_score")
    )

    base_score = number(
        row.get("v27_master_score")
    )

    bonus = number(
        row.get("v31_learning_bonus")
    )

    matched_count = integer(
        row.get("matched_pattern_count")
    )

    positive_count = integer(
        row.get("positive_pattern_count")
    )

    negative_count = integer(
        row.get("negative_pattern_count")
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

    close = number(
        row.get("close")
    )

    explanation = text(
        row.get("learning_explanation")
    )

    patterns = clean_patterns(
        row.get("matched_patterns")
    )

    v27_decision = text(
        row.get("v27_decision")
    )

    return (
        f"{icon} {index}. {symbol}\n"
        f"V31 kararı: {decision}\n"
        f"V31 skoru: {score:.1f}/100\n"
        f"V27 temel skoru: {base_score:.1f}/100\n"
        f"Öğrenme bonusu: {bonus:+.2f} puan\n"
        f"Referans fiyat: {close:.2f}\n"
        f"Önceki V27 kararı: {v27_decision}\n"
        f"Risk: {risk_class} | {risk_score:.1f}/100\n"
        f"Rejim: {regime}\n"
        f"Beklenen ortalama: {expected_return:+.2f}%\n\n"
        f"Örüntü eşleşmeleri:\n"
        f"• Toplam eşleşme: {matched_count}\n"
        f"• Olumlu örüntü: {positive_count}\n"
        f"• Olumsuz örüntü: {negative_count}\n"
        f"• {patterns}\n\n"
        f"Karar açıklaması:\n"
        f"• {explanation}"
    )


# ============================================================
# RAPOR OLUŞTURMA
# ============================================================

def build_report(
    frame: pd.DataFrame,
    status: dict[str, Any],
) -> str:
    if frame.empty:
        status_name = text(
            status.get("status")
        )

        message = text(
            status.get("message")
        )

        return (
            "🧠 LARUS V31 ÖĞRENİLMİŞ ÖRÜNTÜ RAPORU\n\n"
            "Değerlendirilebilecek aday bulunamadı.\n\n"
            f"Durum: {status_name or 'veri yok'}\n"
            f"Açıklama: {message or 'Girdi dosyası boş veya eksik.'}\n\n"
            "RSI kullanımı: DEVRE DIŞI\n\n"
            "⚠️ Bu rapor istatistiksel analiz özetidir. "
            "Otomatik alım-satım emri veya getiri garantisi değildir."
        )

    candidate_count = integer(
        status.get(
            "candidate_count",
            len(frame),
        )
    )

    usable_pattern_count = integer(
        status.get("usable_pattern_count")
    )

    matched_candidate_count = integer(
        status.get(
            "matched_candidate_count"
        )
    )

    approved_count = integer(
        status.get("approved_count")
    )

    positive_bonus_count = integer(
        status.get(
            "positive_bonus_count"
        )
    )

    negative_bonus_count = integer(
        status.get(
            "negative_bonus_count"
        )
    )

    total_bonus = number(
        status.get(
            "total_learning_bonus"
        )
    )

    top_symbol = text(
        status.get("top_symbol")
    )

    top_decision = text(
        status.get("top_decision")
    )

    top_score = number(
        status.get("top_score")
    )

    header = (
        "🧠 LARUS V31 ÖĞRENİLMİŞ ÖRÜNTÜ RAPORU\n\n"
        f"İncelenen aday: {candidate_count}\n"
        f"Kullanılabilir örüntü: {usable_pattern_count}\n"
        f"Örüntü eşleşen aday: {matched_candidate_count}\n"
        f"Öğrenilmiş onay alan: {approved_count}\n"
        f"Pozitif bonus alan: {positive_bonus_count}\n"
        f"Negatif bonus alan: {negative_bonus_count}\n"
        f"Toplam öğrenme etkisi: {total_bonus:+.2f} puan\n"
        f"İlk aday: {top_symbol or '-'}\n"
        f"İlk karar: {top_decision or '-'}\n"
        f"İlk skor: {top_score:.1f}/100\n"
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
        row = frame.iloc[index]

        blocks.append(
            build_candidate_block(
                row,
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

    blocks.append(
        (
            "📌 V31 yalnızca geçmişte tamamlanmış "
            "gözlemlerden öğrenilen örüntüleri kullanır.\n"
            "RSI ve RSI tabanlı örüntüler puana dahil edilmez.\n\n"
            "⚠️ Bu rapor alım-satım emri değildir. "
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
        frame,
        status,
    )

    send_telegram(
        report
    )

    print(
        "V31 Telegram raporu gönderildi."
    )


if __name__ == "__main__":
    main()
