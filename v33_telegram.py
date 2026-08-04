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

SIMILAR_DAYS_FILE = Path("v33_similar_days.csv")
CANDIDATE_FILE = Path("v33_similar_day_candidates.csv")
STATUS_FILE = Path("v33_status.json")

VERSION = "V33.0"


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


def load_json(path: Path) -> dict[str, Any]:
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


def send_telegram(message: str) -> None:
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

def decision_icon(decision: str) -> str:
    icons = {
        "BENZER GÜN GÜÇLÜ TEYİT": "🟢",
        "BENZER GÜN AKTİF İZLEME": "🔵",
        "BENZER GÜN TEYİT BEKLE": "🟡",
        "BENZER GÜN PASİF": "⚪",
        "YETERSİZ BENZER GÜN": "🟣",
        "ELE": "🔴",
    }

    return icons.get(
        text(decision),
        "⚪",
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


def build_similar_day_block(
    row: pd.Series,
) -> str:
    rank = integer(
        row.get("similarity_rank")
    )

    similar_date = text(
        row.get("similar_date")
    )

    score = number(
        row.get("similarity_score")
    )

    distance = number(
        row.get("similarity_distance")
    )

    market_1d = number(
        row.get("market_next_1d_return")
    )

    market_3d = number(
        row.get("market_next_3d_return")
    )

    market_5d = number(
        row.get("market_next_5d_return")
    )

    breadth_1d = number(
        row.get("breadth_1d")
    )

    breadth_5d = number(
        row.get("breadth_5d")
    )

    above_ema20 = number(
        row.get("above_ema20")
    )

    volume_ratio = number(
        row.get("median_volume_ratio")
    )

    return (
        f"{rank}. {similar_date}\n"
        f"Benzerlik skoru: {score:.1f}/100\n"
        f"Uzaklık: {distance:.2f}\n"
        f"Piyasa genişliği 1G: %{breadth_1d:.1f}\n"
        f"Piyasa genişliği 5G: %{breadth_5d:.1f}\n"
        f"EMA20 üzerindeki hisse: %{above_ema20:.1f}\n"
        f"Medyan hacim oranı: {volume_ratio:.2f}x\n"
        f"Sonraki piyasa getirisi:\n"
        f"• 1 gün: {market_1d:+.2f}%\n"
        f"• 3 gün: {market_3d:+.2f}%\n"
        f"• 5 gün: {market_5d:+.2f}%"
    )


def build_candidate_block(
    row: pd.Series,
    index: int,
) -> str:
    symbol = text(
        row.get("symbol")
    )

    decision = text(
        row.get("v33_decision")
    )

    icon = decision_icon(
        decision
    )

    score = number(
        row.get("v33_score")
    )

    similar_count = integer(
        row.get("similar_day_count")
    )

    positive_1d = number(
        row.get("positive_1d_rate")
    )

    positive_3d = number(
        row.get("positive_3d_rate")
    )

    positive_5d = number(
        row.get("positive_5d_rate")
    )

    average_1d = number(
        row.get("average_return_1d")
    )

    average_3d = number(
        row.get("average_return_3d")
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

    v32_decision = text(
        row.get("v32_decision")
    ) or "-"

    v32_score = number(
        row.get("v32_score")
    )

    v32_confidence = number(
        row.get("v32_confidence")
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

    reason = clean_note(
        row.get("v33_reason"),
        "Karar açıklaması bulunamadı",
    )

    return (
        f"{icon} {index}. {symbol}\n"
        f"V33 kararı: {decision}\n"
        f"V33 skoru: {score:.1f}/100\n"
        f"Referans fiyat: {close:.2f}\n"
        f"Benzer gün sayısı: {similar_count}\n\n"
        f"Benzer gün sonrası pozitif oran:\n"
        f"• 1 gün: %{positive_1d:.1f}\n"
        f"• 3 gün: %{positive_3d:.1f}\n"
        f"• 5 gün: %{positive_5d:.1f}\n\n"
        f"Ortalama geçmiş performans:\n"
        f"• 1 gün: {average_1d:+.2f}%\n"
        f"• 3 gün: {average_3d:+.2f}%\n"
        f"• 5 gün: {average_5d:+.2f}%\n"
        f"• 5 günlük medyan: {median_5d:+.2f}%\n"
        f"• En iyi 5 gün: {best_5d:+.2f}%\n"
        f"• En kötü 5 gün: {worst_5d:+.2f}%\n\n"
        f"Önceki katman:\n"
        f"• V32: {v32_decision}\n"
        f"• V32 skoru: {v32_score:.1f}/100\n"
        f"• V32 güveni: {v32_confidence:.1f}/100\n\n"
        f"Risk ve piyasa görünümü:\n"
        f"• Risk: {risk_class} | {risk_score:.1f}/100\n"
        f"• Rejim: {regime}\n"
        f"• Beklenen ortalama: {expected_return:+.2f}%\n"
        f"• Temkinli senaryo: {downside:+.2f}%\n"
        f"• Olumlu senaryo: {upside:+.2f}%\n\n"
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
        "🧭 LARUS V33 BENZER PİYASA GÜNLERİ RAPORU\n\n"
        "Benzer gün analizi için yeterli sonuç oluşmadı.\n\n"
        f"Durum: {text(status.get('status')) or 'veri yok'}\n"
        f"Açıklama: "
        f"{text(status.get('message')) or 'Girdi dosyası boş veya eksik.'}\n"
        f"İndirilen hisse sayısı: "
        f"{integer(status.get('downloaded_symbol_count'))}\n"
        f"Geçmiş gün sayısı: "
        f"{integer(status.get('history_day_count'))}\n"
        "RSI kullanımı: DEVRE DIŞI\n\n"
        "⚠️ Bu rapor otomatik alım-satım emri değildir."
    )


def build_report(
    similar_days: pd.DataFrame,
    candidates: pd.DataFrame,
    status: dict[str, Any],
) -> str:
    if similar_days.empty and candidates.empty:
        return build_empty_report(
            status
        )

    target_date = text(
        status.get("target_date")
    ) or "-"

    symbol_count = integer(
        status.get("symbol_count")
    )

    downloaded_count = integer(
        status.get("downloaded_symbol_count")
    )

    history_count = integer(
        status.get("history_day_count")
    )

    similar_count = integer(
        status.get("similar_day_count")
    )

    candidate_count = integer(
        status.get("candidate_count")
    )

    approved_count = integer(
        status.get("approved_count")
    )

    strong_count = integer(
        status.get("strong_confirmation_count")
    )

    active_count = integer(
        status.get("active_tracking_count")
    )

    waiting_count = integer(
        status.get("waiting_count")
    )

    passive_count = integer(
        status.get("passive_count")
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

    runtime = number(
        status.get("runtime_seconds")
    )

    header = (
        "🧭 LARUS V33 BENZER PİYASA GÜNLERİ RAPORU\n\n"
        f"Analiz tarihi: {target_date}\n"
        f"Tam piyasa sembolü: {symbol_count}\n"
        f"Verisi indirilen sembol: {downloaded_count}\n"
        f"Oluşturulan geçmiş piyasa günü: {history_count}\n"
        f"Bulunan benzer gün: {similar_count}\n"
        f"İncelenen aday: {candidate_count}\n\n"
        f"Güçlü teyit: {strong_count}\n"
        f"Aktif izleme: {active_count}\n"
        f"Teyit bekleyen: {waiting_count}\n"
        f"Pasif: {passive_count}\n"
        f"Toplam onay: {approved_count}\n\n"
        f"İlk aday: {top_symbol}\n"
        f"İlk karar: {top_decision}\n"
        f"İlk skor: {top_score:.1f}/100\n"
        f"Çalışma süresi: {runtime:.1f} saniye\n"
        "Çalışma modu: GÖLGE\n"
        "RSI kullanımı: DEVRE DIŞI"
    )

    blocks: list[str] = [
        header
    ]

    if not similar_days.empty:
        similar_blocks: list[str] = [
            "📅 BUGÜNE EN ÇOK BENZEYEN GEÇMİŞ GÜNLER"
        ]

        max_similar = min(
            len(similar_days),
            5,
        )

        for index in range(
            max_similar
        ):
            similar_blocks.append(
                build_similar_day_block(
                    similar_days.iloc[index]
                )
            )

        blocks.append(
            "\n\n".join(
                similar_blocks
            )
        )

    if not candidates.empty:
        max_candidates = min(
            len(candidates),
            5,
        )

        for index in range(
            max_candidates
        ):
            blocks.append(
                build_candidate_block(
                    candidates.iloc[index],
                    index + 1,
                )
            )

        if len(candidates) > max_candidates:
            blocks.append(
                (
                    f"ℹ️ Toplam {len(candidates)} adaydan "
                    f"ilk {max_candidates} aday gösterildi."
                )
            )

    blocks.append(
        (
            "🟣 V33 şu anda GÖLGE modundadır.\n\n"
            "Benzer gün sonuçları V32 kararlarını henüz değiştirmez. "
            "Önce performansları gözlemlenecek; güvenilirlik oluşursa "
            "ileride kontrollü biçimde puan sistemine bağlanacaktır."
        )
    )

    blocks.append(
        (
            "📌 V33; bugünkü piyasa genişliği, fiyat hareketi, "
            "EMA20 dağılımı, hacim, volatilite ve piyasa dağılımını "
            "geçmiş günlerle karşılaştırır.\n"
            "RSI veya RSI tabanlı örüntü kullanılmaz.\n\n"
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
    similar_days = load_csv(
        SIMILAR_DAYS_FILE
    )

    candidates = load_csv(
        CANDIDATE_FILE
    )

    status = load_json(
        STATUS_FILE
    )

    report = build_report(
        similar_days=similar_days,
        candidates=candidates,
        status=status,
    )

    send_telegram(
        report
    )

    print(
        "V33 Telegram raporu gönderildi."
    )


if __name__ == "__main__":
    main()
