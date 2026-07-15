from __future__ import annotations

from typing import Any

from tg_utf8_common import csv, js, n, send, t


RESULT_FILE = "v22_signal_states.csv"
STATUS_FILE = "v22_status.json"


def state_icon(state: str) -> str:
    return {
        "GÜÇLÜ TEYİT": "🟢",
        "İZLEMEYE AL": "🔵",
        "TEYİT BEKLE": "🟡",
        "PASİF İZLEME": "⚪",
        "ELE": "🔴",
        "RİSKLİ - ELE": "⛔",
    }.get(state, "⚪")


def split_items(value: Any) -> list[str]:
    text = t(value)
    if not text:
        return []

    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


def main() -> None:
    frame = csv(RESULT_FILE)
    status = js(STATUS_FILE)

    if frame.empty:
        send(
            "🚦 LARUS V22 SİNYAL DURUM RAPORU\n\n"
            "Bugün sinyal durumu üretilecek uygun aday bulunamadı."
        )
        return

    message = (
        "🚦 LARUS V22 SİNYAL DURUM RAPORU\n\n"
        f"İncelenen aday: {len(frame)}\n"
        f"İzlemeye alınabilir: {int(n(status.get('actionable_count')))}\n"
        f"Güçlü teyit: {int(n(status.get('strong_confirmation_count')))}\n"
        f"Günün ilk adayı: {t(status.get('top_symbol'))}\n"
        f"İlk durum: {t(status.get('top_state'))}\n"
        f"İlk skor: {n(status.get('top_score')):.1f}/100\n"
        f"V21 öğrenme katkısı: {n(status.get('learning_bonus')):+.1f}\n\n"
    )

    for index, row in frame.head(5).iterrows():
        rank = int(n(row.get("v22_rank"), index + 1))
        state = t(row.get("v22_signal_state"))

        message += (
            f"{state_icon(state)} {rank}. {t(row.get('symbol'))}\n"
            f"V22 durumu: {state}\n"
            f"V22 skoru: {n(row.get('v22_signal_score')):.1f}/100\n"
            f"Referans fiyat: {n(row.get('close')):.2f}\n"
            f"Model ağırlığı: %{n(row.get('model_weight_pct')):.1f}\n"
            f"Top Pick: {n(row.get('top_pick_score')):.1f}/100\n"
            f"AI Final: {n(row.get('ai_final_score')):.1f}/100\n"
            f"Consensus: {n(row.get('consensus_score')):.1f}/100\n"
            f"Risk: {t(row.get('risk_class'))} | "
            f"{n(row.get('risk_score')):.1f}/100\n"
            f"Rejim: {t(row.get('regime'))}\n"
            f"İzleme ufku: {int(n(row.get('best_horizon_days')))} işlem günü\n"
            f"Beklenen ortalama: {n(row.get('expected_return')):+.2f}%\n"
            f"Temkinli senaryo: {n(row.get('downside_20pct')):+.2f}%\n"
            f"Olumlu senaryo: {n(row.get('upside_80pct')):+.2f}%\n"
        )

        reasons = split_items(row.get("v22_reasons"))
        if reasons:
            message += "\nDurumu destekleyenler:\n"
            message += "\n".join(
                f"✓ {reason}"
                for reason in reasons
            )
            message += "\n"

        risks = split_items(row.get("v22_risks"))
        if risks:
            message += "\nRisk ve çelişkiler:\n"
            message += "\n".join(
                f"• {risk}"
                for risk in risks
            )
            message += "\n"

        message += "\n--------------------\n\n"

    message += (
        "📌 V22 durumları alım-satım emri değildir. "
        "Sistem katmanlarının istatistiksel teyit seviyesini gösterir.\n\n"
        "⚠️ Yatırım tavsiyesi veya getiri garantisi değildir."
    )

    send(message)


if __name__ == "__main__":
    main()
