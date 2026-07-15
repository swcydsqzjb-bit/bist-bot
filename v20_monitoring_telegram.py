from __future__ import annotations
from tg_utf8_common import csv, js, n, send, t

RESULT_FILE = "v20_monitoring_plan.csv"
STATUS_FILE = "v20_monitoring_status.json"

def icon(state: str) -> str:
    return {
        "AKTİF TEYİT": "🟢",
        "TEYİT BEKLE": "🔵",
        "SADECE TAKİP": "🟡",
        "PASİF": "🔴",
    }.get(state, "⚪")

def main() -> None:
    frame = csv(RESULT_FILE)
    status = js(STATUS_FILE)

    if frame.empty:
        send(
            "🧭 LARUS V20.4 İZLEME PLANI\n\n"
            "Bugün izleme planı üretilecek uygun aday bulunamadı."
        )
        return

    message = (
        "🧭 LARUS V20.4 İZLEME PLANI\n\n"
        f"Planlanan aday: {len(frame)}\n"
        f"Aktif teyit: {int(n(status.get('active_confirmation_count')))}\n"
        f"Teyit bekleyen: {int(n(status.get('waiting_confirmation_count')))}\n\n"
    )

    for _, row in frame.iterrows():
        state = t(row.get("monitoring_state"))
        message += (
            f"{icon(state)} {int(n(row.get('monitor_rank')))}. {t(row.get('symbol'))}\n"
            f"Durum: {state}\n"
            f"Model ağırlığı: %{n(row.get('model_weight_pct')):.1f}\n"
            f"Referans fiyat: {n(row.get('close')):.2f}\n"
            f"Yeniden değerlendirme: {int(n(row.get('review_horizon_days')))} işlem günü\n"
            f"Kontrol kuralı: {t(row.get('review_rule'))}\n\n"
            "İstatistiksel takip bölgeleri:\n"
            f"• Geçersizlik bölgesi: {n(row.get('statistical_invalidation_price')):.2f} "
            f"({n(row.get('statistical_invalidation_pct')):+.2f}%)\n"
            f"• İlk gözlem bölgesi: {n(row.get('first_objective_price')):.2f} "
            f"({n(row.get('first_objective_pct')):+.2f}%)\n"
            f"• Olumlu senaryo bölgesi: {n(row.get('optimistic_objective_price')):.2f} "
            f"({n(row.get('optimistic_objective_pct')):+.2f}%)\n"
            f"• Risk: {t(row.get('risk_class'))} | {n(row.get('risk_score')):.1f}/100\n"
        )

        conflict = t(row.get("conflict_note"))
        if conflict:
            message += f"• Motor notu: {conflict}\n"

        message += "\n--------------------\n\n"

    message += (
        "⚠️ Fiyat bölgeleri geçmiş örneklerden türetilmiş istatistiksel "
        "izleme seviyeleridir; alım-satım talimatı veya garanti değildir."
    )
    send(message)

if __name__ == "__main__":
    main()
