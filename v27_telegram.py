from tg_utf8_common import csv, js, n, send, t


def main():
    frame = csv("v27_master_decisions.csv")
    status = js("v27_status.json")

    if frame.empty:
        send(
            "🧠 LARUS V27 ANA KARAR SİSTEMİ\n\n"
            "Bugün ana karar üretilecek uygun aday bulunamadı."
        )
        return

    message = (
        "🧠 LARUS V27 ANA KARAR SİSTEMİ\n\n"
        f"İncelenen aday: {int(n(status.get('candidate_count')))}\n"
        f"Onaylanan: {int(n(status.get('approved_count')))}\n"
        f"Üst düzey teyit: {int(n(status.get('top_level_confirmation_count')))}\n"
        f"İlk aday: {t(status.get('top_symbol'))}\n"
        f"İlk karar: {t(status.get('top_decision'))}\n"
        f"İlk skor: {n(status.get('top_score')):.1f}/100\n\n"
    )

    for i, row in frame.head(6).iterrows():
        message += (
            f"{i + 1}. {t(row.get('symbol'))}\n"
            f"V27 kararı: {t(row.get('v27_decision'))}\n"
            f"Ana skor: {n(row.get('v27_master_score')):.1f}/100\n"
            f"Neden: {t(row.get('v27_reason'))}\n"
            f"Model ağırlığı: %{n(row.get('optimized_weight_pct')):.1f}\n"
            f"V22: {t(row.get('v22_signal_state'))}\n"
            f"V24: {t(row.get('v24_state'))}\n"
            f"Performans kalitesi: {n(row.get('quality_score')):.1f}/100\n"
            f"Consensus: {n(row.get('consensus_score')):.1f}/100\n"
            f"Risk: {t(row.get('risk_class'))} | {n(row.get('risk_score')):.1f}/100\n"
            f"Rejim: {t(row.get('regime'))}\n"
            f"İzleme ufku: {int(n(row.get('best_horizon_days')))} işlem günü\n"
            f"Beklenen ortalama: {n(row.get('expected_return')):+.2f}%\n"
            "--------------------\n"
        )

    message += (
        "\n⚠️ V27 yalnızca istatistiksel karar özeti üretir. "
        "Otomatik emir vermez ve yatırım tavsiyesi değildir."
    )
    send(message)


if __name__ == "__main__":
    main()
