from tg_utf8_common import csv, js, n, send, t


def main():
    frame = csv("v25_performance_evaluations.csv")
    status = js("v25_status.json")

    if frame.empty:
        send(
            "📈 LARUS V25 PERFORMANS DEĞERLENDİRME\n\n"
            "Değerlendirilecek aktif izleme kaydı bulunamadı."
        )
        return

    message = (
        "📈 LARUS V25 PERFORMANS DEĞERLENDİRME\n\n"
        f"Değerlendirilen: {int(n(status.get('evaluated_count')))}\n"
        f"Pozitif ilerleyen: {int(n(status.get('positive_count')))}\n"
        f"İlk hedefe ulaşan: {int(n(status.get('first_objective_count')))}\n"
        f"Olumlu senaryoya ulaşan: {int(n(status.get('optimistic_objective_count')))}\n"
        f"Geçersizlik gören: {int(n(status.get('invalidation_count')))}\n"
        f"Ortalama kalite: {n(status.get('average_quality_score')):.1f}/100\n\n"
    )

    for i, row in frame.head(6).iterrows():
        message += (
            f"{i + 1}. {t(row.get('symbol'))}\n"
            f"Durum: {t(row.get('evaluation_state'))}\n"
            f"Kalite: {n(row.get('quality_score')):.1f}/100\n"
            f"Güven sınıfı: {t(row.get('reliability_class'))}\n"
            f"Yaklaşık sonuç: {n(row.get('realized_proxy_return_pct')):+.2f}%\n"
            f"En iyi hareket: {n(row.get('max_gain_pct')):+.2f}%\n"
            f"En kötü hareket: {n(row.get('max_drawdown_pct')):+.2f}%\n"
            "--------------------\n"
        )

    message += "\n⚠️ V25 performans ölçüm katmanıdır; yatırım tavsiyesi değildir."
    send(message)


if __name__ == "__main__":
    main()
