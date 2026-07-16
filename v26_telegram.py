from tg_utf8_common import csv, js, n, send, t


def main():
    frame = csv("v26_optimized_portfolio.csv")
    status = js("v26_status.json")

    if frame.empty:
        send(
            "⚖️ LARUS V26 PORTFÖY OPTİMİZASYONU\n\n"
            "Bugün portföye ayrılabilecek yeterli güçte aday bulunamadı.\n"
            "Model nakit oranı: %100"
        )
        return

    message = (
        "⚖️ LARUS V26 PORTFÖY OPTİMİZASYONU\n\n"
        f"Pozisyon sayısı: {int(n(status.get('position_count')))}\n"
        f"Model yatırım oranı: %{n(status.get('invested_pct')):.1f}\n"
        f"Model nakit oranı: %{n(status.get('cash_pct')):.1f}\n"
        f"En yüksek ağırlık: {t(status.get('top_symbol'))} "
        f"(%{n(status.get('top_weight_pct')):.1f})\n\n"
    )

    for i, row in frame.iterrows():
        message += (
            f"{i + 1}. {t(row.get('symbol'))}\n"
            f"Rol: {t(row.get('portfolio_role'))}\n"
            f"Model ağırlığı: %{n(row.get('optimized_weight_pct')):.1f}\n"
            f"Optimizasyon skoru: {n(row.get('optimizer_score')):.1f}/100\n"
            f"V22: {t(row.get('v22_signal_state'))}\n"
            f"V24: {t(row.get('v24_state'))}\n"
            f"Risk: {t(row.get('risk_class'))} | {n(row.get('risk_score')):.1f}/100\n"
            f"Beklenen ortalama: {n(row.get('expected_return')):+.2f}%\n"
            f"İzleme ufku: {int(n(row.get('best_horizon_days')))} işlem günü\n"
            "--------------------\n"
        )

    message += "\n⚠️ Model ağırlıkları gerçek portföy talimatı değildir."
    send(message)


if __name__ == "__main__":
    main()
