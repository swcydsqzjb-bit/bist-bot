from __future__ import annotations
from tg_utf8_common import csv, js, n, send, t

RESULT_FILE = "v20_portfolio_model.csv"
STATUS_FILE = "v20_portfolio_status.json"

def medal(rank: int) -> str:
    return {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, "⭐")

def main() -> None:
    frame = csv(RESULT_FILE)
    status = js(STATUS_FILE)

    if frame.empty:
        send(
            "📊 LARUS V20.3 MODEL PORTFÖYÜ\n\n"
            "Bugün model portföyüne alınacak uygun aday bulunamadı.\n"
            "Model nakit oranı: %100"
        )
        return

    message = (
        "📊 LARUS V20.3 MODEL PORTFÖYÜ\n\n"
        f"Pozisyon sayısı: {len(frame)}\n"
        f"Model yatırım oranı: %{n(status.get('invested_pct')):.1f}\n"
        f"Model nakit oranı: %{n(status.get('cash_reserve_pct')):.1f}\n"
        f"En yüksek ağırlık: {t(status.get('top_symbol'))} "
        f"(%{n(status.get('top_weight_pct')):.1f})\n\n"
    )

    for index, row in frame.iterrows():
        rank = int(n(row.get("portfolio_rank"), index + 1))
        message += (
            f"{medal(rank)} {rank}. {t(row.get('symbol'))}\n"
            f"Model ağırlığı: %{n(row.get('model_weight_pct')):.1f}\n"
            f"Rol: {t(row.get('allocation_label'))}\n"
            f"Durum: {t(row.get('confirmation_state'))}\n"
            f"Top Pick: {n(row.get('top_pick_score')):.1f}/100\n"
            f"AI Final: {n(row.get('ai_final_score')):.1f}/100\n"
            f"Consensus: {n(row.get('consensus_score')):.1f}/100\n"
            f"Risk: {t(row.get('risk_class'))} | {n(row.get('risk_score')):.1f}/100\n"
            f"İzleme ufku: {int(n(row.get('best_horizon_days')))} işlem günü\n"
            f"Beklenen ortalama: {n(row.get('expected_return')):+.2f}%\n"
            f"Temkinli senaryo: {n(row.get('downside_20pct')):+.2f}%\n"
            f"Olumlu senaryo: {n(row.get('upside_80pct')):+.2f}%\n"
            "\n--------------------\n\n"
        )

    message += (
        "⚠️ Bu dağılım gerçek portföy önerisi değil, sistem katmanlarının "
        "karşılaştırmalı model ağırlığıdır. Yatırım tavsiyesi değildir."
    )
    send(message)

if __name__ == "__main__":
    main()
