from __future__ import annotations

from typing import Any

import pandas as pd

from tg_utf8_common import (
    csv,
    js,
    n,
    send,
    t,
)


RESULT_FILE = "v20_ai_final_decisions.csv"
STATUS_FILE = "v20_ai_status.json"


def medal(
    rank: int,
) -> str:
    return {
        1: "🥇",
        2: "🥈",
        3: "🥉",
    }.get(
        rank,
        "⭐",
    )


def main() -> None:
    frame = csv(
        RESULT_FILE
    )
    status = js(
        STATUS_FILE
    )

    if frame.empty:
        send(
            "🏆 LARUS V20 AI KARAR RAPORU\n\n"
            "Bugün V20'nin değerlendireceği aday bulunamadı."
        )
        return

    message = (
        "🏆 LARUS V20 AI KARAR RAPORU\n\n"
        f"İncelenen aday: {len(frame)}\n"
        f"Güçlü aday: "
        f"{int(n(status.get('approved_count')))}\n"
        f"Bugünün ilk adayı: "
        f"{t(status.get('top_pick'))}\n"
        f"İlk aday skoru: "
        f"{n(status.get('top_pick_score')):.1f}/100\n\n"
    )

    for index, row in frame.head(10).iterrows():
        rank = int(
            n(
                row.get("rank"),
                index + 1,
            )
        )

        message += (
            f"{medal(rank)} {rank}. "
            f"{t(row.get('symbol'))}\n"
            f"V20 kararı: "
            f"{t(row.get('v20_decision'))}\n"
            f"AI Final Score: "
            f"{n(row.get('ai_final_score')):.1f}/100\n"
            f"Consensus: "
            f"{n(row.get('consensus_score')):.1f}/100\n"
            f"Risk: "
            f"{t(row.get('risk_class'))} | "
            f"{n(row.get('risk_score')):.1f}/100\n"
            f"Fiyat: "
            f"{n(row.get('close')):.2f}\n"
            f"Piyasa yüzdeliği: "
            f"%{n(row.get('market_percentile')):.1f}\n"
            f"Piyasa rejimi: "
            f"{t(row.get('regime'))}\n"
            f"Önerilen izleme ufku: "
            f"{int(n(row.get('best_horizon_days')))} işlem günü\n"
            f"Zamanlama güveni: "
            f"{n(row.get('timing_confidence')):.1f}/100\n"
            f"Beklenen ortalama sonuç: "
            f"{n(row.get('expected_return')):+.2f}%\n"
            f"Temkinli senaryo: "
            f"{n(row.get('downside_20pct')):+.2f}%\n"
            f"Olumlu senaryo: "
            f"{n(row.get('upside_80pct')):+.2f}%\n"
        )

        reasons = t(
            row.get("ai_reasons")
        )

        if reasons:
            message += (
                "\nNeden öne çıktı?\n"
                + "\n".join(
                    f"✓ {item.strip()}"
                    for item in reasons.split("|")
                    if item.strip()
                )
                + "\n"
            )

        risk_reasons = t(
            row.get("risk_reasons")
        )

        if risk_reasons:
            message += (
                "\nRisk değerlendirmesi:\n"
                + "\n".join(
                    f"• {item.strip()}"
                    for item in risk_reasons.split("|")
                    if item.strip()
                )
                + "\n"
            )

        message += (
            "\n--------------------\n\n"
        )

    message += (
        "⚠️ V20, önceki analiz katmanlarını birleştiren "
        "istatistiksel karar motorudur. Yatırım tavsiyesi "
        "veya getiri garantisi değildir."
    )

    send(message)


if __name__ == "__main__":
    main()
