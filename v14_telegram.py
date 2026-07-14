from __future__ import annotations
import os
from typing import Any
import numpy as np
import pandas as pd
import requests

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
FILE = "v14_adaptive_decisions.csv"

def f(v: Any, d: float = 0.0) -> float:
    try:
        x = float(v)
        return d if np.isnan(x) or np.isinf(x) else x
    except Exception:
        return d

def t(v: Any) -> str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return str(v).strip()

def send(text: str) -> None:
    if not TOKEN or not CHAT_ID:
        print(text)
        return
    parts, current = [], ""
    for p in text.split("\n\n"):
        c = p if not current else current + "\n\n" + p
        if len(c) <= 3900:
            current = c
        else:
            if current:
                parts.append(current)
            current = p
    if current:
        parts.append(current)
    for part in parts:
        r = requests.post(
            "https://api.telegram.org/bot" + TOKEN + "/sendMessage",
            data={"chat_id": CHAT_ID, "text": part, "disable_web_page_preview": True},
            timeout=30,
        )
        print(r.status_code, r.text[:250])

def reasons(value: Any, marker: str) -> str:
    items = [x.strip() for x in t(value).split("|") if x.strip()]
    return "\n".join(marker + " " + x for x in items)

def emoji(decision: str) -> str:
    if decision == "G\u00dc\u00c7L\u00dc ONAY":
        return "\U0001f7e2"
    if decision == "ONAYLI \u0130ZLEME":
        return "\U0001f535"
    if decision == "TEMK\u0130NL\u0130 \u0130ZLEME":
        return "\U0001f7e1"
    return "\U0001f534"

def main() -> None:
    try:
        df = pd.read_csv(FILE, encoding="utf-8-sig")
    except Exception as exc:
        print(FILE + " okunamadi:", exc)
        return
    if df.empty:
        send("\U0001f9e0 LARUS V14 ADAPT\u0130F KARAR RAPORU\n\nBug\u00fcn incelenecek aday bulunamadi.")
        return

    approved = int(df["v14_decision"].isin(["G\u00dc\u00c7L\u00dc ONAY", "ONAYLI \u0130ZLEME"]).sum())
    msg = (
        "\U0001f9e0 LARUS V14 ADAPT\u0130F KARAR RAPORU\n\n"
        f"\u0130ncelenen aday: {len(df)}\n"
        f"Onaylanan: {approved}\n"
        f"Agirlik modu: {t(df.iloc[0].get('weight_mode'))}\n\n"
    )

    for _, r in df.iterrows():
        d = t(r.get("v14_decision"))
        msg += (
            f"{emoji(d)} {int(f(r.get('rank')))}. {t(r.get('symbol'))}\n"
            f"Karar: {d}\n"
            f"Fiyat: {f(r.get('close')):.2f}\n"
            f"V14 skoru: {f(r.get('v14_score')):.1f}/100\n"
            f"V8 skoru: {f(r.get('v8_score')):.1f}/100\n"
            f"Smart Money: {f(r.get('smart_money_score')):.1f}/100\n"
            f"Kurumsal: {f(r.get('institutional_score')):.1f}/100\n"
            f"DNA: {t(r.get('dna_classification'))} | {f(r.get('dna_confidence')):.1f}/100\n"
            f"5 g\u00fcnde pozitif: %{f(r.get('positive_rate_5d')):.1f}\n"
            f"5 g\u00fcnde en az %3: %{f(r.get('hit_3pct_5d_rate')):.1f}\n"
            f"Ortalama 5 g\u00fcnl\u00fck sonu\u00e7: {f(r.get('average_result_5d')):+.2f}%\n"
            f"Pozitif bonus: +{f(r.get('positive_bonus')):.1f}\n"
            f"Risk kesintisi: -{f(r.get('risk_penalty')):.1f}\n"
        )
        p = reasons(r.get("positive_reasons"), "\u2713")
        q = reasons(r.get("risk_reasons"), "\u2022")
        if p:
            msg += "\nOnay nedenleri:\n" + p + "\n"
        if q:
            msg += "\nRiskler:\n" + q + "\n"
        msg += "\n--------------------\n\n"

    msg += "\u26a0\ufe0f V14 istatistiksel bir karar katmanidir. Yatirim tavsiyesi veya getiri garantisi degildir."
    send(msg)

if __name__ == "__main__":
    main()
