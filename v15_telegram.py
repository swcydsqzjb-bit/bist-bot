from __future__ import annotations
import json
import os
from typing import Any
import numpy as np
import pandas as pd
import requests

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
FILE = "v15_final_decisions.csv"
STATUS = "v15_status.json"

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

def load_status() -> dict:
    if not os.path.exists(STATUS):
        return {}
    try:
        with open(STATUS, "r", encoding="utf-8") as h:
            return json.load(h)
    except Exception:
        return {}

def level(n: int) -> str:
    if n < 10:
        return "BA\u015eLANGI\u00c7"
    if n < 30:
        return "GEL\u0130\u015e\u0130YOR"
    if n < 100:
        return "OLGUNLA\u015eIYOR"
    return "OLGUN"

def emoji(decision: str) -> str:
    if "G\u00dc\u00c7L\u00dc" in decision:
        return "\U0001f7e2"
    if "ONAYLI" in decision:
        return "\U0001f535"
    if "TEMK\u0130NL\u0130" in decision:
        return "\U0001f7e1"
    if "GER\u0130" in decision:
        return "\U0001f7e0"
    return "\U0001f534"

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

def main() -> None:
    try:
        df = pd.read_csv(FILE, encoding="utf-8-sig")
    except Exception as exc:
        print(FILE + " okunamadi:", exc)
        return
    if df.empty:
        send("\U0001f985 LARUS V15 N\u0130HA\u0130 KARAR RAPORU\n\nBug\u00fcn degerlendirilecek aday bulunamadi.")
        return

    status = load_status()
    mode = t(status.get("model_mode", df.iloc[0].get("model_mode")))
    completed = int(f(status.get("completed_5d"), 0))
    approved = int(df["v15_decision"].isin(["V15 G\u00dc\u00c7L\u00dc ONAY", "V15 ONAYLI \u0130ZLEME"]).sum())

    msg = (
        "\U0001f985 LARUS V15 N\u0130HA\u0130 KARAR RAPORU\n\n"
        f"\u0130ncelenen aday: {len(df)}\n"
        f"Onaylanan: {approved}\n"
        f"Model modu: {mode}\n"
        f"Tamamlanmis 5 g\u00fcnl\u00fck sonu\u00e7: {completed}\n"
        f"\u00d6grenme seviyesi: {level(completed)}\n\n"
    )

    if mode == "FALLBACK":
        msg += (
            "V15 hen\u00fcz 30 tamamlanmis sonuca ulasmadigi i\u00e7in "
            "\u00f6grenilmis agirliklar yerine g\u00fcvenli ge\u00e7is modunu kullaniyor. "
            "Bu asamada V15, V14 kararina ek bir dogrulama katmani olarak \u00e7alisir.\n\n"
        )

    for _, r in df.iterrows():
        d = t(r.get("v15_decision"))
        msg += (
            f"{emoji(d)} {int(f(r.get('rank')))}. {t(r.get('symbol'))}\n"
            f"V15 karari: {d}\n"
            f"Fiyat: {f(r.get('close')):.2f}\n"
            f"V15 skoru: {f(r.get('v15_score')):.1f}/100\n"
            f"V14 skoru: {f(r.get('v14_score')):.1f}/100\n"
            f"\u00d6grenme bileseni: {f(r.get('learned_component_score')):.1f}/100\n"
            f"V8 skoru: {f(r.get('v8_score')):.1f}/100\n"
            f"Smart Money: {f(r.get('smart_money_score')):.1f}/100\n"
            f"Kurumsal: {f(r.get('institutional_score')):.1f}/100\n"
            f"DNA: {t(r.get('dna_classification'))} | {f(r.get('dna_confidence')):.1f}/100\n"
            f"5 g\u00fcnde pozitif: %{f(r.get('positive_rate_5d')):.1f}\n"
            f"Ortalama 5 g\u00fcnl\u00fck sonu\u00e7: {f(r.get('average_result_5d')):+.2f}%\n"
            f"\u00d6nceki V14 karari: {t(r.get('v14_decision'))}\n"
            "\n--------------------\n\n"
        )

    msg += "\u26a0\ufe0f V15 ge\u00e7mis sonu\u00e7lardan istatistiksel agirlik \u00f6grenir. Yatirim tavsiyesi veya getiri garantisi degildir."
    send(msg)

if __name__ == "__main__":
    main()
