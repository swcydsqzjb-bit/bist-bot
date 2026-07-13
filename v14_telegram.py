from __future__ import annotations
import os
from typing import Any
import numpy as np
import pandas as pd
import requests

TOKEN=os.getenv("TOKEN"); CHAT_ID=os.getenv("CHAT_ID")
FILE="v14_adaptive_decisions.csv"

def f(v:Any,d=0.0):
    try:
        x=float(v); return d if np.isnan(x) or np.isinf(x) else x
    except Exception:return d

def send(text):
    if not TOKEN or not CHAT_ID:
        print(text); return
    parts=[]; cur=""
    for p in text.split("\n\n"):
        c=p if not cur else cur+"\n\n"+p
        if len(c)<=3900: cur=c
        else:
            if cur: parts.append(cur)
            cur=p
    if cur: parts.append(cur)
    for p in parts:
        r=requests.post("https://api.telegram.org/bot"+TOKEN+"/sendMessage",
                        data={"chat_id":CHAT_ID,"text":p,"disable_web_page_preview":True},timeout=30)
        print(r.status_code,r.text[:200])

def main():
    try: df=pd.read_csv(FILE,encoding="utf-8-sig")
    except Exception:
        print(FILE+" bulunamadi."); return
    if df.empty:
        send("\U0001f9e0 LARUS V14 ADAPTIF KARAR RAPORU\n\nBugun incelenecek aday yok.")
        return
    msg="\U0001f9e0 LARUS V14 ADAPTIF KARAR RAPORU\n\n"
    msg+=f"Incelenen aday: {len(df)}\n"
    msg+=f"Onaylanan: {int(df['v14_decision'].isin(['GUCLU ONAY','ONAYLI IZLEME']).sum())}\n"
    msg+=f"Agirlik modu: {df.iloc[0].get('weight_mode','BASE')}\n\n"
    for _,r in df.iterrows():
        d=str(r.get("v14_decision",""))
        e="\U0001f7e2" if "GUCLU" in d else "\U0001f535" if "ONAYLI" in d else "\U0001f7e1" if "TEMKINLI" in d else "\U0001f534"
        msg+=(
            f"{e} {int(f(r.get('rank')))}. {r.get('symbol','')}\n"
            f"Karar: {d}\nFiyat: {f(r.get('close')):.2f}\n"
            f"V14 skoru: {f(r.get('v14_score')):.1f}/100\n"
            f"V8: {f(r.get('v8_score')):.1f}/100\n"
            f"DNA: {r.get('dna_classification','')} | {f(r.get('dna_confidence')):.1f}/100\n"
            f"5g pozitif: %{f(r.get('positive_rate_5d')):.1f}\n"
            f"5g en az %3: %{f(r.get('hit_3pct_5d_rate')):.1f}\n"
            f"Ort. 5g sonuc: {f(r.get('average_result_5d')):+.2f}%\n"
            f"Pozitif bonus: +{f(r.get('positive_bonus')):.1f}\n"
            f"Risk kesintisi: -{f(r.get('risk_penalty')):.1f}\n"
        )
        if str(r.get("positive_reasons","")).strip():
            msg+="Onay nedenleri:\n"+str(r.get("positive_reasons")).replace(" | ","\nâ ")+"\n"
        if str(r.get("risk_reasons","")).strip():
            msg+="Riskler:\nâ¢ "+str(r.get("risk_reasons")).replace(" | ","\nâ¢ ")+"\n"
        msg+="\n--------------------\n\n"
    msg+="â ï¸ V14 istatistiksel karar katmanidir; yatirim tavsiyesi veya getiri garantisi degildir."
    send(msg)

if __name__=="__main__":
    main()
