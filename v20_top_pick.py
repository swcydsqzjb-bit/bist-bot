from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

INPUT_FILE=Path("v20_ai_final_decisions.csv")
OUTPUT_FILE=Path("v20_top_picks.csv")
STATUS_FILE=Path("v20_top_pick_status.json")

def sf(v:Any,d:float=0.0)->float:
    try:
        x=float(v)
        return d if not np.isfinite(x) else x
    except Exception:
        return d

def tx(v:Any)->str:
    if v is None:return ""
    try:
        if pd.isna(v):return ""
    except Exception:pass
    return str(v).strip()

def load(path:Path)->pd.DataFrame:
    if not path.exists():return pd.DataFrame()
    try:return pd.read_csv(path,encoding="utf-8-sig")
    except UnicodeDecodeError:return pd.read_csv(path,encoding="utf-8")

def clamp(x:float)->float:return float(np.clip(x,0,100))

def dscore(d:str)->float:
    return {"V20 ÇOK GÜÇLÜ ADAY":100,"V20 GÜÇLÜ ADAY":88,"V20 İZLEME ADAYI":72,"V20 TEMKİNLİ TAKİP":55,"V20 ELE":20}.get(d,40)

def state(r:pd.Series)->str:
    s=sf(r.get("ai_final_score")); c=sf(r.get("consensus_score")); risk=tx(r.get("risk_class")).upper()
    t=sf(r.get("timing_confidence")); e=sf(r.get("expected_return")); down=sf(r.get("downside_20pct"))
    if s>=82 and c>=78 and risk=="DÜŞÜK" and t>=70 and e>0:return "TEYİT GELDİ"
    if s>=68 and c>=65 and risk!="YÜKSEK" and e>0:return "İZLEMEDE TUT"
    if s>=55 and risk!="YÜKSEK" and down>-5:return "TEMKİNLİ İZLE"
    return "ELE"

def conflict(r:pd.Series)->str:
    out=[]; c=sf(r.get("consensus_score")); disp=sf(r.get("consensus_dispersion")); risk=tx(r.get("risk_class")).upper()
    e=sf(r.get("expected_return")); p=sf(r.get("market_percentile"))
    if c<55:out.append("Motorlar arasında belirgin görüş ayrılığı var")
    elif c<70:out.append("Motor uyumu orta seviyede")
    if disp>=18:out.append("Katman puanlarının dağılımı geniş")
    if risk=="YÜKSEK":out.append("Risk motoru yüksek risk üretti")
    elif risk=="ORTA":out.append("Risk seviyesi orta")
    if e<=0:out.append("Zamanlama motorunun beklenen getirisi pozitif değil")
    if p<50:out.append("Göreli güç piyasanın alt yarısında")
    return " | ".join(out) if out else "Belirgin motor çelişkisi yok"

def why(r:pd.Series)->str:
    out=[]; p=sf(r.get("market_percentile")); c=sf(r.get("consensus_score")); t=sf(r.get("timing_confidence"))
    e=sf(r.get("expected_return")); regime=tx(r.get("regime")); risk=tx(r.get("risk_class")).upper()
    if p>=90:out.append("Göreli güçte piyasanın üst %10 diliminde")
    elif p>=75:out.append("Göreli güç piyasa ortalamasının üzerinde")
    if c>=75:out.append("Analiz motorları aynı yönde")
    if t>=80:out.append("Zamanlama güveni yüksek")
    elif t>=65:out.append("Zamanlama motoru yeterli teyit verdi")
    if e>=3:out.append("Tarihsel beklenen sonuç güçlü")
    elif e>0:out.append("Tarihsel beklenen sonuç pozitif")
    if regime in {"RALLİ","TREND"}:out.append(f"Piyasa rejimi destekliyor: {regime}")
    elif regime=="YATAY":out.append("Yatay piyasada göreli gücü korunuyor")
    if risk=="DÜŞÜK":out.append("Risk seviyesi düşük")
    if not out:out.append("Çoklu analiz katmanlarının ortak sıralamasında öne çıktı")
    return " | ".join(out[:6])

def topscore(r:pd.Series)->float:
    ai=sf(r.get("ai_final_score")); c=sf(r.get("consensus_score")); risk=sf(r.get("risk_score"))
    t=sf(r.get("timing_confidence")); p=sf(r.get("market_percentile")); e=sf(r.get("expected_return"))
    exp=clamp((e+3)/10*100)
    score=ai*.42+c*.18+t*.12+p*.10+exp*.08+dscore(tx(r.get("v20_decision")))*.10-risk*.16
    return round(clamp(score),2)

def main():
    f=load(INPUT_FILE)
    if f.empty:
        pd.DataFrame().to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")
        STATUS_FILE.write_text(json.dumps({"status":"input_missing","candidate_count":0,"selected_count":0},ensure_ascii=False,indent=2),encoding="utf-8")
        return
    f["top_pick_score"]=f.apply(topscore,axis=1)
    f["confirmation_state"]=f.apply(state,axis=1)
    f["why_now"]=f.apply(why,axis=1)
    f["conflict_note"]=f.apply(conflict,axis=1)
    pri={"TEYİT GELDİ":4,"İZLEMEDE TUT":3,"TEMKİNLİ İZLE":2,"ELE":1}
    f["_p"]=f["confirmation_state"].map(pri).fillna(0)
    f=f.sort_values(["_p","top_pick_score","ai_final_score","consensus_score"],ascending=False).drop(columns="_p").reset_index(drop=True)
    f.insert(0,"top_pick_rank",range(1,len(f)+1))
    f.to_csv(OUTPUT_FILE,index=False,encoding="utf-8-sig")
    s={"status":"ready","candidate_count":len(f),"selected_count":int((f["confirmation_state"]!="ELE").sum()),
       "confirmed_count":int((f["confirmation_state"]=="TEYİT GELDİ").sum()),
       "top_symbol":tx(f.iloc[0]["symbol"]) if len(f) else "","top_score":sf(f.iloc[0]["top_pick_score"]) if len(f) else 0,"version":"V20.2"}
    STATUS_FILE.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(s,ensure_ascii=False,indent=2))
    print(f.head(10).to_string(index=False))
if __name__=="__main__":main()
