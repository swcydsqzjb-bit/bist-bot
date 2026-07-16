from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

SIGNALS=Path("v22_signal_states.csv")
POSITIONS=Path("v23_positions.csv")
HISTORY=Path("v23_position_history.csv")
STATUS=Path("v23_status.json")

COLS=["symbol","position_state","opened_at","last_updated_at","entry_reference","last_price","highest_price","lowest_price","days_in_position","entry_v22_score","latest_v22_score","entry_signal_state","latest_signal_state","model_weight_pct","best_horizon_days","expected_return","statistical_invalidation_price","first_objective_price","optimistic_objective_price","unrealized_return_pct","max_favorable_excursion_pct","max_adverse_excursion_pct","action","action_reason"]

def sf(v:Any,d:float=0.0)->float:
    try:
        x=float(v); return x if np.isfinite(x) else d
    except Exception:return d

def tx(v:Any)->str:
    if v is None:return ""
    try:
        if pd.isna(v):return ""
    except Exception:pass
    return str(v).strip()

def load(p:Path)->pd.DataFrame:
    if not p.exists():return pd.DataFrame()
    try:return pd.read_csv(p,encoding="utf-8-sig")
    except UnicodeDecodeError:return pd.read_csv(p,encoding="utf-8")

def now()->str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def empty()->pd.DataFrame:
    return pd.DataFrame(columns=COLS)

def normalize(f:pd.DataFrame)->pd.DataFrame:
    if f.empty:return empty()
    for c in COLS:
        if c not in f.columns:f[c]=np.nan
    return f[COLS].copy()

def should_open(r:pd.Series)->bool:
    return tx(r.get("v22_signal_state")) in {"GÃÃLÃ TEYÄ°T","Ä°ZLEMEYE AL"} and sf(r.get("v22_signal_score"))>=68 and sf(r.get("risk_score"))<=45 and sf(r.get("expected_return"))>0

def action(state,price,inv,first,opt,score,old_score):
    if state in {"RÄ°SKLÄ° - ELE","ELE"}:return "ÃIKIÅ Ä°ZLE","V22 sinyal durumu zayÄ±fladÄ±"
    if inv>0 and price<=inv:return "GEÃERSÄ°ZLÄ°K","Fiyat istatistiksel geÃ§ersizlik bÃ¶lgesine ulaÅtÄ±"
    if opt>0 and price>=opt:return "KÃR KORU","Olumlu senaryo bÃ¶lgesine ulaÅÄ±ldÄ±"
    if first>0 and price>=first:return "KISMÄ° KÃR Ä°ZLE","Ä°lk gÃ¶zlem bÃ¶lgesine ulaÅÄ±ldÄ±"
    if state=="GÃÃLÃ TEYÄ°T":return "TAÅI","GÃ¼Ã§lÃ¼ teyit devam ediyor"
    if state=="Ä°ZLEMEYE AL":return ("TAÅI","V22 skoru gÃ¼Ã§leniyor") if score-old_score>=3 else ("KORU","Ä°zleme sinyali korunuyor")
    if state=="TEYÄ°T BEKLE":return ("ZAYIFLAMA","V22 skoru belirgin geriledi") if score-old_score<=-5 else ("TEYÄ°T BEKLE","Yeni teyit henÃ¼z oluÅmadÄ±")
    if state=="PASÄ°F Ä°ZLEME":return "PASÄ°F TAKÄ°P","Sinyal aktif teyit Ã¼retmiyor"
    return "Ä°ZLE","Belirgin yeni durum oluÅmadÄ±"

def create(r,t):
    p=sf(r.get("close"))
    return {"symbol":tx(r.get("symbol")),"position_state":"AÃIK Ä°ZLEME","opened_at":t,"last_updated_at":t,"entry_reference":round(p,4),"last_price":round(p,4),"highest_price":round(p,4),"lowest_price":round(p,4),"days_in_position":0,"entry_v22_score":round(sf(r.get("v22_signal_score")),2),"latest_v22_score":round(sf(r.get("v22_signal_score")),2),"entry_signal_state":tx(r.get("v22_signal_state")),"latest_signal_state":tx(r.get("v22_signal_state")),"model_weight_pct":round(sf(r.get("model_weight_pct")),2),"best_horizon_days":int(sf(r.get("best_horizon_days"),1)),"expected_return":round(sf(r.get("expected_return")),2),"statistical_invalidation_price":round(sf(r.get("statistical_invalidation_price")),4),"first_objective_price":round(sf(r.get("first_objective_price")),4),"optimistic_objective_price":round(sf(r.get("optimistic_objective_price")),4),"unrealized_return_pct":0.0,"max_favorable_excursion_pct":0.0,"max_adverse_excursion_pct":0.0,"action":"YENÄ° Ä°ZLEME","action_reason":"V22 aktif izleme koÅullarÄ±nÄ± geÃ§ti"}

def update(old,sig,t):
    d=old.to_dict(); entry=sf(old.get("entry_reference")); price=sf(old.get("last_price"),entry); state=tx(old.get("latest_signal_state")); score=sf(old.get("latest_v22_score")); old_score=score
    if sig is not None:
        price=sf(sig.get("close"),price); state=tx(sig.get("v22_signal_state")) or state; score=sf(sig.get("v22_signal_score"),score)
        d["model_weight_pct"]=round(sf(sig.get("model_weight_pct"),sf(old.get("model_weight_pct"))),2)
        d["best_horizon_days"]=int(sf(sig.get("best_horizon_days"),sf(old.get("best_horizon_days"),1)))
        d["expected_return"]=round(sf(sig.get("expected_return"),sf(old.get("expected_return"))),2)
    high=max(sf(old.get("highest_price"),entry),price); low=min(sf(old.get("lowest_price"),entry),price)
    un=((price/entry)-1)*100 if entry>0 else 0; mfe=((high/entry)-1)*100 if entry>0 else 0; mae=((low/entry)-1)*100 if entry>0 else 0
    a,reason=action(state,price,sf(old.get("statistical_invalidation_price")),sf(old.get("first_objective_price")),sf(old.get("optimistic_objective_price")),score,old_score)
    ps="KAPANMA ADAYI" if a in {"GEÃERSÄ°ZLÄ°K","ÃIKIÅ Ä°ZLE"} else "HEDEF BÃLGESÄ°" if a in {"KÃR KORU","KISMÄ° KÃR Ä°ZLE"} else "AÃIK Ä°ZLEME"
    d.update({"position_state":ps,"last_updated_at":t,"last_price":round(price,4),"highest_price":round(high,4),"lowest_price":round(low,4),"days_in_position":int(sf(old.get("days_in_position")))+1,"latest_v22_score":round(score,2),"latest_signal_state":state,"unrealized_return_pct":round(un,2),"max_favorable_excursion_pct":round(mfe,2),"max_adverse_excursion_pct":round(mae,2),"action":a,"action_reason":reason})
    return d

def hrow(p,event):
    return {"recorded_at":p["last_updated_at"],"symbol":p["symbol"],"event":event,"position_state":p["position_state"],"action":p["action"],"action_reason":p["action_reason"],"last_price":p["last_price"],"unrealized_return_pct":p["unrealized_return_pct"],"latest_v22_score":p["latest_v22_score"],"latest_signal_state":p["latest_signal_state"],"days_in_position":p["days_in_position"]}

def main():
    sigs=load(SIGNALS); old=normalize(load(POSITIONS)); t=now()
    smap={tx(r.get("symbol")):r for _,r in sigs.iterrows()} if not sigs.empty and "symbol" in sigs.columns else {}
    out=[]; hist=[]; known=set()
    for _,r in old.iterrows():
        s=tx(r.get("symbol"))
        if not s:continue
        known.add(s); p=update(r,smap.get(s),t); out.append(p); hist.append(hrow(p,"GÃNCELLEME"))
    if not sigs.empty:
        for _,r in sigs.iterrows():
            s=tx(r.get("symbol"))
            if s and s not in known and should_open(r):
                p=create(r,t); out.append(p); hist.append(hrow(p,"YENÄ° Ä°ZLEME"))
    pf=pd.DataFrame(out) if out else empty()
    if not pf.empty:
        order={"HEDEF BÃLGESÄ°":4,"AÃIK Ä°ZLEME":3,"KAPANMA ADAYI":2}
        pf["_p"]=pf["position_state"].map(order).fillna(0)
        pf=pf.sort_values(["_p","latest_v22_score"],ascending=False).drop(columns="_p").reset_index(drop=True)
    pf.to_csv(POSITIONS,index=False,encoding="utf-8-sig")
    oldh=load(HISTORY); newh=pd.DataFrame(hist)
    allh=newh if oldh.empty else oldh if newh.empty else pd.concat([oldh,newh],ignore_index=True,sort=False)
    allh.to_csv(HISTORY,index=False,encoding="utf-8-sig")
    st={"status":"ready","position_count":int(len(pf)),"new_position_count":sum(x["event"]=="YENÄ° Ä°ZLEME" for x in hist),"target_zone_count":int((pf["position_state"]=="HEDEF BÃLGESÄ°").sum()) if not pf.empty else 0,"closing_candidate_count":int((pf["position_state"]=="KAPANMA ADAYI").sum()) if not pf.empty else 0,"top_symbol":tx(pf.iloc[0]["symbol"]) if len(pf) else "","top_action":tx(pf.iloc[0]["action"]) if len(pf) else "","version":"V23.0"}
    STATUS.write_text(json.dumps(st,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(st,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
