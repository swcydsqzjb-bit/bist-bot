from __future__ import annotations
import json, os
from typing import Any
import numpy as np
import pandas as pd

V8_FILE="v8_today_candidates.csv"
V13_FILE="v13_market_dna_results.csv"
V12_STATUS="v12_status.json"
V12_WEIGHTS="v12_recommended_weights.csv"
OUT="v14_adaptive_decisions.csv"
STATUS="v14_status.json"

def f(v:Any,d=0.0):
    try:
        x=float(v)
        return d if np.isnan(x) or np.isinf(x) else x
    except Exception:
        return d

def b(v:Any):
    return str(v).strip().lower() in {"true","1","yes","evet","on"}

def load_csv(path):
    if not os.path.exists(path): return pd.DataFrame()
    try: return pd.read_csv(path,encoding="utf-8-sig")
    except UnicodeDecodeError: return pd.read_csv(path,encoding="utf-8")
    except Exception: return pd.DataFrame()

def load_json(path):
    if not os.path.exists(path): return {}
    try:
        with open(path,"r",encoding="utf-8") as h:return json.load(h)
    except Exception:return {}

def col(df,*names):
    for n in names:
        if n in df.columns:return df[n]
    return pd.Series([np.nan]*len(df),index=df.index)

def norm(v,lo,hi):
    return float(np.clip((f(v)-lo)/(hi-lo)*100,0,100))

def main():
    v8=load_csv(V8_FILE)
    dna=load_csv(V13_FILE)
    if v8.empty or "symbol" not in v8.columns:
        pd.DataFrame().to_csv(OUT,index=False,encoding="utf-8-sig")
        json.dump({"status":"v8_missing"},open(STATUS,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
        return

    x=pd.DataFrame()
    x["symbol"]=v8["symbol"].astype(str).str.replace(".IS","",regex=False).str.upper()
    x["close"]=pd.to_numeric(col(v8,"close","price","signal_price"),errors="coerce")
    x["v8_score"]=pd.to_numeric(col(v8,"v8_score"),errors="coerce")
    x["smart_money_score"]=pd.to_numeric(col(v8,"smart_money_score"),errors="coerce")
    x["institutional_score"]=pd.to_numeric(col(v8,"institutional_score"),errors="coerce")
    x["historical_support_score"]=pd.to_numeric(col(v8,"historical_support_score"),errors="coerce")
    x["rsi"]=pd.to_numeric(col(v8,"rsi"),errors="coerce")
    x["ema20_distance"]=pd.to_numeric(col(v8,"ema20_distance","ema20_dist"),errors="coerce")

    if not dna.empty and "symbol" in dna.columns:
        dna=dna.copy()
        dna["symbol"]=dna["symbol"].astype(str).str.replace(".IS","",regex=False).str.upper()
        x=x.merge(dna,on="symbol",how="left")

    mode="V12_ADAPTIVE" if load_json(V12_STATUS).get("status")=="recommendations_ready" else "BASE"
    rows=[]
    for _,r in x.iterrows():
        dna_ready=b(r.get("dna_ready"))
        avg5=f(r.get("average_result_5d"))
        score=(
            norm(r.get("v8_score"),0,100)*0.42+
            norm(r.get("dna_confidence"),0,100)*0.18+
            norm(r.get("positive_rate_5d"),0,100)*0.12+
            norm(r.get("hit_3pct_5d_rate"),0,100)*0.08+
            norm(avg5,-5,8)*0.08+
            norm(r.get("smart_money_score"),0,100)*0.05+
            norm(r.get("institutional_score"),0,100)*0.05+
            norm(r.get("historical_support_score"),0,100)*0.02
        )
        bonus=0; pos=[]
        if f(r.get("positive_rate_5d"))>=70: bonus+=4; pos.append("5 gun pozitif orani guclu")
        elif f(r.get("positive_rate_5d"))>=60: bonus+=2; pos.append("5 gun pozitif orani olumlu")
        if f(r.get("hit_3pct_5d_rate"))>=45: bonus+=3; pos.append("%3 hedef gecmisi guclu")
        if avg5>=2: bonus+=3; pos.append("Ortalama 5g getiri guclu")
        elif avg5>=1: bonus+=1.5; pos.append("Ortalama 5g getiri olumlu")

        penalty=0; risks=[]
        dc=str(r.get("dna_classification","")).upper()
        if not dna_ready: penalty+=12; risks.append("DNA hazir degil")
        if "ZAYIF" in dc: penalty+=15; risks.append("Zayif DNA")
        elif "KARISIK" in dc or "KARIÅIK" in dc: penalty+=6; risks.append("Karisik DNA")
        if avg5<0: penalty+=6; risks.append("Benzer ornek ortalamasi negatif")
        if f(r.get("rsi"),50)>=76: penalty+=8; risks.append("RSI asiri yuksek")
        if f(r.get("ema20_distance"))>=15: penalty+=8; risks.append("EMA20'den fazla uzak")

        final=float(np.clip(score+bonus-penalty,0,100))
        if final>=78 and penalty<=8 and dna_ready: decision="GUCLU ONAY"
        elif final>=68 and penalty<=14 and dna_ready: decision="ONAYLI IZLEME"
        elif final>=58: decision="TEMKINLI IZLEME"
        else: decision="ELE"

        rows.append({
            "symbol":r.get("symbol"),"close":f(r.get("close")),
            "v8_score":f(r.get("v8_score")),"smart_money_score":f(r.get("smart_money_score")),
            "institutional_score":f(r.get("institutional_score")),
            "dna_classification":r.get("dna_classification",""),
            "dna_confidence":f(r.get("dna_confidence")),
            "positive_rate_5d":f(r.get("positive_rate_5d")),
            "hit_3pct_5d_rate":f(r.get("hit_3pct_5d_rate")),
            "average_result_5d":avg5,"weight_mode":mode,
            "raw_score":round(score,2),"positive_bonus":bonus,"risk_penalty":penalty,
            "v14_score":round(final,2),"v14_decision":decision,
            "positive_reasons":" | ".join(pos),"risk_reasons":" | ".join(risks)
        })

    out=pd.DataFrame(rows)
    order={"GUCLU ONAY":4,"ONAYLI IZLEME":3,"TEMKINLI IZLEME":2,"ELE":1}
    out["_p"]=out["v14_decision"].map(order).fillna(0)
    out=out.sort_values(["_p","v14_score"],ascending=False).drop(columns="_p").reset_index(drop=True)
    out.insert(0,"rank",range(1,len(out)+1))
    out.to_csv(OUT,index=False,encoding="utf-8-sig")
    json.dump({"status":"ready","weight_mode":mode,"candidate_count":len(out),
               "selected_count":int(out["v14_decision"].isin(["GUCLU ONAY","ONAYLI IZLEME"]).sum())},
              open(STATUS,"w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(out.to_string(index=False))

if __name__=="__main__":
    main()
