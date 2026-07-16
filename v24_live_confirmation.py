from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yfinance as yf

INPUT=Path("v22_signal_states.csv")
OUTPUT=Path("v24_live_confirmations.csv")
STATUS=Path("v24_status.json")

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

def rsi(s:pd.Series,n:int=14)->pd.Series:
    d=s.diff(); g=d.clip(lower=0); l=-d.clip(upper=0)
    ag=g.ewm(alpha=1/n,adjust=False).mean()
    al=l.ewm(alpha=1/n,adjust=False).mean()
    rs=ag/al.replace(0,np.nan)
    return (100-(100/(1+rs))).fillna(50)

def fetch(symbol:str)->pd.DataFrame:
    ticker=symbol if symbol.endswith(".IS") else symbol+".IS"
    f=yf.download(ticker,period="5d",interval="5m",progress=False,auto_adjust=False,threads=False)
    if f.empty:return f
    if isinstance(f.columns,pd.MultiIndex):f.columns=[c[0] for c in f.columns]
    f.columns=[str(c).title() for c in f.columns]
    need=["Open","High","Low","Close","Volume"]
    if not all(c in f.columns for c in need):return pd.DataFrame()
    f=f.dropna(subset=need)
    return f if len(f)>=35 else pd.DataFrame()

def features(f:pd.DataFrame)->dict:
    o=f["Open"].astype(float); h=f["High"].astype(float); l=f["Low"].astype(float)
    c=f["Close"].astype(float); v=f["Volume"].astype(float)
    e20=c.ewm(span=20,adjust=False).mean(); e50=c.ewm(span=50,adjust=False).mean()
    vr=v.iloc[-1]/max(v.rolling(20).mean().iloc[-1],1)
    rh=h.iloc[-21:-1].max()
    rng=max(h.iloc[-1]-l.iloc[-1],1e-9)
    return {
        "live_price":sf(c.iloc[-1]),
        "rsi_5m":sf(rsi(c).iloc[-1],50),
        "volume_ratio_5m":sf(vr),
        "ema20_distance_5m":sf((c.iloc[-1]/e20.iloc[-1]-1)*100),
        "ema20_slope_5m":sf((e20.iloc[-1]/e20.iloc[-4]-1)*100),
        "ema20_above_ema50":float(e20.iloc[-1]>e50.iloc[-1]),
        "breakout_pct":sf((c.iloc[-1]/rh-1)*100 if rh>0 else 0),
        "return_15m":sf((c.iloc[-1]/c.iloc[-4]-1)*100),
        "return_60m":sf((c.iloc[-1]/c.iloc[-13]-1)*100),
        "close_position":sf((c.iloc[-1]-l.iloc[-1])/rng),
        "upper_wick_ratio":sf((h.iloc[-1]-max(c.iloc[-1],o.iloc[-1]))/rng),
    }

def score(x:dict,base:float)->float:
    s=base*.35
    s+=14 if 52<=x["rsi_5m"]<=72 else 7 if 48<=x["rsi_5m"]<52 else -8 if x["rsi_5m"]>78 else 0
    s+=18 if x["volume_ratio_5m"]>=2 else 12 if x["volume_ratio_5m"]>=1.4 else 5 if x["volume_ratio_5m"]>=1 else 0
    s+=10 if x["ema20_distance_5m"]>0 else 0
    s+=6 if x["ema20_slope_5m"]>0 else 0
    s+=5 if x["ema20_above_ema50"] else 0
    s+=10 if x["breakout_pct"]>=0 else 4 if x["breakout_pct"]>-.6 else 0
    s+=4 if x["return_15m"]>0 else 0
    s+=4 if x["return_60m"]>0 else 0
    s+=6 if x["close_position"]>=.65 else 0
    s-=8 if x["upper_wick_ratio"]>.4 else 0
    s-=7 if x["return_15m"]>4 else 0
    return round(float(np.clip(s,0,100)),2)

def state(x:dict,s:float)->str:
    if s>=76 and x["volume_ratio_5m"]>=1.4 and 52<=x["rsi_5m"]<=72 and x["ema20_distance_5m"]>0 and x["ema20_slope_5m"]>0 and x["close_position"]>=.6 and x["upper_wick_ratio"]<.4:
        return "CANLI TEYÄ°T GELDÄ°"
    if s>=66 and x["volume_ratio_5m"]>=1 and 49<=x["rsi_5m"]<=74 and x["ema20_distance_5m"]>-.3 and x["close_position"]>=.5:
        return "ERKEN TEYÄ°T"
    if x["rsi_5m"]>78 or x["upper_wick_ratio"]>=.5 or x["return_15m"]>5:
        return "ÅÄ°ÅKÄ°N / RÄ°SKLÄ°"
    return "TEYÄ°T BEKLE"

def rs(x:dict)->str:
    a=[]
    if x["volume_ratio_5m"]>=1.4:a.append("5 dakikalÄ±k hacim gÃ¼Ã§lÃ¼")
    elif x["volume_ratio_5m"]>=1:a.append("5 dakikalÄ±k hacim normal Ã¼stÃ¼")
    if 52<=x["rsi_5m"]<=72:a.append("RSI saÄlÄ±klÄ± momentumda")
    if x["ema20_distance_5m"]>0:a.append("Fiyat 5 dakikalÄ±k EMA20 Ã¼zerinde")
    if x["ema20_slope_5m"]>0:a.append("EMA20 eÄimi yukarÄ±")
    if x["breakout_pct"]>=0:a.append("KÄ±sa vadeli kÄ±rÄ±lÄ±m oluÅtu")
    if x["close_position"]>=.65:a.append("Son mum kapanÄ±ÅÄ± gÃ¼Ã§lÃ¼")
    return " | ".join(a or ["CanlÄ± teyit koÅullarÄ± henÃ¼z tamamlanmadÄ±"])

def rk(x:dict)->str:
    a=[]
    if x["rsi_5m"]>78:a.append("RSI aÅÄ±rÄ± Ä±sÄ±nmÄ±Å")
    if x["upper_wick_ratio"]>=.4:a.append("Ãst fitil riski yÃ¼ksek")
    if x["return_15m"]>4:a.append("Son 15 dakikalÄ±k hareket ÅiÅkin")
    if x["volume_ratio_5m"]<.8:a.append("CanlÄ± hacim zayÄ±f")
    if x["ema20_distance_5m"]<0:a.append("Fiyat EMA20 altÄ±nda")
    return " | ".join(a or ["Belirgin canlÄ± risk yok"])

def main():
    c=load(INPUT)
    if c.empty:
        pd.DataFrame().to_csv(OUTPUT,index=False,encoding="utf-8-sig")
        STATUS.write_text(json.dumps({"status":"input_missing","candidate_count":0,"confirmed_count":0,"version":"V24.0"},ensure_ascii=False,indent=2),encoding="utf-8"); return
    e=c[c["v22_signal_state"].isin(["TEYÄ°T BEKLE","Ä°ZLEMEYE AL","GÃÃLÃ TEYÄ°T"])].sort_values("v22_signal_score",ascending=False).head(8)
    rows=[]; failed=[]
    for _,r in e.iterrows():
        sym=tx(r.get("symbol")); f=fetch(sym)
        if f.empty:failed.append(sym); continue
        x=features(f); sc=score(x,sf(r.get("v22_signal_score"))); st=state(x,sc)
        rows.append({"symbol":sym,"v24_state":st,"v24_score":sc,"reference_price":sf(r.get("close")),"live_price":x["live_price"],"price_change_from_reference_pct":round((x["live_price"]/max(sf(r.get("close")),1e-9)-1)*100,2),"v22_state":tx(r.get("v22_signal_state")),"v22_score":sf(r.get("v22_signal_score")),"risk_class":tx(r.get("risk_class")),"risk_score":sf(r.get("risk_score")),"rsi_5m":round(x["rsi_5m"],2),"volume_ratio_5m":round(x["volume_ratio_5m"],2),"ema20_distance_5m":round(x["ema20_distance_5m"],2),"ema20_slope_5m":round(x["ema20_slope_5m"],3),"breakout_pct":round(x["breakout_pct"],2),"return_15m":round(x["return_15m"],2),"return_60m":round(x["return_60m"],2),"close_position":round(x["close_position"],2),"upper_wick_ratio":round(x["upper_wick_ratio"],2),"v24_reasons":rs(x),"v24_risks":rk(x),"checked_at_utc":datetime.now(timezone.utc).replace(microsecond=0).isoformat()})
    out=pd.DataFrame(rows)
    if not out.empty:
        p={"CANLI TEYÄ°T GELDÄ°":4,"ERKEN TEYÄ°T":3,"TEYÄ°T BEKLE":2,"ÅÄ°ÅKÄ°N / RÄ°SKLÄ°":1}
        out["_p"]=out["v24_state"].map(p).fillna(0)
        out=out.sort_values(["_p","v24_score"],ascending=False).drop(columns="_p").reset_index(drop=True)
        out.insert(0,"v24_rank",range(1,len(out)+1))
    out.to_csv(OUTPUT,index=False,encoding="utf-8-sig")
    st={"status":"ready","candidate_count":int(len(e)),"analyzed_count":int(len(out)),"confirmed_count":int((out["v24_state"]=="CANLI TEYÄ°T GELDÄ°").sum()) if not out.empty else 0,"early_confirmation_count":int((out["v24_state"]=="ERKEN TEYÄ°T").sum()) if not out.empty else 0,"failed_symbols":failed,"top_symbol":tx(out.iloc[0]["symbol"]) if len(out) else "","top_state":tx(out.iloc[0]["v24_state"]) if len(out) else "","top_score":sf(out.iloc[0]["v24_score"]) if len(out) else 0.0,"version":"V24.0"}
    STATUS.write_text(json.dumps(st,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(st,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
