from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

HISTORY_FILES=[Path("v11_signal_history.csv"),Path("signals_history.csv"),Path("v3_signals_history.csv"),Path("v5_backfill_history.csv")]
OUT_W=Path("v21_learned_weights.csv")
OUT_R=Path("v21_learning_report.csv")
OUT_S=Path("v21_status.json")
MIN_N=30
BASE={"v8_score":20.0,"smart_money_score":18.0,"institutional_score":16.0,"historical_support_score":12.0,"prediction_score":10.0,"live_confirmation_score":8.0,"relationship_score":6.0,"rsi":4.0,"volume_ratio":4.0,"ema20_distance":2.0}

def sf(v:Any,d=float("nan")):
    try:
        x=float(v); return x if np.isfinite(x) else d
    except Exception:return d

def load(p:Path):
    try:return pd.read_csv(p,encoding="utf-8-sig")
    except UnicodeDecodeError:return pd.read_csv(p,encoding="utf-8")

def hist():
    frames=[]
    for p in HISTORY_FILES:
        if p.exists():
            try:
                f=load(p)
                if not f.empty:
                    f["_source"]=p.name; frames.append(f)
            except Exception:pass
    if not frames:return pd.DataFrame(),""
    f=pd.concat(frames,ignore_index=True,sort=False).drop_duplicates()
    return f,",".join(sorted(set(f["_source"].astype(str))))

def target_col(f):
    for c in ["result_5d","return_5d_result","realized_return_5d","actual_return_5d"]:
        if c in f.columns:return c
    return None

def norm(s,feature):
    x=pd.to_numeric(s,errors="coerce")
    if feature=="rsi":return ((x-50)/20).clip(-2.5,2.5)
    if feature=="volume_ratio":return ((x-1)/1.5).clip(-2.5,2.5)
    if feature=="ema20_distance":return (x/10).clip(-2.5,2.5)
    if x.dropna().empty:return x
    if x.dropna().between(0,100).mean()>.8:return ((x-50)/25).clip(-2.5,2.5)
    sd=x.std()
    return x*0 if not np.isfinite(sd) or sd==0 else ((x-x.mean())/sd).clip(-2.5,2.5)

def factor(n):
    return 0 if n<30 else .35 if n<50 else .60 if n<100 else .80 if n<200 else 1.0

def main():
    f,sources=hist()
    if f.empty:
        pd.DataFrame().to_csv(OUT_W,index=False,encoding="utf-8-sig")
        pd.DataFrame().to_csv(OUT_R,index=False,encoding="utf-8-sig")
        OUT_S.write_text(json.dumps({"status":"history_missing","completed_5d":0,"minimum_required":MIN_N},ensure_ascii=False,indent=2),encoding="utf-8"); return
    tc=target_col(f)
    if tc is None:
        pd.DataFrame().to_csv(OUT_W,index=False,encoding="utf-8-sig")
        pd.DataFrame().to_csv(OUT_R,index=False,encoding="utf-8-sig")
        OUT_S.write_text(json.dumps({"status":"target_missing","completed_5d":0,"minimum_required":MIN_N},ensure_ascii=False,indent=2),encoding="utf-8"); return
    f[tc]=pd.to_numeric(f[tc],errors="coerce")
    f=f.dropna(subset=[tc]).copy()
    n=len(f); rel=factor(n); rows=[]; rec={}
    for feat,bw in BASE.items():
        if feat not in f.columns:
            rows.append({"feature":feat,"base_weight":bw,"recommended_weight":bw,"change":0.0,"sample_count":0,"correlation_5d":np.nan,"spread_5d":np.nan,"direction":"VERÄ° YOK"}); rec[feat]=bw; continue
        x=norm(f[feat],feat); m=x.notna() & f[tc].notna(); cnt=int(m.sum())
        if cnt<10:
            corr=spread=np.nan; ch=0.0; direction="VERÄ° YETERSÄ°Z"
        else:
            xx=x[m]; yy=f.loc[m,tc]; med=xx.median()
            corr=sf(xx.corr(yy),0.0); spread=sf(yy[xx>=med].mean()-yy[xx<med].mean(),0.0)
            ch=float(np.clip((np.clip(corr*4,-1.5,1.5)+np.clip(spread/2,-1.5,1.5))*rel,-3,3))
            direction="ARTIR" if ch>.35 else "AZALT" if ch<-.35 else "KORU"
        nw=float(np.clip(bw+ch,2,35)); rec[feat]=nw
        rows.append({"feature":feat,"base_weight":round(bw,3),"recommended_weight":round(nw,3),"change":round(nw-bw,3),"sample_count":cnt,"correlation_5d":round(corr,5) if np.isfinite(corr) else np.nan,"spread_5d":round(spread,3) if np.isfinite(spread) else np.nan,"direction":direction})
    total=sum(rec.values())
    pd.DataFrame([{"feature":k,"learned_weight":round(v/total*100,4)} for k,v in rec.items()]).to_csv(OUT_W,index=False,encoding="utf-8-sig")
    rf=pd.DataFrame(rows); rf.to_csv(OUT_R,index=False,encoding="utf-8-sig")
    status={"status":"learning_active" if n>=MIN_N else "waiting_for_data","history_sources":sources,"completed_5d":n,"minimum_required":MIN_N,"reliability_factor":rel,"strongest_feature":rf.sort_values("change",ascending=False).iloc[0]["feature"],"weakest_feature":rf.sort_values("change").iloc[0]["feature"],"version":"V21.0"}
    OUT_S.write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(status,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
