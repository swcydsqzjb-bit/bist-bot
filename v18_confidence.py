from __future__ import annotations
import json, os
from typing import Any
import numpy as np
import pandas as pd

INPUT_FILE='v17_regime_adjusted_decisions.csv'
V15_STATUS='v15_status.json'
OUT='v18_confidence_decisions.csv'
STATUS='v18_confidence_status.json'

def f(v:Any,d:float=0.0)->float:
    try:
        x=float(v)
        return d if np.isnan(x) or np.isinf(x) else x
    except Exception:return d

def t(v:Any)->str:
    if v is None:return ''
    try:
        if pd.isna(v):return ''
    except Exception:pass
    return str(v).strip()

def load_csv(p:str)->pd.DataFrame:
    if not os.path.exists(p):return pd.DataFrame()
    try:return pd.read_csv(p,encoding='utf-8-sig')
    except Exception:return pd.DataFrame()

def load_json(p:str)->dict:
    try:
        with open(p,'r',encoding='utf-8') as h:return json.load(h)
    except Exception:return {}

def clamp(x:float)->float:return float(np.clip(x,0,100))

def conf_class(x:float)->str:
    if x>=82:return 'ÇOK YÜKSEK GÜVEN'
    if x>=70:return 'YÜKSEK GÜVEN'
    if x>=58:return 'ORTA GÜVEN'
    if x>=45:return 'DÜŞÜK GÜVEN'
    return 'ÇOK DÜŞÜK GÜVEN'

def decision(v17:str,c:float)->str:
    if 'GÜÇLÜ ONAY' in v17 and c>=72:return 'V18 GÜÇLÜ ONAY'
    if 'ONAYLI İZLEME' in v17 and c>=62:return 'V18 ONAYLI İZLEME'
    if 'TEMKİNLİ' in v17 and c>=52:return 'V18 TEMKİNLİ İZLEME'
    if 'ELE' not in v17 and c>=45:return 'V18 ERKEN İZLEME'
    return 'V18 ELE'

def score_row(r:pd.Series)->tuple[float,list[str],list[str]]:
    pos=[]; risk=[]; s=50.0
    vals={k:f(r.get(k)) for k in ['v15_score','v17_score','relative_strength_score','market_percentile','momentum_percentile','trend_percentile','volume_percentile','quality_percentile','regime_confidence','regime_adjustment']}
    if vals['v17_score']>=78:s+=10;pos.append('V17 skoru güçlü')
    elif vals['v17_score']<55:s-=10;risk.append('V17 skoru zayıf')
    if vals['relative_strength_score']>=72:s+=8;pos.append('Göreli güç yüksek')
    elif vals['relative_strength_score']<50:s-=8;risk.append('Göreli güç zayıf')
    if vals['market_percentile']>=85:s+=8;pos.append('Karşılaştırma grubunun üst diliminde')
    elif vals['market_percentile']<45:s-=8;risk.append('Karşılaştırma grubunun alt diliminde')
    dims=[vals['momentum_percentile'],vals['trend_percentile'],vals['volume_percentile'],vals['quality_percentile']]
    strong=sum(x>=70 for x in dims); weak=sum(x<45 for x in dims)
    if strong>=3:s+=10;pos.append('Ana bileşenlerin çoğu güçlü')
    elif strong==2:s+=5;pos.append('İki ana bileşen güçlü')
    if weak>=2:s-=10;risk.append('Birden fazla bileşen zayıf')
    disp=float(np.std(dims))
    if disp<=12:s+=7;pos.append('Bileşenler birbiriyle uyumlu')
    elif disp>=25:s-=7;risk.append('Bileşenler arasında uyumsuzluk var')
    diff=abs(vals['v17_score']-vals['v15_score'])
    if diff<=6:s+=5;pos.append('V15 ve V17 uyumlu')
    elif diff>=18:s-=6;risk.append('V15 ve V17 arasında büyük fark var')
    if vals['regime_confidence']>=75:s+=5;pos.append('Piyasa rejimi güveni yüksek')
    elif vals['regime_confidence']<55:s-=4;risk.append('Piyasa rejimi güveni düşük')
    if vals['regime_adjustment']>=5:s+=3;pos.append('Piyasa rejimi adayı destekliyor')
    elif vals['regime_adjustment']<=-5:s-=5;risk.append('Piyasa rejimi adayı baskılıyor')
    return clamp(s),pos,risk

def main():
    df=load_csv(INPUT_FILE); st=load_json(V15_STATUS); completed=int(f(st.get('completed_5d')))
    if df.empty:
        pd.DataFrame().to_csv(OUT,index=False,encoding='utf-8-sig')
        json.dump({'status':'v17_missing','candidate_count':0,'completed_5d':completed},open(STATUS,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
        return
    rows=[]
    for _,r in df.iterrows():
        c,pos,risk=score_row(r)
        if completed<30:
            c=clamp(c-(12-min(12,completed/30*12)))
            risk.append('Öğrenme hafızası henüz 30 sonuca ulaşmadı')
        else:pos.append('Öğrenme hafızası yeterli')
        v17=t(r.get('v17_decision'))
        rows.append({
            'symbol':t(r.get('symbol')),'close':round(f(r.get('close')),4),'regime':t(r.get('regime')),
            'regime_confidence':round(f(r.get('regime_confidence')),2),'v15_score':round(f(r.get('v15_score')),2),
            'v17_score':round(f(r.get('v17_score')),2),'v17_decision':v17,
            'relative_strength_score':round(f(r.get('relative_strength_score')),2),'market_percentile':round(f(r.get('market_percentile')),2),
            'confidence_score':round(c,2),'confidence_class':conf_class(c),'v18_decision':decision(v17,c),
            'confidence_reasons':' | '.join(pos),'confidence_risks':' | '.join(risk),'completed_5d':completed})
    out=pd.DataFrame(rows)
    pr={'V18 GÜÇLÜ ONAY':5,'V18 ONAYLI İZLEME':4,'V18 TEMKİNLİ İZLEME':3,'V18 ERKEN İZLEME':2,'V18 ELE':1}
    out['_p']=out.v18_decision.map(pr).fillna(0)
    out=out.sort_values(['_p','confidence_score','v17_score'],ascending=False).drop(columns='_p').reset_index(drop=True)
    out.insert(0,'rank',range(1,len(out)+1));out.to_csv(OUT,index=False,encoding='utf-8-sig')
    approved=int(out.v18_decision.isin(['V18 GÜÇLÜ ONAY','V18 ONAYLI İZLEME']).sum())
    with open(STATUS,'w',encoding='utf-8') as h:json.dump({'status':'ready','candidate_count':len(out),'approved_count':approved,'completed_5d':completed,'history_ready':completed>=30},h,ensure_ascii=False,indent=2)
    print(out.to_string(index=False))
if __name__=='__main__':main()
