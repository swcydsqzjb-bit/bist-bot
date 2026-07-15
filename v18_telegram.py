from __future__ import annotations
import json, os
from typing import Any
import numpy as np
import pandas as pd
import requests
TOKEN=os.getenv('TOKEN');CHAT_ID=os.getenv('CHAT_ID')
FILE='v18_confidence_decisions.csv';STATUS='v18_confidence_status.json'
def f(v:Any,d:float=0.0)->float:
    try:
        x=float(v);return d if np.isnan(x) or np.isinf(x) else x
    except Exception:return d
def t(v:Any)->str:
    if v is None:return ''
    try:
        if pd.isna(v):return ''
    except Exception:pass
    return str(v).strip()
def load_status():
    try:return json.load(open(STATUS,'r',encoding='utf-8'))
    except Exception:return {}
def send(text:str):
    if not TOKEN or not CHAT_ID:print(text);return
    parts=[];cur=''
    for p in text.split('\n\n'):
        cand=p if not cur else cur+'\n\n'+p
        if len(cand)<=3900:cur=cand
        else:
            if cur:parts.append(cur)
            cur=p
    if cur:parts.append(cur)
    for part in parts:
        r=requests.post(f'https://api.telegram.org/bot{TOKEN}/sendMessage',data={'chat_id':CHAT_ID,'text':part,'disable_web_page_preview':True},timeout=30);print(r.status_code,r.text[:250])
def em(d:str)->str:
    if 'GÃÃLÃ' in d:return 'ð¢'
    if 'ONAYLI' in d:return 'ðµ'
    if 'TEMKÄ°NLÄ°' in d:return 'ð¡'
    if 'ERKEN' in d:return 'ð '
    return 'ð´'
def fmt(v:Any,m:str)->str:return '\n'.join(f'{m} {x.strip()}' for x in t(v).split('|') if x.strip())
def main():
    try:df=pd.read_csv(FILE,encoding='utf-8-sig')
    except Exception as e:print(e);return
    st=load_status()
    if df.empty:send('ð¡ï¸ LARUS V18 GÃVEN RAPORU\n\nBugÃ¼n gÃ¼ven analizi yapÄ±lacak aday bulunamadÄ±.');return
    msg=(f'ð¡ï¸ LARUS V18 GÃVEN RAPORU\n\nÄ°ncelenen aday: {len(df)}\nOnaylanan: {int(f(st.get("approved_count")))}\nTamamlanmÄ±Å 5 gÃ¼nlÃ¼k sonuÃ§: {int(f(st.get("completed_5d")))}\nÃÄrenme hafÄ±zasÄ± hazÄ±r: {"EVET" if st.get("history_ready") else "HAYIR"}\n\n')
    for _,r in df.iterrows():
        d=t(r.get('v18_decision'))
        msg+=(f'{em(d)} {int(f(r.get("rank")))}. {t(r.get("symbol"))}\nV18 kararÄ±: {d}\nGÃ¼ven sÄ±nÄ±fÄ±: {t(r.get("confidence_class"))}\nGÃ¼ven puanÄ±: {f(r.get("confidence_score")):.1f}/100\nFiyat: {f(r.get("close")):.2f}\nV17 skoru: {f(r.get("v17_score")):.1f}/100\nV15 skoru: {f(r.get("v15_score")):.1f}/100\nGÃ¶reli gÃ¼Ã§: {f(r.get("relative_strength_score")):.1f}/100\nKarÅÄ±laÅtÄ±rma yÃ¼zdeliÄi: %{f(r.get("market_percentile")):.1f}\nRejim: {t(r.get("regime"))} | {f(r.get("regime_confidence")):.1f}/100\n')
        p=fmt(r.get('confidence_reasons'),'â');q=fmt(r.get('confidence_risks'),'â¢')
        if p:msg+='\nGÃ¼veni artÄ±ranlar:\n'+p+'\n'
        if q:msg+='\nGÃ¼veni dÃ¼ÅÃ¼renler:\n'+q+'\n'
        msg+='\n--------------------\n\n'
    msg+='â ï¸ GÃ¼ven puanÄ± istatistiksel uyum Ã¶lÃ§Ã¼sÃ¼dÃ¼r; gerÃ§ek yÃ¼kselme olasÄ±lÄ±ÄÄ± veya getiri garantisi deÄildir.'
    send(msg)
if __name__=='__main__':main()
