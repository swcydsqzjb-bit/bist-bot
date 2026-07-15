from __future__ import annotations
import json, os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import requests

TOKEN=os.getenv("TOKEN","").strip()
CHAT_ID=os.getenv("CHAT_ID","").strip()
LIMIT=3900

def n(v:Any,d:float=0.0)->float:
    try:
        x=float(v)
        return d if not np.isfinite(x) else x
    except Exception:
        return d

def fix(s:str)->str:
    if not s:
        return s
    out=s
    for _ in range(3):
        if not any(c in out for c in ("Ã","Ä","Å","Â","â","ð","�")):
            break
        changed=False
        for enc in ("latin1","cp1252"):
            try:
                candidate=out.encode(enc).decode("utf-8")
                if candidate!=out:
                    out=candidate
                    changed=True
                    break
            except Exception:
                pass
        if not changed:
            break
    return out

def t(v:Any)->str:
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    return fix(str(v).strip())

def csv(path:str)->pd.DataFrame:
    p=Path(path)
    if not p.exists():
        return pd.DataFrame()
    for enc in ("utf-8-sig","utf-8","cp1254","latin1"):
        try:
            f=pd.read_csv(p,encoding=enc)
            for c in f.select_dtypes(include="object").columns:
                f[c]=f[c].map(lambda x:fix(x) if isinstance(x,str) else x)
            return f
        except UnicodeDecodeError:
            continue
        except Exception:
            break
    return pd.DataFrame()

def js(path:str)->dict:
    p=Path(path)
    if not p.exists():
        return {}
    for enc in ("utf-8","utf-8-sig","cp1254","latin1"):
        try:
            data=json.loads(p.read_text(encoding=enc))
            return repair(data)
        except Exception:
            continue
    return {}

def repair(v:Any)->Any:
    if isinstance(v,str): return fix(v)
    if isinstance(v,list): return [repair(x) for x in v]
    if isinstance(v,dict): return {k:repair(x) for k,x in v.items()}
    return v

def first(r:pd.Series,*names:str,default:Any="")->Any:
    for name in names:
        if name in r.index:
            v=r.get(name)
            try:
                if pd.isna(v): continue
            except Exception:
                pass
            if v not in (None,""): return v
    return default

def split(msg:str)->list[str]:
    msg=fix(msg).strip()
    if len(msg)<=LIMIT: return [msg]
    parts=[]; cur=""
    for block in msg.split("\n\n"):
        cand=block if not cur else cur+"\n\n"+block
        if len(cand)<=LIMIT:
            cur=cand
        else:
            if cur: parts.append(cur)
            while len(block)>LIMIT:
                cut=block.rfind("\n",0,LIMIT)
                if cut<500: cut=LIMIT
                parts.append(block[:cut])
                block=block[cut:].lstrip()
            cur=block
    if cur: parts.append(cur)
    return parts

def send(msg:str)->None:
    msg=fix(msg)
    if not TOKEN or not CHAT_ID:
        print(msg); return
    url=f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    for part in split(msg):
        r=requests.post(url,json={"chat_id":CHAT_ID,"text":part,"disable_web_page_preview":True},timeout=30)
        r.raise_for_status()

def listed(value:Any,prefix:str="•")->str:
    value=t(value)
    if not value: return ""
    return "\n".join(f"{prefix} {x.strip()}" for x in value.split("|") if x.strip())
