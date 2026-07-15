from __future__ import annotations
from typing import Any
from tg_utf8_common import csv,js,n,send,t

F="v20_top_picks.csv"; S="v20_top_pick_status.json"

def medal(r:int)->str:return {1:"🥇",2:"🥈",3:"🥉"}.get(r,"⭐")
def icon(s:str)->str:return {"TEYİT GELDİ":"🟢","İZLEMEDE TUT":"🔵","TEMKİNLİ İZLE":"🟡","ELE":"🔴"}.get(s,"⚪")
def items(v:Any)->list[str]:
    x=t(v)
    return [i.strip() for i in x.split("|") if i.strip()] if x else []

def main():
    f=csv(F); s=js(S)
    if f.empty:
        send("🏆 LARUS V20.2 TOP PICKS\n\nBugün sıralanacak aday bulunamadı.");return
    m=f"🏆 LARUS V20.2 TOP PICKS\n\nİncelenen aday: {len(f)}\nTakibe değer aday: {int(n(s.get('selected_count')))}\nTeyit gelen aday: {int(n(s.get('confirmed_count')))}\nGünün ilk adayı: {t(s.get('top_symbol'))}\nTop Pick skoru: {n(s.get('top_score')):.1f}/100\n\n"
    for idx,r in f.head(3).iterrows():
        rank=int(n(r.get("top_pick_rank"),idx+1)); st=t(r.get("confirmation_state"))
        m+=f"{medal(rank)} {rank}. {t(r.get('symbol'))}\n{icon(st)} Durum: {st}\nTop Pick skoru: {n(r.get('top_pick_score')):.1f}/100\nAI Final Score: {n(r.get('ai_final_score')):.1f}/100\nConsensus: {n(r.get('consensus_score')):.1f}/100\nRisk: {t(r.get('risk_class'))} | {n(r.get('risk_score')):.1f}/100\nFiyat: {n(r.get('close')):.2f}\nPiyasa yüzdeliği: %{n(r.get('market_percentile')):.1f}\nRejim: {t(r.get('regime'))}\nİzleme ufku: {int(n(r.get('best_horizon_days')))} işlem günü\nBeklenen ortalama sonuç: {n(r.get('expected_return')):+.2f}%\nTemkinli senaryo: {n(r.get('downside_20pct')):+.2f}%\nOlumlu senaryo: {n(r.get('upside_80pct')):+.2f}%\n"
        w=items(r.get("why_now"))
        if w:m+="\nNeden şimdi?\n"+"\n".join(f"✓ {x}" for x in w)+"\n"
        c=t(r.get("conflict_note"))
        if c:m+=f"\nMotor çelişkisi:\n• {c}\n"
        m+="\n--------------------\n\n"
    m+="📌 Durum açıklaması:\n🟢 Teyit geldi: Katmanlar güçlü ve uyumlu.\n🔵 İzlemede tut: Güçlü görünüm var, ek teyit bekleniyor.\n🟡 Temkinli izle: Bazı katmanlar destekliyor fakat risk/uyum sınırlı.\n🔴 Ele: Mevcut veriler yeterli ortak teyit üretmedi.\n\n⚠️ Bu rapor istatistiksel sıralamadır; yatırım tavsiyesi değildir."
    send(m)
if __name__=="__main__":main()
