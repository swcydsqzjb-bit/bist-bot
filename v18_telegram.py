from tg_utf8_common import *
F="v18_confidence_decisions.csv"; S="v18_confidence_status.json"
def main():
    f=csv(F); s=js(S)
    if f.empty: send("🛡️ LARUS V18 GÜVEN RAPORU\n\nBugün güven analizi yapılacak aday bulunamadı."); return
    m=f"🛡️ LARUS V18 GÜVEN RAPORU\n\nİncelenen aday: {len(f)}\nOnaylanan: {int(n(s.get('approved_count')))}\nTamamlanmış 5 günlük sonuç: {int(n(s.get('completed_5d')))}\nÖğrenme hafızası hazır: {'EVET' if s.get('history_ready') else 'HAYIR'}\n\n"
    for i,r in f.iterrows():
        m+=f"🎯 {i+1}. {t(first(r,'symbol'))}\nV18 kararı: {t(first(r,'v18_decision'))}\nGüven sınıfı: {t(first(r,'confidence_class'))}\nGüven puanı: {n(first(r,'confidence_score')):.1f}/100\nFiyat: {n(first(r,'close','price')):.2f}\nV17 skoru: {n(first(r,'v17_score')):.1f}/100\nV15 skoru: {n(first(r,'v15_score')):.1f}/100\nGöreli güç: {n(first(r,'relative_strength_score')):.1f}/100\nKarşılaştırma yüzdeliği: %{n(first(r,'market_percentile')):.1f}\nRejim: {t(first(r,'regime'))} | {n(first(r,'regime_confidence')):.1f}/100\n"
        plus=listed(first(r,'confidence_boosters','positive_reasons'),"✓")
        minus=listed(first(r,'confidence_reducers','negative_reasons'))
        if plus: m+="\nGüveni artıranlar:\n"+plus+"\n"
        if minus: m+="\nGüveni düşürenler:\n"+minus+"\n"
        m+="\n--------------------\n\n"
    send(m+"⚠️ Güven puanı istatistiksel uyum ölçüsüdür; gerçek yükselme olasılığı değildir.")
if __name__=="__main__": main()
