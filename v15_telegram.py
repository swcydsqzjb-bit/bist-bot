from tg_utf8_common import *
F="v15_final_decisions.csv"; S="v15_status.json"
def main():
    f=csv(F); s=js(S)
    if f.empty: send("🦅 LARUS V15 NİHAİ KARAR RAPORU\n\nBugün incelenecek aday bulunamadı."); return
    completed=int(n(s.get("completed_5d",s.get("completed_count",0))))
    m=f"🦅 LARUS V15 NİHAİ KARAR RAPORU\n\nİncelenen aday: {len(f)}\nOnaylanan: {int(n(s.get('approved_count',0)))}\nModel modu: {t(s.get('model_mode','FALLBACK'))}\nTamamlanmış 5 günlük sonuç: {completed}\nÖğrenme seviyesi: {t(s.get('learning_level','BAŞLANGIÇ'))}\n\n"
    if completed<30: m+="V15 henüz 30 tamamlanmış sonuca ulaşmadığı için güvenli geçiş modunu kullanıyor.\n\n"
    for i,r in f.iterrows():
        m+=f"🎯 {i+1}. {t(first(r,'symbol'))}\nV15 kararı: {t(first(r,'v15_decision','decision'))}\nFiyat: {n(first(r,'close','price')):.2f}\nV15 skoru: {n(first(r,'v15_score','final_score')):.1f}/100\nV14 skoru: {n(first(r,'v14_score')):.1f}/100\nÖğrenme bileşeni: {n(first(r,'learning_component','learning_score')):.1f}/100\nV8 skoru: {n(first(r,'v8_score')):.1f}/100\nSmart Money: {n(first(r,'smart_money_score')):.1f}/100\nKurumsal: {n(first(r,'institutional_score','institutional_accumulation_score')):.1f}/100\nDNA: {t(first(r,'dna_classification','dna_class'))} | {n(first(r,'dna_confidence')):.1f}/100\n5 günde pozitif: %{n(first(r,'positive_rate_5d')):.1f}\nOrtalama 5 günlük sonuç: {n(first(r,'average_result_5d','avg_result_5d')):+.2f}%\nÖnceki V14 kararı: {t(first(r,'v14_decision'))}\n\n--------------------\n\n"
    send(m+"⚠️ V15 geçmiş sonuçlardan istatistiksel ağırlık öğrenir. Yatırım tavsiyesi değildir.")
if __name__=="__main__": main()
