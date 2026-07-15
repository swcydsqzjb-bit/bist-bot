from tg_utf8_common import *
F="v14_adaptive_decisions.csv"; S="v14_status.json"
def main():
    f=csv(F); s=js(S)
    if f.empty: send("🧠 LARUS V14 ADAPTİF KARAR RAPORU\n\nBugün incelenecek aday bulunamadı."); return
    m=f"🧠 LARUS V14 ADAPTİF KARAR RAPORU\n\nİncelenen aday: {len(f)}\nOnaylanan: {int(n(s.get('approved_count',0)))}\nAğırlık modu: {t(s.get('weight_mode','BASE'))}\n\n"
    for i,r in f.iterrows():
        m+=f"🎯 {i+1}. {t(first(r,'symbol'))}\nKarar: {t(first(r,'v14_decision','decision'))}\nFiyat: {n(first(r,'close','price')):.2f}\nV14 skoru: {n(first(r,'v14_score','final_score')):.1f}/100\nV8 skoru: {n(first(r,'v8_score')):.1f}/100\nSmart Money: {n(first(r,'smart_money_score')):.1f}/100\nKurumsal: {n(first(r,'institutional_score','institutional_accumulation_score')):.1f}/100\nDNA: {t(first(r,'dna_classification','dna_class'))} | {n(first(r,'dna_confidence')):.1f}/100\n5 günde pozitif: %{n(first(r,'positive_rate_5d')):.1f}\n5 günde en az %3: %{n(first(r,'hit_3pct_5d_rate','hit_3_rate')):.1f}\nOrtalama 5 günlük sonuç: {n(first(r,'average_result_5d','avg_result_5d')):+.2f}%\nPozitif bonus: {n(first(r,'positive_bonus')):+.1f}\nRisk kesintisi: {n(first(r,'risk_penalty')):+.1f}\n"
        reasons=listed(first(r,'approval_reasons','positive_reasons','reasons'),"✓")
        risks=listed(first(r,'risks','risk_reasons','risk_notes'))
        if reasons: m+="\nOnay nedenleri:\n"+reasons+"\n"
        if risks: m+="\nRiskler:\n"+risks+"\n"
        m+="\n--------------------\n\n"
    send(m+"⚠️ V14 istatistiksel bir karar katmanıdır. Yatırım tavsiyesi değildir.")
if __name__=="__main__": main()
