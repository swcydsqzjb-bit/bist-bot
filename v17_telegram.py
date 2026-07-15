from tg_utf8_common import *
F="v17_regime_adjusted_decisions.csv"; S="v17_market_regime_status.json"
def main():
    f=csv(F); s=js(S)
    if f.empty: send("🧭 LARUS V17 PİYASA REJİM RAPORU\n\nBugün değerlendirilecek aday bulunamadı."); return
    m=f"🧭 LARUS V17 PİYASA REJİM RAPORU\n\nPiyasa rejimi: {t(s.get('regime'))}\nRejim güveni: {n(s.get('regime_confidence')):.1f}/100\nKarşılaştırılan hisse: {int(n(s.get('market_count')))}\nİncelenen aday: {len(f)}\nOnaylanan: {int(n(s.get('approved_count')))}\n\n1 günlük pozitif genişlik: %{n(s.get('breadth_1d_positive_pct')):.1f}\n5 günlük pozitif genişlik: %{n(s.get('breadth_5d_positive_pct')):.1f}\nEMA20 üzerindeki hisseler: %{n(s.get('above_ema20_pct')):.1f}\n\n"
    for i,r in f.iterrows():
        m+=f"🎯 {i+1}. {t(first(r,'symbol'))}\nV17 kararı: {t(first(r,'v17_decision'))}\nFiyat: {n(first(r,'close','price')):.2f}\nV17 skoru: {n(first(r,'v17_score')):.1f}/100\nRejim etkisi: {n(first(r,'regime_adjustment')):+.1f} puan\nV15 skoru: {n(first(r,'v15_score')):.1f}/100\nGöreli güç: {n(first(r,'relative_strength_score')):.1f}/100\nPiyasa yüzdeliği: %{n(first(r,'market_percentile')):.1f}\nMomentum: {n(first(r,'momentum_percentile')):.1f}/100\nTrend: {n(first(r,'trend_percentile')):.1f}/100\nHacim: {n(first(r,'volume_percentile')):.1f}/100\nKalite: {n(first(r,'quality_percentile')):.1f}/100\n"
        reasons=listed(first(r,'regime_reasons'))
        if reasons: m+="\nRejim değerlendirmesi:\n"+reasons+"\n"
        m+="\n--------------------\n\n"
    send(m+"⚠️ V17, sinyalleri piyasa rejimine göre yeniden ağırlıklandırır. Yatırım tavsiyesi değildir.")
if __name__=="__main__": main()
