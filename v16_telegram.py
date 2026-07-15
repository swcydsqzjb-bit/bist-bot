from tg_utf8_common import *
F="v16_relative_strength.csv"; S="v16_status.json"
def main():
    f=csv(F); s=js(S)
    if f.empty: send("📊 LARUS V16 GÖRELİ GÜÇ RAPORU\n\nBugün karşılaştırılacak aday bulunamadı."); return
    total=int(n(s.get("market_count",0)))
    m=f"📊 LARUS V16 GÖRELİ GÜÇ RAPORU\n\nKarşılaştırılan piyasa: {total} hisse\nİncelenen V15 adayı: {len(f)}\n\nV15 adayları; momentum, trend, hacim ve kalite açısından tüm BIST ile karşılaştırıldı.\n\n"
    for i,r in f.iterrows():
        m+=f"🎯 {i+1}. {t(first(r,'symbol'))}\nV16 sınıfı: {t(first(r,'relative_class'))}\nPiyasa sırası: {int(n(first(r,'market_rank')))}/{total}\nPiyasa yüzdeliği: %{n(first(r,'market_percentile')):.1f}\nGöreli güç skoru: {n(first(r,'relative_strength_score')):.1f}/100\nMomentum: {n(first(r,'momentum_percentile')):.1f}/100\nTrend: {n(first(r,'trend_percentile')):.1f}/100\nHacim: {n(first(r,'volume_percentile')):.1f}/100\nKalite: {n(first(r,'quality_percentile')):.1f}/100\nV15 kararı: {t(first(r,'v15_decision'))}\nV15 skoru: {n(first(r,'v15_score')):.1f}/100\n1 günlük değişim: {n(first(r,'return_1d')):+.1f}%\n5 günlük değişim: {n(first(r,'return_5d')):+.1f}%\n20 günlük değişim: {n(first(r,'return_20d')):+.1f}%\n\n--------------------\n\n"
    send(m+"⚠️ Göreli güç, hissenin diğer hisselere göre istatistiksel konumudur. Yatırım tavsiyesi değildir.")
if __name__=="__main__": main()
