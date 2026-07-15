from tg_utf8_common import *
F="v19_timing_forecasts.csv"; S="v19_timing_status.json"
def main():
    f=csv(F); s=js(S)
    if f.empty: send("⏱️ LARUS V19 ZAMANLAMA RAPORU\n\nBugün zamanlama tahmini yapılacak aday bulunamadı."); return
    files=s.get("history_files",[])
    files=", ".join(t(x) for x in files) if isinstance(files,list) else t(files)
    m=f"⏱️ LARUS V19 ZAMANLAMA RAPORU\n\nİncelenen aday: {len(f)}\nZamanlaması hesaplanan: {int(n(s.get('timing_ready_count')))}\nTarihsel hafıza: {int(n(s.get('history_count')))} örnek\nKullanılan hafıza: {files or 'bulunamadı'}\n\nV19, geçmiş benzer sinyallerin 1, 3, 5 ve 10 işlem günlük sonuçlarını karşılaştırır.\n\n"
    for i,r in f.iterrows():
        ready=str(first(r,'timing_ready')).lower() in {"true","1","evet"}
        m+=f"🎯 {i+1}. {t(first(r,'symbol'))}\n"
        if not ready:
            m+=f"Durum: VERİ YETERSİZ\nBenzer örnek: {int(n(first(r,'neighbor_count')))}\nAçıklama: {t(first(r,'timing_message'))}\n\n--------------------\n\n"; continue
        m+=f"Zamanlama sınıfı: {t(first(r,'timing_class'))}\nÖnerilen izleme ufku: {int(n(first(r,'best_horizon_days')))} işlem günü\nZamanlama güveni: {n(first(r,'timing_confidence')):.1f}/100\nBenzer tarihsel örnek: {int(n(first(r,'neighbor_count')))}\nBeklenen ortalama sonuç: {n(first(r,'expected_return')):+.2f}%\nMedyan sonuç: {n(first(r,'median_return')):+.2f}%\nPozitif sonuç oranı: %{n(first(r,'positive_rate')):.1f}\nEn az %3 oranı: %{n(first(r,'hit_3_rate')):.1f}\nTemkinli senaryo: {n(first(r,'downside_20pct')):+.2f}%\nOlumlu senaryo: {n(first(r,'upside_80pct')):+.2f}%\n\nUfuk karşılaştırması:\n• 1 gün: {n(first(r,'result_1d_mean')):+.2f}%\n• 3 gün: {n(first(r,'result_3d_mean')):+.2f}%\n• 5 gün: {n(first(r,'result_5d_mean')):+.2f}%\n• 10 gün: {n(first(r,'result_10d_mean')):+.2f}%\n\n--------------------\n\n"
    send(m+"⚠️ V19 zamanlama çıktısı geçmiş benzer örneklerin istatistiksel özetidir; alım-satım talimatı değildir.")
if __name__=="__main__": main()
