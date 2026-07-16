from tg_utf8_common import csv,js,n,send,t
def main():
    f=csv("v23_positions.csv"); s=js("v23_status.json")
    if f.empty:
        send("🧭 LARUS V23 POZİSYON YÖNETİCİSİ\n\nBugün aktif izleme yaşam döngüsüne alınan aday bulunamadı."); return
    m=f"🧭 LARUS V23 POZİSYON YÖNETİCİSİ\n\nAktif izleme kaydı: {len(f)}\nYeni izleme: {int(n(s.get('new_position_count')))}\nHedef bölgesinde: {int(n(s.get('target_zone_count')))}\nKapanma adayı: {int(n(s.get('closing_candidate_count')))}\nİlk aday: {t(s.get('top_symbol'))}\nİlk aksiyon: {t(s.get('top_action'))}\n\n"
    for i,r in f.head(8).iterrows():
        m+=f"{i+1}. {t(r.get('symbol'))}\nYaşam döngüsü: {t(r.get('position_state'))}\nSistem aksiyonu: {t(r.get('action'))}\nNedeni: {t(r.get('action_reason'))}\nReferans giriş: {n(r.get('entry_reference')):.2f}\nSon fiyat: {n(r.get('last_price')):.2f}\nGerçekleşmemiş değişim: {n(r.get('unrealized_return_pct')):+.2f}%\nEn iyi hareket: {n(r.get('max_favorable_excursion_pct')):+.2f}%\nEn kötü hareket: {n(r.get('max_adverse_excursion_pct')):+.2f}%\nİzlemede geçen gün: {int(n(r.get('days_in_position')))}\nSon V22 durumu: {t(r.get('latest_signal_state'))}\nSon V22 skoru: {n(r.get('latest_v22_score')):.1f}/100\nModel ağırlığı: %{n(r.get('model_weight_pct')):.1f}\n\nİstatistiksel bölgeler:\n• Geçersizlik: {n(r.get('statistical_invalidation_price')):.2f}\n• İlk gözlem: {n(r.get('first_objective_price')):.2f}\n• Olumlu senaryo: {n(r.get('optimistic_objective_price')):.2f}\n\n--------------------\n\n"
    m+="⚠️ V23 gerçek emir üretmez. Adayların istatistiksel izleme yaşam döngüsünü yönetir; yatırım tavsiyesi değildir."
    send(m)
if __name__=="__main__":main()
