from tg_utf8_common import csv,js,n,send,t
def main():
    s=js("v21_status.json"); r=csv("v21_learning_report.csv"); w=csv("v21_learned_weights.csv")
    done=int(n(s.get("completed_5d"))); need=int(n(s.get("minimum_required"),30))
    if t(s.get("status"))!="learning_active":
        send(f"🧠 LARUS V21 SELF LEARNING\n\nÖğrenme durumu: VERİ BEKLİYOR\nTamamlanmış 5 günlük sonuç: {done}\nGerekli minimum sonuç: {need}\n\nYeterli gerçek sonuç oluşana kadar mevcut ağırlıklar korunacak."); return
    m=f"🧠 LARUS V21 SELF LEARNING\n\nÖğrenme durumu: AKTİF\nTamamlanmış 5 günlük sonuç: {done}\nGüven katsayısı: %{n(s.get('reliability_factor'))*100:.0f}\nEn çok güçlenen özellik: {t(s.get('strongest_feature'))}\nEn çok zayıflayan özellik: {t(s.get('weakest_feature'))}\n\nAğırlık değişimleri:\n"
    if not r.empty:
        for _,x in r.reindex(r["change"].abs().sort_values(ascending=False).index).head(8).iterrows():
            ch=n(x.get("change")); ico="⬆️" if ch>.35 else "⬇️" if ch<-.35 else "➡️"
            m+=f"{ico} {t(x.get('feature'))}: {n(x.get('base_weight')):.1f} → {n(x.get('recommended_weight')):.1f} ({ch:+.2f})\n"
    m+="\nYeni normalize ağırlıklar:\n"
    if not w.empty:
        for _,x in w.sort_values("learned_weight",ascending=False).iterrows():
            m+=f"• {t(x.get('feature'))}: %{n(x.get('learned_weight')):.1f}\n"
    m+="\n⚠️ Bu katman geçmiş tamamlanmış sonuçlara göre istatistiksel ağırlık ayarı yapar; getiri garantisi değildir."
    send(m)
if __name__=="__main__":main()
