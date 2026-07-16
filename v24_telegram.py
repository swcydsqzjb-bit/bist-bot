from tg_utf8_common import csv,js,n,send,t
def main():
    f=csv("v24_live_confirmations.csv"); s=js("v24_status.json")
    if f.empty:
        send("⚡ LARUS V24 CANLI TEYİT RAPORU\n\nCanlı veriyle analiz edilebilen uygun aday bulunamadı."); return
    m=f"⚡ LARUS V24 CANLI TEYİT RAPORU\n\nİncelenen aday: {int(n(s.get('candidate_count')))}\nCanlı verisi gelen: {int(n(s.get('analyzed_count')))}\nCanlı teyit: {int(n(s.get('confirmed_count')))}\nErken teyit: {int(n(s.get('early_confirmation_count')))}\nİlk aday: {t(s.get('top_symbol'))}\nİlk durum: {t(s.get('top_state'))}\nİlk skor: {n(s.get('top_score')):.1f}/100\n\n"
    for i,r in f.head(6).iterrows():
        m+=f"{i+1}. {t(r.get('symbol'))}\nV24 durumu: {t(r.get('v24_state'))}\nCanlı teyit skoru: {n(r.get('v24_score')):.1f}/100\nReferans fiyat: {n(r.get('reference_price')):.2f}\nCanlı fiyat: {n(r.get('live_price')):.2f}\nReferansa göre değişim: {n(r.get('price_change_from_reference_pct')):+.2f}%\nÖnceki V22 durumu: {t(r.get('v22_state'))}\nV22 skoru: {n(r.get('v22_score')):.1f}/100\nRisk: {t(r.get('risk_class'))} | {n(r.get('risk_score')):.1f}/100\n\nCanlı göstergeler:\n• RSI 5dk: {n(r.get('rsi_5m')):.1f}\n• Hacim 5dk: {n(r.get('volume_ratio_5m')):.2f}x\n• EMA20 farkı: {n(r.get('ema20_distance_5m')):+.2f}%\n• EMA20 eğimi: {n(r.get('ema20_slope_5m')):+.3f}%\n• Kırılım farkı: {n(r.get('breakout_pct')):+.2f}%\n• Son 15dk: {n(r.get('return_15m')):+.2f}%\n• Son 60dk: {n(r.get('return_60m')):+.2f}%\n\nCanlı teyit nedenleri:\n"
        for x in [z.strip() for z in t(r.get("v24_reasons")).split("|") if z.strip()]:m+=f"✓ {x}\n"
        m+="\nCanlı riskler:\n"
        for x in [z.strip() for z in t(r.get("v24_risks")).split("|") if z.strip()]:m+=f"• {x}\n"
        m+="\n--------------------\n\n"
    m+="⚠️ V24 canlı teknik teyit katmanıdır. Gerçek emir üretmez ve yatırım tavsiyesi değildir."
    send(m)
if __name__=="__main__":main()
