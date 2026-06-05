import os
import time
import requests
import pandas as pd
import yfinance as yf
import ta
import numpy as np
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
MODE = os.getenv("BOT_MODE", "daily")

def send_message(text):
    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text}
    )
    print("Telegram cevap:", r.status_code, r.text)


def get_symbols():
    url = "https://stockanalysis.com/list/borsa-istanbul/"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers, timeout=20)
    tables = pd.read_html(StringIO(response.text))
    symbols = tables[0]["Symbol"].dropna().astype(str).tolist()
    return [s.strip() + ".IS" for s in symbols]

def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def formasyon_etiketi(close, high, low, volume):
    etiketler = []

    son20_range = (
        (float(high.tail(20).max()) - float(low.tail(20).min()))
        / float(low.tail(20).min()) * 100
    )

    son10_getiri = ((float(close.iloc[-1]) / float(close.iloc[-11])) - 1) * 100
    son20_getiri = ((float(close.iloc[-1]) / float(close.iloc[-21])) - 1) * 100

    # Çanak: önce düşüş, sonra toparlanma, direnç bölgesine yaklaşma
    sol = float(close.iloc[-30])
    dip = float(close.iloc[-15:-5].min())
    son = float(close.iloc[-1])
    if sol > dip and son > dip * 1.08 and son < sol * 1.08:
        etiketler.append("🟢 ÇANAK")

    # Üçgen / sıkışma: dar bant
    if son20_range < 12:
        etiketler.append("🔺 ÜÇGEN/SIKIŞMA")

    # Flama: önce sert yükseliş, sonra dar bantta dinlenme
    if son20_getiri > 12 and son10_getiri < 6 and son20_range < 16:
        etiketler.append("🟡 FLAMA")

    if not etiketler:
        return "Yok"

    return " ".join(etiketler)

def daily_scan():
    hisseler = get_symbols()
    sonuclar = []
    basarili = 0
    hata = 0

    for hisse in hisseler:
        try:
            df = yf.download(hisse, period="6mo", interval="1d", progress=False, auto_adjust=False)
            if df.empty or len(df) < 80:
                continue

            df = clean_df(df)

            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            volume = df["Volume"].astype(float)

            ema20 = ta.trend.ema_indicator(close, window=20)
            ema50 = ta.trend.ema_indicator(close, window=50)
            rsi = ta.momentum.rsi(close, window=14)
            volavg20 = volume.rolling(20).mean()
            ema20_egim = ema20.iloc[-1] > ema20.iloc[-4]
            son3_kapanis_destek = close.iloc[-1] >= close.iloc[-3] * 0.97
            hacim_kalite = volume.tail(5).mean() > volume.iloc[-20:-5].mean() * 1.10
            haftalik_momentum = close.iloc[-1] > close.iloc[-10]

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            last_high = float(high.iloc[-1])
            last_low = float(low.iloc[-1])
            last_volume = float(volume.iloc[-1])

            last_ema20 = float(ema20.iloc[-1])
            last_ema50 = float(ema50.iloc[-1])
            last_rsi = float(rsi.iloc[-1])
            last_volavg20 = float(volavg20.iloc[-1])
            formasyon = formasyon_etiketi(close, high, low, volume)

            ai_skor = 0

            if hacim_orani > 1.5:
                ai_skor += 20

            if 50 < last_rsi < 65:
                ai_skor += 15

            if last_close > last_ema20:
                ai_skor += 15

            if last_ema20 > last_ema50:
                ai_skor += 10

            if duseni_kirdi:
                ai_skor += 15

            if sikisma:
                ai_skor += 10

            if formasyon != "Yok":
                ai_skor += 10

            if sahte_toplama_riski:
                ai_skor -= 20

            if perf_20g > 25:
                ai_skor -= 15

            ai_skor = max(0, min(100, ai_skor))
            
            prev_high_20 = float(high.tail(20).max())

            hacim_orani = last_volume / last_volavg20 if last_volavg20 > 0 else 0
            zirve_uzaklik = ((prev_high_20 - last_close) / prev_high_20) * 100
            ema20_uzaklik = ((last_close - last_ema20) / last_ema20) * 100

            perf_5g = ((last_close / float(close.iloc[-6])) - 1) * 100
            perf_10g = ((last_close / float(close.iloc[-11])) - 1) * 100
            perf_20g = ((last_close / float(close.iloc[-21])) - 1) * 100
            perf_60g = ((last_close / float(close.iloc[-61])) - 1) * 100

            # Çok yükselmiş / aşırı şişmişleri ele
            cok_yukselmis = (
                perf_5g > 18 or
                perf_10g > 28 or
                perf_20g > 45 or
                perf_60g > 90 or
                ema20_uzaklik > 18 or
                last_rsi >= 78
            )
            if cok_yukselmis:
                continue

            # Düşen trend kırılımı
            trend_high = high.tail(30).values
            x = np.arange(len(trend_high))
            egim, kesisim = np.polyfit(x, trend_high, 1)
            bugunku_trend_direnci = egim * (len(trend_high) - 1) + kesisim

            duseni_kirdi = (
                egim < 0 and
                last_close > bugunku_trend_direnci and
                hacim_orani > 1.05
            )

            # Fake breakout filtresi
            mum_araligi = last_high - last_low
            ust_fitil = last_high - last_close
            govde = abs(last_close - prev_close)

            uzun_ust_fitil = mum_araligi > 0 and (ust_fitil / mum_araligi) > 0.45
            kapanis_zayif = last_close < last_high * 0.97
            hacim_yetersiz = hacim_orani < 0.9

            fake_breakout = (
                last_high >= prev_high_20 * 0.98 and
                (uzun_ust_fitil or kapanis_zayif or hacim_yetersiz)
            )
            if fake_breakout:
                continue

            # Sıkışma / erken patlama
            son20_range = (
                (float(high.tail(20).max()) - float(low.tail(20).min()))
                / float(low.tail(20).min()) * 100
            )

            son5_hacim = float(volume.tail(5).mean())
            onceki15_hacim = float(volume.iloc[-20:-5].mean())

            sikisma = son20_range < 22

            haber_oncesi = (
                hacim_orani > 1.3 and
                50 < last_rsi < 65 and
                son20_range < 18 and
                last_close > last_ema20 and
                perf_5g > 0 and
                perf_20g < 15
            )
            trend_destegi = (
                last_close > last_ema20 and
                last_ema20 > last_ema50 and
                ema20_egim and
                son3_kapanis_destek
            )
            hacim_sessiz_toplanma = son5_hacim > onceki15_hacim * 1.15 if onceki15_hacim > 0 else False

            erken_patlama = (
                sikisma and
                trend_destegi and
                hacim_sessiz_toplanma and
                45 < last_rsi < 70
            )

            # Toplanma skoru
            toplanma_skor = 0
            if sikisma:
                toplanma_skor += 1
            if trend_destegi:
                toplanma_skor += 1
            if hacim_sessiz_toplanma:
                toplanma_skor += 2
            if 45 < last_rsi < 65:
                toplanma_skor += 1
            if perf_20g < 20:
                toplanma_skor += 1

            # Takas benzeri gerçek toplama ayrımı
            son10_fiyat_degisim = ((last_close / float(close.iloc[-11])) - 1) * 100
            son10_hacim_ort = float(volume.tail(10).mean())
            onceki30_hacim_ort = float(volume.iloc[-40:-10].mean())

            hacim_artiyor = son10_hacim_ort > onceki30_hacim_ort * 1.25 if onceki30_hacim_ort > 0 else False
            fiyat_cok_gitmemis = son10_fiyat_degisim < 18
            ema20_ustu_tutunma = last_close > last_ema20
            rsi_saglikli = 45 < last_rsi < 70

            gercek_toplama = (
                hacim_artiyor and
                fiyat_cok_gitmemis and
                ema20_ustu_tutunma and
                rsi_saglikli and
                not fake_breakout
            )

            sahte_toplama_riski = (
                hacim_orani > 2.5 and
                perf_5g > 12 and
                uzun_ust_fitil
            )
            # Ani hacim kontrolü
            volume_ratio_3 = volume / volume.rolling(3).mean()
            ani_hacim = volume_ratio_3.iloc[-1] > 2.2

            # İlk patlama skoru
            patlama_skor = 0
            if duseni_kirdi:
                patlama_skor += 3
            if toplanma_skor >= 4:
                patlama_skor += 2
            if hacim_orani > 1.5:
                patlama_skor += 2
            if ani_hacim:
                patlama_skor += 1
            if zirve_uzaklik <= 5:
                patlama_skor += 1
            if 45 < last_rsi < 68:
                patlama_skor += 1
            if perf_20g < 20:
                patlama_skor += 1
            if gercek_toplama:
                patlama_skor += 2
            if erken_patlama:
                patlama_skor += 2
            if sahte_toplama_riski:
                patlama_skor -= 2
            if hacim_kalite:
                patlama_skor += 1
            if haftalik_momentum:
                patlama_skor += 1

            if patlama_skor < 7:
                continue

            durumlar = []
            if gercek_toplama:
                durumlar.append("🟢 GERÇEK TOPLANMA")
            if duseni_kirdi:
                durumlar.append("🔥 DÜŞEN KIRILIM")
            if erken_patlama:
                durumlar.append("🚀 ERKEN ADAY")
            if perf_20g < 20:
                durumlar.append("✅ FAZLA ŞİŞMEMİŞ")
            if sahte_toplama_riski:
                durumlar.append("⚠️ SAHTE RİSK")
            if haber_oncesi:
                durumlar.append("🕵️ HABER ÖNCESİ")

            basarili += 1

            sonuclar.append((
                hisse.replace(".IS", ""),
                round(last_close, 2),
                round(last_rsi, 1),
                patlama_skor,
                toplanma_skor,
                round(hacim_orani, 2),
                round(zirve_uzaklik, 2),
                round(ema20_uzaklik, 2),
                round(perf_5g, 1),
                round(perf_20g, 1),
                round(son20_range, 1),
                " ".join(durumlar),
                formasyon,
                ai_skor
    
            ))

            time.sleep(0.03)

        except Exception:
            hata += 1

    sonuclar = sorted(
        sonuclar,
        key=lambda x: (
            1 if "GERÇEK TOPLANMA" in x[11] else 0,
            1 if "DÜŞEN" in x[11] else 0,
            1 if "ERKEN" in x[11] else 0,
            x[3],
            x[4],
            x[5],
            -x[9]
        ),
        reverse=True
    )
    with open("adaylar.txt", "w", encoding="utf-8") as f:
        for item in sonuclar[:10]:
            f.write(item[0] + "\n")

    if not sonuclar:
        with open("adaylar.txt", "w", encoding="utf-8") as f:
            f.write("SNICA\n")
            
    mesaj = "📊 BIST SABAH 09:30 PRO TARAMA\n"
    mesaj += f"Toplam liste: {len(hisseler)}\n"
    mesaj += f"Başarılı aday: {basarili}\n"
    mesaj += f"Hata sayısı: {hata}\n\n"
    mesaj += "🔥 İlk patlama adayları:\n\n"

    for kod, fiyat, rsi, pskor, tskor, hacim, zirve, ema20fark, perf5, perf20, range20, durum, formasyon, ai_skor in sonuclar[:10]:
        mesaj += (
            f"{kod} | Fiyat: {fiyat}\n"
            f"AI Skor: {ai_skor}/100\n"
            f"RSI: {rsi} | Patlama: {pskor}/12 | Toplanma: {tskor}/6\n"
            f"Hacim: {hacim}x | Zirve uzaklık: %{zirve}\n"
            f"EMA20 fark: %{ema20fark} | 5g: %{perf5} | 20g: %{perf20}\n"
            f"20g Bant: %{range20}\n"
            f"Sinyal: {durum}\n\n"
            f"Formasyon: {formasyon}\n\n"
        )

    if not sonuclar:
        mesaj += "Bugün uygun aday çıkmadı."

    with open("adaylar.txt", "w", encoding="utf-8") as f:
        if sonuclar:
            for item in sonuclar[:10]:
                f.write(item[0] + "\n")
        else:
            f.write("SNICA\n")


        send_message(mesaj)
    

def intraday_scan():
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    if now.weekday() >= 5:
        return

    if not (10 <= now.hour <= 18):
        return

    hisseler = get_symbols()
    alarm_listesi = []

    for hisse in hisseler:
        try:
            df = yf.download(hisse, period="5d", interval="15m", progress=False, auto_adjust=False)
            df1h = yf.download(hisse, period="10d", interval="1h", progress=False, auto_adjust=False)

            if df.empty or df1h.empty or len(df) < 40 or len(df1h) < 30:
                continue

            df = clean_df(df)
            df1h = clean_df(df1h)

            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            volume = df["Volume"].astype(float)

            ema20 = ta.trend.ema_indicator(close, window=20)
            rsi = ta.momentum.rsi(close, window=14)
            volavg20 = volume.rolling(20).mean()

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            last_high = float(high.iloc[-1])
            last_low = float(low.iloc[-1])
            last_ema20 = float(ema20.iloc[-1])
            last_rsi = float(rsi.iloc[-1])
            last_volume = float(volume.iloc[-1])
            last_volavg20 = float(volavg20.iloc[-1])

            gunici_direnc = float(high.iloc[-20:-1].max())
            hacim_orani = last_volume / last_volavg20 if last_volavg20 > 0 else 0
            son3_getiri = ((last_close / float(close.iloc[-4])) - 1) * 100
            # İntraday ani hacim kontrolü
            volume_ratio_3 = volume / volume.rolling(3).mean()
            ani_hacim = volume_ratio_3.iloc[-1] > 1.8

            # Son mum hacim patlaması
            onceki5_ort = float(volume.iloc[-6:-1].mean())
            mum_patlamasi = last_volume > onceki5_ort * 2

            mum_araligi = last_high - last_low
            ust_fitil = last_high - last_close
            uzun_ust_fitil = mum_araligi > 0 and (ust_fitil / mum_araligi) > 0.45
            kapanis_zayif = last_close < last_high * 0.97

            fake_intraday = uzun_ust_fitil or kapanis_zayif

            kapanis_gucu = (last_close - last_low) / (last_high - last_low) if (last_high - last_low) > 0 else 0

            ust_fitil_orani = (last_high - last_close) / (last_high - last_low) if (last_high - last_low) > 0 else 1

            son4_yukselis = (
                close.iloc[-1] > close.iloc[-2] >
                close.iloc[-3] > close.iloc[-4]
            )

            son12_getiri = ((last_close / float(close.iloc[-13])) - 1) * 100

            c1h = df1h["Close"].astype(float)
            ema1h = ta.trend.ema_indicator(c1h, window=20)

            last1h = float(c1h.iloc[-1])
            last_ema1h = float(ema1h.iloc[-1])

            trend_1h_guclu = last1h > last_ema1h
           
            alarm = (
                last_close > gunici_direnc and
                last_close > last_ema20 and
                trend_1h_guclu and
                52 < last_rsi < 72 and
                (hacim_orani > 1.8 or ani_hacim or mum_patlamasi) and
                son3_getiri < 6 and
                son12_getiri < 10 and
                kapanis_gucu > 0.60 and
                ust_fitil_orani < 0.35 and
                son4_yukselis and
                not fake_intraday
            )

            if alarm:
                alarm_listesi.append((
                    hisse.replace(".IS", ""),
                    round(last_close, 2),
                    round(last_rsi, 1),
                    round(hacim_orani, 2),
                    round(son3_getiri, 2)
                ))

            time.sleep(0.03)

        except Exception:
            pass

    if alarm_listesi:
        mesaj = "🚨 GÜN İÇİ 15DK PRO ALARM\n\n"
        for kod, fiyat, rsi, hacim, son3 in alarm_listesi[:10]:
            mesaj += (
                f"{kod} | Fiyat: {fiyat}\n"
                f"15dk kırılım | RSI: {rsi} | Hacim: {hacim}x\n"
                f"Son 3 mum: %{son3}\n"
                f"Filtre: Fake breakout elendi ✅\n\n"
            )
        with open("adaylar.txt", "a", encoding="utf-8") as f:
            for item in alarm_listesi[:10]:
                f.write(item[0] + "\n")
        
        send_message(mesaj)

if __name__ == "__main__":
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    if now.weekday() >= 5 and MODE != "test_akd":
        exit()

    if MODE == "daily":
        daily_scan()

elif MODE == "intraday":
    intraday_scan()


def hazirlik_15dk_5dk_tetik_scan():
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    if now.weekday() >= 5:
        return

    if not (10 <= now.hour <= 18):
        return

    hisseler = get_symbols()
    adaylar = []

    for hisse in hisseler:
        try:
            # 1H hazırlık: son 10 gün
            df1h = yf.download(hisse, period="10d", interval="1h", progress=False, auto_adjust=False)
            # 15DK hazır ol: son 5 gün
            df15 = yf.download(hisse, period="5d", interval="15m", progress=False, auto_adjust=False)
            # 5DK son tetik: son 1 gün
            df5 = yf.download(hisse, period="1d", interval="5m", progress=False, auto_adjust=False)
            df1d = yf.download(hisse, period="3mo", interval="1d", progress=False, auto_adjust=False)

            if df1h.empty or df15.empty or df5.empty or df1d.empty:
                continue

            if len(df1h) < 45 or len(df15) < 60 or len(df5) < 35:
                continue

            df1h = clean_df(df1h)
            df15 = clean_df(df15)
            df5 = clean_df(df5)
            df1d = clean_df(df1d)

            c1d = df1d["Close"].astype(float)
            ema20_1d = ta.trend.ema_indicator(c1d, window=20)

            last1d = float(c1d.iloc[-1])
            ema1d = float(ema20_1d.iloc[-1])

            gunluk_guclu = last1d > ema1d

            if not gunluk_guclu:
                continue

            # ---------- 1H HAZIRLIK ----------
            c1 = df1h["Close"].astype(float)
            h1 = df1h["High"].astype(float)
            l1 = df1h["Low"].astype(float)
            v1 = df1h["Volume"].astype(float)

            ema20_1h = ta.trend.ema_indicator(c1, window=20)

            last1 = float(c1.iloc[-1])
            ema1 = float(ema20_1h.iloc[-1])

            range_1h = (float(h1.tail(35).max()) - float(l1.tail(35).min())) / float(l1.tail(35).min()) * 100
            perf_1h_7gun = ((last1 / float(c1.iloc[-35])) - 1) * 100

            son8_hacim_1h = float(v1.tail(8).mean())
            onceki25_hacim_1h = float(v1.iloc[-33:-8].mean())
            hacim_birikim_1h = son8_hacim_1h > onceki25_hacim_1h * 1.15 if onceki25_hacim_1h > 0 else False

            hazirlik_1h = (
                range_1h < 22 and
                perf_1h_7gun < 28 and
                last1 > ema1 and
                hacim_birikim_1h
            )

            if not hazirlik_1h:
                continue

            # ---------- 15DK HAZIR OL ----------
            c15 = df15["Close"].astype(float)
            h15 = df15["High"].astype(float)
            l15 = df15["Low"].astype(float)
            v15 = df15["Volume"].astype(float)

            ema20_15 = ta.trend.ema_indicator(c15, window=20)

            last15 = float(c15.iloc[-1])
            prev15 = float(c15.iloc[-2])
            high15 = float(h15.iloc[-1])
            low15 = float(l15.iloc[-1])
            ema15 = float(ema20_15.iloc[-1])

            direnc15 = float(h15.iloc[-30:-1].max())
            hacim15_ort = float(v15.iloc[-31:-1].mean())
            hacim15 = float(v15.iloc[-1]) / hacim15_ort if hacim15_ort > 0 else 0

            mum_aralik15 = high15 - low15
            kapanis_gucu15 = (last15 - low15) / mum_aralik15 if mum_aralik15 > 0 else 0
            ust_fitil15 = (high15 - last15) / mum_aralik15 if mum_aralik15 > 0 else 1

            son3_15_getiri = ((last15 / float(c15.iloc[-4])) - 1) * 100

            hazir_15dk = (
                last15 >= direnc15 * 0.985 and
                last15 > ema15 and
                hacim15 > 1.25 and
                kapanis_gucu15 > 0.55 and
                ust_fitil15 < 0.45 and
                son3_15_getiri < 6
            )

            if not hazir_15dk:
                continue

            # ---------- 5DK SON TETİK ----------
            c5 = df5["Close"].astype(float)
            h5 = df5["High"].astype(float)
            l5 = df5["Low"].astype(float)
            v5 = df5["Volume"].astype(float)

            ema20_5 = ta.trend.ema_indicator(c5, window=20)

            last5 = float(c5.iloc[-1])
            prev5 = float(c5.iloc[-2])
            high5 = float(h5.iloc[-1])
            low5 = float(l5.iloc[-1])
            ema5 = float(ema20_5.iloc[-1])

            direnc5 = float(h5.iloc[-20:-1].max())
            hacim5_ort = float(v5.iloc[-21:-1].mean())
            hacim5 = float(v5.iloc[-1]) / hacim5_ort if hacim5_ort > 0 else 0

            son3_hacim5 = float(v5.tail(3).mean())
            onceki15_hacim5 = float(v5.iloc[-18:-3].mean())
            hacim_ivme5 = son3_hacim5 / onceki15_hacim5 if onceki15_hacim5 > 0 else 0

            mum_aralik5 = high5 - low5
            kapanis_gucu5 = (last5 - low5) / mum_aralik5 if mum_aralik5 > 0 else 0
            ust_fitil5 = (high5 - last5) / mum_aralik5 if mum_aralik5 > 0 else 1

            son3_5_getiri = ((last5 / float(c5.iloc[-4])) - 1) * 100
            son12_5_getiri = ((last5 / float(c5.iloc[-13])) - 1) * 100

            son5_yesil = int((c5.tail(5).diff() > 0).sum())

            son4_kapanis_yukseliyor = (
                c5.iloc[-1] > c5.iloc[-2] >
                c5.iloc[-3] > c5.iloc[-4]
            )

            tetik_5dk = (
                last5 > direnc5 and
                last5 > ema5 and
                hacim5 > 1.8 and
                hacim_ivme5 > 1.15 and
                kapanis_gucu5 > 0.58 and
                ust_fitil5 < 0.35 and
                son3_5_getiri < 4.2 and
                son12_5_getiri < 9 and
                son5_yesil >= 3
             )

            if not tetik_5dk:
                continue

            skor = 0
            if hazirlik_1h:
                skor += 4
            if hazir_15dk:
                skor += 4
            if tetik_5dk:
                skor += 5
            if hacim5 > 2:
                skor += 1
            if hacim_ivme5 > 1.5:
                skor += 1
            if kapanis_gucu5 > 0.75:
                skor += 1

            adaylar.append((
                hisse.replace(".IS", ""),
                round(last5, 2),
                skor,
                round(range_1h, 2),
                round(perf_1h_7gun, 2),
                round(hacim15, 2),
                round(hacim5, 2),
                round(hacim_ivme5, 2),
                round(son3_5_getiri, 2),
                round(kapanis_gucu5, 2)
            ))

            time.sleep(0.03)

        except Exception:
            pass

    adaylar = sorted(adaylar, key=lambda x: (x[2], x[6], x[7], -x[8]), reverse=True)

    if adaylar:
        mesaj = "🟣 1H HAZIRLIK + 🟡 15DK HAZIR OL + 🔴 5DK TETİK\n\n"
        mesaj += "Filtre: 1 haftalık hazırlık + 15dk direnç yakınlığı + 5dk son patlama tetik\n\n"

        for kod, fiyat, skor, range1h, perf1h, hacim15, hacim5, ivme5, son3, kapanis in adaylar[:8]:
            mesaj += (
                f"{kod} | Fiyat: {fiyat}\n"
                f"Skor: {skor}/16\n"
                f"1H Bant: %{range1h} | 1H 7g getiri: %{perf1h}\n"
                f"15DK Hacim: {hacim15}x\n"
                f"5DK Hacim: {hacim5}x | İvme: {ivme5}x\n"
                f"5DK Son 3 mum: %{son3} | Kapanış gücü: {kapanis}\n"
                f"Sinyal: 🔴 ARTIK HAZIR OL / TETİK GELDİ\n\n"
            )
        with open("adaylar.txt", "w", encoding="utf-8") as f:
            for item in adaylar[:10]:
                f.write(item[0] + "\n")

        send_message(mesaj)
    else:
        print("1H+15DK+5DK filtre çalıştı ama aday bulamadı")


if __name__ == "__main__":
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    #send_message(f"🧪 TEST | Bot çalıştı | MODE={MODE} | Saat={now.strftime('%H:%M')}")
    #send_message("🧪 AUTO BLOĞUNA GİRDİ")
    
    if now.weekday() >= 5 and MODE != "daily":
        exit()

    if MODE == "daily":
        daily_scan()

    elif MODE == "intraday":
        intraday_scan()
        hazirlik_15dk_5dk_tetik_scan()

    else:
        #send_message("🧪 AUTO MODE ÇALIŞIYOR")

        if now.hour == 9:
            #send_message("🧪 DAILY ÇALIŞACAK")
            daily_scan()

        if 0 <= now.hour <= 23:
            #send_message("🧪 INTRADAY ÇALIŞACAK")
            intraday_scan()
            hazirlik_15dk_5dk_tetik_scan() 
            
time.sleep(15)
