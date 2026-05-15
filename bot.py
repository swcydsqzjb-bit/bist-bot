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

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            last_high = float(high.iloc[-1])
            last_low = float(low.iloc[-1])
            last_volume = float(volume.iloc[-1])

            last_ema20 = float(ema20.iloc[-1])
            last_ema50 = float(ema50.iloc[-1])
            last_rsi = float(rsi.iloc[-1])
            last_volavg20 = float(volavg20.iloc[-1])

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
            trend_destegi = last_close > last_ema20 and last_ema20 > last_ema50
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

            if patlama_skor < 6:
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
                " ".join(durumlar)
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

    mesaj = "📊 BIST SABAH 09:30 PRO TARAMA\n"
    mesaj += f"Toplam liste: {len(hisseler)}\n"
    mesaj += f"Başarılı aday: {basarili}\n"
    mesaj += f"Hata sayısı: {hata}\n\n"
    mesaj += "🔥 İlk patlama adayları:\n\n"

    for kod, fiyat, rsi, pskor, tskor, hacim, zirve, ema20fark, perf5, perf20, range20, durum in sonuclar[:10]:
        mesaj += (
            f"{kod} | Fiyat: {fiyat}\n"
            f"RSI: {rsi} | Patlama: {pskor}/12 | Toplanma: {tskor}/6\n"
            f"Hacim: {hacim}x | Zirve uzaklık: %{zirve}\n"
            f"EMA20 fark: %{ema20fark} | 5g: %{perf5} | 20g: %{perf20}\n"
            f"20g Bant: %{range20}\n"
            f"Sinyal: {durum}\n\n"
        )

    if not sonuclar:
        mesaj += "Bugün uygun aday çıkmadı."

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

            if df.empty or len(df) < 40:
                continue

            df = clean_df(df)

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

            alarm = (
                last_close > gunici_direnc and
                last_close > last_ema20 and
                50 < last_rsi < 75 and
                (hacim_orani > 1.8 or ani_hacim or mum_patlamasi) and
                son3_getiri < 8 and
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
        send_message(mesaj)

if __name__ == "__main__":
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    if now.weekday() >= 5:
        exit()

    if MODE == "daily":
        daily_scan()

    elif MODE == "intraday":
        intraday_scan()
def tavan_oncesi_momentum_scan():
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    if now.weekday() >= 5:
        return

    if not (10 <= now.hour <= 18):
        return

    hisseler = get_symbols()
    adaylar = []

    for hisse in hisseler:
        try:
            df = yf.download(hisse, period="5d", interval="15m", progress=False, auto_adjust=False)

            if df.empty or len(df) < 60:
                continue

            df = clean_df(df)

            close = df["Close"].astype(float)
            high = df["High"].astype(float)
            low = df["Low"].astype(float)
            volume = df["Volume"].astype(float)

            ema20 = ta.trend.ema_indicator(close, window=20)

            last_close = float(close.iloc[-1])
            prev_close = float(close.iloc[-2])
            last_high = float(high.iloc[-1])
            last_low = float(low.iloc[-1])
            last_volume = float(volume.iloc[-1])
            last_ema20 = float(ema20.iloc[-1])
            prev_ema20 = float(ema20.iloc[-4])

            gunici_direnc = float(high.iloc[-25:-1].max())
            hacim_ort20 = float(volume.iloc[-21:-1].mean())
            hacim_orani = last_volume / hacim_ort20 if hacim_ort20 > 0 else 0

            son3_getiri = ((last_close / float(close.iloc[-4])) - 1) * 100
            son12_getiri = ((last_close / float(close.iloc[-13])) - 1) * 100

            son3_hacim = float(volume.tail(3).mean())
            onceki20_hacim = float(volume.iloc[-23:-3].mean())
            hacim_ivmesi = son3_hacim / onceki20_hacim if onceki20_hacim > 0 else 0

            son20_range = (
                (float(high.iloc[-21:-1].max()) - float(low.iloc[-21:-1].min()))
                / float(low.iloc[-21:-1].min()) * 100
            )

            mum_araligi = last_high - last_low
            ust_fitil = last_high - last_close
            kapanis_gucu = (last_close - last_low) / mum_araligi if mum_araligi > 0 else 0
            ust_fitil_orani = ust_fitil / mum_araligi if mum_araligi > 0 else 1

            son_mum_yesil = last_close > prev_close
            ema20_ustu = last_close > last_ema20
            ema20_yukari = last_ema20 > prev_ema20
            direnc_kirildi = last_close > gunici_direnc
            hacim_patlamasi = hacim_orani > 2.0 and hacim_ivmesi > 1.4
            # İntraday ani hacim kontrolü
            volume_ratio_3 = volume / volume.rolling(3).mean()
            ani_hacim = volume_ratio_3.iloc[-1] > 1.8

            # Son mum hacim patlaması
            onceki5_ort = float(volume.iloc[-6:-1].mean())
            mum_patlamasi = last_volume > onceki5_ort * 2
            guclu_kapanis = kapanis_gucu > 0.65 and ust_fitil_orani < 0.35
            fazla_ucmamis = son3_getiri < 7 and son12_getiri < 14
            sikismadan_cikis = son20_range < 10

            skor = 0
            if direnc_kirildi:
                skor += 3
            if hacim_patlamasi:
                skor += 3
            if ani_hacim:
                skor += 1
            if mum_patlamasi:
                skor += 2
            if guclu_kapanis:
                skor += 2
            if ema20_ustu and ema20_yukari:
                skor += 2
            if fazla_ucmamis:
                skor += 1
            if sikismadan_cikis:
                skor += 1
            if son_mum_yesil:
                skor += 1

            zorunlu_sart = (
    direnc_kirildi and
    guclu_kapanis and
    ema20_ustu and
    fazla_ucmamis and
    (hacim_patlamasi or ani_hacim or mum_patlamasi)
)

if not zorunlu_sart:
    continue

if skor < 10:
    continue

            adaylar.append((
                hisse.replace(".IS", ""),
                round(last_close, 2),
                skor,
                round(hacim_orani, 2),
                round(hacim_ivmesi, 2),
                round(son3_getiri, 2),
                round(son12_getiri, 2),
                round(son20_range, 2),
                round(kapanis_gucu, 2),
            ))

            time.sleep(0.03)

        except Exception:
            pass

    adaylar = sorted(adaylar, key=lambda x: (x[2], x[3], x[4], -x[5]), reverse=True)

    if adaylar:
        mesaj = "🚀 TAVAN ÖNCESİ MOMENTUM ALARMI\n"
        mesaj += "RSI kullanılmadı ✅\n"
        mesaj += "Filtre: hacim + direnç kırılımı + güçlü kapanış + EMA20\n\n"

        for kod, fiyat, skor, hacim, ivme, son3, son12, bant, kapanis in adaylar[:8]:
            mesaj += (
                f"{kod} | Fiyat: {fiyat}\n"
                f"Momentum skoru: {skor}/13\n"
                f"Hacim: {hacim}x | Hacim ivmesi: {ivme}x\n"
                f"Son 3 mum: %{son3} | Son 12 mum: %{son12}\n"
                f"20 mum bant: %{bant} | Kapanış gücü: {kapanis}\n"
                f"Sinyal: 🚀 TAVAN ÖNCESİ MOMENTUM ADAYI\n\n"
            )

        send_message(mesaj)
    else:
        print("Momentum filtresi çalıştı ama aday bulamadı")


if __name__ == "__main__":
    now = datetime.now(ZoneInfo("Europe/Istanbul"))

    if now.weekday() >= 5:
        exit()

    if MODE == "daily":
        daily_scan()

    elif MODE == "intraday":
        intraday_scan()
        tavan_oncesi_momentum_scan()

    else:
        if now.hour == 9 and 30 <= now.minute < 45:
            daily_scan()
        elif 10 <= now.hour <= 18:
            intraday_scan()
            tavan_oncesi_momentum_scan()
