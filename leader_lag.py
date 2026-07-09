import os
import time
import requests
import pandas as pd
import yfinance as yf
from io import StringIO

TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

PERIOD = "3y"
LEADER_GAIN = 5
FOLLOWER_GAIN = 3
LAGS = [0, 1, 2, 3, 5]
MIN_EVENTS = 8
MIN_SUCCESS_RATE = 60


def send_message(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return

    r = requests.post(
        f"https://api.telegram.org/bot{TOKEN}/sendMessage",
        data={"chat_id": CHAT_ID, "text": text[:4000]}
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


def download_data(symbols):
    data = {}

    for i, symbol in enumerate(symbols, 1):
        try:
            print(f"{i}/{len(symbols)} indiriliyor: {symbol}")

            df = yf.download(
                symbol,
                period=PERIOD,
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if df.empty or len(df) < 250:
                continue

            df = clean_df(df)
            df["change_pct"] = df["Close"].pct_change() * 100
            df = df.dropna()

            data[symbol] = df[["change_pct"]]

            time.sleep(0.03)

        except Exception as e:
            print(f"{symbol} hata: {e}")

    return data


def analyze(data):
    results = []
    symbols = list(data.keys())

    for i, leader in enumerate(symbols, 1):
        print(f"Analiz ediliyor: {i}/{len(symbols)} {leader}")

        leader_df = data[leader]
        leader_days = leader_df[leader_df["change_pct"] >= LEADER_GAIN]

        if len(leader_days) < MIN_EVENTS:
            continue

        for follower in symbols:
            if leader == follower:
                continue

            follower_df = data[follower]

            common_dates = leader_days.index.intersection(follower_df.index)

            if len(common_dates) < MIN_EVENTS:
                continue

            for lag in LAGS:
                total = 0
                success = 0
                success_returns = []

                for d in common_dates:
                    try:
                        pos = follower_df.index.get_loc(d)
                        target_pos = pos + lag

                        if target_pos >= len(follower_df):
                            continue

                        ret = float(follower_df.iloc[target_pos]["change_pct"])

                        total += 1

                        if ret >= FOLLOWER_GAIN:
                            success += 1
                            success_returns.append(ret)

                    except Exception:
                        continue

                if total < MIN_EVENTS:
                    continue

                success_rate = success / total * 100

                if success_rate >= MIN_SUCCESS_RATE:
                    avg_return = sum(success_returns) / len(success_returns) if success_returns else 0

                    results.append({
                        "lider": leader.replace(".IS", ""),
                        "takipci": follower.replace(".IS", ""),
                        "gecikme_gun": lag,
                        "olay_sayisi": total,
                        "basari_orani": round(success_rate, 2),
                        "ortalama_getiri": round(avg_return, 2)
                    })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    return df.sort_values(
        by=["basari_orani", "olay_sayisi", "ortalama_getiri"],
        ascending=False
    )


def main():
    send_message("🧠 BIST LEADER-LAG analizi başladı.\nBu işlem biraz uzun sürebilir.")

    symbols = get_symbols()
    print(f"Toplam sembol: {len(symbols)}")

    data = download_data(symbols)
    print(f"Verisi gelen sembol: {len(data)}")

    result = analyze(data)

    if result.empty:
        send_message("Leader-lag analizinde güçlü ilişki bulunamadı.")
        return

    result.to_csv("leader_lag_results.csv", index=False)

    mesaj = "🧠 BIST LEADER-LAG SONUÇLARI\n\n"
    mesaj += f"Toplam sembol: {len(symbols)}\n"
    mesaj += f"Verisi gelen: {len(data)}\n\n"
    mesaj += "En güçlü ilişkiler:\n\n"

    for _, row in result.head(20).iterrows():
        mesaj += (
            f"{row['lider']} ➜ {row['takipci']}\n"
            f"Gecikme: {row['gecikme_gun']} gün\n"
            f"Başarı: %{row['basari_orani']}\n"
            f"Olay: {row['olay_sayisi']}\n"
            f"Ort. getiri: %{row['ortalama_getiri']}\n\n"
        )

    send_message(mesaj)
    print(result.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
