# leader_lag.py

import yfinance as yf
import pandas as pd
from itertools import product

PERIOD = "3y"
LEADER_GAIN = 5
FOLLOWER_GAIN = 3
LAGS = [0, 1, 2, 3, 5]

MIN_EVENTS = 10
MIN_SUCCESS_RATE = 60


def load_symbols():
    # Buraya senin mevcut botundaki 500+ BIST sembol listesini bağlayacağız.
    # Örnek format:
    # GARAN.IS, THYAO.IS, ASTOR.IS gibi olmalı.
    from bist_symbols import SYMBOLS
    return SYMBOLS


def download_all_data(symbols):
    all_data = {}

    for symbol in symbols:
        try:
            df = yf.download(
                symbol,
                period=PERIOD,
                interval="1d",
                progress=False,
                auto_adjust=True
            )

            if df.empty or len(df) < 200:
                continue

            df["change_pct"] = df["Close"].pct_change() * 100
            df = df.dropna()

            all_data[symbol] = df[["Close", "change_pct"]]

        except Exception as e:
            print(f"{symbol} veri hatası: {e}")

    return all_data


def analyze_leader_lag(all_data):
    results = []
    symbols = list(all_data.keys())

    for leader, follower in product(symbols, symbols):
        if leader == follower:
            continue

        leader_df = all_data[leader]
        follower_df = all_data[follower]

        leader_days = leader_df[leader_df["change_pct"] >= LEADER_GAIN]

        if len(leader_days) < MIN_EVENTS:
            continue

        for lag in LAGS:
            total_events = 0
            success_events = 0
            follower_returns = []

            for event_date in leader_days.index:
                if event_date not in follower_df.index:
                    continue

                event_pos = follower_df.index.get_loc(event_date)
                target_pos = event_pos + lag

                if target_pos >= len(follower_df):
                    continue

                follower_return = follower_df.iloc[target_pos]["change_pct"]

                total_events += 1

                if follower_return >= FOLLOWER_GAIN:
                    success_events += 1
                    follower_returns.append(follower_return)

            if total_events < MIN_EVENTS:
                continue

            success_rate = success_events / total_events * 100

            if success_rate >= MIN_SUCCESS_RATE:
                avg_return = sum(follower_returns) / len(follower_returns) if follower_returns else 0

                results.append({
                    "leader": leader.replace(".IS", ""),
                    "follower": follower.replace(".IS", ""),
                    "lag_days": lag,
                    "events": total_events,
                    "success_rate": round(success_rate, 2),
                    "avg_success_return": round(avg_return, 2)
                })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        return result_df

    return result_df.sort_values(
        by=["success_rate", "events", "avg_success_return"],
        ascending=False
    )


if __name__ == "__main__":
    symbols = load_symbols()
    print(f"Toplam sembol: {len(symbols)}")

    all_data = download_all_data(symbols)
    print(f"Verisi gelen sembol: {len(all_data)}")

    result = analyze_leader_lag(all_data)

    if result.empty:
        print("Güçlü leader-lag ilişkisi bulunamadı.")
    else:
        print(result.head(50).to_string(index=False))
        result.to_csv("leader_lag_results.csv", index=False)
        print("Sonuç dosyası oluşturuldu: leader_lag_results.csv")
