import os
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


SIGNALS_FILE = "signals_history.csv"
CANDIDATES_FILE = "adaylar.txt"

CHECK_DAYS = [1, 3, 5, 10]


def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def read_candidates():
    if not os.path.exists(CANDIDATES_FILE):
        print("adaylar.txt bulunamadı.")
        return []

    with open(CANDIDATES_FILE, "r", encoding="utf-8") as file:
        symbols = [
            line.strip().upper()
            for line in file
            if line.strip()
        ]

    # Aynı hisseyi iki kez kaydetme
    return list(dict.fromkeys(symbols))


def download_price(symbol, period="3mo"):
    ticker = symbol if symbol.endswith(".IS") else symbol + ".IS"

    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=True,
        threads=False
    )

    if df.empty:
        return pd.DataFrame()

    df = clean_df(df)

    if "Close" not in df.columns:
        return pd.DataFrame()

    df = df[["Close"]].copy()
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    return df.dropna()


def load_history():
    if not os.path.exists(SIGNALS_FILE):
        columns = [
            "signal_date",
            "symbol",
            "signal_price",
            "return_1d",
            "return_3d",
            "return_5d",
            "return_10d",
            "max_return_10d",
            "min_return_10d",
            "status"
        ]
        return pd.DataFrame(columns=columns)

    try:
        return pd.read_csv(SIGNALS_FILE)
    except Exception as exc:
        print("Geçmiş dosyası okunamadı:", exc)
        return pd.DataFrame()


def add_new_signals(history):
    now = datetime.now(ZoneInfo("Europe/Istanbul"))
    today = now.strftime("%Y-%m-%d")

    candidates = read_candidates()

    if not candidates:
        print("Yeni aday bulunamadı.")
        return history

    new_rows = []

    for symbol in candidates:
        # Aynı hisse aynı gün yalnızca bir kez kaydedilsin
        duplicate = (
            (history["signal_date"].astype(str) == today) &
            (history["symbol"].astype(str) == symbol)
        )

        if duplicate.any():
            continue

        df = download_price(symbol, period="10d")

        if df.empty:
            print(symbol, "için fiyat alınamadı.")
            continue

        signal_price = float(df["Close"].iloc[-1])

        new_rows.append({
            "signal_date": today,
            "symbol": symbol,
            "signal_price": round(signal_price, 4),
            "return_1d": np.nan,
            "return_3d": np.nan,
            "return_5d": np.nan,
            "return_10d": np.nan,
            "max_return_10d": np.nan,
            "min_return_10d": np.nan,
            "status": "bekliyor"
        })

        print(
            f"Yeni sinyal kaydedildi: "
            f"{symbol} | {signal_price:.2f}"
        )

    if new_rows:
        history = pd.concat(
            [history, pd.DataFrame(new_rows)],
            ignore_index=True
        )

    return history


def update_old_signals(history):
    if history.empty:
        return history

    today = pd.Timestamp.now(tz="Europe/Istanbul").tz_localize(None).normalize()

    for index, row in history.iterrows():
        signal_date = pd.to_datetime(
            row["signal_date"],
            errors="coerce"
        )

        if pd.isna(signal_date):
            continue

        # Yeni kaydedilmiş sinyali aynı gün değerlendirme
        if signal_date.normalize() >= today:
            continue

        symbol = str(row["symbol"]).strip().upper()
        signal_price = float(row["signal_price"])

        df = download_price(symbol, period="3mo")

        if df.empty:
            continue

        price_dates = pd.to_datetime(df.index).tz_localize(None)

        # Sinyal tarihinden sonraki işlem günleri
        future_df = df.loc[price_dates > signal_date.normalize()].copy()

        if future_df.empty:
            continue

        future_prices = future_df["Close"].astype(float)

        for day in CHECK_DAYS:
            column = f"return_{day}d"

            if len(future_prices) >= day:
                future_price = float(future_prices.iloc[day - 1])

                result = (
                    (future_price / signal_price) - 1
                ) * 100

                history.at[index, column] = round(result, 2)

        first_10_days = future_prices.head(10)

        if not first_10_days.empty:
            max_return = (
                (float(first_10_days.max()) / signal_price) - 1
            ) * 100

            min_return = (
                (float(first_10_days.min()) / signal_price) - 1
            ) * 100

            history.at[index, "max_return_10d"] = round(
                max_return, 2
            )

            history.at[index, "min_return_10d"] = round(
                min_return, 2
            )

        if len(future_prices) >= 10:
            history.at[index, "status"] = "tamamlandı"
        elif len(future_prices) >= 5:
            history.at[index, "status"] = "5g_güncellendi"
        elif len(future_prices) >= 3:
            history.at[index, "status"] = "3g_güncellendi"
        elif len(future_prices) >= 1:
            history.at[index, "status"] = "1g_güncellendi"

    return history


def print_summary(history):
    completed = history[
        history["return_5d"].notna()
    ].copy()

    print("\n📊 PERFORMANS ÖZETİ")

    print("Toplam kayıt:", len(history))
    print("5 günlük sonucu oluşan:", len(completed))

    if completed.empty:
        print("Henüz yeterli sonuç oluşmadı.")
        return

    returns_5d = pd.to_numeric(
        completed["return_5d"],
        errors="coerce"
    ).dropna()

    success_3 = (returns_5d >= 3).mean() * 100
    success_5 = (returns_5d >= 5).mean() * 100
    positive = (returns_5d > 0).mean() * 100

    print(f"5 günde pozitif kapanan: %{positive:.1f}")
    print(f"5 günde en az %3 yükselen: %{success_3:.1f}")
    print(f"5 günde en az %5 yükselen: %{success_5:.1f}")
    print(f"Ortalama 5 günlük getiri: %{returns_5d.mean():.2f}")
    print(f"Medyan 5 günlük getiri: %{returns_5d.median():.2f}")


def main():
    history = load_history()
    history = update_old_signals(history)
    history = add_new_signals(history)

    history = history.drop_duplicates(
        subset=["signal_date", "symbol"],
        keep="last"
    )

    history = history.sort_values(
        by=["signal_date", "symbol"],
        ascending=[False, True]
    )

    history.to_csv(
        SIGNALS_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print_summary(history)
    print(f"\nDosya güncellendi: {SIGNALS_FILE}")


if __name__ == "__main__":
    main()
