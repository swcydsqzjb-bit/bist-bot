import os
import time
import math
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from io import StringIO
from collections import defaultdict


TOKEN = os.getenv("TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# ---------------- AYARLAR ----------------

PERIOD = "5y"

# Lider bir günde en az yüzde kaç yükselmeli?
LEADER_GAIN = 5.0

# Takipçi hedef günde en az yüzde kaç yükselmeli?
FOLLOWER_GAIN = 3.0

# Aynı gün yok. Sadece gelecek işlem günleri.
LAGS = [1, 2, 3, 5]

# Eğitim döneminde minimum lider olayı
MIN_TRAIN_EVENTS = 12

# Son 1 yıllık test döneminde minimum olay
MIN_TEST_EVENTS = 4

# Eğitim döneminde minimum başarı
MIN_TRAIN_SUCCESS = 55.0

# Son 1 yıllık test döneminde minimum başarı
MIN_TEST_SUCCESS = 50.0

# Lider sonrasındaki başarı, normal başarıdan en az
# kaç yüzde puan yüksek olmalı?
MIN_TRAIN_UPLIFT = 15.0
MIN_TEST_UPLIFT = 10.0

# En az kaç günlük geçmişi olan hisseler alınsın?
MIN_HISTORY_DAYS = 500

# Test dönemi: son kaç takvim günü?
TEST_DAYS = 365

# Telegram'da gösterilecek ilişki sayısı
TOP_RESULTS = 20

# Aynı takipçi en fazla kaç kere listede görünsün?
MAX_SAME_FOLLOWER = 2


def send_message(text):
    if not TOKEN or not CHAT_ID:
        print(text)
        return

    # Telegram mesaj sınırına karşı parçalara ayır
    max_length = 3900

    for start in range(0, len(text), max_length):
        part = text[start:start + max_length]

        try:
            response = requests.post(
                f"https://api.telegram.org/bot{TOKEN}/sendMessage",
                data={
                    "chat_id": CHAT_ID,
                    "text": part
                },
                timeout=30
            )

            print(
                "Telegram cevap:",
                response.status_code,
                response.text[:300]
            )

        except Exception as exc:
            print("Telegram gönderim hatası:", exc)


def get_symbols():
    url = "https://stockanalysis.com/list/borsa-istanbul/"
    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(
        url,
        headers=headers,
        timeout=30
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    if not tables:
        raise RuntimeError("BIST sembol tablosu bulunamadı.")

    symbols = (
        tables[0]["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    result = []

    for symbol in symbols:
        if not symbol:
            continue

        if symbol.endswith(".IS"):
            result.append(symbol)
        else:
            result.append(symbol + ".IS")

    return sorted(set(result))


def clean_df(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    return df


def download_all_returns(symbols):
    returns_dict = {}
    downloaded = 0
    failed = 0

    for number, symbol in enumerate(symbols, start=1):
        try:
            print(
                f"[{number}/{len(symbols)}] "
                f"İndiriliyor: {symbol}"
            )

            df = yf.download(
                symbol,
                period=PERIOD,
                interval="1d",
                progress=False,
                auto_adjust=True,
                actions=False,
                threads=False,
                timeout=20
            )

            if df.empty:
                failed += 1
                continue

            df = clean_df(df)

            if "Close" not in df.columns:
                failed += 1
                continue

            close = pd.to_numeric(
                df["Close"],
                errors="coerce"
            )

            daily_return = close.pct_change(
                fill_method=None
            ) * 100

            daily_return = daily_return.replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            if len(daily_return) < MIN_HISTORY_DAYS:
                print(
                    f"{symbol}: geçmiş kısa "
                    f"({len(daily_return)} gün), atlandı."
                )
                continue

            daily_return.name = symbol.replace(".IS", "")
            returns_dict[daily_return.name] = daily_return
            downloaded += 1

            time.sleep(0.02)

        except Exception as exc:
            failed += 1
            print(f"{symbol} veri hatası: {exc}")

    print("Verisi alınan:", downloaded)
    print("Veri hatası/kısa geçmiş:", failed)

    if not returns_dict:
        return pd.DataFrame()

    returns_df = pd.concat(
        returns_dict.values(),
        axis=1,
        join="outer"
    )

    returns_df = returns_df.sort_index()

    # Tamamen boş satırları kaldır
    returns_df = returns_df.dropna(how="all")

    return returns_df


def wilson_lower_bound(successes, total, z=1.0):
    """
    Başarı oranının temkinli alt sınırı.

    z=1.0 kullanılması yaklaşık olarak daha ılımlı,
    fakat ham başarı oranından daha temkinli bir sıralama sağlar.
    """
    if total <= 0:
        return 0.0

    p = successes / total
    denominator = 1 + (z * z / total)

    centre = (
        p +
        (z * z / (2 * total))
    )

    adjustment = z * math.sqrt(
        (p * (1 - p) / total) +
        (z * z / (4 * total * total))
    )

    lower = (
        centre - adjustment
    ) / denominator

    return max(0.0, lower * 100)


def calculate_period_stats(
    leader_event,
    future_success,
    future_return,
    follower_eligible,
    period_mask
):
    """
    Belirli bir dönemde:
    - lider olay sayısını
    - takipçi başarı oranını
    - takipçinin normal başarı oranını
    - ek avantajı
    hesaplar.
    """

    valid_event = (
        leader_event &
        follower_eligible &
        future_success.notna() &
        period_mask
    )

    event_count = int(valid_event.sum())

    if event_count == 0:
        return None

    event_success_count = int(
        future_success[valid_event].sum()
    )

    event_success_rate = (
        event_success_count /
        event_count *
        100
    )

    successful_returns = future_return[
        valid_event &
        (future_success == 1)
    ]

    if successful_returns.empty:
        average_success_return = 0.0
    else:
        average_success_return = float(
            successful_returns.mean()
        )

    # Normal günlerde takipçinin aynı şartlar altında
    # yükselme olasılığı
    baseline_mask = (
        follower_eligible &
        future_success.notna() &
        period_mask
    )

    baseline_count = int(baseline_mask.sum())

    if baseline_count == 0:
        baseline_rate = 0.0
    else:
        baseline_rate = float(
            future_success[baseline_mask].mean() * 100
        )

    uplift = event_success_rate - baseline_rate

    confidence_lower = wilson_lower_bound(
        event_success_count,
        event_count,
        z=1.0
    )

    return {
        "events": event_count,
        "successes": event_success_count,
        "success_rate": event_success_rate,
        "baseline_rate": baseline_rate,
        "uplift": uplift,
        "avg_success_return": average_success_return,
        "confidence_lower": confidence_lower
    }


def analyze_leader_lag(returns_df):
    if returns_df.empty:
        return pd.DataFrame()

    latest_date = returns_df.index.max()
    test_start = latest_date - pd.Timedelta(days=TEST_DAYS)

    train_mask = pd.Series(
        returns_df.index < test_start,
        index=returns_df.index
    )

    test_mask = pd.Series(
        returns_df.index >= test_start,
        index=returns_df.index
    )

    print("Son veri tarihi:", latest_date.date())
    print("Test dönemi başlangıcı:", test_start.date())
    print("Hisse sayısı:", len(returns_df.columns))

    results = []
    symbols = list(returns_df.columns)

    # Lider olay matrisi
    leader_events = returns_df >= LEADER_GAIN

    for leader_number, leader in enumerate(symbols, start=1):
        print(
            f"[{leader_number}/{len(symbols)}] "
            f"Lider analiz ediliyor: {leader}"
        )

        leader_event = leader_events[leader].fillna(False)

        total_leader_events = int(leader_event.sum())

        if total_leader_events < (
            MIN_TRAIN_EVENTS + MIN_TEST_EVENTS
        ):
            continue

        for follower in symbols:
            if follower == leader:
                continue

            follower_return = returns_df[follower]

            # Takipçi lider gününden önce zaten hareketlenmişse
            # o olay değerlendirmeye alınmaz.
            previous_day_return = follower_return.shift(1)

            previous_3day_return = (
                (1 + follower_return / 100)
                .rolling(3)
                .apply(np.prod, raw=True)
                .shift(1)
                .sub(1)
                .mul(100)
            )

            follower_eligible = (
                previous_day_return.notna() &
                previous_3day_return.notna() &
                (previous_day_return < FOLLOWER_GAIN) &
                (previous_3day_return < 6.0)
            )

            for lag in LAGS:
                # Lider tarihi t iken takipçinin t+lag getirisi
                future_return = follower_return.shift(-lag)

                future_success = (
                    future_return >= FOLLOWER_GAIN
                ).astype(float)

                # Veri olmayan gelecek günleri yanlışlıkla
                # başarısız saymamak için tekrar NaN yap
                future_success = future_success.where(
                    future_return.notna(),
                    np.nan
                )

                train_stats = calculate_period_stats(
                    leader_event=leader_event,
                    future_success=future_success,
                    future_return=future_return,
                    follower_eligible=follower_eligible,
                    period_mask=train_mask
                )

                test_stats = calculate_period_stats(
                    leader_event=leader_event,
                    future_success=future_success,
                    future_return=future_return,
                    follower_eligible=follower_eligible,
                    period_mask=test_mask
                )

                if train_stats is None or test_stats is None:
                    continue

                if train_stats["events"] < MIN_TRAIN_EVENTS:
                    continue

                if test_stats["events"] < MIN_TEST_EVENTS:
                    continue

                if (
                    train_stats["success_rate"]
                    < MIN_TRAIN_SUCCESS
                ):
                    continue

                if (
                    test_stats["success_rate"]
                    < MIN_TEST_SUCCESS
                ):
                    continue

                if (
                    train_stats["uplift"]
                    < MIN_TRAIN_UPLIFT
                ):
                    continue

                if (
                    test_stats["uplift"]
                    < MIN_TEST_UPLIFT
                ):
                    continue

                # Eğitim ve test birlikte güçlü olmalı.
                # Test dönemi daha fazla ağırlık alıyor.
                validation_score = (
                    test_stats["confidence_lower"] * 0.45 +
                    test_stats["uplift"] * 0.30 +
                    train_stats["confidence_lower"] * 0.15 +
                    train_stats["uplift"] * 0.10
                )

                results.append({
                    "lider": leader,
                    "takipci": follower,
                    "gecikme_gun": lag,

                    "egitim_olay": train_stats["events"],
                    "egitim_basari": round(
                        train_stats["success_rate"], 2
                    ),
                    "egitim_normal": round(
                        train_stats["baseline_rate"], 2
                    ),
                    "egitim_avantaj": round(
                        train_stats["uplift"], 2
                    ),

                    "test_olay": test_stats["events"],
                    "test_basari": round(
                        test_stats["success_rate"], 2
                    ),
                    "test_normal": round(
                        test_stats["baseline_rate"], 2
                    ),
                    "test_avantaj": round(
                        test_stats["uplift"], 2
                    ),

                    "test_ort_getiri": round(
                        test_stats["avg_success_return"], 2
                    ),

                    "guven_alt_sinir": round(
                        test_stats["confidence_lower"], 2
                    ),

                    "dogrulama_skoru": round(
                        validation_score, 2
                    )
                })

    if not results:
        return pd.DataFrame()

    result_df = pd.DataFrame(results)

    # Aynı lider-takipçi çifti için yalnızca
    # en güçlü gecikmeyi bırak
    result_df = result_df.sort_values(
        by=[
            "dogrulama_skoru",
            "test_avantaj",
            "test_olay",
            "egitim_olay"
        ],
        ascending=False
    )

    result_df = result_df.drop_duplicates(
        subset=["lider", "takipci"],
        keep="first"
    )

    return result_df.reset_index(drop=True)


def diversify_results(result_df):
    """
    Tek bir takipçinin bütün listeyi doldurmasını engeller.
    """
    selected_rows = []
    follower_counts = defaultdict(int)
    leader_counts = defaultdict(int)

    for _, row in result_df.iterrows():
        follower = row["takipci"]
        leader = row["lider"]

        if follower_counts[follower] >= MAX_SAME_FOLLOWER:
            continue

        # Bir liderin de Telegram listesini çok doldurmasını önle
        if leader_counts[leader] >= 3:
            continue

        selected_rows.append(row)

        follower_counts[follower] += 1
        leader_counts[leader] += 1

        if len(selected_rows) >= TOP_RESULTS:
            break

    if not selected_rows:
        return pd.DataFrame()

    return pd.DataFrame(selected_rows)


def build_telegram_message(
    symbols_count,
    data_count,
    result_df,
    selected_df
):
    message = "🧠 BIST LEADER-LAG V2 SONUÇLARI\n\n"

    message += f"Toplam sembol: {symbols_count}\n"
    message += f"Yeterli verisi gelen: {data_count}\n"
    message += f"Filtreyi geçen ilişki: {len(result_df)}\n\n"

    message += (
        "Koşul: Lider en az %5 yükseliyor; "
        "takipçi 1–5 işlem günü sonra en az %3 yükseliyor.\n"
    )

    message += (
        "İlişki eski dönemde bulunup son 1 yılda "
        "ayrıca doğrulanmıştır.\n\n"
    )

    if selected_df.empty:
        message += (
            "Gerekli eğitim ve test şartlarını geçen "
            "güçlü ilişki bulunamadı."
        )
        return message

    message += "🏆 En güçlü doğrulanmış ilişkiler:\n\n"

    for _, row in selected_df.iterrows():
        message += (
            f"{row['lider']} ➜ {row['takipci']}\n"
            f"Gecikme: {int(row['gecikme_gun'])} işlem günü\n"
            f"Son 1 yıl başarı: %{row['test_basari']}\n"
            f"Normal başarı: %{row['test_normal']}\n"
            f"Ek avantaj: +{row['test_avantaj']} puan\n"
            f"Son 1 yıl olay: {int(row['test_olay'])}\n"
            f"Eski dönem: %{row['egitim_basari']} "
            f"/ {int(row['egitim_olay'])} olay\n"
            f"Başarılı gün ort. getiri: "
            f"%{row['test_ort_getiri']}\n"
            f"Doğrulama skoru: {row['dogrulama_skoru']}\n\n"
        )

    message += (
        "⚠️ İstatistiksel ilişki alım garantisi değildir. "
        "Likidite, haber ve piyasa koşulları ayrıca incelenmelidir."
    )

    return message


def main():
    send_message(
        "🧠 BIST LEADER-LAG V2 analizi başladı.\n"
        "Eski dönem + son 1 yıl doğrulaması yapılacak."
    )

    try:
        symbols = get_symbols()
    except Exception as exc:
        send_message(
            f"❌ BIST sembol listesi alınamadı:\n{exc}"
        )
        raise

    print("Toplam sembol:", len(symbols))

    returns_df = download_all_returns(symbols)

    if returns_df.empty:
        send_message(
            "❌ Leader-Lag V2 için fiyat verisi alınamadı."
        )
        return

    result_df = analyze_leader_lag(returns_df)

    if result_df.empty:
        send_message(
            "🧠 Leader-Lag V2 tamamlandı.\n\n"
            "Eğitim ve son 1 yıl test şartlarının tamamını "
            "geçen güçlü ilişki bulunamadı."
        )
        return

    # Bütün sonuçları CSV olarak kaydet
    result_df.to_csv(
        "leader_lag_v2_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    selected_df = diversify_results(result_df)

    message = build_telegram_message(
        symbols_count=len(symbols),
        data_count=len(returns_df.columns),
        result_df=result_df,
        selected_df=selected_df
    )

    send_message(message)

    print("\nEN GÜÇLÜ SONUÇLAR\n")
    print(
        selected_df.to_string(index=False)
        if not selected_df.empty
        else "Sonuç yok."
    )

    print(
        "\nCSV oluşturuldu: "
        "leader_lag_v2_results.csv"
    )


if __name__ == "__main__":
    main()
