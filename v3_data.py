import time
from io import StringIO
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from v3_config import (
    DAILY_INTERVAL,
    DAILY_PERIOD,
    MIN_AVG_TURNOVER_TL,
    MIN_DAILY_BARS,
    SYMBOL_SOURCE_URL,
)


REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    )
}


# ============================================================
# GENEL TEMİZLEME
# ============================================================

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo Finance verilerindeki MultiIndex sütunlarını temizler
    ve tarih sırasına koyar.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        result.columns = result.columns.get_level_values(0)

    result = result.loc[:, ~result.columns.duplicated()].copy()
    result.index = pd.to_datetime(result.index, errors="coerce")
    result = result[~result.index.isna()].sort_index()

    if getattr(result.index, "tz", None) is not None:
        result.index = result.index.tz_localize(None)

    return result


def normalize_symbol(symbol: str) -> str:
    """
    THYAO veya THYAO.IS girişini THYAO.IS biçimine getirir.
    """
    clean_symbol = str(symbol).strip().upper()

    if not clean_symbol:
        return ""

    if clean_symbol.endswith(".IS"):
        return clean_symbol

    return clean_symbol + ".IS"


def symbol_without_suffix(symbol: str) -> str:
    return normalize_symbol(symbol).replace(".IS", "")


# ============================================================
# BIST SEMBOL LİSTESİ
# ============================================================

def get_bist_symbols() -> List[str]:
    """
    BIST sembollerini dinamik olarak çeker.
    """
    response = requests.get(
        SYMBOL_SOURCE_URL,
        headers=REQUEST_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    tables = pd.read_html(StringIO(response.text))

    if not tables:
        raise RuntimeError("BIST sembol tablosu bulunamadı.")

    symbol_table = None

    for table in tables:
        if "Symbol" in table.columns:
            symbol_table = table
            break

    if symbol_table is None:
        raise RuntimeError(
            "BIST tablosunda Symbol sütunu bulunamadı."
        )

    raw_symbols = (
        symbol_table["Symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    symbols = []

    for raw_symbol in raw_symbols:
        symbol = normalize_symbol(raw_symbol)

        if not symbol:
            continue

        symbols.append(symbol)

    # Sıralamayı koruyarak tekrarları kaldır.
    return list(dict.fromkeys(symbols))


# ============================================================
# TEK HİSSE VERİSİ
# ============================================================

def download_daily_data(
    symbol: str,
    period: str = DAILY_PERIOD,
    interval: str = DAILY_INTERVAL,
    retries: int = 2,
) -> pd.DataFrame:
    """
    Tek hisse için günlük fiyat verisi indirir.
    Hata olursa belirlenen sayıda yeniden dener.
    """
    ticker = normalize_symbol(symbol)

    if not ticker:
        return pd.DataFrame()

    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 2):
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=False,
                actions=False,
                threads=False,
                timeout=25,
            )

            df = prepare_ohlcv_data(df)

            if not df.empty:
                return df

        except Exception as exc:
            last_error = exc
            print(
                f"{ticker} veri denemesi "
                f"{attempt}/{retries + 1} başarısız: {exc}"
            )

        if attempt <= retries:
            time.sleep(1.5 * attempt)

    if last_error is not None:
        print(f"{ticker} veri alınamadı: {last_error}")

    return pd.DataFrame()


def prepare_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV verisini sayısal hale getirir ve bozuk satırları temizler.
    """
    result = clean_dataframe(df)

    if result.empty:
        return pd.DataFrame()

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
    ]

    if not all(column in result.columns for column in required_columns):
        return pd.DataFrame()

    result = result[required_columns].copy()

    for column in required_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = result.dropna(
        subset=["High", "Low", "Close", "Volume"]
    )

    result = result[
        (result["Close"] > 0)
        & (result["High"] > 0)
        & (result["Low"] > 0)
        & (result["Volume"] >= 0)
    ]

    return result


# ============================================================
# VERİ KALİTE VE LİKİDİTE KONTROLÜ
# ============================================================

def calculate_average_turnover(
    df: pd.DataFrame,
    window: int = 20,
) -> float:
    """
    Yaklaşık günlük TL işlem hacmini hesaplar:
    kapanış fiyatı × adet hacmi.
    """
    if df.empty or len(df) < window:
        return 0.0

    turnover = (
        df["Close"].astype(float)
        * df["Volume"].astype(float)
    )

    average_turnover = turnover.tail(window).mean()

    if pd.isna(average_turnover):
        return 0.0

    return float(average_turnover)


def validate_daily_data(
    df: pd.DataFrame,
) -> Tuple[bool, str]:
    """
    Verinin V3 taramasına uygun olup olmadığını kontrol eder.
    """
    if df.empty:
        return False, "veri_yok"

    if len(df) < MIN_DAILY_BARS:
        return False, "gecmis_kisa"

    recent_close = df["Close"].tail(20)

    if recent_close.isna().any():
        return False, "eksik_fiyat"

    if float(recent_close.iloc[-1]) <= 0:
        return False, "gecersiz_fiyat"

    average_turnover = calculate_average_turnover(df)

    if average_turnover < MIN_AVG_TURNOVER_TL:
        return False, "dusuk_likidite"

    return True, "uygun"


# ============================================================
# TÜM BIST VERİLERİNİ TOPLAMA
# ============================================================

def download_bist_daily_data(
    symbols: Optional[List[str]] = None,
    sleep_seconds: float = 0.03,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Bütün BIST hisselerini indirir.

    Dönüş:
    1. Uygun hisselerin veri sözlüğü
    2. Veri kalite raporu
    """
    if symbols is None:
        symbols = get_bist_symbols()

    valid_data: Dict[str, pd.DataFrame] = {}
    report_rows = []

    total = len(symbols)

    for index, raw_symbol in enumerate(symbols, start=1):
        symbol = normalize_symbol(raw_symbol)

        print(
            f"[{index}/{total}] "
            f"V3 günlük veri: {symbol}"
        )

        try:
            df = download_daily_data(symbol)
            is_valid, reason = validate_daily_data(df)

            average_turnover = calculate_average_turnover(df)

            report_rows.append({
                "symbol": symbol_without_suffix(symbol),
                "downloaded_bars": len(df),
                "average_turnover_tl": round(
                    average_turnover,
                    2,
                ),
                "valid": is_valid,
                "reason": reason,
            })

            if is_valid:
                valid_data[
                    symbol_without_suffix(symbol)
                ] = df

        except Exception as exc:
            print(f"{symbol} genel veri hatası: {exc}")

            report_rows.append({
                "symbol": symbol_without_suffix(symbol),
                "downloaded_bars": 0,
                "average_turnover_tl": 0.0,
                "valid": False,
                "reason": f"hata:{type(exc).__name__}",
            })

        time.sleep(sleep_seconds)

    report_df = pd.DataFrame(report_rows)

    return valid_data, report_df


# ============================================================
# HAZIR GÖSTERGELER
# ============================================================

def add_basic_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    V3 tarayıcılarında tekrar tekrar kullanılacak temel sütunları ekler.
    """
    if df.empty:
        return pd.DataFrame()

    result = df.copy()

    result["daily_return_pct"] = (
        result["Close"].pct_change(fill_method=None) * 100
    )

    result["turnover_tl"] = (
        result["Close"] * result["Volume"]
    )

    result["volume_avg_20"] = (
        result["Volume"].rolling(20).mean()
    )

    result["turnover_avg_20"] = (
        result["turnover_tl"].rolling(20).mean()
    )

    result["volume_ratio_20"] = (
        result["Volume"]
        / result["volume_avg_20"]
    )

    result["high_20"] = (
        result["High"].rolling(20).max()
    )

    result["low_20"] = (
        result["Low"].rolling(20).min()
    )

    result["range_20_pct"] = (
        (
            result["high_20"] -
            result["low_20"]
        )
        / result["low_20"]
        * 100
    )

    candle_range = (
        result["High"] -
        result["Low"]
    )

    result["close_position"] = np.where(
        candle_range > 0,
        (
            result["Close"] -
            result["Low"]
        )
        / candle_range,
        0.5,
    )

    result["upper_wick_ratio"] = np.where(
        candle_range > 0,
        (
            result["High"] -
            result["Close"]
        )
        / candle_range,
        1.0,
    )

    result = result.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return result


# ============================================================
# DOĞRUDAN TEST
# ============================================================

def main():
    print("V3 veri motoru testi başladı.")

    try:
        symbols = get_bist_symbols()
    except Exception as exc:
        print("Sembol listesi alınamadı:", exc)
        return

    print("Toplam sembol:", len(symbols))

    # İlk testte tüm BIST yerine ilk 5 hisseyi kontrol eder.
    test_symbols = symbols[:5]

    data, report = download_bist_daily_data(
        symbols=test_symbols,
        sleep_seconds=0.05,
    )

    print("\nV3 VERİ RAPORU")
    print(report.to_string(index=False))

    print("\nUygun veri gelen hisseler:")
    print(list(data.keys()))

    for symbol, df in data.items():
        enriched = add_basic_columns(df)

        print(
            f"\n{symbol} son veri:"
        )

        columns = [
            "Close",
            "Volume",
            "volume_ratio_20",
            "range_20_pct",
            "close_position",
            "upper_wick_ratio",
        ]

        print(
            enriched[columns]
            .tail(1)
            .to_string()
        )


if __name__ == "__main__":
    main()
