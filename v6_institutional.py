from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ta

from v3_data import add_basic_columns


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def prepare_institutional_data(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Kurumsal birikim benzeri fiyat-hacim davranışlarını
    ölçmek için gerekli sütunları oluşturur.
    """
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = add_basic_columns(raw_df)

    if df.empty or len(df) < 120:
        return pd.DataFrame()

    close = pd.to_numeric(
        df["Close"],
        errors="coerce",
    )

    high = pd.to_numeric(
        df["High"],
        errors="coerce",
    )

    low = pd.to_numeric(
        df["Low"],
        errors="coerce",
    )

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce",
    )

    open_price = pd.to_numeric(
        df["Open"],
        errors="coerce",
    )

    # --------------------------------------------------------
    # TEMEL GÖSTERGELER
    # --------------------------------------------------------

    df["ema20"] = ta.trend.ema_indicator(
        close,
        window=20,
    )

    df["atr14"] = ta.volatility.average_true_range(
        high,
        low,
        close,
        window=14,
    )

    df["atr_pct"] = (
        df["atr14"] /
        close *
        100
    )

    df["atr_pct_avg_20"] = (
        df["atr_pct"]
        .rolling(20)
        .mean()
    )

    df["obv"] = ta.volume.on_balance_volume(
        close,
        volume,
    )

    df["cmf"] = ta.volume.chaikin_money_flow(
        high,
        low,
        close,
        volume,
        window=20,
    )

    df["mfi"] = ta.volume.money_flow_index(
        high,
        low,
        close,
        volume,
        window=14,
    )

    # --------------------------------------------------------
    # HACİM DAVRANIŞI
    # --------------------------------------------------------

    daily_change = close.pct_change(
        fill_method=None
    )

    green_day = daily_change > 0
    red_day = daily_change < 0

    df["green_volume"] = np.where(
        green_day,
        volume,
        0.0,
    )

    df["red_volume"] = np.where(
        red_day,
        volume,
        0.0,
    )

    df["green_volume_10"] = (
        pd.Series(
            df["green_volume"],
            index=df.index,
        )
        .rolling(10)
        .sum()
    )

    df["red_volume_10"] = (
        pd.Series(
            df["red_volume"],
            index=df.index,
        )
        .rolling(10)
        .sum()
    )

    df["green_red_volume_ratio"] = (
        df["green_volume_10"]
        /
        df["red_volume_10"].replace(
            0,
            np.nan,
        )
    )

    df["volume_avg_5"] = (
        volume.rolling(5).mean()
    )

    df["volume_avg_10"] = (
        volume.rolling(10).mean()
    )

    df["previous_volume_avg_20"] = (
        volume.shift(10)
        .rolling(20)
        .mean()
    )

    df["stable_volume_accumulation"] = (
        df["volume_avg_10"]
        /
        df["previous_volume_avg_20"]
    )

    # Tek günlük hacim ile 5 günlük ortalama hacim kıyası.
    df["single_day_volume_excess"] = (
        volume
        /
        df["volume_avg_5"]
    )

    # --------------------------------------------------------
    # FİYAT HENÜZ GİTMEDEN HACİM BİRİKİMİ
    # --------------------------------------------------------

    df["return_5d"] = (
        close.pct_change(
            periods=5,
            fill_method=None,
        ) * 100
    )

    df["return_10d"] = (
        close.pct_change(
            periods=10,
            fill_method=None,
        ) * 100
    )

    df["return_20d"] = (
        close.pct_change(
            periods=20,
            fill_method=None,
        ) * 100
    )

    df["price_volume_divergence"] = (
        (
            df["stable_volume_accumulation"] > 1.15
        )
        &
        (
            df["return_10d"] < 10
        )
    )

    # --------------------------------------------------------
    # MUM KALİTESİ VE DAĞITIM DAVRANIŞI
    # --------------------------------------------------------

    candle_range = high - low

    df["close_position"] = np.where(
        candle_range > 0,
        (
            close - low
        ) / candle_range,
        0.5,
    )

    df["upper_wick_ratio"] = np.where(
        candle_range > 0,
        (
            high -
            np.maximum(close, open_price)
        ) / candle_range,
        1.0,
    )

    df["body_ratio"] = np.where(
        candle_range > 0,
        np.abs(
            close - open_price
        ) / candle_range,
        0.0,
    )

    # Hacimli ama zayıf kapanan gün: dağıtım ihtimali.
    volume_avg_20 = volume.rolling(20).mean()

    df["distribution_day"] = (
        (
            volume >
            volume_avg_20 * 1.40
        )
        &
        (
            df["close_position"] < 0.40
        )
        &
        (
            daily_change < 0
        )
    )

    df["distribution_days_10"] = (
        df["distribution_day"]
        .astype(int)
        .rolling(10)
        .sum()
    )

    # Hacimli ve güçlü kapanan gün: talep ihtimali.
    df["demand_day"] = (
        (
            volume >
            volume_avg_20 * 1.15
        )
        &
        (
            df["close_position"] > 0.65
        )
        &
        (
            daily_change > 0
        )
    )

    df["demand_days_10"] = (
        df["demand_day"]
        .astype(int)
        .rolling(10)
        .sum()
    )

    # --------------------------------------------------------
    # OBV VE TREND
    # --------------------------------------------------------

    df["obv_ema10"] = ta.trend.ema_indicator(
        df["obv"],
        window=10,
    )

    df["obv_ema20"] = ta.trend.ema_indicator(
        df["obv"],
        window=20,
    )

    df["obv_slope_5"] = (
        (
            df["obv"]
            /
            df["obv"].shift(5)
        ) - 1
    ) * 100

    df["ema20_distance"] = (
        (
            close -
            df["ema20"]
        )
        /
        df["ema20"]
        * 100
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


def calculate_institutional_score(
    symbol: str,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    0-100 arasında kurumsal birikim benzeri davranış skoru.

    Bu skor kurum adı, takas veya gerçek emir defteri içermez.
    """
    df = prepare_institutional_data(
        raw_df
    )

    if df.empty:
        return {
            "symbol": symbol,
            "institutional_valid": False,
            "institutional_score": 0,
            "institutional_classification": (
                "YETERSİZ VERİ"
            ),
            "institutional_reasons": "",
            "institutional_risks": (
                "Yetersiz veri"
            ),
        }

    last = df.iloc[-1]

    score = 0
    reasons: List[str] = []
    risks: List[str] = []

    green_red_ratio = safe_float(
        last["green_red_volume_ratio"]
    )

    accumulation_ratio = safe_float(
        last["stable_volume_accumulation"]
    )

    single_day_excess = safe_float(
        last["single_day_volume_excess"]
    )

    return_5d = safe_float(
        last["return_5d"]
    )

    return_10d = safe_float(
        last["return_10d"]
    )

    return_20d = safe_float(
        last["return_20d"]
    )

    cmf = safe_float(
        last["cmf"]
    )

    mfi = safe_float(
        last["mfi"]
    )

    obv = safe_float(
        last["obv"]
    )

    obv_ema10 = safe_float(
        last["obv_ema10"]
    )

    obv_ema20 = safe_float(
        last["obv_ema20"]
    )

    close_position = safe_float(
        last["close_position"]
    )

    upper_wick_ratio = safe_float(
        last["upper_wick_ratio"]
    )

    atr_pct = safe_float(
        last["atr_pct"]
    )

    atr_pct_avg_20 = safe_float(
        last["atr_pct_avg_20"]
    )

    demand_days = safe_float(
        last["demand_days_10"]
    )

    distribution_days = safe_float(
        last["distribution_days_10"]
    )

    ema20_distance = safe_float(
        last["ema20_distance"]
    )

    price_volume_divergence = bool(
        last.get(
            "price_volume_divergence",
            False,
        )
    )

    # --------------------------------------------------------
    # 1. YÜKSELEN / DÜŞEN GÜN HACMİ — 20 PUAN
    # --------------------------------------------------------

    if green_red_ratio >= 1.80:
        score += 20
        reasons.append(
            "Yükselen gün hacmi çok güçlü"
        )

    elif green_red_ratio >= 1.40:
        score += 15
        reasons.append(
            "Yükselen gün hacmi baskın"
        )

    elif green_red_ratio >= 1.10:
        score += 8
        reasons.append(
            "Alım günleri hafif baskın"
        )

    elif green_red_ratio < 0.75:
        score -= 8
        risks.append(
            "Satış günleri hacmi baskın"
        )

    # --------------------------------------------------------
    # 2. İSTİKRARLI HACİM BİRİKİMİ — 20 PUAN
    # --------------------------------------------------------

    if 1.25 <= accumulation_ratio <= 2.20:
        score += 20
        reasons.append(
            "İstikrarlı hacim birikimi"
        )

    elif 1.10 <= accumulation_ratio < 1.25:
        score += 12
        reasons.append(
            "Hacim birikimi başlıyor"
        )

    elif accumulation_ratio > 2.20:
        score += 8
        risks.append(
            "Hacim artışı çok hızlı"
        )

    # --------------------------------------------------------
    # 3. FİYAT GİTMEDEN HACİM — 15 PUAN
    # --------------------------------------------------------

    if price_volume_divergence:
        score += 15
        reasons.append(
            "Fiyat gitmeden hacim artıyor"
        )

    elif (
        accumulation_ratio > 1.10
        and return_10d < 15
    ):
        score += 8

    # --------------------------------------------------------
    # 4. PARA AKIŞI — 15 PUAN
    # --------------------------------------------------------

    if cmf >= 0.15:
        score += 10
        reasons.append(
            "Chaikin para akışı güçlü"
        )

    elif cmf >= 0.05:
        score += 6
        reasons.append(
            "Chaikin para akışı pozitif"
        )

    elif cmf < -0.10:
        score -= 8
        risks.append(
            "Chaikin para akışı negatif"
        )

    if 50 <= mfi <= 75:
        score += 5
        reasons.append(
            "Para akış endeksi sağlıklı"
        )

    elif mfi > 85:
        score -= 5
        risks.append(
            "Para akış endeksi aşırı yüksek"
        )

    # --------------------------------------------------------
    # 5. OBV — 10 PUAN
    # --------------------------------------------------------

    if (
        obv > obv_ema10 >
        obv_ema20
    ):
        score += 10
        reasons.append(
            "OBV birikimi pozitif"
        )

    elif obv > obv_ema20:
        score += 5

    else:
        score -= 4
        risks.append(
            "OBV eğilimi zayıf"
        )

    # --------------------------------------------------------
    # 6. TALEP / DAĞITIM GÜNLERİ — 10 PUAN
    # --------------------------------------------------------

    demand_difference = (
        demand_days -
        distribution_days
    )

    if demand_difference >= 3:
        score += 10
        reasons.append(
            "Talep günleri belirgin baskın"
        )

    elif demand_difference >= 1:
        score += 5
        reasons.append(
            "Talep günleri satıştan fazla"
        )

    elif distribution_days >= 3:
        score -= 10
        risks.append(
            "Dağıtım günleri artmış"
        )

    # --------------------------------------------------------
    # 7. MUM VE VOLATİLİTE — 10 PUAN
    # --------------------------------------------------------

    if close_position >= 0.70:
        score += 5
        reasons.append(
            "Güçlü kapanış davranışı"
        )

    elif close_position < 0.35:
        score -= 6
        risks.append(
            "Kapanış günün alt bölümünde"
        )

    if upper_wick_ratio <= 0.25:
        score += 3

    elif upper_wick_ratio > 0.50:
        score -= 8
        risks.append(
            "Uzun üst fitil / satış baskısı"
        )

    if (
        atr_pct_avg_20 > 0
        and atr_pct <
        atr_pct_avg_20 * 0.90
    ):
        score += 2
        reasons.append(
            "Volatilite kontrollü daralıyor"
        )

    # --------------------------------------------------------
    # 8. PUMP / AŞIRI UZAKLIK CEZALARI
    # --------------------------------------------------------

    if (
        single_day_excess > 3
        and return_5d > 10
    ):
        score -= 15
        risks.append(
            "Tek günlük hacim patlaması riski"
        )

    if return_5d > 18:
        score -= 12
        risks.append(
            "5 günlük yükseliş fazla"
        )

    if return_20d > 35:
        score -= 12
        risks.append(
            "20 günlük yükseliş fazla"
        )

    if ema20_distance > 15:
        score -= 10
        risks.append(
            "EMA20'den fazla uzaklaşmış"
        )

    final_score = int(
        max(
            0,
            min(
                100,
                round(score),
            ),
        )
    )

    if final_score >= 80:
        classification = (
            "ÇOK GÜÇLÜ BİRİKİM"
        )

    elif final_score >= 65:
        classification = (
            "GÜÇLÜ BİRİKİM"
        )

    elif final_score >= 50:
        classification = (
            "ORTA BİRİKİM"
        )

    elif final_score >= 35:
        classification = (
            "ZAYIF BİRİKİM"
        )

    else:
        classification = (
            "BİRİKİM YOK"
        )

    return {
        "symbol": symbol,
        "institutional_valid": True,
        "institutional_score": final_score,
        "institutional_classification": (
            classification
        ),

        "institutional_green_red_ratio": round(
            green_red_ratio,
            2,
        ),

        "institutional_accumulation_ratio": round(
            accumulation_ratio,
            2,
        ),

        "institutional_cmf": round(
            cmf,
            3,
        ),

        "institutional_mfi": round(
            mfi,
            2,
        ),

        "institutional_demand_days": int(
            demand_days
        ),

        "institutional_distribution_days": int(
            distribution_days
        ),

        "institutional_reasons": (
            " | ".join(reasons)
            if reasons
            else "Belirgin birikim nedeni yok"
        ),

        "institutional_risks": (
            " | ".join(risks)
            if risks
            else "Belirgin kurumsal risk yok"
        ),
    }


def main():
    """
    Dosyanın doğrudan çalıştırılması için küçük test.
    """
    from v3_data import (
        download_bist_daily_data,
        get_bist_symbols,
    )

    symbols = get_bist_symbols()[:20]

    market_data, _ = (
        download_bist_daily_data(
            symbols=symbols,
            sleep_seconds=0.03,
        )
    )

    rows = []

    for symbol, dataframe in (
        market_data.items()
    ):
        result = calculate_institutional_score(
            symbol,
            dataframe,
        )

        rows.append(result)

    result_df = pd.DataFrame(rows)

    if result_df.empty:
        print(
            "Institutional Score sonucu oluşmadı."
        )
        return

    result_df = result_df.sort_values(
        by="institutional_score",
        ascending=False,
    )

    columns = [
        "symbol",
        "institutional_score",
        "institutional_classification",
        "institutional_green_red_ratio",
        "institutional_accumulation_ratio",
        "institutional_cmf",
        "institutional_mfi",
        "institutional_demand_days",
        "institutional_distribution_days",
        "institutional_reasons",
        "institutional_risks",
    ]

    print(
        result_df[columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
