from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import ta

from v3_config import (
    ELITE_SCORE,
    MAX_EMA20_DISTANCE,
    MAX_HEALTHY_RSI,
    MAX_RANGE_20D,
    MAX_RETURN_20D,
    MAX_RETURN_5D,
    MAX_RSI,
    MAX_UPPER_WICK_RATIO,
    MIN_CLOSE_POSITION,
    MIN_HEALTHY_RSI,
    MIN_SMART_MONEY_SCORE,
    MIN_VOLUME_RATIO,
    STRONG_COMPRESSION_RANGE,
    STRONG_VOLUME_RATIO,
)
from v3_data import (
    add_basic_columns,
    download_bist_daily_data,
    get_bist_symbols,
)


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)

        if np.isnan(number) or np.isinf(number):
            return default

        return number

    except (TypeError, ValueError):
        return default


def percentage_return(
    series: pd.Series,
    periods: int,
) -> float:
    if len(series) <= periods:
        return 0.0

    old_value = safe_float(series.iloc[-periods - 1])
    new_value = safe_float(series.iloc[-1])

    if old_value <= 0:
        return 0.0

    return ((new_value / old_value) - 1) * 100


def format_reasons(reasons: List[str]) -> str:
    if not reasons:
        return "Belirgin güçlü neden yok"

    return " | ".join(reasons)


def format_risks(risks: List[str]) -> str:
    if not risks:
        return "Belirgin risk yok"

    return " | ".join(risks)


# ============================================================
# GÖSTERGE HAZIRLAMA
# ============================================================

def prepare_scoring_data(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:
    if raw_df is None or raw_df.empty:
        return pd.DataFrame()

    df = add_basic_columns(raw_df)

    if df.empty or len(df) < 120:
        return pd.DataFrame()

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    df["ema_10"] = ta.trend.ema_indicator(
        close,
        window=10,
    )

    df["ema_20"] = ta.trend.ema_indicator(
        close,
        window=20,
    )

    df["ema_50"] = ta.trend.ema_indicator(
        close,
        window=50,
    )

    df["rsi_14"] = ta.momentum.rsi(
        close,
        window=14,
    )

    df["atr_14"] = ta.volatility.average_true_range(
        high,
        low,
        close,
        window=14,
    )

    df["atr_pct"] = (
        df["atr_14"] /
        df["Close"] *
        100
    )

    df["volume_avg_5"] = (
        volume.rolling(5).mean()
    )

    df["volume_avg_10"] = (
        volume.rolling(10).mean()
    )

    df["previous_volume_avg_20"] = (
        volume.shift(10).rolling(20).mean()
    )

    df["volume_accumulation_ratio"] = (
        df["volume_avg_10"] /
        df["previous_volume_avg_20"]
    )

    # Fiyat yükselen günlerdeki hacim ile
    # fiyat düşen günlerdeki hacmin karşılaştırılması.
    daily_change = close.diff()

    df["up_volume"] = np.where(
        daily_change > 0,
        volume,
        0.0,
    )

    df["down_volume"] = np.where(
        daily_change < 0,
        volume,
        0.0,
    )

    df["up_volume_10"] = (
        pd.Series(
            df["up_volume"],
            index=df.index,
        )
        .rolling(10)
        .sum()
    )

    df["down_volume_10"] = (
        pd.Series(
            df["down_volume"],
            index=df.index,
        )
        .rolling(10)
        .sum()
    )

    df["up_down_volume_ratio"] = (
        df["up_volume_10"] /
        df["down_volume_10"].replace(0, np.nan)
    )

    # Son 20 günlük zirve, bugünkü mum hariç.
    df["previous_high_20"] = (
        high.shift(1).rolling(20).max()
    )

    df["distance_to_high_20"] = (
        (
            df["previous_high_20"] -
            close
        )
        / df["previous_high_20"]
        * 100
    )

    df["ema20_distance"] = (
        (
            close -
            df["ema_20"]
        )
        / df["ema_20"]
        * 100
    )

    df["ema20_slope_5"] = (
        (
            df["ema_20"] /
            df["ema_20"].shift(5)
        ) - 1
    ) * 100

    # Sıkışmanın önceki döneme göre iyileşip
    # iyileşmediğini ölçer.
    df["range_20_previous"] = (
        df["range_20_pct"].shift(10)
    )

    df["compression_improving"] = (
        df["range_20_pct"] <
        df["range_20_previous"]
    )

    # Hacim yükselirken fiyatın henüz fazla gitmediği durum.
    df["quiet_accumulation"] = (
        (df["volume_accumulation_ratio"] > 1.15)
        & (
            close.pct_change(
                periods=10,
                fill_method=None,
            ) * 100 < 12
        )
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return df


# ============================================================
# SMART MONEY BENZERİ PUANLAMA
# ============================================================

def calculate_smart_money_score(
    symbol: str,
    raw_df: pd.DataFrame,
) -> Dict[str, Any]:
    df = prepare_scoring_data(raw_df)

    if df.empty:
        return {
            "symbol": symbol,
            "valid": False,
            "reason": "yetersiz_veri",
        }

    last = df.iloc[-1]

    close = safe_float(last["Close"])
    volume = safe_float(last["Volume"])

    ema10 = safe_float(last["ema_10"])
    ema20 = safe_float(last["ema_20"])
    ema50 = safe_float(last["ema_50"])

    rsi = safe_float(last["rsi_14"])
    atr_pct = safe_float(last["atr_pct"])

    volume_ratio = safe_float(
        last["volume_ratio_20"]
    )

    volume_accumulation = safe_float(
        last["volume_accumulation_ratio"]
    )

    up_down_volume_ratio = safe_float(
        last["up_down_volume_ratio"]
    )

    range_20 = safe_float(
        last["range_20_pct"]
    )

    close_position = safe_float(
        last["close_position"]
    )

    upper_wick_ratio = safe_float(
        last["upper_wick_ratio"]
    )

    distance_to_high = safe_float(
        last["distance_to_high_20"]
    )

    ema20_distance = safe_float(
        last["ema20_distance"]
    )

    ema20_slope = safe_float(
        last["ema20_slope_5"]
    )

    average_turnover = safe_float(
        last["turnover_avg_20"]
    )

    compression_improving = bool(
        last.get("compression_improving", False)
    )

    quiet_accumulation = bool(
        last.get("quiet_accumulation", False)
    )

    return_1d = percentage_return(
        df["Close"],
        1,
    )

    return_5d = percentage_return(
        df["Close"],
        5,
    )

    return_10d = percentage_return(
        df["Close"],
        10,
    )

    return_20d = percentage_return(
        df["Close"],
        20,
    )

    score = 0
    positive_reasons: List[str] = []
    risks: List[str] = []

    component_scores = {
        "trend_score": 0,
        "volume_score": 0,
        "compression_score": 0,
        "candle_score": 0,
        "momentum_score": 0,
        "liquidity_score": 0,
        "risk_penalty": 0,
    }

    # --------------------------------------------------------
    # 1. TREND PUANI — MAKSİMUM 20
    # --------------------------------------------------------

    if close > ema20:
        score += 6
        component_scores["trend_score"] += 6
        positive_reasons.append("EMA20 üzerinde")

    if ema20 > ema50:
        score += 6
        component_scores["trend_score"] += 6
        positive_reasons.append("Orta vadeli trend pozitif")

    if ema10 > ema20:
        score += 4
        component_scores["trend_score"] += 4

    if ema20_slope > 0:
        score += 4
        component_scores["trend_score"] += 4
        positive_reasons.append("EMA20 eğimi yukarı")

    # --------------------------------------------------------
    # 2. HACİM VE BİRİKİM — MAKSİMUM 30
    # --------------------------------------------------------

    if volume_ratio >= STRONG_VOLUME_RATIO:
        score += 10
        component_scores["volume_score"] += 10
        positive_reasons.append(
            f"Güçlü hacim {volume_ratio:.2f}x"
        )

    elif volume_ratio >= MIN_VOLUME_RATIO:
        score += 6
        component_scores["volume_score"] += 6
        positive_reasons.append(
            f"Hacim ortalamanın üzerinde {volume_ratio:.2f}x"
        )

    if volume_accumulation >= 1.35:
        score += 8
        component_scores["volume_score"] += 8
        positive_reasons.append("10 günlük hacim birikimi güçlü")

    elif volume_accumulation >= 1.15:
        score += 5
        component_scores["volume_score"] += 5
        positive_reasons.append("Hacim birikimi var")

    if up_down_volume_ratio >= 1.50:
        score += 7
        component_scores["volume_score"] += 7
        positive_reasons.append("Yükselen gün hacmi baskın")

    elif up_down_volume_ratio >= 1.15:
        score += 4
        component_scores["volume_score"] += 4

    if quiet_accumulation:
        score += 5
        component_scores["volume_score"] += 5
        positive_reasons.append(
            "Fiyat gitmeden sessiz hacim artışı"
        )

    # --------------------------------------------------------
    # 3. SIKIŞMA — MAKSİMUM 15
    # --------------------------------------------------------

    if range_20 <= STRONG_COMPRESSION_RANGE:
        score += 10
        component_scores["compression_score"] += 10
        positive_reasons.append(
            f"Güçlü sıkışma %{range_20:.1f}"
        )

    elif range_20 <= MAX_RANGE_20D:
        score += 6
        component_scores["compression_score"] += 6
        positive_reasons.append(
            f"Kontrollü bant %{range_20:.1f}"
        )

    if compression_improving:
        score += 5
        component_scores["compression_score"] += 5
        positive_reasons.append("Volatilite daralıyor")

    # --------------------------------------------------------
    # 4. MUM KALİTESİ — MAKSİMUM 10
    # --------------------------------------------------------

    if close_position >= 0.75:
        score += 6
        component_scores["candle_score"] += 6
        positive_reasons.append("Güçlü kapanış")

    elif close_position >= MIN_CLOSE_POSITION:
        score += 4
        component_scores["candle_score"] += 4

    if upper_wick_ratio <= 0.20:
        score += 4
        component_scores["candle_score"] += 4

    elif upper_wick_ratio <= MAX_UPPER_WICK_RATIO:
        score += 2
        component_scores["candle_score"] += 2

    # --------------------------------------------------------
    # 5. MOMENTUM — MAKSİMUM 20
    # --------------------------------------------------------

    if MIN_HEALTHY_RSI <= rsi <= MAX_HEALTHY_RSI:
        score += 7
        component_scores["momentum_score"] += 7
        positive_reasons.append(
            f"RSI sağlıklı bölgede {rsi:.1f}"
        )

    if 0 < return_5d <= 8:
        score += 5
        component_scores["momentum_score"] += 5
        positive_reasons.append(
            "Erken momentum"
        )

    elif -3 <= return_5d <= 0:
        score += 2
        component_scores["momentum_score"] += 2

    if 0 <= distance_to_high <= 5:
        score += 5
        component_scores["momentum_score"] += 5
        positive_reasons.append(
            "20 günlük dirence yakın"
        )

    elif 5 < distance_to_high <= 10:
        score += 3
        component_scores["momentum_score"] += 3

    if 0 < return_20d < 20:
        score += 3
        component_scores["momentum_score"] += 3

    # --------------------------------------------------------
    # 6. LİKİDİTE — MAKSİMUM 5
    # --------------------------------------------------------

    if average_turnover >= 100_000_000:
        score += 5
        component_scores["liquidity_score"] += 5

    elif average_turnover >= 25_000_000:
        score += 3
        component_scores["liquidity_score"] += 3

    else:
        score += 1
        component_scores["liquidity_score"] += 1

    # --------------------------------------------------------
    # 7. RİSK CEZALARI
    # --------------------------------------------------------

    penalty = 0

    if return_5d > MAX_RETURN_5D:
        penalty += 15
        risks.append(
            f"5 günde fazla yükselmiş %{return_5d:.1f}"
        )

    if return_20d > MAX_RETURN_20D:
        penalty += 15
        risks.append(
            f"20 günde fazla yükselmiş %{return_20d:.1f}"
        )

    if ema20_distance > MAX_EMA20_DISTANCE:
        penalty += 12
        risks.append(
            f"EMA20'den uzak %{ema20_distance:.1f}"
        )

    if rsi > MAX_RSI:
        penalty += 12
        risks.append(
            f"RSI yüksek {rsi:.1f}"
        )

    if upper_wick_ratio > 0.50:
        penalty += 10
        risks.append("Uzun üst fitil")

    if close_position < 0.35:
        penalty += 8
        risks.append("Zayıf kapanış")

    if volume_ratio > 3.5 and return_1d > 7:
        penalty += 12
        risks.append("Tek günlük hacim/pump riski")

    if atr_pct > 8:
        penalty += 8
        risks.append(
            f"Volatilite yüksek %{atr_pct:.1f}"
        )

    if return_1d < -5:
        penalty += 8
        risks.append("Günlük sert satış")

    score -= penalty
    component_scores["risk_penalty"] = -penalty

    final_score = int(
        max(0, min(100, round(score)))
    )

    if final_score >= ELITE_SCORE:
        classification = "ELİT ADAY"

    elif final_score >= MIN_SMART_MONEY_SCORE:
        classification = "GÜÇLÜ ADAY"

    elif final_score >= 50:
        classification = "İZLEME ADAYI"

    else:
        classification = "ZAYIF"

    eligible = (
        final_score >= MIN_SMART_MONEY_SCORE
        and return_5d <= MAX_RETURN_5D
        and return_20d <= MAX_RETURN_20D
        and ema20_distance <= MAX_EMA20_DISTANCE
        and rsi <= MAX_RSI
        and upper_wick_ratio <= 0.50
        and close_position >= 0.35
    )

    return {
        "symbol": symbol,
        "valid": True,
        "eligible": eligible,
        "classification": classification,
        "smart_money_score": final_score,

        "close": round(close, 4),
        "volume": int(volume),
        "average_turnover_tl": round(
            average_turnover,
            2,
        ),

        "rsi": round(rsi, 2),
        "atr_pct": round(atr_pct, 2),
        "ema20_distance": round(
            ema20_distance,
            2,
        ),

        "volume_ratio": round(
            volume_ratio,
            2,
        ),
        "volume_accumulation_ratio": round(
            volume_accumulation,
            2,
        ),
        "up_down_volume_ratio": round(
            up_down_volume_ratio,
            2,
        ),

        "range_20_pct": round(
            range_20,
            2,
        ),
        "close_position": round(
            close_position,
            2,
        ),
        "upper_wick_ratio": round(
            upper_wick_ratio,
            2,
        ),
        "distance_to_high_20": round(
            distance_to_high,
            2,
        ),

        "return_1d": round(
            return_1d,
            2,
        ),
        "return_5d": round(
            return_5d,
            2,
        ),
        "return_10d": round(
            return_10d,
            2,
        ),
        "return_20d": round(
            return_20d,
            2,
        ),

        "positive_reasons": format_reasons(
            positive_reasons
        ),
        "risk_reasons": format_risks(
            risks
        ),

        **component_scores,
    }


# ============================================================
# BİRDEN FAZLA HİSSEYİ PUANLAMA
# ============================================================

def score_market(
    market_data: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []

    total = len(market_data)

    for number, (symbol, df) in enumerate(
        market_data.items(),
        start=1,
    ):
        print(
            f"[{number}/{total}] "
            f"V3 puanlama: {symbol}"
        )

        try:
            result = calculate_smart_money_score(
                symbol,
                df,
            )

            if result.get("valid"):
                rows.append(result)

        except Exception as exc:
            print(
                f"{symbol} puanlama hatası: {exc}"
            )

    if not rows:
        return pd.DataFrame()

    result_df = pd.DataFrame(rows)

    result_df = result_df.sort_values(
        by=[
            "eligible",
            "smart_money_score",
            "volume_score",
            "trend_score",
            "compression_score",
        ],
        ascending=False,
    )

    return result_df.reset_index(drop=True)


# ============================================================
# TEST
# ============================================================

def main():
    print("V3 puanlama motoru testi başladı.")

    symbols = get_bist_symbols()

    # İlk testte yalnızca ilk 15 sembol.
    test_symbols = symbols[:15]

    market_data, quality_report = (
        download_bist_daily_data(
            symbols=test_symbols,
            sleep_seconds=0.03,
        )
    )

    print("\nVERİ KALİTE RAPORU")
    print(
        quality_report.to_string(index=False)
    )

    scored = score_market(market_data)

    if scored.empty:
        print(
            "\nPuanlanabilecek uygun hisse bulunamadı."
        )
        return

    display_columns = [
        "symbol",
        "smart_money_score",
        "classification",
        "eligible",
        "close",
        "rsi",
        "volume_ratio",
        "volume_accumulation_ratio",
        "range_20_pct",
        "return_5d",
        "return_20d",
        "positive_reasons",
        "risk_reasons",
    ]

    print("\nV3 PUANLAMA SONUÇLARI")

    print(
        scored[display_columns]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
