from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# DOSYA YOLLARI
# ============================================================

MARKET_FILE = Path("v16_full_market_snapshot.csv")
CANDIDATE_FILE = Path("v16_relative_strength.csv")

RESULT_FILE = Path("v17_regime_adjusted_decisions.csv")
STATUS_FILE = Path("v17_market_regime_status.json")


# ============================================================
# SONUÇ DOSYASI SÜTUNLARI
# ============================================================

RESULT_COLUMNS = [
    "rank",
    "symbol",
    "close",
    "regime",
    "regime_confidence",
    "market_rank",
    "market_percentile",
    "relative_strength_score",
    "relative_class",
    "momentum_percentile",
    "trend_percentile",
    "volume_percentile",
    "quality_percentile",
    "v15_score",
    "v15_decision",
    "regime_adjustment",
    "v17_score",
    "v17_decision",
    "regime_reasons",
]


# ============================================================
# TEMEL YARDIMCI FONKSİYONLAR
# ============================================================

def text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (TypeError, ValueError):
        return default


def integer(
    value: Any,
    default: int = 0,
) -> int:
    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return int(result)

    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(
            f"UYARI: {path} bulunamadı."
        )
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            print(
                f"UYARI: {path} tamamen boş."
            )
            return pd.DataFrame()
    except OSError as exc:
        print(
            f"UYARI: {path} kontrol edilemedi: {exc}"
        )
        return pd.DataFrame()

    encodings = [
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ]

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                path,
                encoding=encoding,
            )

            return dataframe

        except pd.errors.EmptyDataError:
            print(
                f"UYARI: {path} içinde okunabilir "
                "sütun veya satır bulunamadı."
            )
            return pd.DataFrame()

        except UnicodeDecodeError:
            continue

        except pd.errors.ParserError as exc:
            print(
                f"UYARI: {path} ayrıştırılamadı: "
                f"{exc}"
            )
            return pd.DataFrame()

        except Exception as exc:
            print(
                f"UYARI: {path} okunamadı: {exc}"
            )
            return pd.DataFrame()

    print(
        f"UYARI: {path} desteklenen "
        "kodlamalarla okunamadı."
    )

    return pd.DataFrame()


def save_status(
    status: dict[str, Any],
) -> None:
    STATUS_FILE.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_empty_result() -> pd.DataFrame:
    result = pd.DataFrame(
        columns=RESULT_COLUMNS
    )

    result.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return result


# ============================================================
# PİYASA REJİMİ HESAPLAMA
# ============================================================

def positive_ratio(
    series: pd.Series,
) -> float:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    return float(
        (values > 0).mean() * 100
    )


def detect_regime(
    market: pd.DataFrame,
) -> dict[str, Any]:
    numeric_columns = [
        "return_1d",
        "return_5d",
        "return_20d",
        "ema20_distance",
        "rsi",
        "volume_ratio",
    ]

    market = market.copy()

    for column in numeric_columns:
        if column not in market.columns:
            market[column] = np.nan

        market[column] = pd.to_numeric(
            market[column],
            errors="coerce",
        )

    breadth_1d = positive_ratio(
        market["return_1d"]
    )

    breadth_5d = positive_ratio(
        market["return_5d"]
    )

    breadth_20d = positive_ratio(
        market["return_20d"]
    )

    above_ema20 = positive_ratio(
        market["ema20_distance"]
    )

    median_1d = number(
        market["return_1d"].median()
    )

    median_5d = number(
        market["return_5d"].median()
    )

    median_20d = number(
        market["return_20d"].median()
    )

    median_rsi = number(
        market["rsi"].median(),
        50.0,
    )

    if (
        breadth_1d <= 32
        or median_1d <= -2
    ):
        regime = "PANİK"

        confidence = min(
            100.0,
            60
            + (50 - breadth_1d) * 1.2
            + abs(median_1d) * 6,
        )

    elif (
        breadth_1d >= 68
        and breadth_5d >= 60
        and median_5d > 1.5
    ):
        regime = "RALLİ"

        confidence = min(
            100.0,
            60
            + (breadth_1d - 60)
            + (breadth_5d - 55) * 0.5,
        )

    elif (
        breadth_20d >= 58
        and above_ema20 >= 58
        and median_20d > 2
    ):
        regime = "TREND"

        confidence = min(
            100.0,
            60
            + (breadth_20d - 55) * 0.8
            + (above_ema20 - 55) * 0.8,
        )

    else:
        regime = "YATAY"

        confidence = max(
            50.0,
            82
            - abs(breadth_1d - 50) * 0.5
            - abs(breadth_5d - 50) * 0.4,
        )

    return {
        "regime": regime,
        "regime_confidence": round(
            confidence,
            2,
        ),
        "market_count": int(
            len(market)
        ),
        "breadth_1d_positive_pct": round(
            breadth_1d,
            2,
        ),
        "breadth_5d_positive_pct": round(
            breadth_5d,
            2,
        ),
        "breadth_20d_positive_pct": round(
            breadth_20d,
            2,
        ),
        "above_ema20_pct": round(
            above_ema20,
            2,
        ),
        "median_return_1d": round(
            median_1d,
            2,
        ),
        "median_return_5d": round(
            median_5d,
            2,
        ),
        "median_return_20d": round(
            median_20d,
            2,
        ),
        "median_rsi": round(
            median_rsi,
            2,
        ),
        "comparison_scope": "FULL_MARKET",
    }


# ============================================================
# REJİM AYARLAMASI
# ============================================================

def adjustment(
    row: pd.Series,
    regime: str,
) -> tuple[float, str]:
    momentum = number(
        row.get("momentum_percentile"),
        50.0,
    )

    trend = number(
        row.get("trend_percentile"),
        50.0,
    )

    volume = number(
        row.get("volume_percentile"),
        50.0,
    )

    quality = number(
        row.get("quality_percentile"),
        50.0,
    )

    market_percentile = number(
        row.get("market_percentile"),
        50.0,
    )

    score = 0.0
    reasons: list[str] = []

    if regime == "RALLİ":
        if momentum >= 75:
            score += 5
            reasons.append(
                "Ralli rejiminde güçlü momentum"
            )

        if volume >= 70:
            score += 3
            reasons.append(
                "Ralli rejiminde hacim desteği"
            )

        if market_percentile < 55:
            score -= 4
            reasons.append(
                "Ralli rejiminde piyasa gerisinde"
            )

    elif regime == "TREND":
        if trend >= 75:
            score += 5
            reasons.append(
                "Trend rejiminde güçlü trend"
            )

        if quality >= 70:
            score += 3
            reasons.append(
                "Trend rejiminde kalite desteği"
            )

    elif regime == "YATAY":
        if quality >= 70:
            score += 4
            reasons.append(
                "Yatay rejimde kalite avantajı"
            )

        if volume >= 70:
            score += 2
            reasons.append(
                "Yatay rejimde hacim birikimi"
            )

        if momentum >= 90:
            score -= 3
            reasons.append(
                "Yatay rejimde aşırı momentum"
            )

    elif regime == "PANİK":
        score -= 8
        reasons.append(
            "Panik rejimi genel risk kesintisi"
        )

        if quality >= 80:
            score += 3
            reasons.append(
                "Panik rejiminde yüksek kalite"
            )

        if market_percentile >= 90:
            score += 2
            reasons.append(
                "Panik rejiminde piyasa liderliği"
            )

    if not reasons:
        reasons.append(
            "Rejim kaynaklı ek puan değişimi yok"
        )

    return (
        round(score, 2),
        " | ".join(reasons),
    )


# ============================================================
# V17 KARAR SINIFLANDIRMASI
# ============================================================

def classify(
    score: float,
    regime: str,
) -> str:
    thresholds = {
        "PANİK": (
            86.0,
            77.0,
            64.0,
        ),
        "RALLİ": (
            77.0,
            67.0,
            56.0,
        ),
        "TREND": (
            78.0,
            68.0,
            57.0,
        ),
        "YATAY": (
            80.0,
            70.0,
            58.0,
        ),
    }

    strong, approved, cautious = (
        thresholds.get(
            regime,
            thresholds["YATAY"],
        )
    )

    if score >= strong:
        return "V17 GÜÇLÜ ONAY"

    if score >= approved:
        return "V17 ONAYLI İZLEME"

    if score >= cautious:
        return "V17 TEMKİNLİ İZLEME"

    return "V17 ELE"


# ============================================================
# ADAY İŞLEME
# ============================================================

def build_result_row(
    row: pd.Series,
    status: dict[str, Any],
) -> dict[str, Any]:
    regime = text(
        status.get("regime")
    ) or "YATAY"

    regime_effect, reasons = adjustment(
        row,
        regime,
    )

    v15_score = number(
        row.get("v15_score")
    )

    relative_score = number(
        row.get(
            "relative_strength_score"
        )
    )

    v17_score = float(
        np.clip(
            v15_score * 0.72
            + relative_score * 0.28
            + regime_effect,
            0,
            100,
        )
    )

    return {
        "symbol": text(
            row.get("symbol")
        ),
        "close": round(
            number(
                row.get("close")
            ),
            4,
        ),
        "regime": regime,
        "regime_confidence": number(
            status.get(
                "regime_confidence"
            )
        ),
        "market_rank": integer(
            row.get("market_rank")
        ),
        "market_percentile": round(
            number(
                row.get(
                    "market_percentile"
                )
            ),
            2,
        ),
        "relative_strength_score": round(
            relative_score,
            2,
        ),
        "relative_class": text(
            row.get("relative_class")
        ),
        "momentum_percentile": round(
            number(
                row.get(
                    "momentum_percentile"
                )
            ),
            2,
        ),
        "trend_percentile": round(
            number(
                row.get(
                    "trend_percentile"
                )
            ),
            2,
        ),
        "volume_percentile": round(
            number(
                row.get(
                    "volume_percentile"
                )
            ),
            2,
        ),
        "quality_percentile": round(
            number(
                row.get(
                    "quality_percentile"
                )
            ),
            2,
        ),
        "v15_score": round(
            v15_score,
            2,
        ),
        "v15_decision": text(
            row.get("v15_decision")
        ),
        "regime_adjustment": round(
            regime_effect,
            2,
        ),
        "v17_score": round(
            v17_score,
            2,
        ),
        "v17_decision": classify(
            v17_score,
            regime,
        ),
        "regime_reasons": reasons,
    }


def sort_results(
    result: pd.DataFrame,
) -> pd.DataFrame:
    if result.empty:
        return pd.DataFrame(
            columns=RESULT_COLUMNS
        )

    decision_order = {
        "V17 GÜÇLÜ ONAY": 4,
        "V17 ONAYLI İZLEME": 3,
        "V17 TEMKİNLİ İZLEME": 2,
        "V17 ELE": 1,
    }

    result = result.copy()

    result["_decision_order"] = (
        result["v17_decision"]
        .map(decision_order)
        .fillna(0)
    )

    result = result.sort_values(
        by=[
            "_decision_order",
            "v17_score",
        ],
        ascending=[
            False,
            False,
        ],
    )

    result = result.drop(
        columns=["_decision_order"]
    )

    result = result.reset_index(
        drop=True
    )

    result.insert(
        0,
        "rank",
        range(
            1,
            len(result) + 1,
        ),
    )

    return result[
        RESULT_COLUMNS
    ]


# ============================================================
# ANA MOTOR
# ============================================================

def main() -> None:
    print(
        "===== V17 PİYASA REJİM MOTORU BAŞLADI ====="
    )

    market = load_csv(
        MARKET_FILE
    )

    candidates = load_csv(
        CANDIDATE_FILE
    )

    if market.empty:
        status = {
            "status": "market_data_missing",
            "message": (
                "v16_full_market_snapshot.csv "
                "bulunamadı veya boş."
            ),
            "market_count": 0,
            "candidate_count": 0,
            "approved_count": 0,
        }

        save_empty_result()
        save_status(status)

        print(
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
            )
        )

        raise RuntimeError(
            "v16_full_market_snapshot.csv "
            "bulunamadı veya boş."
        )

    status = detect_regime(
        market
    )

    print(
        "Tespit edilen piyasa rejimi: "
        f"{status['regime']}"
    )

    print(
        "Rejim güveni: "
        f"%{status['regime_confidence']:.2f}"
    )

    # --------------------------------------------------------
    # ADAY DOSYASI BOŞSA NORMAL SONUÇ OLARAK DEVAM ET
    # --------------------------------------------------------

    if candidates.empty:
        result = save_empty_result()

        status.update(
            {
                "status": "ready",
                "candidate_file_status": "empty",
                "candidate_count": 0,
                "approved_count": 0,
                "strong_approved_count": 0,
                "cautious_count": 0,
                "eliminated_count": 0,
                "message": (
                    "V16 aday dosyasında "
                    "değerlendirilecek aday bulunamadı."
                ),
            }
        )

        save_status(
            status
        )

        print(
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
            )
        )

        print(
            "V17 sonucu: Değerlendirilecek aday yok."
        )

        print(
            "Başlıklı boş sonuç dosyası üretildi: "
            f"{RESULT_FILE}"
        )

        return

    # --------------------------------------------------------
    # ADAYLARI İŞLE
    # --------------------------------------------------------

    rows: list[dict[str, Any]] = []

    for _, row in candidates.iterrows():
        symbol = text(
            row.get("symbol")
        )

        if not symbol:
            continue

        rows.append(
            build_result_row(
                row,
                status,
            )
        )

    if rows:
        result = pd.DataFrame(
            rows
        )

        result = sort_results(
            result
        )

    else:
        result = pd.DataFrame(
            columns=RESULT_COLUMNS
        )

    result.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    decision_series = result.get(
        "v17_decision",
        pd.Series(dtype=str),
    )

    strong_approved_count = int(
        decision_series.eq(
            "V17 GÜÇLÜ ONAY"
        ).sum()
    )

    approved_monitoring_count = int(
        decision_series.eq(
            "V17 ONAYLI İZLEME"
        ).sum()
    )

    cautious_count = int(
        decision_series.eq(
            "V17 TEMKİNLİ İZLEME"
        ).sum()
    )

    eliminated_count = int(
        decision_series.eq(
            "V17 ELE"
        ).sum()
    )

    approved_count = (
        strong_approved_count
        + approved_monitoring_count
    )

    status.update(
        {
            "status": "ready",
            "candidate_file_status": "ready",
            "candidate_count": int(
                len(result)
            ),
            "approved_count": approved_count,
            "strong_approved_count": (
                strong_approved_count
            ),
            "approved_monitoring_count": (
                approved_monitoring_count
            ),
            "cautious_count": cautious_count,
            "eliminated_count": eliminated_count,
        }
    )

    save_status(
        status
    )

    print(
        "===== V17 DURUMU ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "===== V17 SONUÇLARI ====="
    )

    if result.empty:
        print(
            "Değerlendirilebilir aday bulunamadı."
        )
    else:
        print(
            result.to_string(
                index=False
            )
        )

    print(
        "===== V17 PİYASA REJİM MOTORU TAMAMLANDI ====="
    )


if __name__ == "__main__":
    main()
