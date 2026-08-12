from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# V33.3 - IKINCI DOGRULAMA / FINAL CONFIRMATION
# ============================================================

VERSION = "V33.3"

PRESCAN_FILE = Path("v33_prescan_candidates.csv")
V33_FILE = Path("v33_similar_day_candidates.csv")

OUTPUT_FILE = Path("v33_3_confirmed_candidates.csv")
STATUS_FILE = Path("v33_3_status.json")


RSI_USAGE = "DISABLED"

MAX_FINAL_CANDIDATES = 10
MAX_STRONG_CONFIRMATIONS = 5

HARD_RISK_LIMIT = 70.0


OUTPUT_COLUMNS = [
    "v33_3_rank",
    "symbol",

    "v33_3_decision",
    "v33_3_score",
    "v33_3_confidence",

    "v33_score",
    "prescan_score",

    "similar_day_count",

    "positive_1d_rate",
    "positive_3d_rate",
    "positive_5d_rate",

    "average_return_1d",
    "average_return_3d",
    "average_return_5d",

    "median_return_5d",
    "best_return_5d",
    "worst_return_5d",

    "return_range_5d",
    "downside_penalty",
    "consistency_score",

    "current_technical_score",
    "historical_quality_score",

    "risk_score",
    "risk_class",
    "risk_available",
    "risk_source",

    "regime",
    "regime_confidence",

    "market_percentile",

    "timing_confidence",
    "timing_available",
    "timing_source",

    "close",
    "return_1d",
    "return_5d",
    "return_20d",
    "ema20_distance",
    "volume_ratio",

    "prescan_class",

    "supporting_factors",
    "risk_notes",
    "v33_3_reason",

    "rsi_usage",
]


# ============================================================
# YARDIMCILAR
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

        if np.isfinite(result):
            return result

        return default

    except (TypeError, ValueError):
        return default


def optional_number(
    value: Any,
) -> float:
    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return np.nan


def boolean(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except Exception:
        pass

    value_text = str(value).strip().lower()

    if value_text in {
        "true",
        "1",
        "yes",
        "evet",
        "var",
    }:
        return True

    if value_text in {
        "false",
        "0",
        "no",
        "hayir",
        "hayır",
        "yok",
        "",
    }:
        return False

    return default


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        print(
            f"UYARI: Dosya bulunamadi: {path}"
        )
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            print(
                f"UYARI: Dosya bos: {path}"
            )
            return pd.DataFrame()

    except OSError:
        return pd.DataFrame()

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        except UnicodeDecodeError:
            continue

        except Exception as exc:
            print(
                f"UYARI: {path} okunamadi: {exc}"
            )
            return pd.DataFrame()

    return pd.DataFrame()


def normalize_symbol(
    value: Any,
) -> str:
    symbol = (
        text(value)
        .upper()
        .replace(" ", "")
    )

    if symbol.endswith(".IS"):
        symbol = symbol[:-3]

    return symbol


def normalize_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    if "symbol" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()

    result["symbol"] = (
        result["symbol"]
        .apply(normalize_symbol)
    )

    result = result[
        result["symbol"].ne("")
    ].copy()

    result = result.drop_duplicates(
        subset=["symbol"],
        keep="first",
    )

    return result.reset_index(
        drop=True
    )


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


# ============================================================
# VERI HAZIRLAMA - PRESCAN
# ============================================================

def prepare_prescan() -> pd.DataFrame:
    frame = normalize_frame(
        load_csv(
            PRESCAN_FILE
        )
    )

    if frame.empty:
        return pd.DataFrame()

    defaults = {
        "prescan_score": 0.0,
        "prescan_class": "",

        "close": 0.0,
        "return_1d": 0.0,
        "return_5d": 0.0,
        "return_20d": 0.0,
        "ema20_distance": 0.0,
        "volume_ratio": 0.0,
        "market_percentile": 0.0,

        "timing_confidence": np.nan,
        "timing_available": False,
        "timing_source": "VERI_YOK",

        "risk_score": np.nan,
        "risk_class": "VERI YOK",
        "risk_available": False,
        "risk_source": "VERI_YOK",

        "regime": "BILINMIYOR",
        "regime_confidence": np.nan,

        "supporting_factors": "",
        "risk_notes": "",
    }

    for column, default in defaults.items():
        ensure_column(
            frame,
            column,
            default,
        )

    normal_numeric_columns = [
        "prescan_score",
        "close",
        "return_1d",
        "return_5d",
        "return_20d",
        "ema20_distance",
        "volume_ratio",
        "market_percentile",
    ]

    for column in normal_numeric_columns:
        frame[column] = (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            .fillna(
                defaults[column]
            )
        )

    # Bu alanlarda eksik bilgi NaN olarak korunur.
    frame["timing_confidence"] = pd.to_numeric(
        frame["timing_confidence"],
        errors="coerce",
    )

    frame["risk_score"] = pd.to_numeric(
        frame["risk_score"],
        errors="coerce",
    )

    frame["regime_confidence"] = pd.to_numeric(
        frame["regime_confidence"],
        errors="coerce",
    )

    frame["timing_available"] = (
        frame["timing_available"]
        .apply(boolean)
    )

    frame["risk_available"] = (
        frame["risk_available"]
        .apply(boolean)
    )

    # Güvenlik:
    # sayı gerçekten varsa availability True olsun.
    frame.loc[
        frame["timing_confidence"].notna(),
        "timing_available",
    ] = True

    frame.loc[
        frame["risk_score"].notna(),
        "risk_available",
    ] = True

    frame["timing_source"] = (
        frame["timing_source"]
        .fillna("VERI_YOK")
        .astype(str)
        .str.strip()
    )

    frame["risk_source"] = (
        frame["risk_source"]
        .fillna("VERI_YOK")
        .astype(str)
        .str.strip()
    )

    frame["risk_class"] = (
        frame["risk_class"]
        .fillna("VERI YOK")
        .astype(str)
        .str.strip()
    )

    frame["regime"] = (
        frame["regime"]
        .fillna("BILINMIYOR")
        .astype(str)
        .str.strip()
    )

    frame["supporting_factors"] = (
        frame["supporting_factors"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    frame["risk_notes"] = (
        frame["risk_notes"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return frame


# ============================================================
# VERI HAZIRLAMA - V33 BENZER GUN
# ============================================================

def prepare_v33() -> pd.DataFrame:
    frame = normalize_frame(
        load_csv(
            V33_FILE
        )
    )

    if frame.empty:
        return pd.DataFrame()

    defaults = {
        "v33_decision": "",
        "v33_score": 0.0,

        "similar_day_count": 0.0,

        "positive_1d_rate": 0.0,
        "positive_3d_rate": 0.0,
        "positive_5d_rate": 0.0,

        "average_return_1d": 0.0,
        "average_return_3d": 0.0,
        "average_return_5d": 0.0,

        "median_return_5d": 0.0,
        "best_return_5d": 0.0,
        "worst_return_5d": 0.0,
    }

    for column, default in defaults.items():
        ensure_column(
            frame,
            column,
            default,
        )

    numeric_columns = [
        "v33_score",
        "similar_day_count",

        "positive_1d_rate",
        "positive_3d_rate",
        "positive_5d_rate",

        "average_return_1d",
        "average_return_3d",
        "average_return_5d",

        "median_return_5d",
        "best_return_5d",
        "worst_return_5d",
    ]

    for column in numeric_columns:
        frame[column] = (
            pd.to_numeric(
                frame[column],
                errors="coerce",
            )
            .fillna(
                defaults[column]
            )
        )

    return frame


# ============================================================
# MEVCUT TEKNIK PUAN
# ============================================================

def current_technical_score(
    row: pd.Series,
) -> float:
    score = 0.0

    return_1d = number(
        row.get("return_1d")
    )

    return_5d = number(
        row.get("return_5d")
    )

    return_20d = number(
        row.get("return_20d")
    )

    ema20 = number(
        row.get("ema20_distance")
    )

    volume = number(
        row.get("volume_ratio"),
        1.0,
    )

    market_pct = number(
        row.get("market_percentile"),
        50.0,
    )

    timing = optional_number(
        row.get("timing_confidence")
    )

    timing_available = boolean(
        row.get("timing_available")
    )

    # --------------------------------------------------------
    # 1 GUNLUK HAREKET
    # --------------------------------------------------------

    if -1.5 <= return_1d <= 3.5:
        score += 12

    elif 3.5 < return_1d <= 6:
        score += 8

    elif return_1d > 8:
        score -= 8

    # --------------------------------------------------------
    # 5 GUNLUK MOMENTUM
    # --------------------------------------------------------

    if 0 <= return_5d <= 8:
        score += 18

    elif 8 < return_5d <= 14:
        score += 10

    elif return_5d > 18:
        score -= 10

    # --------------------------------------------------------
    # 20 GUNLUK YAPI
    # --------------------------------------------------------

    if -5 <= return_20d <= 18:
        score += 10

    elif 18 < return_20d <= 28:
        score += 5

    elif return_20d > 35:
        score -= 8

    # --------------------------------------------------------
    # EMA20
    # --------------------------------------------------------

    if -3 <= ema20 <= 6:
        score += 20

    elif 6 < ema20 <= 10:
        score += 12

    elif -6 <= ema20 < -3:
        score += 7

    elif ema20 > 15:
        score -= 10

    # --------------------------------------------------------
    # HACIM
    # --------------------------------------------------------

    if 1.2 <= volume <= 3.5:
        score += 18

    elif 0.9 <= volume < 1.2:
        score += 10

    elif 3.5 < volume <= 5:
        score += 8

    elif volume > 6:
        score -= 6

    # --------------------------------------------------------
    # PIYASA GORELI GUC
    # --------------------------------------------------------

    if market_pct >= 90:
        score += 14

    elif market_pct >= 80:
        score += 10

    elif market_pct >= 70:
        score += 6

    # --------------------------------------------------------
    # ZAMANLAMA
    #
    # Sadece gerçek veri varsa kullanılır.
    # Veri yoksa 0 kabul edilmez ve ceza verilmez.
    # --------------------------------------------------------

    if (
        timing_available
        and np.isfinite(timing)
    ):
        if timing >= 85:
            score += 8

        elif timing >= 75:
            score += 5

        elif timing < 40:
            score -= 3

    return round(
        float(
            np.clip(
                score,
                0,
                100,
            )
        ),
        2,
    )


# ============================================================
# TARIHSEL KALITE
# ============================================================

def historical_quality_score(
    row: pd.Series,
) -> float:
    positive_1d = number(
        row.get("positive_1d_rate")
    )

    positive_3d = number(
        row.get("positive_3d_rate")
    )

    positive_5d = number(
        row.get("positive_5d_rate")
    )

    average_1d = number(
        row.get("average_return_1d")
    )

    average_3d = number(
        row.get("average_return_3d")
    )

    average_5d = number(
        row.get("average_return_5d")
    )

    median_5d = number(
        row.get("median_return_5d")
    )

    similar_count = number(
        row.get("similar_day_count")
    )

    score = (
        positive_1d * 0.10
        + positive_3d * 0.18
        + positive_5d * 0.28
    )

    score += (
        np.clip(
            50 + average_1d * 7,
            0,
            100,
        )
        * 0.08
    )

    score += (
        np.clip(
            50 + average_3d * 5,
            0,
            100,
        )
        * 0.12
    )

    score += (
        np.clip(
            50 + average_5d * 4,
            0,
            100,
        )
        * 0.16
    )

    score += (
        np.clip(
            50 + median_5d * 5,
            0,
            100,
        )
        * 0.08
    )

    if similar_count < 8:
        score -= 6

    elif similar_count >= 12:
        score += 2

    return round(
        float(
            np.clip(
                score,
                0,
                100,
            )
        ),
        2,
    )


# ============================================================
# TUTARLILIK / DAGILIM
# ============================================================

def consistency_score(
    row: pd.Series,
) -> tuple[float, float, float]:
    best_5d = number(
        row.get("best_return_5d")
    )

    worst_5d = number(
        row.get("worst_return_5d")
    )

    average_5d = number(
        row.get("average_return_5d")
    )

    median_5d = number(
        row.get("median_return_5d")
    )

    positive_5d = number(
        row.get("positive_5d_rate")
    )

    return_range = (
        best_5d - worst_5d
    )

    downside_penalty = 0.0

    if worst_5d <= -20:
        downside_penalty = 22.0

    elif worst_5d <= -15:
        downside_penalty = 16.0

    elif worst_5d <= -10:
        downside_penalty = 10.0

    elif worst_5d <= -7:
        downside_penalty = 6.0

    elif worst_5d <= -4:
        downside_penalty = 3.0

    consistency = 50.0

    if positive_5d >= 75:
        consistency += 20

    elif positive_5d >= 66:
        consistency += 15

    elif positive_5d >= 58:
        consistency += 9

    elif positive_5d >= 50:
        consistency += 3

    else:
        consistency -= 10

    if average_5d > 0:
        consistency += min(
            average_5d * 3,
            15,
        )

    else:
        consistency -= 10

    if median_5d > 0:
        consistency += min(
            median_5d * 3,
            12,
        )

    else:
        consistency -= 8

    if (
        average_5d >= 5
        and median_5d < 1
    ):
        consistency -= 12

    if return_range >= 45:
        consistency -= 15

    elif return_range >= 35:
        consistency -= 10

    elif return_range >= 25:
        consistency -= 5

    consistency -= (
        downside_penalty * 0.45
    )

    return (
        round(
            float(
                np.clip(
                    consistency,
                    0,
                    100,
                )
            ),
            2,
        ),

        round(
            return_range,
            2,
        ),

        round(
            downside_penalty,
            2,
        ),
    )


# ============================================================
# FINAL SKOR
# ============================================================

def calculate_final_score(
    row: pd.Series,
) -> dict[str, float]:
    v33_score = number(
        row.get("v33_score")
    )

    prescan_score = number(
        row.get("prescan_score")
    )

    risk_score = optional_number(
        row.get("risk_score")
    )

    risk_available = boolean(
        row.get("risk_available")
    )

    technical = current_technical_score(
        row
    )

    historical = historical_quality_score(
        row
    )

    (
        consistency,
        return_range,
        downside_penalty,
    ) = consistency_score(
        row
    )

    # --------------------------------------------------------
    # ANA BILESENLER
    # --------------------------------------------------------

    final_score = (
        v33_score * 0.24
        + prescan_score * 0.22
        + historical * 0.22
        + technical * 0.18
        + consistency * 0.14
    )

    # --------------------------------------------------------
    # GERCEK RISK VERISI VARSA UYGULA
    #
    # Risk verisi yoksa:
    # - 50 varsayılmaz
    # - bonus verilmez
    # - ceza verilmez
    # --------------------------------------------------------

    if (
        risk_available
        and np.isfinite(risk_score)
    ):
        if risk_score > 40:
            final_score -= (
                risk_score - 40
            ) * 0.20

        elif risk_score <= 30:
            final_score += 1.0

    # --------------------------------------------------------
    # TARIHSEL DOWNSIDE
    # --------------------------------------------------------

    final_score -= (
        downside_penalty * 0.30
    )

    # --------------------------------------------------------
    # TARIHSEL BONUSLAR
    # --------------------------------------------------------

    if (
        number(
            row.get(
                "average_return_5d"
            )
        ) > 1.0
        and number(
            row.get(
                "median_return_5d"
            )
        ) > 1.0
    ):
        final_score += 3.0

    if number(
        row.get("positive_5d_rate")
    ) >= 70:
        final_score += 3.0

    return {
        "v33_3_score": round(
            float(
                np.clip(
                    final_score,
                    0,
                    100,
                )
            ),
            2,
        ),

        "current_technical_score": technical,

        "historical_quality_score": historical,

        "consistency_score": consistency,

        "return_range_5d": return_range,

        "downside_penalty": downside_penalty,
    }


# ============================================================
# KARAR
# ============================================================

def determine_decision(
    row: pd.Series,
) -> tuple[str, str, float]:
    score = number(
        row.get("v33_3_score")
    )

    risk = optional_number(
        row.get("risk_score")
    )

    risk_available = boolean(
        row.get("risk_available")
    )

    timing_available = boolean(
        row.get("timing_available")
    )

    positive_5d = number(
        row.get("positive_5d_rate")
    )

    avg_5d = number(
        row.get("average_return_5d")
    )

    median_5d = number(
        row.get("median_return_5d")
    )

    worst_5d = number(
        row.get("worst_return_5d")
    )

    technical = number(
        row.get("current_technical_score")
    )

    historical = number(
        row.get("historical_quality_score")
    )

    consistency = number(
        row.get("consistency_score")
    )

    # --------------------------------------------------------
    # GERCEK RISK VAR VE LIMIT USTUNDEYSE ELE
    # --------------------------------------------------------

    if (
        risk_available
        and np.isfinite(risk)
        and risk >= HARD_RISK_LIMIT
    ):
        return (
            "ELE",
            (
                "Gerçek V20 risk puanı kabul edilebilir "
                "seviyenin üzerinde"
            ),
            25.0,
        )

    # --------------------------------------------------------
    # UST DUZEY TEYIT
    #
    # Burada gerçek risk doğrulaması zorunlu.
    # Risk verisi olmayan hisse en yüksek etiketi alamaz.
    # --------------------------------------------------------

    if (
        risk_available
        and np.isfinite(risk)
        and risk <= 50
        and score >= 74
        and positive_5d >= 66
        and avg_5d >= 1.5
        and median_5d > 0
        and worst_5d > -15
        and technical >= 55
        and historical >= 60
        and consistency >= 55
    ):
        confidence = (
            score * 0.55
            + historical * 0.20
            + consistency * 0.15
            + technical * 0.10
        )

        return (
            "ÜST DÜZEY TEYİT",
            (
                "Benzer gün geçmişi, mevcut teknik yapı, "
                "dağılım kalitesi ve gerçek risk verisi birlikte güçlü"
            ),
            round(
                confidence,
                2,
            ),
        )

    # --------------------------------------------------------
    # AKTIF IZLEME
    #
    # Risk verisi yoksa yine aktif izleme mümkün.
    # Ancak güçlü teyit verilmez.
    # --------------------------------------------------------

    risk_ok_for_active = (
        (
            risk_available
            and np.isfinite(risk)
            and risk <= 58
        )
        or (
            not risk_available
        )
    )

    if (
        score >= 64
        and positive_5d >= 55
        and avg_5d > 0
        and median_5d >= 0
        and worst_5d > -20
        and risk_ok_for_active
    ):
        confidence = (
            score * 0.60
            + historical * 0.20
            + consistency * 0.12
            + technical * 0.08
        )

        if not risk_available:
            confidence = min(
                confidence,
                79.0,
            )

            reason = (
                "Tarihsel benzerlik ve mevcut teknik yapı aktif takip "
                "için yeterli; ancak gerçek V20 risk verisi olmadığı "
                "için üst düzey teyit verilmedi"
            )

        elif not timing_available:
            reason = (
                "Tarihsel benzerlik ve mevcut teknik yapı aktif takip "
                "için yeterli; zamanlama verisi bulunmadığından "
                "zamanlama bonusu kullanılmadı"
            )

        else:
            reason = (
                "Tarihsel benzerlik, mevcut ön tarama ve doğrulama "
                "verileri aktif takip için yeterli ortak güç üretti"
            )

        return (
            "AKTİF İZLEME",
            reason,
            round(
                confidence,
                2,
            ),
        )

    # --------------------------------------------------------
    # TEYIT BEKLE
    # --------------------------------------------------------

    risk_ok_for_waiting = (
        (
            risk_available
            and np.isfinite(risk)
            and risk <= 62
        )
        or (
            not risk_available
        )
    )

    if (
        score >= 54
        and risk_ok_for_waiting
    ):
        confidence = (
            score * 0.70
            + historical * 0.20
            + consistency * 0.10
        )

        missing_notes: list[str] = []

        if not risk_available:
            missing_notes.append(
                "gerçek risk verisi yok"
            )

        if not timing_available:
            missing_notes.append(
                "zamanlama verisi yok"
            )

        if missing_notes:
            reason = (
                "Genel görünüm olumlu ancak "
                + " ve ".join(missing_notes)
                + "; daha güçlü doğrulama bekleniyor"
            )

        else:
            reason = (
                "Genel görünüm olumlu fakat güçlü doğrulama "
                "için bazı şartlar eksik"
            )

        return (
            "TEYİT BEKLE",
            reason,
            round(
                confidence,
                2,
            ),
        )

    # --------------------------------------------------------
    # PASIF IZLEME
    # --------------------------------------------------------

    confidence = (
        score * 0.75
        + consistency * 0.25
    )

    reason_parts = [
        (
            "Toplam doğrulama skoru aktif takip "
            "için yeterli seviyeye ulaşmadı"
        )
    ]

    if not risk_available:
        reason_parts.append(
            "V20 risk verisi yok"
        )

    if not timing_available:
        reason_parts.append(
            "V19/V20 zamanlama verisi yok"
        )

    return (
        "PASİF İZLEME",
        " | ".join(
            reason_parts
        ),
        round(
            confidence,
            2,
        ),
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    prescan = prepare_prescan()
    v33 = prepare_v33()

    if prescan.empty or v33.empty:
        empty = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

        empty.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        status = {
            "status": "input_missing",

            "candidate_count": 0,

            "strong_confirmation_count": 0,
            "active_tracking_count": 0,
            "waiting_count": 0,
            "passive_count": 0,
            "eliminated_count": 0,

            "risk_available_count": 0,
            "risk_missing_count": 0,

            "timing_available_count": 0,
            "timing_missing_count": 0,

            "top_symbol": "",
            "top_decision": "",
            "top_score": 0.0,

            "regime": "BILINMIYOR",
            "regime_confidence": None,

            "rsi_usage": RSI_USAGE,
            "version": VERSION,
        }

        STATUS_FILE.write_text(
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    # --------------------------------------------------------
    # V33 + PRESCAN BIRLESTIR
    # --------------------------------------------------------

    merged = v33.merge(
        prescan,
        on="symbol",
        how="left",
        suffixes=(
            "_v33",
            "",
        ),
    )

    # --------------------------------------------------------
    # GEREKLI ALANLARI GARANTI ET
    # --------------------------------------------------------

    defaults = {
        "prescan_score": 0.0,
        "prescan_class": "",

        "close": 0.0,
        "return_1d": 0.0,
        "return_5d": 0.0,
        "return_20d": 0.0,
        "ema20_distance": 0.0,
        "volume_ratio": 0.0,
        "market_percentile": 0.0,

        "timing_confidence": np.nan,
        "timing_available": False,
        "timing_source": "VERI_YOK",

        "risk_score": np.nan,
        "risk_class": "VERI YOK",
        "risk_available": False,
        "risk_source": "VERI_YOK",

        "regime": "BILINMIYOR",
        "regime_confidence": np.nan,

        "supporting_factors": "",
        "risk_notes": "",
    }

    for column, default in defaults.items():
        ensure_column(
            merged,
            column,
            default,
        )

    # Sayılar
    normal_numeric_columns = [
        "prescan_score",
        "close",
        "return_1d",
        "return_5d",
        "return_20d",
        "ema20_distance",
        "volume_ratio",
        "market_percentile",
    ]

    for column in normal_numeric_columns:
        merged[column] = (
            pd.to_numeric(
                merged[column],
                errors="coerce",
            )
            .fillna(
                defaults[column]
            )
        )

    # Eksik kalması gereken sayılar
    merged["risk_score"] = pd.to_numeric(
        merged["risk_score"],
        errors="coerce",
    )

    merged["timing_confidence"] = pd.to_numeric(
        merged["timing_confidence"],
        errors="coerce",
    )

    merged["regime_confidence"] = pd.to_numeric(
        merged["regime_confidence"],
        errors="coerce",
    )

    # Availability
    merged["risk_available"] = (
        merged["risk_available"]
        .apply(boolean)
    )

    merged["timing_available"] = (
        merged["timing_available"]
        .apply(boolean)
    )

    merged.loc[
        merged["risk_score"].notna(),
        "risk_available",
    ] = True

    merged.loc[
        merged["timing_confidence"].notna(),
        "timing_available",
    ] = True

    # Metinler
    merged["risk_class"] = (
        merged["risk_class"]
        .fillna("VERI YOK")
        .astype(str)
        .str.strip()
    )

    merged["risk_source"] = (
        merged["risk_source"]
        .fillna("VERI_YOK")
        .astype(str)
        .str.strip()
    )

    merged["timing_source"] = (
        merged["timing_source"]
        .fillna("VERI_YOK")
        .astype(str)
        .str.strip()
    )

    merged["regime"] = (
        merged["regime"]
        .fillna("BILINMIYOR")
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # FINAL SKORLAR
    # --------------------------------------------------------

    score_rows = merged.apply(
        calculate_final_score,
        axis=1,
    )

    merged[
        "v33_3_score"
    ] = [
        item[
            "v33_3_score"
        ]
        for item in score_rows
    ]

    merged[
        "current_technical_score"
    ] = [
        item[
            "current_technical_score"
        ]
        for item in score_rows
    ]

    merged[
        "historical_quality_score"
    ] = [
        item[
            "historical_quality_score"
        ]
        for item in score_rows
    ]

    merged[
        "consistency_score"
    ] = [
        item[
            "consistency_score"
        ]
        for item in score_rows
    ]

    merged[
        "return_range_5d"
    ] = [
        item[
            "return_range_5d"
        ]
        for item in score_rows
    ]

    merged[
        "downside_penalty"
    ] = [
        item[
            "downside_penalty"
        ]
        for item in score_rows
    ]

    # --------------------------------------------------------
    # KARARLAR
    # --------------------------------------------------------

    decisions = merged.apply(
        determine_decision,
        axis=1,
    )

    merged[
        "v33_3_decision"
    ] = [
        item[0]
        for item in decisions
    ]

    merged[
        "v33_3_reason"
    ] = [
        item[1]
        for item in decisions
    ]

    merged[
        "v33_3_confidence"
    ] = [
        item[2]
        for item in decisions
    ]

    # --------------------------------------------------------
    # ONCELIK SIRASI
    # --------------------------------------------------------

    priority = {
        "ÜST DÜZEY TEYİT": 5,
        "AKTİF İZLEME": 4,
        "TEYİT BEKLE": 3,
        "PASİF İZLEME": 2,
        "ELE": 1,
    }

    merged["_priority"] = (
        merged[
            "v33_3_decision"
        ]
        .map(priority)
        .fillna(0)
    )

    merged = (
        merged.sort_values(
            [
                "_priority",
                "v33_3_score",
                "v33_3_confidence",
                "positive_5d_rate",
            ],
            ascending=False,
        )
        .drop(
            columns="_priority"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # UST DUZEY TEYIT EN FAZLA 5
    # --------------------------------------------------------

    strong_mask = (
        merged[
            "v33_3_decision"
        ]
        == "ÜST DÜZEY TEYİT"
    )

    strong_indices = (
        merged[
            strong_mask
        ]
        .index
        .tolist()
    )

    if (
        len(strong_indices)
        > MAX_STRONG_CONFIRMATIONS
    ):
        for index in strong_indices[
            MAX_STRONG_CONFIRMATIONS:
        ]:
            merged.at[
                index,
                "v33_3_decision",
            ] = "AKTİF İZLEME"

            merged.at[
                index,
                "v33_3_reason",
            ] = (
                "Üst düzey teyit kriterlerini karşıladı ancak "
                "günlük maksimum güçlü teyit kotası nedeniyle "
                "aktif izlemeye alındı"
            )

    # --------------------------------------------------------
    # ILK 10 FINAL ADAY
    # --------------------------------------------------------

    merged = (
        merged.head(
            MAX_FINAL_CANDIDATES
        )
        .reset_index(
            drop=True
        )
    )

    merged.insert(
        0,
        "v33_3_rank",
        range(
            1,
            len(merged) + 1,
        ),
    )

    merged[
        "rsi_usage"
    ] = RSI_USAGE

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    result = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column in merged.columns:
            result[column] = (
                merged[column]
            )

        else:
            result[column] = np.nan

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # SAYACLAR
    # --------------------------------------------------------

    strong_count = int(
        (
            result[
                "v33_3_decision"
            ]
            == "ÜST DÜZEY TEYİT"
        ).sum()
    )

    active_count = int(
        (
            result[
                "v33_3_decision"
            ]
            == "AKTİF İZLEME"
        ).sum()
    )

    waiting_count = int(
        (
            result[
                "v33_3_decision"
            ]
            == "TEYİT BEKLE"
        ).sum()
    )

    passive_count = int(
        (
            result[
                "v33_3_decision"
            ]
            == "PASİF İZLEME"
        ).sum()
    )

    eliminated_count = int(
        (
            result[
                "v33_3_decision"
            ]
            == "ELE"
        ).sum()
    )

    risk_available_count = int(
        result[
            "risk_available"
        ].apply(
            boolean
        ).sum()
    )

    timing_available_count = int(
        result[
            "timing_available"
        ].apply(
            boolean
        ).sum()
    )

    # --------------------------------------------------------
    # REJIM STATUS
    # --------------------------------------------------------

    if len(result):
        regime_value = (
            text(
                result.iloc[0][
                    "regime"
                ]
            )
            or "BILINMIYOR"
        )

        regime_confidence = (
            optional_number(
                result.iloc[0][
                    "regime_confidence"
                ]
            )
        )

    else:
        regime_value = "BILINMIYOR"
        regime_confidence = np.nan

    # --------------------------------------------------------
    # STATUS JSON
    # --------------------------------------------------------

    status = {
        "status": "ready",

        "candidate_count": int(
            len(result)
        ),

        "strong_confirmation_count": strong_count,

        "active_tracking_count": active_count,

        "waiting_count": waiting_count,

        "passive_count": passive_count,

        "eliminated_count": eliminated_count,

        "approved_count": (
            strong_count
            + active_count
        ),

        "risk_available_count": (
            risk_available_count
        ),

        "risk_missing_count": int(
            len(result)
            - risk_available_count
        ),

        "timing_available_count": (
            timing_available_count
        ),

        "timing_missing_count": int(
            len(result)
            - timing_available_count
        ),

        "regime": regime_value,

        "regime_confidence": (
            round(
                float(
                    regime_confidence
                ),
                2,
            )
            if np.isfinite(
                regime_confidence
            )
            else None
        ),

        "top_symbol": (
            text(
                result.iloc[0][
                    "symbol"
                ]
            )
            if len(result)
            else ""
        ),

        "top_decision": (
            text(
                result.iloc[0][
                    "v33_3_decision"
                ]
            )
            if len(result)
            else ""
        ),

        "top_score": (
            round(
                number(
                    result.iloc[0][
                        "v33_3_score"
                    ]
                ),
                2,
            )
            if len(result)
            else 0.0
        ),

        "rsi_usage": RSI_USAGE,

        "version": VERSION,
    }

    STATUS_FILE.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # LOG
    # --------------------------------------------------------

    print(
        "===== V33.3 STATUS ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "===== V33.3 FINAL CANDIDATES ====="
    )

    if result.empty:
        print(
            "Final aday bulunamadi."
        )

    else:
        display_columns = [
            "v33_3_rank",
            "symbol",
            "v33_3_decision",
            "v33_3_score",
            "v33_3_confidence",

            "v33_score",
            "prescan_score",

            "positive_5d_rate",
            "average_return_5d",
            "median_return_5d",
            "worst_return_5d",

            "risk_score",
            "risk_available",
            "risk_source",

            "timing_confidence",
            "timing_available",
            "timing_source",

            "regime",
            "regime_confidence",
        ]

        print(
            result[
                display_columns
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
