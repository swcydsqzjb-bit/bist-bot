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
    "regime",
    "market_percentile",
    "timing_confidence",
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


def load_csv(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
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

    result["symbol"] = result["symbol"].apply(
        normalize_symbol
    )

    result = result[
        result["symbol"].ne("")
    ].copy()

    result = result.drop_duplicates(
        subset=["symbol"],
        keep="first",
    )

    return result.reset_index(drop=True)


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


# ============================================================
# VERI HAZIRLAMA
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
        "timing_confidence": 0.0,
        "risk_score": 50.0,
        "risk_class": "ORTA",
        "regime": "",
        "supporting_factors": "",
        "risk_notes": "",
    }

    for column, default in defaults.items():
        ensure_column(
            frame,
            column,
            default,
        )

    numeric_columns = [
        "prescan_score",
        "close",
        "return_1d",
        "return_5d",
        "return_20d",
        "ema20_distance",
        "volume_ratio",
        "market_percentile",
        "timing_confidence",
        "risk_score",
    ]

    for column in numeric_columns:
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(
            defaults[column]
        )

    return frame


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
        frame[column] = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).fillna(
            defaults[column]
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
        row.get("volume_ratio")
    )

    market_pct = number(
        row.get("market_percentile")
    )

    timing = number(
        row.get("timing_confidence")
    )

    # Günlük hareket
    if -1.5 <= return_1d <= 3.5:
        score += 12
    elif 3.5 < return_1d <= 6:
        score += 8
    elif return_1d > 8:
        score -= 8

    # 5 günlük momentum
    if 0 <= return_5d <= 8:
        score += 18
    elif 8 < return_5d <= 14:
        score += 10
    elif return_5d > 18:
        score -= 10

    # 20 günlük yapı
    if -5 <= return_20d <= 18:
        score += 10
    elif 18 < return_20d <= 28:
        score += 5
    elif return_20d > 35:
        score -= 8

    # EMA20
    if -3 <= ema20 <= 6:
        score += 20
    elif 6 < ema20 <= 10:
        score += 12
    elif -6 <= ema20 < -3:
        score += 7
    elif ema20 > 15:
        score -= 10

    # Hacim
    if 1.2 <= volume <= 3.5:
        score += 18
    elif 0.9 <= volume < 1.2:
        score += 10
    elif 3.5 < volume <= 5:
        score += 8
    elif volume > 6:
        score -= 6

    # Piyasa göreli güç
    if market_pct >= 90:
        score += 14
    elif market_pct >= 80:
        score += 10
    elif market_pct >= 70:
        score += 6

    # Zamanlama güveni varsa kullan
    if timing >= 85:
        score += 8
    elif timing >= 75:
        score += 5

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

    # Benzer gün sayısı azsa hafif kesinti
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

    # Ortalama çok yüksek ama medyan düşükse
    # birkaç uç değer ortalamayı taşıyor olabilir.
    if (
        average_5d >= 5
        and median_5d < 1
    ):
        consistency -= 12

    # Dağılım aşırı genişse azalt
    if return_range >= 45:
        consistency -= 15

    elif return_range >= 35:
        consistency -= 10

    elif return_range >= 25:
        consistency -= 5

    consistency -= downside_penalty * 0.45

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

    risk_score = number(
        row.get("risk_score"),
        50.0,
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

    # Ana bileşenler
    final_score = (
        v33_score * 0.24
        + prescan_score * 0.22
        + historical * 0.22
        + technical * 0.18
        + consistency * 0.14
    )

    # Risk kesintisi
    if risk_score > 40:
        final_score -= (
            risk_score - 40
        ) * 0.20

    # Çok kötü tarihsel senaryo varsa
    final_score -= (
        downside_penalty * 0.30
    )

    # 5 günlük ortalama + medyan ikisi de pozitifse bonus
    if (
        number(
            row.get("average_return_5d")
        ) > 1.0
        and number(
            row.get("median_return_5d")
        ) > 1.0
    ):
        final_score += 3.0

    # Pozitif oran çok yüksekse bonus
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

    risk = number(
        row.get("risk_score"),
        50.0,
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

    if risk >= HARD_RISK_LIMIT:
        return (
            "ELE",
            "Risk puanı kabul edilebilir seviyenin üzerinde",
            25.0,
        )

    # --------------------------------------------------------
    # UST DUZEY TEYIT
    # --------------------------------------------------------

    if (
        score >= 74
        and positive_5d >= 66
        and avg_5d >= 1.5
        and median_5d > 0
        and worst_5d > -15
        and technical >= 55
        and historical >= 60
        and consistency >= 55
        and risk <= 50
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
                "Benzer gün geçmişi, mevcut teknik yapı ve "
                "dağılım kalitesi birlikte güçlü"
            ),
            round(
                confidence,
                2,
            ),
        )

    # --------------------------------------------------------
    # AKTIF IZLEME
    # --------------------------------------------------------

    if (
        score >= 64
        and positive_5d >= 55
        and avg_5d > 0
        and median_5d >= 0
        and worst_5d > -20
        and risk <= 58
    ):
        confidence = (
            score * 0.60
            + historical * 0.20
            + consistency * 0.12
            + technical * 0.08
        )

        return (
            "AKTİF İZLEME",
            (
                "Tarihsel benzerlik ve mevcut ön tarama "
                "aktif takip için yeterli ortak güç üretti"
            ),
            round(
                confidence,
                2,
            ),
        )

    # --------------------------------------------------------
    # TEYIT BEKLE
    # --------------------------------------------------------

    if (
        score >= 54
        and risk <= 62
    ):
        confidence = (
            score * 0.70
            + historical * 0.20
            + consistency * 0.10
        )

        return (
            "TEYİT BEKLE",
            (
                "Genel görünüm olumlu fakat güçlü doğrulama "
                "için bazı şartlar eksik"
            ),
            round(
                confidence,
                2,
            ),
        )

    confidence = (
        score * 0.75
        + consistency * 0.25
    )

    return (
        "PASİF İZLEME",
        (
            "Toplam doğrulama skoru aktif takip için "
            "yeterli seviyeye ulaşmadı"
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
            "top_symbol": "",
            "top_decision": "",
            "top_score": 0.0,
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
    # Gerekli alanların varlığını garanti et
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
        "timing_confidence": 0.0,
        "risk_score": 50.0,
        "risk_class": "ORTA",
        "regime": "",
        "supporting_factors": "",
        "risk_notes": "",
    }

    for column, default in defaults.items():
        ensure_column(
            merged,
            column,
            default,
        )

    # --------------------------------------------------------
    # Final skorlar
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
    # Öncelik sırası
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

    merged = merged.sort_values(
        [
            "_priority",
            "v33_3_score",
            "v33_3_confidence",
            "positive_5d_rate",
        ],
        ascending=False,
    ).drop(
        columns="_priority"
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Üst düzey teyit en fazla 5 tane
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

    if len(
        strong_indices
    ) > MAX_STRONG_CONFIRMATIONS:
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
                "Üst düzey teyit kriterlerini karşıladı "
                "ancak günlük maksimum güçlü teyit kotası nedeniyle "
                "aktif izlemeye alındı"
            )

    # --------------------------------------------------------
    # Sadece ilk 10 final aday
    # --------------------------------------------------------

    merged = merged.head(
        MAX_FINAL_CANDIDATES
    ).reset_index(
        drop=True
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
    # Çıktı
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
        print(
            result[
                [
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
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
