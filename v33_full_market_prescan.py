from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# V33.2 - TAM PIYASA ON TARAMA
# ============================================================

VERSION = "V33.2"

MARKET_FILE = Path("v16_full_market_snapshot.csv")

V22_FILE = Path("v22_signal_states.csv")
V27_FILE = Path("v27_master_decisions.csv")
V32_FILE = Path("v32_adaptive_decisions.csv")

REGIME_FILE = Path("v17_market_regime_status.json")
V19_FILE = Path("v19_timing_forecasts.csv")
V20_FILE = Path("v20_ai_final_decisions.csv")

OUTPUT_FILE = Path("v33_prescan_candidates.csv")
STATUS_FILE = Path("v33_prescan_status.json")


# ============================================================
# AYARLAR
# ============================================================

MAX_CANDIDATES = 30
MIN_CANDIDATES = 10

MAX_RISK_SCORE = 70.0

RSI_USAGE = "DISABLED"


OUTPUT_COLUMNS = [
    "prescan_rank",
    "symbol",
    "prescan_score",
    "prescan_class",

    "close",
    "return_1d",
    "return_5d",
    "return_20d",
    "ema20_distance",
    "volume_ratio",
    "market_percentile",

    "timing_confidence",
    "timing_available",
    "timing_source",

    "risk_score",
    "risk_class",
    "risk_available",
    "risk_source",

    "regime",
    "regime_confidence",

    "v22_signal_state",
    "v22_signal_score",

    "v27_decision",
    "v27_master_score",

    "v32_decision",
    "v32_score",
    "v32_confidence",

    "supporting_factors",
    "risk_notes",

    "rsi_usage",
]


# ============================================================
# TEMEL YARDIMCILAR
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


def load_json(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        print(
            f"UYARI: JSON bulunamadi: {path}"
        )
        return {}

    try:
        raw = path.read_text(
            encoding="utf-8",
        )

        if not raw.strip():
            return {}

        result = json.loads(raw)

        if isinstance(result, dict):
            return result

    except Exception as exc:
        print(
            f"UYARI: {path} okunamadi: {exc}"
        )

    return {}


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


def first_existing_column(
    frame: pd.DataFrame,
    names: list[str],
) -> str | None:
    for name in names:
        if name in frame.columns:
            return name

    return None


def numeric_from_aliases(
    frame: pd.DataFrame,
    aliases: list[str],
    default: float = np.nan,
) -> pd.Series:
    column = first_existing_column(
        frame,
        aliases,
    )

    if column is None:
        return pd.Series(
            default,
            index=frame.index,
            dtype=float,
        )

    return pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def text_from_aliases(
    frame: pd.DataFrame,
    aliases: list[str],
    default: str = "",
) -> pd.Series:
    column = first_existing_column(
        frame,
        aliases,
    )

    if column is None:
        return pd.Series(
            default,
            index=frame.index,
            dtype=object,
        )

    return (
        frame[column]
        .fillna("")
        .astype(str)
        .str.strip()
    )


# ============================================================
# PIYASA VERISI
# ============================================================

def prepare_market() -> pd.DataFrame:
    raw = normalize_frame(
        load_csv(
            MARKET_FILE
        )
    )

    if raw.empty:
        return pd.DataFrame()

    market = pd.DataFrame()

    market["symbol"] = raw["symbol"]

    market["close"] = numeric_from_aliases(
        raw,
        [
            "close",
            "price",
            "last",
            "last_price",
        ],
        0.0,
    ).fillna(0.0)

    market["return_1d"] = numeric_from_aliases(
        raw,
        [
            "return_1d",
            "ret_1d",
            "change_1d",
            "daily_return",
        ],
        0.0,
    ).fillna(0.0)

    market["return_5d"] = numeric_from_aliases(
        raw,
        [
            "return_5d",
            "ret_5d",
            "change_5d",
        ],
        0.0,
    ).fillna(0.0)

    market["return_20d"] = numeric_from_aliases(
        raw,
        [
            "return_20d",
            "ret_20d",
            "change_20d",
        ],
        0.0,
    ).fillna(0.0)

    market["ema20_distance"] = numeric_from_aliases(
        raw,
        [
            "ema20_distance",
            "ema20_dist",
            "distance_ema20",
        ],
        0.0,
    ).fillna(0.0)

    market["volume_ratio"] = numeric_from_aliases(
        raw,
        [
            "volume_ratio",
            "relative_volume",
            "rel_volume",
            "volume_ratio_20d",
        ],
        1.0,
    ).fillna(1.0)

    market_pct = numeric_from_aliases(
        raw,
        [
            "market_percentile",
            "relative_percentile",
            "market_pct",
        ],
        np.nan,
    )

    if market_pct.notna().sum() >= max(
        10,
        int(len(market) * 0.20),
    ):
        market[
            "market_percentile"
        ] = market_pct.fillna(
            market_pct.median()
        )

    else:
        # V16 dosyasında doğrudan piyasa yüzdeliği yoksa,
        # yalnızca V16 verileriyle sıralama üret.
        composite = (
            market["return_5d"] * 0.45
            + market["return_20d"] * 0.20
            + market["ema20_distance"] * 0.20
            + (
                market["volume_ratio"] - 1.0
            ) * 10.0 * 0.15
        )

        market[
            "market_percentile"
        ] = (
            composite.rank(
                pct=True,
                method="average",
            )
            * 100.0
        )

    market[
        "market_percentile"
    ] = (
        market["market_percentile"]
        .clip(
            lower=0,
            upper=100,
        )
    )

    return market


# ============================================================
# V22
# ============================================================

def prepare_v22() -> pd.DataFrame:
    raw = normalize_frame(
        load_csv(
            V22_FILE
        )
    )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v22_signal_state",
                "v22_signal_score",
            ]
        )

    result = pd.DataFrame()

    result["symbol"] = raw["symbol"]

    result[
        "v22_signal_state"
    ] = text_from_aliases(
        raw,
        [
            "v22_signal_state",
            "signal_state",
            "v22_state",
            "decision",
        ],
        "",
    )

    result[
        "v22_signal_score"
    ] = numeric_from_aliases(
        raw,
        [
            "v22_signal_score",
            "signal_score",
            "v22_score",
            "score",
        ],
        0.0,
    ).fillna(0.0)

    return result


# ============================================================
# V27
# ============================================================

def prepare_v27() -> pd.DataFrame:
    raw = normalize_frame(
        load_csv(
            V27_FILE
        )
    )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v27_decision",
                "v27_master_score",
            ]
        )

    result = pd.DataFrame()

    result["symbol"] = raw["symbol"]

    result[
        "v27_decision"
    ] = text_from_aliases(
        raw,
        [
            "v27_decision",
            "entry_decision",
            "decision",
        ],
        "",
    )

    result[
        "v27_master_score"
    ] = numeric_from_aliases(
        raw,
        [
            "v27_master_score",
            "master_score",
            "v27_score",
            "score",
        ],
        0.0,
    ).fillna(0.0)

    return result


# ============================================================
# V32
# ============================================================

def prepare_v32() -> pd.DataFrame:
    raw = normalize_frame(
        load_csv(
            V32_FILE
        )
    )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v32_decision",
                "v32_score",
                "v32_confidence",
            ]
        )

    result = pd.DataFrame()

    result["symbol"] = raw["symbol"]

    result[
        "v32_decision"
    ] = text_from_aliases(
        raw,
        [
            "v32_decision",
            "decision",
        ],
        "",
    )

    result[
        "v32_score"
    ] = numeric_from_aliases(
        raw,
        [
            "v32_score",
            "adaptive_score",
            "score",
        ],
        0.0,
    ).fillna(0.0)

    result[
        "v32_confidence"
    ] = numeric_from_aliases(
        raw,
        [
            "v32_confidence",
            "confidence",
            "confidence_score",
        ],
        0.0,
    ).fillna(0.0)

    return result


# ============================================================
# V19 - ZAMANLAMA
# ============================================================

def prepare_v19() -> pd.DataFrame:
    raw = normalize_frame(
        load_csv(
            V19_FILE
        )
    )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v19_timing_confidence",
            ]
        )

    result = pd.DataFrame()

    result["symbol"] = raw["symbol"]

    # Ekranda doğruladığımız gerçek kolon:
    # confidence_score
    result[
        "v19_timing_confidence"
    ] = numeric_from_aliases(
        raw,
        [
            "confidence_score",
            "timing_confidence",
            "v19_score",
        ],
        np.nan,
    )

    return result


# ============================================================
# V20 - RISK + ZAMANLAMA
# ============================================================

def prepare_v20() -> pd.DataFrame:
    raw = normalize_frame(
        load_csv(
            V20_FILE
        )
    )

    if raw.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v20_risk_score",
                "v20_risk_class",
                "v20_timing_confidence",
                "v20_regime",
                "v20_ai_reasons",
                "v20_risk_reasons",
            ]
        )

    result = pd.DataFrame()

    result["symbol"] = raw["symbol"]

    result[
        "v20_risk_score"
    ] = numeric_from_aliases(
        raw,
        [
            "risk_score",
        ],
        np.nan,
    )

    result[
        "v20_risk_class"
    ] = text_from_aliases(
        raw,
        [
            "risk_class",
        ],
        "",
    )

    result[
        "v20_timing_confidence"
    ] = numeric_from_aliases(
        raw,
        [
            "timing_confidence",
        ],
        np.nan,
    )

    result[
        "v20_regime"
    ] = text_from_aliases(
        raw,
        [
            "regime",
        ],
        "",
    )

    result[
        "v20_ai_reasons"
    ] = text_from_aliases(
        raw,
        [
            "ai_reasons",
        ],
        "",
    )

    result[
        "v20_risk_reasons"
    ] = text_from_aliases(
        raw,
        [
            "risk_reasons",
        ],
        "",
    )

    return result


# ============================================================
# V17 - GENEL PIYASA REJIMI
# ============================================================

def prepare_regime() -> dict[str, Any]:
    status = load_json(
        REGIME_FILE
    )

    regime = (
        text(
            status.get("regime")
        )
        or "BILINMIYOR"
    )

    confidence = optional_number(
        status.get(
            "regime_confidence"
        )
    )

    return {
        "regime": regime,
        "regime_confidence": confidence,
    }


# ============================================================
# PRESCAN ALT PUANLARI
# ============================================================

def score_market_strength(
    percentile: float,
) -> float:
    return float(
        np.clip(
            percentile,
            0,
            100,
        )
    )


def score_ema20(
    distance: float,
) -> float:
    if -2.0 <= distance <= 4.0:
        return 100.0

    if 4.0 < distance <= 7.0:
        return 88.0

    if -4.0 <= distance < -2.0:
        return 78.0

    if 7.0 < distance <= 10.0:
        return 68.0

    if -7.0 <= distance < -4.0:
        return 58.0

    if 10.0 < distance <= 14.0:
        return 45.0

    if distance > 18.0:
        return 10.0

    return 30.0


def score_volume(
    ratio: float,
) -> float:
    if 1.20 <= ratio <= 2.50:
        return 100.0

    if 1.00 <= ratio < 1.20:
        return 82.0

    if 2.50 < ratio <= 3.50:
        return 85.0

    if 0.80 <= ratio < 1.00:
        return 62.0

    if 3.50 < ratio <= 5.00:
        return 62.0

    if ratio > 6.0:
        return 30.0

    return 42.0


def score_return_1d(
    value: float,
) -> float:
    if -1.5 <= value <= 2.5:
        return 100.0

    if 2.5 < value <= 4.5:
        return 82.0

    if -3.0 <= value < -1.5:
        return 72.0

    if 4.5 < value <= 6.0:
        return 55.0

    if value > 8.0:
        return 15.0

    return 40.0


def score_return_5d(
    value: float,
) -> float:
    if 0.0 <= value <= 6.0:
        return 100.0

    if -3.0 <= value < 0.0:
        return 78.0

    if 6.0 < value <= 10.0:
        return 82.0

    if 10.0 < value <= 15.0:
        return 58.0

    if value > 18.0:
        return 20.0

    return 45.0


def score_return_20d(
    value: float,
) -> float:
    if -5.0 <= value <= 15.0:
        return 100.0

    if 15.0 < value <= 25.0:
        return 72.0

    if -10.0 <= value < -5.0:
        return 62.0

    if 25.0 < value <= 35.0:
        return 45.0

    if value > 40.0:
        return 15.0

    return 35.0


# ============================================================
# ONCEKI MOTOR BONUSU
# ============================================================

def upstream_bonus(
    row: pd.Series,
) -> float:
    bonus = 0.0

    v22_score = number(
        row.get("v22_signal_score")
    )

    v27_score = number(
        row.get("v27_master_score")
    )

    v32_score = number(
        row.get("v32_score")
    )

    v22_state = text(
        row.get("v22_signal_state")
    ).upper()

    v27_decision = text(
        row.get("v27_decision")
    ).upper()

    v32_decision = text(
        row.get("v32_decision")
    ).upper()

    if v22_score >= 65:
        bonus += 2.0

    if (
        "AKTİF" in v22_state
        or "AKTIF" in v22_state
        or "ONAY" in v22_state
        or "GÜÇLÜ" in v22_state
        or "GUCLU" in v22_state
    ):
        bonus += 1.5

    if v27_score >= 65:
        bonus += 2.0

    if (
        "AKTİF" in v27_decision
        or "AKTIF" in v27_decision
        or "ONAY" in v27_decision
        or "TEYİT" in v27_decision
    ):
        bonus += 1.0

    if v32_score >= 65:
        bonus += 1.5

    if (
        "AKTİF" in v32_decision
        or "AKTIF" in v32_decision
        or "ONAY" in v32_decision
        or "TEYİT" in v32_decision
    ):
        bonus += 1.0

    return min(
        bonus,
        8.0,
    )


# ============================================================
# ANA PRESCAN SKORU
# ============================================================

def calculate_prescan_score(
    row: pd.Series,
) -> float:
    market_strength = score_market_strength(
        number(
            row.get(
                "market_percentile"
            ),
            50.0,
        )
    )

    ema_score = score_ema20(
        number(
            row.get(
                "ema20_distance"
            )
        )
    )

    volume_score = score_volume(
        number(
            row.get(
                "volume_ratio"
            ),
            1.0,
        )
    )

    return_1d_score = score_return_1d(
        number(
            row.get(
                "return_1d"
            )
        )
    )

    return_5d_score = score_return_5d(
        number(
            row.get(
                "return_5d"
            )
        )
    )

    return_20d_score = score_return_20d(
        number(
            row.get(
                "return_20d"
            )
        )
    )

    score = (
        market_strength * 0.25
        + ema_score * 0.20
        + volume_score * 0.18
        + return_5d_score * 0.15
        + return_20d_score * 0.12
        + return_1d_score * 0.10
    )

    score += upstream_bonus(
        row
    )

    # Gerçek risk verisi VARSA kullan.
    # Risk verisi yoksa cezalandırma/bonus verme.
    risk = optional_number(
        row.get("risk_score")
    )

    if np.isfinite(risk):
        if risk >= 65:
            score -= 8.0

        elif risk >= 55:
            score -= 4.0

        elif risk <= 30:
            score += 2.0

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


def classify_prescan(
    score: float,
) -> str:
    if score >= 88:
        return "UST ADAY"

    if score >= 78:
        return "GUCLU ADAY"

    if score >= 68:
        return "IZLEME ADAYI"

    if score >= 58:
        return "YEDEK ADAY"

    return "ZAYIF"


# ============================================================
# DESTEK VE RISK NOTLARI
# ============================================================

def build_supporting_factors(
    row: pd.Series,
) -> str:
    factors: list[str] = []

    return_5d = number(
        row.get("return_5d")
    )

    ema20 = number(
        row.get("ema20_distance")
    )

    volume = number(
        row.get("volume_ratio"),
        1.0,
    )

    percentile = number(
        row.get("market_percentile"),
        50.0,
    )

    if 0 <= return_5d <= 10:
        factors.append(
            "5 günlük momentum kontrollü"
        )

    if -3 <= ema20 <= 7:
        factors.append(
            "EMA20 konumu uygun"
        )

    if 1.2 <= volume <= 3.5:
        factors.append(
            "Hacim artışı sağlıklı"
        )

    if percentile >= 85:
        factors.append(
            "Piyasa göreli gücü yüksek"
        )

    ai_reason = text(
        row.get(
            "v20_ai_reasons"
        )
    )

    if ai_reason:
        factors.append(
            ai_reason
        )

    if not factors:
        return (
            "Belirgin ek destekleyici "
            "faktor bulunamadi"
        )

    return " | ".join(
        factors
    )


def build_risk_notes(
    row: pd.Series,
) -> str:
    risks: list[str] = []

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

    if return_5d > 15:
        risks.append(
            "5 günlük hareket şişkin"
        )

    if return_20d > 35:
        risks.append(
            "20 günlük hareket yüksek"
        )

    if ema20 > 14:
        risks.append(
            "EMA20 uzaklığı yüksek"
        )

    if volume > 5:
        risks.append(
            "Hacim aşırı yüksek"
        )

    if not bool(
        row.get("risk_available")
    ):
        risks.append(
            "V20 risk verisi mevcut değil"
        )

    if not bool(
        row.get("timing_available")
    ):
        risks.append(
            "V19/V20 zamanlama verisi mevcut değil"
        )

    v20_risk_reason = text(
        row.get(
            "v20_risk_reasons"
        )
    )

    if v20_risk_reason:
        risks.append(
            v20_risk_reason
        )

    if not risks:
        return "Belirgin ek risk notu yok"

    return " | ".join(
        risks
    )


# ============================================================
# ANA BIRLESTIRME
# ============================================================

def build_full_market_pool() -> pd.DataFrame:
    market = prepare_market()

    if market.empty:
        return pd.DataFrame()

    v22 = prepare_v22()
    v27 = prepare_v27()
    v32 = prepare_v32()
    v19 = prepare_v19()
    v20 = prepare_v20()

    regime_info = prepare_regime()

    pool = market.copy()

    for source in (
        v22,
        v27,
        v32,
        v19,
        v20,
    ):
        if not source.empty:
            pool = pool.merge(
                source,
                on="symbol",
                how="left",
            )

    # --------------------------------------------------------
    # Önceki motor alanları
    # --------------------------------------------------------

    text_defaults = {
        "v22_signal_state": "",
        "v27_decision": "",
        "v32_decision": "",
        "v20_risk_class": "",
        "v20_regime": "",
        "v20_ai_reasons": "",
        "v20_risk_reasons": "",
    }

    numeric_defaults = {
        "v22_signal_score": 0.0,
        "v27_master_score": 0.0,
        "v32_score": 0.0,
        "v32_confidence": 0.0,
    }

    for column, default in (
        text_defaults.items()
    ):
        ensure_column(
            pool,
            column,
            default,
        )

        pool[column] = (
            pool[column]
            .fillna(default)
            .astype(str)
            .str.strip()
        )

    for column, default in (
        numeric_defaults.items()
    ):
        ensure_column(
            pool,
            column,
            default,
        )

        pool[column] = (
            pd.to_numeric(
                pool[column],
                errors="coerce",
            )
            .fillna(default)
        )

    # --------------------------------------------------------
    # GERCEK REJIM
    # --------------------------------------------------------

    pool["regime"] = (
        regime_info["regime"]
    )

    pool[
        "regime_confidence"
    ] = regime_info[
        "regime_confidence"
    ]

    # --------------------------------------------------------
    # GERCEK ZAMANLAMA
    #
    # Öncelik:
    # 1) V20 timing_confidence
    # 2) V19 confidence_score
    # --------------------------------------------------------

    ensure_column(
        pool,
        "v20_timing_confidence",
        np.nan,
    )

    ensure_column(
        pool,
        "v19_timing_confidence",
        np.nan,
    )

    pool[
        "v20_timing_confidence"
    ] = pd.to_numeric(
        pool[
            "v20_timing_confidence"
        ],
        errors="coerce",
    )

    pool[
        "v19_timing_confidence"
    ] = pd.to_numeric(
        pool[
            "v19_timing_confidence"
        ],
        errors="coerce",
    )

    pool[
        "timing_confidence"
    ] = pool[
        "v20_timing_confidence"
    ].combine_first(
        pool[
            "v19_timing_confidence"
        ]
    )

    pool[
        "timing_available"
    ] = pool[
        "timing_confidence"
    ].notna()

    pool[
        "timing_source"
    ] = np.select(
        [
            pool[
                "v20_timing_confidence"
            ].notna(),

            pool[
                "v19_timing_confidence"
            ].notna(),
        ],
        [
            "V20",
            "V19",
        ],
        default="VERI_YOK",
    )

    # --------------------------------------------------------
    # GERCEK RISK
    #
    # V20'de hisse varsa gerçek değer.
    # Yoksa uydurma 50 YOK.
    # --------------------------------------------------------

    ensure_column(
        pool,
        "v20_risk_score",
        np.nan,
    )

    ensure_column(
        pool,
        "v20_risk_class",
        "",
    )

    pool[
        "risk_score"
    ] = pd.to_numeric(
        pool[
            "v20_risk_score"
        ],
        errors="coerce",
    )

    pool[
        "risk_available"
    ] = pool[
        "risk_score"
    ].notna()

    pool[
        "risk_class"
    ] = (
        pool[
            "v20_risk_class"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    pool.loc[
        ~pool[
            "risk_available"
        ],
        "risk_class",
    ] = "VERI YOK"

    pool[
        "risk_source"
    ] = np.where(
        pool[
            "risk_available"
        ],
        "V20",
        "VERI_YOK",
    )

    # --------------------------------------------------------
    # PRESCAN SCORE
    # --------------------------------------------------------

    pool[
        "prescan_score"
    ] = pool.apply(
        calculate_prescan_score,
        axis=1,
    )

    pool[
        "prescan_class"
    ] = pool[
        "prescan_score"
    ].apply(
        classify_prescan
    )

    # --------------------------------------------------------
    # Metin alanları
    # --------------------------------------------------------

    pool[
        "supporting_factors"
    ] = pool.apply(
        build_supporting_factors,
        axis=1,
    )

    pool[
        "risk_notes"
    ] = pool.apply(
        build_risk_notes,
        axis=1,
    )

    pool[
        "rsi_usage"
    ] = RSI_USAGE

    return pool


# ============================================================
# ADAY SECIMI
# ============================================================

def select_candidates(
    pool: pd.DataFrame,
) -> pd.DataFrame:
    if pool.empty:
        return pd.DataFrame()

    result = pool.copy()

    # Gerçek risk verisi olan ve risk > 70 olanları
    # ana seçimden çıkar.
    eligible = result[
        (
            ~result["risk_available"]
        )
        |
        (
            result["risk_score"]
            <= MAX_RISK_SCORE
        )
    ].copy()

    eligible = eligible.sort_values(
        [
            "prescan_score",
            "market_percentile",
            "volume_ratio",
        ],
        ascending=False,
    )

    selected = eligible.head(
        MAX_CANDIDATES
    ).copy()

    # Olağanüstü durumda 10 adaydan az kalırsa,
    # güvenlik için en yüksek skorlu adaylardan tamamla.
    if len(selected) < MIN_CANDIDATES:
        used_symbols = set(
            selected["symbol"]
        )

        extras = result[
            ~result["symbol"].isin(
                used_symbols
            )
        ].sort_values(
            "prescan_score",
            ascending=False,
        )

        needed = (
            MIN_CANDIDATES
            - len(selected)
        )

        selected = pd.concat(
            [
                selected,
                extras.head(
                    needed
                ),
            ],
            ignore_index=True,
        )

    selected = selected.sort_values(
        [
            "prescan_score",
            "market_percentile",
        ],
        ascending=False,
    ).head(
        MAX_CANDIDATES
    ).reset_index(
        drop=True
    )

    selected.insert(
        0,
        "prescan_rank",
        range(
            1,
            len(selected) + 1,
        ),
    )

    return selected


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    pool = build_full_market_pool()

    if pool.empty:
        empty = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

        empty.to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        status = {
            "status": "market_input_missing",
            "market_count": 0,
            "selected_count": 0,

            "regime": "BILINMIYOR",
            "regime_confidence": None,

            "timing_data_count": 0,
            "risk_data_count": 0,

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

    selected = select_candidates(
        pool
    )

    # --------------------------------------------------------
    # Output kolonları
    # --------------------------------------------------------

    output = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column in selected.columns:
            output[column] = (
                selected[column]
            )
        else:
            output[column] = np.nan

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    regime_value = (
        text(
            selected.iloc[0][
                "regime"
            ]
        )
        if len(selected)
        else "BILINMIYOR"
    )

    regime_confidence = (
        optional_number(
            selected.iloc[0][
                "regime_confidence"
            ]
        )
        if len(selected)
        else np.nan
    )

    top_symbol = (
        text(
            output.iloc[0][
                "symbol"
            ]
        )
        if len(output)
        else ""
    )

    top_score = (
        number(
            output.iloc[0][
                "prescan_score"
            ]
        )
        if len(output)
        else 0.0
    )

    status = {
        "status": "ready",

        "market_count": int(
            len(pool)
        ),

        "selected_count": int(
            len(output)
        ),

        "max_candidates": MAX_CANDIDATES,

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

        "timing_data_count": int(
            pool[
                "timing_available"
            ].sum()
        ),

        "selected_timing_data_count": int(
            output[
                "timing_available"
            ].fillna(False).sum()
        ),

        "risk_data_count": int(
            pool[
                "risk_available"
            ].sum()
        ),

        "selected_risk_data_count": int(
            output[
                "risk_available"
            ].fillna(False).sum()
        ),

        "top_symbol": top_symbol,

        "top_score": round(
            top_score,
            2,
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
        "===== V33.2 PRESCAN STATUS ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "===== V33.2 TOP CANDIDATES ====="
    )

    if output.empty:
        print(
            "Aday bulunamadi."
        )

    else:
        display_columns = [
            "prescan_rank",
            "symbol",
            "prescan_score",
            "prescan_class",
            "market_percentile",
            "timing_confidence",
            "timing_source",
            "risk_score",
            "risk_class",
            "risk_source",
            "regime",
        ]

        print(
            output[
                display_columns
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
