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
    "risk_score",
    "risk_class",
    "regime",
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


def load_csv(path: Path) -> pd.DataFrame:
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
                f"UYARI: {path} okunamadı: {exc}"
            )
            return pd.DataFrame()

    return pd.DataFrame()


def normalize_symbol(value: Any) -> str:
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
# EK MOTOR VERILERI
# ============================================================

def prepare_v22() -> pd.DataFrame:
    frame = normalize_frame(
        load_csv(V22_FILE)
    )

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v22_signal_state",
                "v22_signal_score",
            ]
        )

    ensure_column(
        frame,
        "v22_signal_state",
        "",
    )

    ensure_column(
        frame,
        "v22_signal_score",
        0.0,
    )

    return frame[
        [
            "symbol",
            "v22_signal_state",
            "v22_signal_score",
        ]
    ].copy()


def prepare_v27() -> pd.DataFrame:
    frame = normalize_frame(
        load_csv(V27_FILE)
    )

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v27_decision",
                "v27_master_score",
            ]
        )

    ensure_column(
        frame,
        "v27_decision",
        "",
    )

    ensure_column(
        frame,
        "v27_master_score",
        0.0,
    )

    return frame[
        [
            "symbol",
            "v27_decision",
            "v27_master_score",
        ]
    ].copy()


def prepare_v32() -> pd.DataFrame:
    frame = normalize_frame(
        load_csv(V32_FILE)
    )

    if frame.empty:
        return pd.DataFrame(
            columns=[
                "symbol",
                "v32_decision",
                "v32_score",
                "v32_confidence",
            ]
        )

    ensure_column(
        frame,
        "v32_decision",
        "",
    )

    ensure_column(
        frame,
        "v32_score",
        0.0,
    )

    ensure_column(
        frame,
        "v32_confidence",
        0.0,
    )

    return frame[
        [
            "symbol",
            "v32_decision",
            "v32_score",
            "v32_confidence",
        ]
    ].copy()


# ============================================================
# ANA PIYASA VERISI
# ============================================================

def prepare_market() -> pd.DataFrame:
    market = normalize_frame(
        load_csv(MARKET_FILE)
    )

    if market.empty:
        return market

    numeric_defaults = {
        "close": 0.0,
        "return_1d": 0.0,
        "return_5d": 0.0,
        "return_20d": 0.0,
        "ema20_distance": 0.0,
        "volume_ratio": 0.0,
        "market_percentile": 0.0,
        "timing_confidence": 0.0,
        "risk_score": 50.0,
    }

    text_defaults = {
        "risk_class": "ORTA",
        "regime": "",
    }

    for column, default in numeric_defaults.items():
        ensure_column(
            market,
            column,
            default,
        )

        market[column] = pd.to_numeric(
            market[column],
            errors="coerce",
        ).fillna(default)

    for column, default in text_defaults.items():
        ensure_column(
            market,
            column,
            default,
        )

        market[column] = (
            market[column]
            .fillna(default)
            .astype(str)
            .str.strip()
        )

    return market


# ============================================================
# PUAN BILESENLERI
# ============================================================

def score_momentum(
    return_1d: float,
    return_5d: float,
    return_20d: float,
) -> float:
    score = 0.0

    # 1 günlük hareket
    if -1.5 <= return_1d <= 4.0:
        score += 12.0
    elif 4.0 < return_1d <= 7.0:
        score += 7.0
    elif return_1d > 9.0:
        score -= 8.0

    # 5 günlük yapı
    if 0.0 <= return_5d <= 8.0:
        score += 16.0
    elif 8.0 < return_5d <= 14.0:
        score += 10.0
    elif return_5d > 18.0:
        score -= 10.0

    # 20 günlük yapı
    if -5.0 <= return_20d <= 18.0:
        score += 10.0
    elif 18.0 < return_20d <= 28.0:
        score += 5.0
    elif return_20d > 35.0:
        score -= 8.0

    return score


def score_ema(
    ema20_distance: float,
) -> float:
    if -3.0 <= ema20_distance <= 6.0:
        return 18.0

    if 6.0 < ema20_distance <= 10.0:
        return 12.0

    if -6.0 <= ema20_distance < -3.0:
        return 8.0

    if 10.0 < ema20_distance <= 15.0:
        return 5.0

    if ema20_distance > 18.0:
        return -10.0

    return 0.0


def score_volume(
    volume_ratio: float,
) -> float:
    if 1.2 <= volume_ratio <= 3.5:
        return 18.0

    if 0.9 <= volume_ratio < 1.2:
        return 10.0

    if 3.5 < volume_ratio <= 5.0:
        return 9.0

    if volume_ratio > 6.0:
        return -5.0

    return 3.0


def score_relative_strength(
    market_percentile: float,
) -> float:
    if market_percentile >= 90:
        return 18.0

    if market_percentile >= 80:
        return 14.0

    if market_percentile >= 70:
        return 10.0

    if market_percentile >= 60:
        return 6.0

    return 0.0


def score_timing(
    timing_confidence: float,
) -> float:
    if timing_confidence >= 85:
        return 12.0

    if timing_confidence >= 75:
        return 9.0

    if timing_confidence >= 65:
        return 6.0

    if timing_confidence >= 55:
        return 3.0

    return 0.0


def score_risk(
    risk_score: float,
) -> float:
    if risk_score <= 20:
        return 8.0

    if risk_score <= 35:
        return 5.0

    if risk_score <= 50:
        return 0.0

    if risk_score <= 60:
        return -5.0

    return -12.0


def score_previous_layers(
    row: pd.Series,
) -> float:
    score = 0.0

    v22 = number(
        row.get("v22_signal_score")
    )

    v27 = number(
        row.get("v27_master_score")
    )

    v32 = number(
        row.get("v32_score")
    )

    # Eski katmanlar sadece bonus.
    # Aday olmanın şartı değiller.
    if v22 >= 70:
        score += 5.0
    elif v22 >= 60:
        score += 3.0

    if v27 >= 60:
        score += 4.0
    elif v27 >= 50:
        score += 2.0

    if v32 >= 60:
        score += 4.0
    elif v32 >= 50:
        score += 2.0

    return score


# ============================================================
# ACIKLAMA
# ============================================================

def build_notes(
    row: pd.Series,
) -> tuple[str, str]:
    positives: list[str] = []
    risks: list[str] = []

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

    risk = number(
        row.get("risk_score"),
        50.0,
    )

    if 0 <= return_5d <= 8:
        positives.append(
            "5 günlük momentum kontrollü"
        )

    if -3 <= ema20 <= 6:
        positives.append(
            "EMA20 konumu uygun"
        )

    if 1.2 <= volume <= 3.5:
        positives.append(
            "Hacim artışı sağlıklı"
        )

    if market_pct >= 80:
        positives.append(
            "Piyasa göreli gücü yüksek"
        )

    if timing >= 75:
        positives.append(
            "Zamanlama güveni yüksek"
        )

    if risk <= 35:
        positives.append(
            "Risk puanı düşük"
        )

    if return_1d > 7:
        risks.append(
            "Günlük hareket hızlı"
        )

    if return_5d > 14:
        risks.append(
            "5 günlük hareket şişkin"
        )

    if return_20d > 28:
        risks.append(
            "20 günlük hareket yüksek"
        )

    if ema20 > 12:
        risks.append(
            "EMA20'den uzak"
        )

    if volume > 5:
        risks.append(
            "Hacim aşırı yüksek"
        )

    if risk > 55:
        risks.append(
            "Risk puanı yüksek"
        )

    if not positives:
        positives.append(
            "Belirgin güçlü ön tarama özelliği yok"
        )

    if not risks:
        risks.append(
            "Belirgin ek risk notu yok"
        )

    return (
        " | ".join(positives),
        " | ".join(risks),
    )


# ============================================================
# SINIFLANDIRMA
# ============================================================

def classify(
    score: float,
) -> str:
    if score >= 78:
        return "A+"

    if score >= 68:
        return "A"

    if score >= 58:
        return "B"

    if score >= 48:
        return "C"

    return "ZAYIF"


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    market = prepare_market()

    if market.empty:
        status = {
            "status": "market_input_missing",
            "market_count": 0,
            "selected_count": 0,
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

        pd.DataFrame(
            columns=OUTPUT_COLUMNS
        ).to_csv(
            OUTPUT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

        print(
            json.dumps(
                status,
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    v22 = prepare_v22()
    v27 = prepare_v27()
    v32 = prepare_v32()

    merged = market.copy()

    if not v22.empty:
        merged = merged.merge(
            v22,
            on="symbol",
            how="left",
        )

    if not v27.empty:
        merged = merged.merge(
            v27,
            on="symbol",
            how="left",
        )

    if not v32.empty:
        merged = merged.merge(
            v32,
            on="symbol",
            how="left",
        )

    defaults = {
        "v22_signal_state": "",
        "v22_signal_score": 0.0,
        "v27_decision": "",
        "v27_master_score": 0.0,
        "v32_decision": "",
        "v32_score": 0.0,
        "v32_confidence": 0.0,
    }

    for column, default in defaults.items():
        ensure_column(
            merged,
            column,
            default,
        )

    numeric_columns = [
        "v22_signal_score",
        "v27_master_score",
        "v32_score",
        "v32_confidence",
    ]

    for column in numeric_columns:
        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(0.0)

    # --------------------------------------------------------
    # HARD FILTER
    # --------------------------------------------------------

    merged = merged[
        merged["close"] > 0
    ].copy()

    merged = merged[
        merged["risk_score"]
        < MAX_RISK_SCORE
    ].copy()

    # --------------------------------------------------------
    # PUAN
    # --------------------------------------------------------

    scores: list[float] = []
    supporting_notes: list[str] = []
    risk_notes: list[str] = []

    for _, row in merged.iterrows():
        momentum_score = score_momentum(
            number(
                row.get("return_1d")
            ),
            number(
                row.get("return_5d")
            ),
            number(
                row.get("return_20d")
            ),
        )

        ema_score = score_ema(
            number(
                row.get("ema20_distance")
            )
        )

        volume_score = score_volume(
            number(
                row.get("volume_ratio")
            )
        )

        relative_score = score_relative_strength(
            number(
                row.get("market_percentile")
            )
        )

        timing_score = score_timing(
            number(
                row.get("timing_confidence")
            )
        )

        risk_component = score_risk(
            number(
                row.get("risk_score"),
                50.0,
            )
        )

        layer_bonus = score_previous_layers(
            row
        )

        raw_score = (
            momentum_score
            + ema_score
            + volume_score
            + relative_score
            + timing_score
            + risk_component
            + layer_bonus
        )

        final_score = float(
            np.clip(
                raw_score,
                0,
                100,
            )
        )

        positive_note, risk_note = build_notes(
            row
        )

        scores.append(
            round(
                final_score,
                2,
            )
        )

        supporting_notes.append(
            positive_note
        )

        risk_notes.append(
            risk_note
        )

    merged["prescan_score"] = scores
    merged["supporting_factors"] = (
        supporting_notes
    )
    merged["risk_notes"] = risk_notes

    merged["prescan_class"] = (
        merged["prescan_score"]
        .apply(
            classify
        )
    )

    # --------------------------------------------------------
    # SIRALAMA
    # --------------------------------------------------------

    merged = merged.sort_values(
        [
            "prescan_score",
            "market_percentile",
            "timing_confidence",
            "volume_ratio",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    # Önce 48+ olanları al.
    selected = merged[
        merged["prescan_score"] >= 48
    ].copy()

    # Çok az kaldıysa ilk MIN_CANDIDATES'a tamamla.
    if len(selected) < MIN_CANDIDATES:
        selected = merged.head(
            MIN_CANDIDATES
        ).copy()

    selected = selected.head(
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

    selected["rsi_usage"] = RSI_USAGE

    # --------------------------------------------------------
    # ÇIKTI
    # --------------------------------------------------------

    result = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column in selected.columns:
            result[column] = selected[column]
        else:
            result[column] = np.nan

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": "ready",
        "market_count": int(
            len(market)
        ),
        "eligible_count": int(
            len(merged)
        ),
        "selected_count": int(
            len(result)
        ),
        "top_symbol": (
            text(
                result.iloc[0]["symbol"]
            )
            if len(result)
            else ""
        ),
        "top_score": (
            round(
                number(
                    result.iloc[0][
                        "prescan_score"
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
        "===== V33.2 FULL MARKET PRESCAN STATUS ====="
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

    print(
        result[
            [
                "prescan_rank",
                "symbol",
                "prescan_score",
                "prescan_class",
                "return_5d",
                "ema20_distance",
                "volume_ratio",
                "market_percentile",
                "timing_confidence",
                "risk_score",
            ]
        ].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
