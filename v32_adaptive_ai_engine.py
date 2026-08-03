from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# DOSYA YOLLARI
# ============================================================

V27_FILE = Path("v27_master_decisions.csv")
V31_FILE = Path("v31_learned_decisions.csv")
V30_STATUS_FILE = Path("v30_status.json")
V31_STATUS_FILE = Path("v31_status.json")

OUTPUT_FILE = Path("v32_adaptive_decisions.csv")
STATUS_FILE = Path("v32_status.json")


# ============================================================
# SÜRÜM VE AYARLAR
# ============================================================

VERSION = "V32.0"

# V32 başlangıçta güvenli gölge modunda çalışır.
SHADOW_MODE = True

# Öğrenmenin gerçek karar puanına etki edebilmesi için
# en az bu kadar tamamlanmış gözlem gerekir.
MIN_COMPLETED_OBSERVATIONS = 30

# En az bu kadar kullanılabilir örüntü olmalıdır.
MIN_USABLE_PATTERNS = 3

# En az bu kadar aday örüntüyle eşleşmelidir.
MIN_MATCHED_CANDIDATES = 1

# V32 öğrenme etkisinin üst ve alt sınırı.
MAX_AI_BONUS = 10.0
MAX_AI_PENALTY = -8.0

# RSI hiçbir koşulda kullanılmaz.
IGNORED_FEATURES = {
    "rsi",
    "rsi14",
    "rsi_14",
    "rsi_score",
    "rsi_percentile",
}


OUTPUT_COLUMNS = [
    "v32_rank",
    "symbol",
    "v32_decision",
    "v32_score",
    "v32_ai_adjustment",
    "v32_confidence",
    "v32_mode",
    "v32_reason",
    "v32_supports",
    "v32_risks",
    "rsi_usage",
    "v31_decision",
    "v31_score",
    "v31_learning_bonus",
    "matched_pattern_count",
    "positive_pattern_count",
    "negative_pattern_count",
    "v27_decision",
    "v27_master_score",
    "v22_signal_state",
    "v22_signal_score",
    "v24_state",
    "v24_score",
    "optimized_weight_pct",
    "optimizer_score",
    "quality_score",
    "consensus_score",
    "risk_class",
    "risk_score",
    "regime",
    "market_percentile",
    "timing_confidence",
    "expected_return",
    "downside_20pct",
    "upside_80pct",
    "close",
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


def integer(
    value: Any,
    default: int = 0,
) -> int:
    try:
        result = float(value)

        if np.isfinite(result):
            return int(result)

        return default

    except (TypeError, ValueError):
        return default


def boolean_value(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    normalized = text(value).lower()

    if normalized in {
        "true",
        "1",
        "yes",
        "evet",
        "active",
        "aktif",
    }:
        return True

    if normalized in {
        "false",
        "0",
        "no",
        "hayır",
        "hayir",
        "inactive",
        "pasif",
    }:
        return False

    return default


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"UYARI: {path} bulunamadı.")
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            print(f"UYARI: {path} boş.")
            return pd.DataFrame()
    except OSError as exc:
        print(f"UYARI: {path} kontrol edilemedi: {exc}")
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

        except pd.errors.ParserError as exc:
            print(
                f"UYARI: {path} ayrıştırılamadı: {exc}"
            )
            return pd.DataFrame()

        except Exception as exc:
            print(
                f"UYARI: {path} okunamadı: {exc}"
            )
            return pd.DataFrame()

    return pd.DataFrame()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    try:
        raw = path.read_text(
            encoding="utf-8",
        )

        if not raw.strip():
            return {}

        data = json.loads(raw)

        if isinstance(data, dict):
            return data

        return {}

    except Exception as exc:
        print(
            f"UYARI: {path} okunamadı: {exc}"
        )
        return {}


def normalize_symbol_column(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    if "symbol" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()

    result["symbol"] = (
        result["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(
            ".IS",
            "",
            regex=False,
        )
    )

    result = result[
        result["symbol"].ne("")
    ].copy()

    result = result.drop_duplicates(
        subset=["symbol"],
        keep="first",
    )

    return result


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


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


def save_empty_result(
    status_name: str,
    message: str,
) -> None:
    pd.DataFrame(
        columns=OUTPUT_COLUMNS
    ).to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": status_name,
        "message": message,
        "candidate_count": 0,
        "approved_count": 0,
        "shadow_tracking_count": 0,
        "waiting_count": 0,
        "passive_count": 0,
        "eliminated_count": 0,
        "learning_ready": False,
        "completed_observation_count": 0,
        "usable_pattern_count": 0,
        "matched_candidate_count": 0,
        "top_symbol": "",
        "top_decision": "",
        "top_score": 0.0,
        "mode": "SHADOW",
        "rsi_usage": "DISABLED",
        "version": VERSION,
    }

    save_status(status)

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# ÖĞRENME HAZIRLIK KONTROLÜ
# ============================================================

def detect_learning_readiness(
    v30_status: dict[str, Any],
    v31_status: dict[str, Any],
) -> dict[str, Any]:
    completed = integer(
        v30_status.get(
            "completed_observation_count",
            v30_status.get(
                "completed_count",
                0,
            ),
        )
    )

    usable_patterns = integer(
        v31_status.get(
            "usable_pattern_count",
            0,
        )
    )

    matched_candidates = integer(
        v31_status.get(
            "matched_candidate_count",
            0,
        )
    )

    v30_learning_active = boolean_value(
        v30_status.get(
            "learning_active",
            False,
        )
    )

    ready = (
        completed >= MIN_COMPLETED_OBSERVATIONS
        and usable_patterns >= MIN_USABLE_PATTERNS
        and matched_candidates >= MIN_MATCHED_CANDIDATES
    )

    return {
        "learning_ready": ready,
        "completed_observation_count": completed,
        "usable_pattern_count": usable_patterns,
        "matched_candidate_count": matched_candidates,
        "v30_learning_active": v30_learning_active,
    }


# ============================================================
# REJİM ETKİSİ
# ============================================================

def regime_adjustment(
    regime: str,
    expected_return: float,
    risk_score: float,
    market_percentile: float,
) -> tuple[float, list[str], list[str]]:
    regime = text(regime).upper()

    adjustment = 0.0
    supports: list[str] = []
    risks: list[str] = []

    if regime == "RALLİ":
        if expected_return > 2:
            adjustment += 1.5
            supports.append(
                "Ralli rejiminde beklenen getiri pozitif"
            )

        if market_percentile >= 80:
            adjustment += 1.0
            supports.append(
                "Ralli rejiminde göreli güç yüksek"
            )

    elif regime == "TREND":
        if market_percentile >= 75:
            adjustment += 1.0
            supports.append(
                "Trend rejiminde piyasa göreli gücü yüksek"
            )

        if risk_score <= 30:
            adjustment += 0.5
            supports.append(
                "Trend rejiminde risk kontrollü"
            )

    elif regime == "YATAY":
        if risk_score <= 25:
            adjustment += 0.5
            supports.append(
                "Yatay rejimde düşük risk avantajı"
            )

        if expected_return < 1:
            adjustment -= 1.0
            risks.append(
                "Yatay rejimde beklenen getiri zayıf"
            )

    elif regime == "PANİK":
        adjustment -= 4.0
        risks.append(
            "Panik rejimi genel risk kesintisi"
        )

        if market_percentile >= 90:
            adjustment += 1.5
            supports.append(
                "Panik rejiminde göreli güç liderliği"
            )

        if risk_score <= 20:
            adjustment += 0.5
            supports.append(
                "Panik rejiminde risk seviyesi düşük"
            )

    return (
        round(adjustment, 2),
        supports,
        risks,
    )


# ============================================================
# ADAPTİF PUAN HESABI
# ============================================================

def calculate_ai_adjustment(
    row: pd.Series,
    learning_ready: bool,
) -> dict[str, Any]:
    v27_score = number(
        row.get("v27_master_score")
    )

    v31_score = number(
        row.get(
            "v31_score",
            v27_score,
        )
    )

    v31_bonus = number(
        row.get("v31_learning_bonus")
    )

    matched_count = integer(
        row.get("matched_pattern_count")
    )

    positive_patterns = integer(
        row.get("positive_pattern_count")
    )

    negative_patterns = integer(
        row.get("negative_pattern_count")
    )

    risk_score = number(
        row.get("risk_score"),
        100.0,
    )

    consensus = number(
        row.get("consensus_score")
    )

    timing = number(
        row.get("timing_confidence")
    )

    market_percentile = number(
        row.get("market_percentile")
    )

    expected_return = number(
        row.get("expected_return")
    )

    downside = number(
        row.get("downside_20pct")
    )

    v24_score = number(
        row.get("v24_score")
    )

    v24_state = text(
        row.get("v24_state")
    ).upper()

    regime = text(
        row.get("regime")
    )

    adjustment = 0.0
    supports: list[str] = []
    risks: list[str] = []

    # --------------------------------------------------------
    # ÖĞRENME ETKİSİ
    # --------------------------------------------------------

    if learning_ready:
        adjustment += v31_bonus

        if v31_bonus > 0:
            supports.append(
                f"Geçmiş örüntü desteği {v31_bonus:+.2f} puan"
            )

        elif v31_bonus < 0:
            risks.append(
                f"Geçmiş örüntü uyarısı {v31_bonus:+.2f} puan"
            )

        if matched_count >= 3:
            adjustment += 1.0
            supports.append(
                "Birden fazla öğrenilmiş örüntü eşleşti"
            )

        if positive_patterns >= 2:
            adjustment += 1.0
            supports.append(
                "Olumlu örüntü sayısı güçlü"
            )

        if negative_patterns >= 2:
            adjustment -= 2.0
            risks.append(
                "Birden fazla olumsuz örüntü eşleşti"
            )

    else:
        # Veri azsa öğrenme puanı karar puanına eklenmez.
        risks.append(
            "Öğrenilmiş veri henüz gerçek puan etkisi için yeterli değil"
        )

    # --------------------------------------------------------
    # CANLI TEYİT
    # --------------------------------------------------------

    if v24_state in {
        "CANLI TEYİT GELDİ",
        "GÜÇLÜ CANLI TEYİT",
    }:
        adjustment += 3.0
        supports.append(
            "Canlı teknik teyit güçlü"
        )

    elif v24_state == "ERKEN TEYİT":
        adjustment += 1.5
        supports.append(
            "Erken canlı teyit oluştu"
        )

    elif v24_state == "TEYİT BEKLE":
        adjustment -= 1.0
        risks.append(
            "Canlı teknik teyit henüz oluşmadı"
        )

    elif v24_state in {
        "ŞİŞKİN / RİSKLİ",
        "ELE",
    }:
        adjustment -= 6.0
        risks.append(
            "Canlı görünüm şişkin veya riskli"
        )

    # V24 durumu eksik olsa bile skor destekleyebilir.
    if v24_score >= 75:
        adjustment += 1.0
        supports.append(
            "Canlı teyit skoru yüksek"
        )

    # --------------------------------------------------------
    # MOTOR UYUMU
    # --------------------------------------------------------

    if consensus >= 80:
        adjustment += 2.0
        supports.append(
            "Motorlar arasında yüksek uyum"
        )

    elif consensus >= 68:
        adjustment += 1.0
        supports.append(
            "Motor uyumu olumlu"
        )

    elif consensus < 50:
        adjustment -= 2.0
        risks.append(
            "Motorlar arasında görüş ayrılığı yüksek"
        )

    # --------------------------------------------------------
    # ZAMANLAMA
    # --------------------------------------------------------

    if timing >= 80:
        adjustment += 1.5
        supports.append(
            "Zamanlama güveni yüksek"
        )

    elif timing >= 65:
        adjustment += 0.5
        supports.append(
            "Zamanlama görünümü olumlu"
        )

    elif timing < 45:
        adjustment -= 1.0
        risks.append(
            "Zamanlama güveni zayıf"
        )

    # --------------------------------------------------------
    # RİSK
    # --------------------------------------------------------

    if risk_score <= 20:
        adjustment += 1.5
        supports.append(
            "Risk puanı çok düşük"
        )

    elif risk_score <= 30:
        adjustment += 0.5
        supports.append(
            "Risk puanı düşük"
        )

    elif risk_score >= 55:
        adjustment -= 4.0
        risks.append(
            "Risk puanı yüksek"
        )

    elif risk_score >= 40:
        adjustment -= 1.5
        risks.append(
            "Risk puanı orta-yüksek"
        )

    if downside <= -7:
        adjustment -= 5.0
        risks.append(
            "Temkinli senaryoda aşağı yön riski yüksek"
        )

    elif downside <= -4:
        adjustment -= 2.0
        risks.append(
            "Temkinli senaryo zayıf"
        )

    # --------------------------------------------------------
    # BEKLENEN SONUÇ
    # --------------------------------------------------------

    if expected_return >= 5:
        adjustment += 1.5
        supports.append(
            "Beklenen istatistiksel getiri güçlü"
        )

    elif expected_return >= 2:
        adjustment += 0.5
        supports.append(
            "Beklenen istatistiksel getiri pozitif"
        )

    elif expected_return <= 0:
        adjustment -= 3.0
        risks.append(
            "Beklenen istatistiksel getiri pozitif değil"
        )

    # --------------------------------------------------------
    # PİYASA GÖRELİ GÜCÜ
    # --------------------------------------------------------

    if market_percentile >= 90:
        adjustment += 1.5
        supports.append(
            "Piyasa göreli gücü üst yüzde 10 diliminde"
        )

    elif market_percentile >= 75:
        adjustment += 0.5
        supports.append(
            "Piyasa göreli gücü yüksek"
        )

    elif market_percentile < 40:
        adjustment -= 1.0
        risks.append(
            "Piyasa göreli gücü zayıf"
        )

    # --------------------------------------------------------
    # REJİM
    # --------------------------------------------------------

    regime_effect, regime_supports, regime_risks = (
        regime_adjustment(
            regime=regime,
            expected_return=expected_return,
            risk_score=risk_score,
            market_percentile=market_percentile,
        )
    )

    adjustment += regime_effect
    supports.extend(regime_supports)
    risks.extend(regime_risks)

    # --------------------------------------------------------
    # PUAN FARKI KONTROLÜ
    # --------------------------------------------------------

    learned_difference = v31_score - v27_score

    if learning_ready and learned_difference > 0:
        adjustment += min(
            learned_difference * 0.25,
            2.0,
        )

    elif learning_ready and learned_difference < 0:
        adjustment += max(
            learned_difference * 0.25,
            -2.0,
        )

    adjustment = float(
        np.clip(
            adjustment,
            MAX_AI_PENALTY,
            MAX_AI_BONUS,
        )
    )

    # Güven puanı
    confidence = (
        consensus * 0.30
        + timing * 0.20
        + max(
            0.0,
            100.0 - risk_score,
        ) * 0.20
        + market_percentile * 0.10
        + min(
            matched_count * 10.0,
            100.0,
        ) * 0.20
    )

    if not learning_ready:
        confidence *= 0.70

    confidence = float(
        np.clip(
            confidence,
            0.0,
            100.0,
        )
    )

    return {
        "adjustment": round(
            adjustment,
            2,
        ),
        "confidence": round(
            confidence,
            2,
        ),
        "supports": (
            " | ".join(supports)
            if supports
            else "Belirgin destekleyici unsur yok"
        ),
        "risks": (
            " | ".join(risks)
            if risks
            else "Belirgin ek risk notu yok"
        ),
    }


# ============================================================
# V32 KARAR MANTIĞI
# ============================================================

def determine_v32_decision(
    row: pd.Series,
    learning_ready: bool,
) -> tuple[str, str]:
    score = number(
        row.get("v32_score")
    )

    confidence = number(
        row.get("v32_confidence")
    )

    risk_score = number(
        row.get("risk_score"),
        100.0,
    )

    downside = number(
        row.get("downside_20pct")
    )

    expected_return = number(
        row.get("expected_return")
    )

    v24_state = text(
        row.get("v24_state")
    ).upper()

    v31_bonus = number(
        row.get("v31_learning_bonus")
    )

    if (
        risk_score >= 65
        or downside <= -8
        or v24_state in {
            "ŞİŞKİN / RİSKLİ",
            "ELE",
        }
    ):
        return (
            "ELE",
            (
                "Risk veya canlı teknik görünüm "
                "kabul edilebilir seviyenin dışında"
            ),
        )

    # Öğrenme henüz güvenilir değilse gölge modunda kal.
    if not learning_ready:
        if score >= 65 and risk_score <= 45:
            return (
                "GÖLGE İZLEME",
                (
                    "Temel görünüm olumlu ancak "
                    "öğrenilmiş veri henüz yeterli değil"
                ),
            )

        if score >= 52 and risk_score <= 55:
            return (
                "TEYİT BEKLE",
                (
                    "Temel skor takip edilebilir fakat "
                    "öğrenme güveni henüz oluşmadı"
                ),
            )

        return (
            "PASİF İZLEME",
            (
                "Toplam görünüm ve öğrenme güveni "
                "aktif takip için yeterli değil"
            ),
        )

    if (
        score >= 84
        and confidence >= 75
        and v31_bonus > 0
        and risk_score <= 30
        and expected_return > 0
        and v24_state in {
            "CANLI TEYİT GELDİ",
            "GÜÇLÜ CANLI TEYİT",
            "ERKEN TEYİT",
        }
    ):
        return (
            "ADAPTİF GÜÇLÜ TEYİT",
            (
                "Temel analiz, öğrenilmiş örüntü "
                "ve canlı teyit birlikte güçlü"
            ),
        )

    if (
        score >= 74
        and confidence >= 65
        and risk_score <= 40
        and expected_return > 0
    ):
        return (
            "ADAPTİF AKTİF İZLEME",
            (
                "Öğrenilmiş veri ve temel analiz "
                "aktif takibi destekliyor"
            ),
        )

    if (
        score >= 60
        and risk_score <= 50
    ):
        return (
            "TEYİT BEKLE",
            (
                "Toplam görünüm olumlu ancak "
                "bütün teyit koşulları tamamlanmadı"
            ),
        )

    return (
        "PASİF İZLEME",
        (
            "Adaptif skor aktif takip için "
            "yeterli seviyeye ulaşmadı"
        ),
    )


# ============================================================
# ANA MOTOR
# ============================================================

def main() -> None:
    print(
        "===== V32 ADAPTİF KARAR MOTORU BAŞLADI ====="
    )

    v27 = normalize_symbol_column(
        load_csv(V27_FILE)
    )

    v31 = normalize_symbol_column(
        load_csv(V31_FILE)
    )

    v30_status = load_json(
        V30_STATUS_FILE
    )

    v31_status = load_json(
        V31_STATUS_FILE
    )

    if v27.empty:
        save_empty_result(
            status_name="v27_input_missing",
            message=(
                "v27_master_decisions.csv "
                "bulunamadı veya boş."
            ),
        )
        return

    readiness = detect_learning_readiness(
        v30_status=v30_status,
        v31_status=v31_status,
    )

    merged = v27.copy()

    if not v31.empty:
        v31_columns = [
            column
            for column in [
                "symbol",
                "v31_decision",
                "v31_score",
                "v31_learning_bonus",
                "matched_pattern_count",
                "positive_pattern_count",
                "negative_pattern_count",
            ]
            if column in v31.columns
        ]

        merged = merged.merge(
            v31[
                v31_columns
            ],
            on="symbol",
            how="left",
        )

    text_defaults = {
        "v31_decision": "",
        "v27_decision": "",
        "v22_signal_state": "",
        "v24_state": "TEYİT BEKLE",
        "risk_class": "BİLİNMİYOR",
        "regime": "BİLİNMİYOR",
    }

    for column, default in text_defaults.items():
        ensure_column(
            merged,
            column,
            default,
        )

        merged[column] = (
            merged[column]
            .fillna(default)
            .astype(str)
            .str.strip()
        )

    numeric_defaults = {
        "v31_score": 0.0,
        "v31_learning_bonus": 0.0,
        "matched_pattern_count": 0.0,
        "positive_pattern_count": 0.0,
        "negative_pattern_count": 0.0,
        "v27_master_score": 0.0,
        "v22_signal_score": 0.0,
        "v24_score": 0.0,
        "optimized_weight_pct": 0.0,
        "optimizer_score": 0.0,
        "quality_score": 50.0,
        "consensus_score": 0.0,
        "risk_score": 100.0,
        "market_percentile": 0.0,
        "timing_confidence": 0.0,
        "expected_return": 0.0,
        "downside_20pct": 0.0,
        "upside_80pct": 0.0,
        "close": 0.0,
    }

    for column, default in numeric_defaults.items():
        ensure_column(
            merged,
            column,
            default,
        )

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(default)

    # V31 skoru yoksa V27 skoru kullanılır.
    missing_v31_score = (
        merged["v31_score"] <= 0
    )

    merged.loc[
        missing_v31_score,
        "v31_score",
    ] = merged.loc[
        missing_v31_score,
        "v27_master_score",
    ]

    evaluations = merged.apply(
        lambda row: calculate_ai_adjustment(
            row=row,
            learning_ready=readiness[
                "learning_ready"
            ],
        ),
        axis=1,
    )

    merged["v32_ai_adjustment"] = [
        item["adjustment"]
        for item in evaluations
    ]

    merged["v32_confidence"] = [
        item["confidence"]
        for item in evaluations
    ]

    merged["v32_supports"] = [
        item["supports"]
        for item in evaluations
    ]

    merged["v32_risks"] = [
        item["risks"]
        for item in evaluations
    ]

    # Öğrenme hazır değilse V32 etkisi gölge puanı olarak hesaplanır.
    merged["v32_score"] = (
        merged["v27_master_score"]
        + merged["v32_ai_adjustment"]
    ).clip(
        0,
        100,
    ).round(
        2
    )

    decisions = merged.apply(
        lambda row: determine_v32_decision(
            row=row,
            learning_ready=readiness[
                "learning_ready"
            ],
        ),
        axis=1,
    )

    merged["v32_decision"] = [
        item[0]
        for item in decisions
    ]

    merged["v32_reason"] = [
        item[1]
        for item in decisions
    ]

    merged["v32_mode"] = (
        "ACTIVE"
        if readiness["learning_ready"]
        else "SHADOW"
    )

    merged["rsi_usage"] = "DISABLED"

    priority = {
        "ADAPTİF GÜÇLÜ TEYİT": 6,
        "ADAPTİF AKTİF İZLEME": 5,
        "GÖLGE İZLEME": 4,
        "TEYİT BEKLE": 3,
        "PASİF İZLEME": 2,
        "ELE": 1,
    }

    merged["_priority"] = (
        merged["v32_decision"]
        .map(priority)
        .fillna(0)
    )

    merged = merged.sort_values(
        by=[
            "_priority",
            "v32_score",
            "v32_confidence",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).drop(
        columns=["_priority"]
    ).reset_index(
        drop=True
    )

    merged.insert(
        0,
        "v32_rank",
        range(
            1,
            len(merged) + 1,
        ),
    )

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

    approved_states = {
        "ADAPTİF GÜÇLÜ TEYİT",
        "ADAPTİF AKTİF İZLEME",
    }

    status = {
        "status": "ready",
        "candidate_count": int(
            len(result)
        ),
        "approved_count": int(
            result["v32_decision"]
            .isin(
                approved_states
            )
            .sum()
        ),
        "strong_confirmation_count": int(
            (
                result["v32_decision"]
                == "ADAPTİF GÜÇLÜ TEYİT"
            ).sum()
        ),
        "active_tracking_count": int(
            (
                result["v32_decision"]
                == "ADAPTİF AKTİF İZLEME"
            ).sum()
        ),
        "shadow_tracking_count": int(
            (
                result["v32_decision"]
                == "GÖLGE İZLEME"
            ).sum()
        ),
        "waiting_count": int(
            (
                result["v32_decision"]
                == "TEYİT BEKLE"
            ).sum()
        ),
        "passive_count": int(
            (
                result["v32_decision"]
                == "PASİF İZLEME"
            ).sum()
        ),
        "eliminated_count": int(
            (
                result["v32_decision"]
                == "ELE"
            ).sum()
        ),
        "learning_ready": bool(
            readiness["learning_ready"]
        ),
        "completed_observation_count": int(
            readiness[
                "completed_observation_count"
            ]
        ),
        "minimum_completed_required": (
            MIN_COMPLETED_OBSERVATIONS
        ),
        "usable_pattern_count": int(
            readiness[
                "usable_pattern_count"
            ]
        ),
        "minimum_pattern_required": (
            MIN_USABLE_PATTERNS
        ),
        "matched_candidate_count": int(
            readiness[
                "matched_candidate_count"
            ]
        ),
        "mode": (
            "ACTIVE"
            if readiness["learning_ready"]
            else "SHADOW"
        ),
        "top_symbol": (
            text(
                result.iloc[0]["symbol"]
            )
            if len(result)
            else ""
        ),
        "top_decision": (
            text(
                result.iloc[0][
                    "v32_decision"
                ]
            )
            if len(result)
            else ""
        ),
        "top_score": (
            round(
                number(
                    result.iloc[0][
                        "v32_score"
                    ]
                ),
                2,
            )
            if len(result)
            else 0.0
        ),
        "top_confidence": (
            round(
                number(
                    result.iloc[0][
                        "v32_confidence"
                    ]
                ),
                2,
            )
            if len(result)
            else 0.0
        ),
        "rsi_usage": "DISABLED",
        "version": VERSION,
    }

    save_status(
        status
    )

    print(
        "===== V32 STATUS ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "===== V32 SONUÇLARI ====="
    )

    print(
        result.to_string(
            index=False,
        )
    )

    print(
        "===== V32 ADAPTİF KARAR MOTORU TAMAMLANDI ====="
    )


if __name__ == "__main__":
    main()
