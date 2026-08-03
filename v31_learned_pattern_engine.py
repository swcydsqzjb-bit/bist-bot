from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# DOSYA YOLLARI
# ============================================================

INPUT_CANDIDATES = Path("v27_master_decisions.csv")
INPUT_PATTERNS = Path("v30_pattern_library.csv")
INPUT_FEATURE_BONUS = Path("v30_feature_bonus.csv")
INPUT_V30_STATUS = Path("v30_status.json")

OUTPUT_FILE = Path("v31_learned_decisions.csv")
STATUS_FILE = Path("v31_status.json")


# ============================================================
# AYARLAR
# ============================================================

VERSION = "V31.0"

MAX_POSITIVE_BONUS = 12.0
MAX_NEGATIVE_PENALTY = -8.0

MIN_SAMPLE_COUNT = 5
MIN_SUCCESS_RATE = 55.0

IGNORED_FEATURES = {
    "rsi",
    "rsi14",
    "rsi_14",
    "rsi_score",
    "rsi_percentile",
}


OUTPUT_COLUMNS = [
    "v31_rank",
    "symbol",
    "v31_decision",
    "v31_score",
    "v31_learning_bonus",
    "matched_pattern_count",
    "positive_pattern_count",
    "negative_pattern_count",
    "matched_patterns",
    "learning_explanation",
    "rsi_usage",
    "v27_decision",
    "v27_master_score",
    "v22_signal_state",
    "v22_signal_score",
    "v24_state",
    "v24_score",
    "optimized_weight_pct",
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

        if np.isfinite(result):
            return result

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
            print(f"UYARI: {path} tamamen boş.")
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
            print(
                f"UYARI: {path} içinde okunabilir "
                "sütun veya kayıt bulunamadı."
            )
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


def normalize_feature_name(
    feature: str,
) -> str:
    return (
        text(feature)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace("-", "_")
    )


def is_ignored_feature(
    feature: str,
) -> bool:
    normalized = normalize_feature_name(
        feature
    )

    if normalized in IGNORED_FEATURES:
        return True

    return normalized.startswith("rsi")


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
        "pattern_count": 0,
        "active_pattern_count": 0,
        "matched_candidate_count": 0,
        "total_learning_bonus": 0.0,
        "top_symbol": "",
        "top_decision": "",
        "top_score": 0.0,
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
# ÖRÜNTÜ HAZIRLAMA
# ============================================================

def prepare_patterns(
    patterns: pd.DataFrame,
) -> pd.DataFrame:
    if patterns.empty:
        return pd.DataFrame()

    result = patterns.copy()

    defaults = {
        "feature": "",
        "direction": "",
        "threshold_low": np.nan,
        "threshold_high": np.nan,
        "sample_count": 0,
        "success_count": 0,
        "success_rate": 0.0,
        "recommended_bonus": 0.0,
        "confidence_class": "",
        "active": False,
        "pattern_id": "",
        "source_decision": "",
    }

    for column, default in defaults.items():
        if column not in result.columns:
            result[column] = default

    result["feature"] = (
        result["feature"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["direction"] = (
        result["direction"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    for column in [
        "threshold_low",
        "threshold_high",
        "sample_count",
        "success_count",
        "success_rate",
        "recommended_bonus",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        ).fillna(0.0)

    result["active"] = result["active"].apply(
        boolean_value
    )

    # RSI örüntülerini tamamen çıkar
    result = result[
        ~result["feature"].apply(
            is_ignored_feature
        )
    ].copy()

    # Yalnızca güvenilir örüntüler
    result = result[
        (
            result["active"]
            | (
                (
                    result["sample_count"]
                    >= MIN_SAMPLE_COUNT
                )
                & (
                    result["success_rate"]
                    >= MIN_SUCCESS_RATE
                )
            )
        )
    ].copy()

    return result.reset_index(
        drop=True
    )


# ============================================================
# EŞLEŞME MANTIĞI
# ============================================================

def value_matches_pattern(
    candidate_value: float,
    direction: str,
    threshold_low: float,
    threshold_high: float,
) -> bool:
    direction = text(
        direction
    ).upper()

    low = number(
        threshold_low,
        np.nan,
    )

    high = number(
        threshold_high,
        np.nan,
    )

    if not np.isfinite(candidate_value):
        return False

    if direction in {
        "ABOVE",
        "HIGH",
        "UP",
        "YUKARI",
        "ÜSTÜ",
        "USTU",
        "GREATER",
    }:
        threshold = (
            low
            if np.isfinite(low)
            else high
        )

        return (
            np.isfinite(threshold)
            and candidate_value >= threshold
        )

    if direction in {
        "BELOW",
        "LOW",
        "DOWN",
        "AŞAĞI",
        "ASAGI",
        "ALTI",
        "LESS",
    }:
        threshold = (
            high
            if np.isfinite(high)
            else low
        )

        return (
            np.isfinite(threshold)
            and candidate_value <= threshold
        )

    if direction in {
        "BETWEEN",
        "RANGE",
        "ARALIK",
        "INSIDE",
    }:
        if (
            np.isfinite(low)
            and np.isfinite(high)
        ):
            lower = min(
                low,
                high,
            )

            upper = max(
                low,
                high,
            )

            return (
                lower
                <= candidate_value
                <= upper
            )

        return False

    if (
        np.isfinite(low)
        and np.isfinite(high)
        and low != high
    ):
        lower = min(
            low,
            high,
        )

        upper = max(
            low,
            high,
        )

        return (
            lower
            <= candidate_value
            <= upper
        )

    if np.isfinite(low):
        return candidate_value >= low

    if np.isfinite(high):
        return candidate_value <= high

    return False


def calculate_pattern_bonus(
    pattern: pd.Series,
) -> float:
    recommended = number(
        pattern.get(
            "recommended_bonus"
        )
    )

    success_rate = number(
        pattern.get("success_rate")
    )

    sample_count = number(
        pattern.get("sample_count")
    )

    if recommended != 0:
        raw_bonus = recommended

    elif success_rate >= 80:
        raw_bonus = 5.0

    elif success_rate >= 70:
        raw_bonus = 3.0

    elif success_rate >= 60:
        raw_bonus = 1.5

    elif success_rate >= 55:
        raw_bonus = 0.5

    elif success_rate < 40:
        raw_bonus = -3.0

    elif success_rate < 50:
        raw_bonus = -1.5

    else:
        raw_bonus = 0.0

    if sample_count < 10:
        raw_bonus *= 0.60

    elif sample_count < 20:
        raw_bonus *= 0.80

    return round(
        raw_bonus,
        2,
    )


def evaluate_candidate(
    row: pd.Series,
    patterns: pd.DataFrame,
) -> dict[str, Any]:
    total_bonus = 0.0
    matched: list[str] = []
    positive_count = 0
    negative_count = 0

    for _, pattern in patterns.iterrows():
        feature = normalize_feature_name(
            pattern.get("feature")
        )

        if not feature:
            continue

        if is_ignored_feature(feature):
            continue

        if feature not in row.index:
            continue

        candidate_value = number(
            row.get(feature),
            np.nan,
        )

        if not np.isfinite(candidate_value):
            continue

        is_match = value_matches_pattern(
            candidate_value=candidate_value,
            direction=text(
                pattern.get("direction")
            ),
            threshold_low=number(
                pattern.get("threshold_low"),
                np.nan,
            ),
            threshold_high=number(
                pattern.get("threshold_high"),
                np.nan,
            ),
        )

        if not is_match:
            continue

        bonus = calculate_pattern_bonus(
            pattern
        )

        total_bonus += bonus

        if bonus > 0:
            positive_count += 1

        elif bonus < 0:
            negative_count += 1

        pattern_id = text(
            pattern.get("pattern_id")
        )

        if not pattern_id:
            pattern_id = feature

        matched.append(
            (
                f"{pattern_id}: "
                f"{feature} "
                f"{candidate_value:.2f} "
                f"({bonus:+.2f})"
            )
        )

    total_bonus = float(
        np.clip(
            total_bonus,
            MAX_NEGATIVE_PENALTY,
            MAX_POSITIVE_BONUS,
        )
    )

    return {
        "bonus": round(
            total_bonus,
            2,
        ),
        "matched_count": len(
            matched
        ),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "matched_text": (
            " | ".join(matched)
            if matched
            else "Eşleşen güvenilir örüntü yok"
        ),
    }


# ============================================================
# V31 KARAR MANTIĞI
# ============================================================

def determine_v31_decision(
    row: pd.Series,
) -> tuple[str, str]:
    score = number(
        row.get("v31_score")
    )

    risk_score = number(
        row.get("risk_score"),
        100.0,
    )

    expected_return = number(
        row.get("expected_return")
    )

    downside = number(
        row.get("downside_20pct")
    )

    learning_bonus = number(
        row.get(
            "v31_learning_bonus"
        )
    )

    original_decision = text(
        row.get("v27_decision")
    )

    if (
        risk_score >= 65
        or downside <= -7
    ):
        return (
            "ELE",
            "Risk seviyesi öğrenme bonusundan bağımsız olarak yüksek",
        )

    if (
        score >= 80
        and learning_bonus > 0
        and risk_score <= 35
        and expected_return > 0
    ):
        return (
            "ÖĞRENİLMİŞ GÜÇLÜ TEYİT",
            (
                "Geçmiş başarılı örüntüler "
                "mevcut adayla güçlü biçimde eşleşti"
            ),
        )

    if (
        score >= 70
        and learning_bonus > 0
        and risk_score <= 45
        and expected_return > 0
    ):
        return (
            "ÖĞRENİLMİŞ AKTİF İZLEME",
            (
                "Geçmiş örüntüler toplam "
                "görünümü destekledi"
            ),
        )

    if (
        score >= 58
        and risk_score <= 55
    ):
        return (
            "TEYİT BEKLE",
            (
                "Genel görünüm olumlu ancak "
                "yeterli öğrenilmiş teyit oluşmadı"
            ),
        )

    if original_decision == "ELE":
        return (
            "ELE",
            "Önceki karar ve öğrenilmiş görünüm zayıf",
        )

    return (
        "PASİF İZLEME",
        (
            "Öğrenilmiş örüntüler aktif "
            "izleme için yeterli destek vermedi"
        ),
    )


# ============================================================
# ANA MOTOR
# ============================================================

def main() -> None:
    print(
        "===== V31 ÖĞRENİLMİŞ ÖRÜNTÜ MOTORU BAŞLADI ====="
    )

    candidates = normalize_symbol_column(
        load_csv(INPUT_CANDIDATES)
    )

    raw_patterns = load_csv(
        INPUT_PATTERNS
    )

    feature_bonus = load_csv(
        INPUT_FEATURE_BONUS
    )

    v30_status = load_json(
        INPUT_V30_STATUS
    )

    if candidates.empty:
        save_empty_result(
            status_name="candidate_input_missing",
            message=(
                "v27_master_decisions.csv "
                "bulunamadı veya boş."
            ),
        )
        return

    patterns = prepare_patterns(
        raw_patterns
    )

    # Feature bonus dosyası aynı yapıya sahipse ekle
    if not feature_bonus.empty:
        extra_patterns = prepare_patterns(
            feature_bonus
        )

        if not extra_patterns.empty:
            patterns = pd.concat(
                [
                    patterns,
                    extra_patterns,
                ],
                ignore_index=True,
            )

            patterns = patterns.drop_duplicates(
                subset=[
                    "feature",
                    "direction",
                    "threshold_low",
                    "threshold_high",
                ],
                keep="first",
            )

    working = candidates.copy()

    numeric_columns = [
        "v27_master_score",
        "v22_signal_score",
        "v24_score",
        "optimized_weight_pct",
        "optimizer_score",
        "quality_score",
        "consensus_score",
        "risk_score",
        "market_percentile",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "close",
        "volume_ratio",
        "ema20_distance",
        "ema20_slope",
        "momentum_percentile",
        "trend_percentile",
        "volume_percentile",
        "quality_percentile",
        "relative_strength_score",
        "smart_money_score",
        "institutional_score",
        "historical_support_score",
        "live_confirmation_score",
        "prediction_score",
        "relationship_score",
        "v8_score",
    ]

    for column in numeric_columns:
        if column not in working.columns:
            working[column] = np.nan

        working[column] = pd.to_numeric(
            working[column],
            errors="coerce",
        )

    evaluations = working.apply(
        lambda row: evaluate_candidate(
            row,
            patterns,
        ),
        axis=1,
    )

    working["v31_learning_bonus"] = [
        item["bonus"]
        for item in evaluations
    ]

    working["matched_pattern_count"] = [
        item["matched_count"]
        for item in evaluations
    ]

    working["positive_pattern_count"] = [
        item["positive_count"]
        for item in evaluations
    ]

    working["negative_pattern_count"] = [
        item["negative_count"]
        for item in evaluations
    ]

    working["matched_patterns"] = [
        item["matched_text"]
        for item in evaluations
    ]

    base_score = pd.to_numeric(
        working["v27_master_score"],
        errors="coerce",
    ).fillna(0.0)

    working["v31_score"] = (
        base_score
        + working[
            "v31_learning_bonus"
        ]
    ).clip(
        0,
        100,
    ).round(
        2
    )

    decisions = working.apply(
        determine_v31_decision,
        axis=1,
    )

    working["v31_decision"] = [
        item[0]
        for item in decisions
    ]

    working["learning_explanation"] = [
        item[1]
        for item in decisions
    ]

    working["rsi_usage"] = "DISABLED"

    priority = {
        "ÖĞRENİLMİŞ GÜÇLÜ TEYİT": 5,
        "ÖĞRENİLMİŞ AKTİF İZLEME": 4,
        "TEYİT BEKLE": 3,
        "PASİF İZLEME": 2,
        "ELE": 1,
    }

    working["_priority"] = (
        working["v31_decision"]
        .map(priority)
        .fillna(0)
    )

    working = working.sort_values(
        by=[
            "_priority",
            "v31_score",
            "v31_learning_bonus",
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

    working.insert(
        0,
        "v31_rank",
        range(
            1,
            len(working) + 1,
        ),
    )

    result = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column in working.columns:
            result[column] = (
                working[column]
            )
        else:
            result[column] = np.nan

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    matched_candidate_count = int(
        (
            result[
                "matched_pattern_count"
            ]
            > 0
        ).sum()
    )

    positive_bonus_count = int(
        (
            result[
                "v31_learning_bonus"
            ]
            > 0
        ).sum()
    )

    negative_bonus_count = int(
        (
            result[
                "v31_learning_bonus"
            ]
            < 0
        ).sum()
    )

    approved_states = {
        "ÖĞRENİLMİŞ GÜÇLÜ TEYİT",
        "ÖĞRENİLMİŞ AKTİF İZLEME",
    }

    status = {
        "status": "ready",
        "candidate_count": int(
            len(result)
        ),
        "pattern_count": int(
            len(raw_patterns)
        ),
        "usable_pattern_count": int(
            len(patterns)
        ),
        "matched_candidate_count": (
            matched_candidate_count
        ),
        "positive_bonus_count": (
            positive_bonus_count
        ),
        "negative_bonus_count": (
            negative_bonus_count
        ),
        "approved_count": int(
            result["v31_decision"]
            .isin(
                approved_states
            )
            .sum()
        ),
        "total_learning_bonus": round(
            number(
                result[
                    "v31_learning_bonus"
                ].sum()
            ),
            2,
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
                    "v31_decision"
                ]
            )
            if len(result)
            else ""
        ),
        "top_score": (
            number(
                result.iloc[0][
                    "v31_score"
                ]
            )
            if len(result)
            else 0.0
        ),
        "v30_learning_active": boolean_value(
            v30_status.get(
                "learning_active"
            )
        ),
        "rsi_usage": "DISABLED",
        "version": VERSION,
    }

    save_status(
        status
    )

    print(
        "===== V31 STATUS ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "===== V31 SONUÇLARI ====="
    )

    print(
        result.to_string(
            index=False
        )
    )

    print(
        "===== V31 ÖĞRENİLMİŞ ÖRÜNTÜ MOTORU TAMAMLANDI ====="
    )


if __name__ == "__main__":
    main()
