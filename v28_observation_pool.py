from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# =========================================================
# LARUS V28 — GÖZLEM HAVUZU
#
# Amaç:
# V27 kararlarını ve karar anındaki özellikleri saklamak.
#
# Daha sonra V29:
# - 1 işlem günü
# - 3 işlem günü
# - 5 işlem günü
# - 10 işlem günü
# - 15 işlem günü
#
# sonuçlarını bu kayıtlar üzerinden ölçecek.
# =========================================================


V27_FILE = Path("v27_master_decisions.csv")
V22_FILE = Path("v22_signal_states.csv")
V20_PORTFOLIO_FILE = Path("v20_portfolio_model.csv")

HISTORY_FILE = Path("v28_observation_history.csv")
LATEST_FILE = Path("v28_latest_observations.csv")
STATUS_FILE = Path("v28_status.json")


ISTANBUL_TZ = ZoneInfo("Europe/Istanbul")


# =========================================================
# TAKİP EDİLECEK KARARLAR
# =========================================================

TRACKED_DECISIONS = {
    "ÜST DÜZEY TEYİT",
    "AKTİF İZLEME",
    "TEYİT BEKLE",
    "PASİF İZLEME",
}


# ELE kararlarını da geçmiş analizinde saklarız.
# Fakat bunlar aktif gözlem sınıfına girmez.
ANALYSIS_ONLY_DECISIONS = {
    "ELE",
    "RİSKLİ - ELE",
}


# =========================================================
# ÇIKTI SÜTUNLARI
# =========================================================

OUTPUT_COLUMNS = [
    "observation_id",
    "observation_date",
    "observation_datetime",
    "symbol",

    "entry_decision",
    "tracking_class",
    "is_active_observation",

    "reference_price",
    "entry_price",

    "v27_master_score",
    "v27_reason",

    "v22_signal_state",
    "v22_signal_score",

    "v24_state",
    "v24_score",

    "optimized_weight_pct",
    "optimizer_score",
    "portfolio_role",

    "top_pick_score",
    "ai_final_score",
    "consensus_score",

    "quality_score",
    "reliability_class",

    "risk_class",
    "risk_score",

    "regime",
    "market_percentile",

    "best_horizon_days",
    "timing_confidence",

    "expected_return",
    "downside_20pct",
    "upside_80pct",

    "rsi",
    "volume_ratio",
    "ema20_distance",

    "smart_money_score",
    "institutional_score",
    "historical_support_score",
    "prediction_score",
    "live_confirmation_score",
    "relationship_score",
    "v8_score",

    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_15d",

    "max_return_15d",
    "max_drawdown_15d",

    "completed_1d",
    "completed_3d",
    "completed_5d",
    "completed_10d",
    "completed_15d",

    "missed_opportunity",
    "successful_observation",

    "last_evaluated_date",
    "record_status",
    "version",
]


# =========================================================
# YARDIMCI FONKSİYONLAR
# =========================================================

def sf(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return default

    except (TypeError, ValueError):
        return default


def tx(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def normalize_symbol(
    value: Any,
) -> str:
    symbol = tx(value).upper()

    if symbol.endswith(".IS"):
        symbol = symbol[:-3]

    return symbol


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

    try:
        return pd.read_csv(
            path,
            encoding="utf-8-sig",
        )

    except pd.errors.EmptyDataError:
        return pd.DataFrame()

    except UnicodeDecodeError:
        try:
            return pd.read_csv(
                path,
                encoding="utf-8",
            )
        except Exception as exc:
            print(
                f"Uyarı: {path} okunamadı: {exc}"
            )
            return pd.DataFrame()

    except Exception as exc:
        print(
            f"Uyarı: {path} okunamadı: {exc}"
        )
        return pd.DataFrame()


def normalize_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()

    if "symbol" not in result.columns:
        return pd.DataFrame()

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

    return result


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


def current_times() -> tuple[str, str]:
    now = datetime.now(
        ISTANBUL_TZ
    )

    observation_date = now.strftime(
        "%Y-%m-%d"
    )

    observation_datetime = now.isoformat(
        timespec="seconds"
    )

    return (
        observation_date,
        observation_datetime,
    )


def determine_tracking_class(
    decision: str,
) -> tuple[str, bool]:
    normalized = tx(
        decision
    ).upper()

    if normalized == "ÜST DÜZEY TEYİT":
        return (
            "STRONG_ACTIVE",
            True,
        )

    if normalized == "AKTİF İZLEME":
        return (
            "ACTIVE",
            True,
        )

    if normalized == "TEYİT BEKLE":
        return (
            "WAITING",
            True,
        )

    if normalized == "PASİF İZLEME":
        return (
            "PASSIVE",
            True,
        )

    if normalized in ANALYSIS_ONLY_DECISIONS:
        return (
            "ELIMINATED_ANALYSIS",
            False,
        )

    return (
        "UNKNOWN",
        False,
    )


def save_empty_status(
    status_name: str,
) -> None:
    if not HISTORY_FILE.exists():
        pd.DataFrame(
            columns=OUTPUT_COLUMNS
        ).to_csv(
            HISTORY_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    pd.DataFrame(
        columns=OUTPUT_COLUMNS
    ).to_csv(
        LATEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": status_name,
        "new_observation_count": 0,
        "history_count": 0,
        "tracked_count": 0,
        "analysis_only_count": 0,
        "version": "V28.0",
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


# =========================================================
# ANA ÇALIŞMA
# =========================================================

def main() -> None:
    v27 = normalize_frame(
        load_csv(V27_FILE)
    )

    v22 = normalize_frame(
        load_csv(V22_FILE)
    )

    v20 = normalize_frame(
        load_csv(V20_PORTFOLIO_FILE)
    )

    if v27.empty:
        save_empty_status(
            "v27_input_missing"
        )
        return

    merged = v27.copy()

    # =====================================================
    # V22 ÖZELLİKLERİNİ EKLE
    # =====================================================

    if not v22.empty:
        v22_columns = [
            column
            for column in [
                "symbol",
                "top_pick_score",
                "ai_final_score",
                "consensus_score",
                "risk_class",
                "risk_score",
                "regime",
                "market_percentile",
                "best_horizon_days",
                "timing_confidence",
                "expected_return",
                "downside_20pct",
                "upside_80pct",
                "close",
                "reference_price",
            ]
            if column in v22.columns
        ]

        merged = merged.merge(
            v22[v22_columns],
            on="symbol",
            how="left",
            suffixes=("", "_v22"),
        )

    # =====================================================
    # V20.3 / ALT MOTOR ÖZELLİKLERİNİ EKLE
    # =====================================================

    if not v20.empty:
        v20_columns = [
            column
            for column in [
                "symbol",
                "top_pick_score",
                "ai_final_score",
                "consensus_score",
                "risk_class",
                "risk_score",
                "regime",
                "market_percentile",
                "best_horizon_days",
                "timing_confidence",
                "expected_return",
                "downside_20pct",
                "upside_80pct",
                "close",
                "rsi",
                "volume_ratio",
                "ema20_distance",
                "smart_money_score",
                "institutional_score",
                "historical_support_score",
                "prediction_score",
                "live_confirmation_score",
                "relationship_score",
                "v8_score",
            ]
            if column in v20.columns
        ]

        merged = merged.merge(
            v20[v20_columns],
            on="symbol",
            how="left",
            suffixes=("", "_v20"),
        )

    # =====================================================
    # AYNI ALANIN FARKLI KAYNAKLARINI BİRLEŞTİR
    # =====================================================

    fallback_columns = [
        "top_pick_score",
        "ai_final_score",
        "consensus_score",
        "risk_score",
        "risk_class",
        "regime",
        "market_percentile",
        "best_horizon_days",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "close",
        "reference_price",
    ]

    for column in fallback_columns:
        for suffix in [
            "_v22",
            "_v20",
        ]:
            fallback_column = (
                f"{column}{suffix}"
            )

            if fallback_column not in merged.columns:
                continue

            if column not in merged.columns:
                merged[column] = merged[
                    fallback_column
                ]

            else:
                merged[column] = merged[
                    column
                ].combine_first(
                    merged[
                        fallback_column
                    ]
                )

    # =====================================================
    # VARSAYILAN SÜTUNLAR
    # =====================================================

    text_defaults = {
        "v27_decision": "",
        "v27_reason": "",
        "v22_signal_state": "",
        "v24_state": "TEYİT BEKLE",
        "portfolio_role": "",
        "reliability_class": "YENİ",
        "risk_class": "ORTA",
        "regime": "",
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
        "reference_price": 0.0,
        "close": 0.0,

        "v27_master_score": 0.0,
        "v22_signal_score": 0.0,
        "v24_score": 0.0,

        "optimized_weight_pct": 0.0,
        "optimizer_score": 0.0,

        "top_pick_score": 0.0,
        "ai_final_score": 0.0,
        "consensus_score": 0.0,

        "quality_score": 50.0,
        "risk_score": 50.0,

        "market_percentile": 0.0,
        "best_horizon_days": 5.0,
        "timing_confidence": 0.0,

        "expected_return": 0.0,
        "downside_20pct": 0.0,
        "upside_80pct": 0.0,

        "rsi": 0.0,
        "volume_ratio": 0.0,
        "ema20_distance": 0.0,

        "smart_money_score": 0.0,
        "institutional_score": 0.0,
        "historical_support_score": 0.0,
        "prediction_score": 0.0,
        "live_confirmation_score": 0.0,
        "relationship_score": 0.0,
        "v8_score": 0.0,
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

    merged["reference_price"] = np.where(
        merged["reference_price"] > 0,
        merged["reference_price"],
        merged["close"],
    )

    merged["close"] = np.where(
        merged["close"] > 0,
        merged["close"],
        merged["reference_price"],
    )

    # =====================================================
    # TARİH VE KİMLİK
    # =====================================================

    (
        observation_date,
        observation_datetime,
    ) = current_times()

    new_rows: list[dict[str, Any]] = []

    for _, row in merged.iterrows():
        symbol = normalize_symbol(
            row.get("symbol")
        )

        decision = tx(
            row.get("v27_decision")
        ).upper()

        if not symbol:
            continue

        if (
            decision not in TRACKED_DECISIONS
            and decision not in ANALYSIS_ONLY_DECISIONS
        ):
            continue

        (
            tracking_class,
            is_active_observation,
        ) = determine_tracking_class(
            decision
        )

        observation_id = (
            f"{observation_date}_"
            f"{symbol}_"
            f"{decision.replace(' ', '_')}"
        )

        new_rows.append(
            {
                "observation_id": observation_id,
                "observation_date": observation_date,
                "observation_datetime": observation_datetime,
                "symbol": symbol,

                "entry_decision": decision,
                "tracking_class": tracking_class,
                "is_active_observation": bool(
                    is_active_observation
                ),

                "reference_price": round(
                    sf(
                        row.get(
                            "reference_price"
                        )
                    ),
                    4,
                ),

                "entry_price": round(
                    sf(
                        row.get(
                            "reference_price"
                        )
                    ),
                    4,
                ),

                "v27_master_score": round(
                    sf(
                        row.get(
                            "v27_master_score"
                        )
                    ),
                    2,
                ),

                "v27_reason": tx(
                    row.get(
                        "v27_reason"
                    )
                ),

                "v22_signal_state": tx(
                    row.get(
                        "v22_signal_state"
                    )
                ),

                "v22_signal_score": round(
                    sf(
                        row.get(
                            "v22_signal_score"
                        )
                    ),
                    2,
                ),

                "v24_state": tx(
                    row.get(
                        "v24_state"
                    )
                ),

                "v24_score": round(
                    sf(
                        row.get(
                            "v24_score"
                        )
                    ),
                    2,
                ),

                "optimized_weight_pct": round(
                    sf(
                        row.get(
                            "optimized_weight_pct"
                        )
                    ),
                    2,
                ),

                "optimizer_score": round(
                    sf(
                        row.get(
                            "optimizer_score"
                        )
                    ),
                    2,
                ),

                "portfolio_role": tx(
                    row.get(
                        "portfolio_role"
                    )
                ),

                "top_pick_score": round(
                    sf(
                        row.get(
                            "top_pick_score"
                        )
                    ),
                    2,
                ),

                "ai_final_score": round(
                    sf(
                        row.get(
                            "ai_final_score"
                        )
                    ),
                    2,
                ),

                "consensus_score": round(
                    sf(
                        row.get(
                            "consensus_score"
                        )
                    ),
                    2,
                ),

                "quality_score": round(
                    sf(
                        row.get(
                            "quality_score"
                        ),
                        50.0,
                    ),
                    2,
                ),

                "reliability_class": tx(
                    row.get(
                        "reliability_class"
                    )
                ),

                "risk_class": tx(
                    row.get(
                        "risk_class"
                    )
                ),

                "risk_score": round(
                    sf(
                        row.get(
                            "risk_score"
                        ),
                        50.0,
                    ),
                    2,
                ),

                "regime": tx(
                    row.get(
                        "regime"
                    )
                ),

                "market_percentile": round(
                    sf(
                        row.get(
                            "market_percentile"
                        )
                    ),
                    2,
                ),

                "best_horizon_days": int(
                    max(
                        1,
                        sf(
                            row.get(
                                "best_horizon_days"
                            ),
                            5.0,
                        ),
                    )
                ),

                "timing_confidence": round(
                    sf(
                        row.get(
                            "timing_confidence"
                        )
                    ),
                    2,
                ),

                "expected_return": round(
                    sf(
                        row.get(
                            "expected_return"
                        )
                    ),
                    2,
                ),

                "downside_20pct": round(
                    sf(
                        row.get(
                            "downside_20pct"
                        )
                    ),
                    2,
                ),

                "upside_80pct": round(
                    sf(
                        row.get(
                            "upside_80pct"
                        )
                    ),
                    2,
                ),

                "rsi": round(
                    sf(
                        row.get("rsi")
                    ),
                    2,
                ),

                "volume_ratio": round(
                    sf(
                        row.get(
                            "volume_ratio"
                        )
                    ),
                    2,
                ),

                "ema20_distance": round(
                    sf(
                        row.get(
                            "ema20_distance"
                        )
                    ),
                    2,
                ),

                "smart_money_score": round(
                    sf(
                        row.get(
                            "smart_money_score"
                        )
                    ),
                    2,
                ),

                "institutional_score": round(
                    sf(
                        row.get(
                            "institutional_score"
                        )
                    ),
                    2,
                ),

                "historical_support_score": round(
                    sf(
                        row.get(
                            "historical_support_score"
                        )
                    ),
                    2,
                ),

                "prediction_score": round(
                    sf(
                        row.get(
                            "prediction_score"
                        )
                    ),
                    2,
                ),

                "live_confirmation_score": round(
                    sf(
                        row.get(
                            "live_confirmation_score"
                        )
                    ),
                    2,
                ),

                "relationship_score": round(
                    sf(
                        row.get(
                            "relationship_score"
                        )
                    ),
                    2,
                ),

                "v8_score": round(
                    sf(
                        row.get(
                            "v8_score"
                        )
                    ),
                    2,
                ),

                "return_1d": np.nan,
                "return_3d": np.nan,
                "return_5d": np.nan,
                "return_10d": np.nan,
                "return_15d": np.nan,

                "max_return_15d": np.nan,
                "max_drawdown_15d": np.nan,

                "completed_1d": False,
                "completed_3d": False,
                "completed_5d": False,
                "completed_10d": False,
                "completed_15d": False,

                "missed_opportunity": False,
                "successful_observation": False,

                "last_evaluated_date": "",
                "record_status": "OPEN",
                "version": "V28.0",
            }
        )

    new_frame = pd.DataFrame(
        new_rows
    )

    if new_frame.empty:
        save_empty_status(
            "no_observations"
        )
        return

    # =====================================================
    # ESKİ GEÇMİŞİ OKU
    # =====================================================

    history = load_csv(
        HISTORY_FILE
    )

    if history.empty:
        history = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    for column in OUTPUT_COLUMNS:
        if column not in history.columns:
            history[column] = np.nan

        if column not in new_frame.columns:
            new_frame[column] = np.nan

    history = history[
        OUTPUT_COLUMNS
    ].copy()

    new_frame = new_frame[
        OUTPUT_COLUMNS
    ].copy()

    # Aynı observation_id daha önce varsa yeniden ekleme.
    existing_ids = set(
        history[
            "observation_id"
        ]
        .fillna("")
        .astype(str)
        .tolist()
    )

    additions = new_frame[
        ~new_frame[
            "observation_id"
        ]
        .astype(str)
        .isin(existing_ids)
    ].copy()

    combined = pd.concat(
        [
            history,
            additions,
        ],
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "observation_id",
        ],
        keep="first",
    )

    combined.to_csv(
        HISTORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    new_frame.to_csv(
        LATEST_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    tracked_count = int(
        additions[
            "is_active_observation"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    )

    analysis_only_count = int(
        len(additions)
        - tracked_count
    )

    status = {
        "status": "ready",
        "new_observation_count": int(
            len(additions)
        ),
        "latest_candidate_count": int(
            len(new_frame)
        ),
        "history_count": int(
            len(combined)
        ),
        "tracked_count": tracked_count,
        "analysis_only_count": analysis_only_count,
        "top_symbol": (
            tx(
                new_frame.iloc[0][
                    "symbol"
                ]
            )
            if len(new_frame)
            else ""
        ),
        "top_decision": (
            tx(
                new_frame.iloc[0][
                    "entry_decision"
                ]
            )
            if len(new_frame)
            else ""
        ),
        "version": "V28.0",
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

    print(
        additions[
            [
                "symbol",
                "entry_decision",
                "tracking_class",
                "entry_price",
                "v27_master_score",
                "v22_signal_score",
                "expected_return",
            ]
        ].to_string(
            index=False
        )
        if not additions.empty
        else "V28: Bugün için yeni gözlem kaydı bulunmadı."
    )


if __name__ == "__main__":
    main()
