from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# =========================================================
# V20.3 GERÇEK GİRİŞ DOSYASI
# =========================================================

INPUT_FILE = Path("v20_portfolio_model.csv")


# =========================================================
# ANA ÇIKTILAR
# Sonraki motorların kullandığı gerçek dosya adları
# =========================================================

OUTPUT_FILE = Path("v20_monitoring_plan.csv")
STATUS_FILE = Path("v20_monitoring_status.json")


# =========================================================
# YEDEK / UYUMLULUK ÇIKTILARI
# Telegram veya eski workflow adımları için
# =========================================================

MIRROR_OUTPUT_FILE = Path("v20_4_monitoring_plan.csv")
MIRROR_STATUS_FILE = Path("v20_4_status.json")


OUTPUT_COLUMNS = [
    "plan_rank",
    "rank",
    "portfolio_rank",
    "symbol",

    "monitoring_state",
    "status",
    "confirmation_state",
    "allocation_label",

    "model_weight_pct",

    "reference_price",
    "close",

    "review_horizon_days",
    "best_horizon_days",
    "review_rule",

    "invalidation_price",
    "first_observation_price",
    "optimistic_price",

    "expected_return",
    "downside_20pct",
    "upside_80pct",

    "risk_class",
    "risk_score",

    "top_pick_score",
    "ai_final_score",
    "consensus_score",

    "regime",
    "why_now",
    "conflict_note",
    "motor_note",
]


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


def text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def normalize_symbol(value: Any) -> str:
    symbol = text(value).upper()

    if symbol.endswith(".IS"):
        symbol = symbol[:-3]

    return symbol


def load_csv(path: Path) -> pd.DataFrame:
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
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
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


def monitoring_decision(
    confirmation_state: str,
    risk_score: float,
    top_pick_score: float,
    ai_final_score: float,
    consensus_score: float,
) -> tuple[str, str]:
    state = text(
        confirmation_state
    ).upper()

    if risk_score >= 65:
        return (
            "PASİF",
            "Risk puanı yüksek",
        )

    if state in {
        "TEYİT GELDİ",
        "GÜÇLÜ TEYİT",
        "AKTİF İZLEME",
        "ÜST DÜZEY TEYİT",
    }:
        return (
            "AKTİF",
            "Ana model teyidi oluştu",
        )

    if state in {
        "İZLEMEDE TUT",
        "TEMKİNLİ İZLE",
        "TEYİT BEKLE",
        "İZLEME ADAYI",
    }:
        return (
            "TEYİT BEKLE",
            "Canlı teknik teyit bekleniyor",
        )

    combined_score = (
        top_pick_score * 0.35
        + ai_final_score * 0.35
        + consensus_score * 0.30
    )

    if (
        combined_score >= 65
        and risk_score <= 45
    ):
        return (
            "TEYİT BEKLE",
            "Model görünümü olumlu, canlı teyit bekleniyor",
        )

    return (
        "PASİF",
        "Aktif izleme için yeterli ortak güç oluşmadı",
    )


def create_empty_output() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )


def save_outputs(
    frame: pd.DataFrame,
) -> None:
    frame.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    frame.to_csv(
        MIRROR_OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )


def save_status(
    status: dict[str, Any],
) -> None:
    encoded = json.dumps(
        status,
        ensure_ascii=False,
        indent=2,
    )

    STATUS_FILE.write_text(
        encoded,
        encoding="utf-8",
    )

    MIRROR_STATUS_FILE.write_text(
        encoded,
        encoding="utf-8",
    )

    print(encoded)


def save_empty(
    status_name: str,
) -> None:
    empty = create_empty_output()

    save_outputs(empty)

    status = {
        "status": status_name,
        "plan_count": 0,
        "planned_candidate_count": 0,
        "active_confirmation_count": 0,
        "waiting_confirmation_count": 0,
        "passive_count": 0,
        "top_symbol": "",
        "version": "V20.4.2",
    }

    save_status(status)

    print(
        "V20.4: İzleme planına alınacak aday bulunamadı."
    )


def main() -> None:
    print(
        f"V20.4 giriş dosyası: {INPUT_FILE}"
    )

    frame = load_csv(
        INPUT_FILE
    )

    if frame.empty:
        save_empty(
            "no_candidates"
        )
        return

    if "symbol" not in frame.columns:
        save_empty(
            "symbol_column_missing"
        )
        return

    rows: list[dict[str, Any]] = []

    for index, source in frame.iterrows():
        symbol = normalize_symbol(
            source.get("symbol")
        )

        if not symbol:
            continue

        portfolio_rank = int(
            max(
                1,
                number(
                    source.get(
                        "portfolio_rank"
                    ),
                    index + 1,
                ),
            )
        )

        model_weight = number(
            source.get(
                "model_weight_pct"
            )
        )

        allocation_label = text(
            source.get(
                "allocation_label"
            )
        )

        confirmation_state = text(
            source.get(
                "confirmation_state"
            )
        )

        top_pick_score = number(
            source.get(
                "top_pick_score"
            )
        )

        ai_final_score = number(
            source.get(
                "ai_final_score"
            )
        )

        consensus_score = number(
            source.get(
                "consensus_score"
            )
        )

        risk_class = text(
            source.get(
                "risk_class"
            )
        )

        risk_score = number(
            source.get(
                "risk_score"
            ),
            50.0,
        )

        reference_price = number(
            source.get("close")
        )

        horizon = int(
            max(
                1,
                number(
                    source.get(
                        "best_horizon_days"
                    ),
                    5,
                ),
            )
        )

        expected_return = number(
            source.get(
                "expected_return"
            )
        )

        downside = number(
            source.get(
                "downside_20pct"
            )
        )

        upside = number(
            source.get(
                "upside_80pct"
            )
        )

        why_now = text(
            source.get("why_now")
        )

        conflict_note = text(
            source.get(
                "conflict_note"
            )
        )

        regime = text(
            source.get("regime")
        )

        monitoring_state, motor_note = (
            monitoring_decision(
                confirmation_state=confirmation_state,
                risk_score=risk_score,
                top_pick_score=top_pick_score,
                ai_final_score=ai_final_score,
                consensus_score=consensus_score,
            )
        )

        if reference_price > 0:
            invalidation_price = (
                reference_price
                * (1 + downside / 100)
            )

            first_observation_price = (
                reference_price
                * (
                    1
                    + expected_return / 100
                )
            )

            optimistic_price = (
                reference_price
                * (1 + upside / 100)
            )
        else:
            invalidation_price = 0.0
            first_observation_price = 0.0
            optimistic_price = 0.0

        review_rule = (
            f"{horizon} işlem günü içinde "
            "sonuç ve risk görünümünü yenile"
        )

        rows.append(
            {
                "plan_rank": portfolio_rank,
                "rank": portfolio_rank,
                "portfolio_rank": portfolio_rank,
                "symbol": symbol,

                "monitoring_state": monitoring_state,
                "status": monitoring_state,
                "confirmation_state": confirmation_state,
                "allocation_label": allocation_label,

                "model_weight_pct": round(
                    model_weight,
                    2,
                ),

                "reference_price": round(
                    reference_price,
                    4,
                ),
                "close": round(
                    reference_price,
                    4,
                ),

                "review_horizon_days": horizon,
                "best_horizon_days": horizon,
                "review_rule": review_rule,

                "invalidation_price": round(
                    invalidation_price,
                    4,
                ),
                "first_observation_price": round(
                    first_observation_price,
                    4,
                ),
                "optimistic_price": round(
                    optimistic_price,
                    4,
                ),

                "expected_return": round(
                    expected_return,
                    2,
                ),
                "downside_20pct": round(
                    downside,
                    2,
                ),
                "upside_80pct": round(
                    upside,
                    2,
                ),

                "risk_class": risk_class,
                "risk_score": round(
                    risk_score,
                    2,
                ),

                "top_pick_score": round(
                    top_pick_score,
                    2,
                ),
                "ai_final_score": round(
                    ai_final_score,
                    2,
                ),
                "consensus_score": round(
                    consensus_score,
                    2,
                ),

                "regime": regime,
                "why_now": why_now,
                "conflict_note": conflict_note,
                "motor_note": motor_note,
            }
        )

    if not rows:
        save_empty(
            "no_valid_candidates"
        )
        return

    result = pd.DataFrame(
        rows
    )

    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    result = result[
        OUTPUT_COLUMNS
    ].copy()

    priority = {
        "AKTİF": 3,
        "TEYİT BEKLE": 2,
        "PASİF": 1,
    }

    result["_priority"] = (
        result["monitoring_state"]
        .map(priority)
        .fillna(0)
    )

    result = result.sort_values(
        [
            "_priority",
            "model_weight_pct",
            "top_pick_score",
        ],
        ascending=False,
    ).drop(
        columns="_priority"
    ).reset_index(
        drop=True
    )

    result["plan_rank"] = range(
        1,
        len(result) + 1,
    )

    result["rank"] = result[
        "plan_rank"
    ]

    save_outputs(result)

    active_count = int(
        (
            result["monitoring_state"]
            == "AKTİF"
        ).sum()
    )

    waiting_count = int(
        (
            result["monitoring_state"]
            == "TEYİT BEKLE"
        ).sum()
    )

    passive_count = int(
        (
            result["monitoring_state"]
            == "PASİF"
        ).sum()
    )

    status = {
        "status": "ready",
        "plan_count": int(
            len(result)
        ),
        "planned_candidate_count": int(
            len(result)
        ),
        "active_confirmation_count": (
            active_count
        ),
        "waiting_confirmation_count": (
            waiting_count
        ),
        "passive_count": passive_count,
        "top_symbol": text(
            result.iloc[0]["symbol"]
        ),
        "input_file": str(
            INPUT_FILE
        ),
        "output_file": str(
            OUTPUT_FILE
        ),
        "version": "V20.4.2",
    }

    save_status(status)

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
