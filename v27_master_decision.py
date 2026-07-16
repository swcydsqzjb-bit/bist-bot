from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


V22_FILE = Path("v22_signal_states.csv")
V24_FILE = Path("v24_live_confirmations.csv")
V25_FILE = Path("v25_performance_evaluations.csv")
V26_FILE = Path("v26_optimized_portfolio.csv")

OUTPUT_FILE = Path("v27_master_decisions.csv")
STATUS_FILE = Path("v27_status.json")


OUTPUT_COLUMNS = [
    "v27_rank",
    "symbol",
    "v27_decision",
    "v27_master_score",
    "v27_reason",
    "optimized_weight_pct",
    "optimizer_score",
    "v22_signal_state",
    "v22_signal_score",
    "v24_state",
    "v24_score",
    "quality_score",
    "reliability_class",
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
]


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
                f"Uyarı: {path} UTF-8 olarak okunamadı: {exc}"
            )
            return pd.DataFrame()

    except Exception as exc:
        print(
            f"Uyarı: {path} okunamadı: {exc}"
        )
        return pd.DataFrame()


def normalize_symbol_column(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()

    if "symbol" not in result.columns:
        return pd.DataFrame()

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


def live_state_bonus(
    state: str,
) -> float:
    state = tx(state)

    bonuses = {
        "CANLI TEYİT GELDİ": 12.0,
        "GÜÇLÜ CANLI TEYİT": 12.0,
        "ERKEN TEYİT": 6.0,
        "TEYİT BEKLE": -4.0,
        "PASİF": -8.0,
        "ŞİŞKİN / RİSKLİ": -18.0,
        "ELE": -20.0,
    }

    return bonuses.get(
        state,
        -6.0,
    )


def determine_decision(
    row: pd.Series,
) -> tuple[str, str]:
    master_score = sf(
        row.get(
            "v27_master_score"
        )
    )

    live_state = tx(
        row.get("v24_state")
    )

    v22_state = tx(
        row.get(
            "v22_signal_state"
        )
    )

    risk_score = sf(
        row.get("risk_score"),
        100.0,
    )

    optimized_weight = sf(
        row.get(
            "optimized_weight_pct"
        )
    )

    quality_score = sf(
        row.get("quality_score"),
        50.0,
    )

    if live_state in {
        "ŞİŞKİN / RİSKLİ",
        "ELE",
    }:
        return (
            "ELE",
            (
                "Canlı teknik görünümde "
                "şişkinlik veya belirgin "
                "risk tespit edildi"
            ),
        )

    if risk_score >= 65:
        return (
            "ELE",
            "Risk puanı kabul edilebilir seviyenin üzerinde",
        )

    if quality_score < 35:
        return (
            "ELE",
            "Performans kalite görünümü yetersiz",
        )

    if (
        master_score >= 80
        and live_state
        in {
            "CANLI TEYİT GELDİ",
            "GÜÇLÜ CANLI TEYİT",
        }
        and v22_state
        in {
            "GÜÇLÜ TEYİT",
            "İZLEMEYE AL",
        }
        and risk_score <= 35
        and optimized_weight >= 10
    ):
        return (
            "ÜST DÜZEY TEYİT",
            (
                "Ana analiz katmanları "
                "ve canlı teknik teyit "
                "aynı yönde güçlü"
            ),
        )

    if (
        master_score >= 70
        and live_state
        in {
            "CANLI TEYİT GELDİ",
            "GÜÇLÜ CANLI TEYİT",
            "ERKEN TEYİT",
        }
        and risk_score <= 45
        and optimized_weight >= 8
    ):
        return (
            "AKTİF İZLEME",
            (
                "Canlı teyit ve portföy "
                "uygunluğu birlikte oluştu"
            ),
        )

    if live_state == "TEYİT BEKLE":
        if (
            master_score >= 58
            and risk_score <= 55
        ):
            return (
                "TEYİT BEKLE",
                (
                    "Genel görünüm olumlu "
                    "fakat canlı giriş "
                    "teyidi oluşmadı"
                ),
            )

        return (
            "PASİF İZLEME",
            (
                "Canlı teyit yok ve toplam "
                "skor aktif takip için "
                "yeterli değil"
            ),
        )

    if (
        master_score >= 58
        and risk_score <= 55
    ):
        return (
            "TEYİT BEKLE",
            (
                "Toplam görünüm olumlu "
                "fakat bütün şartlar "
                "tamamlanmadı"
            ),
        )

    return (
        "PASİF İZLEME",
        (
            "Analiz katmanları ortak "
            "güçlü karar üretmedi"
        ),
    )


def save_empty_status(
    status_name: str,
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
        "candidate_count": 0,
        "approved_count": 0,
        "top_level_confirmation_count": 0,
        "active_tracking_count": 0,
        "waiting_count": 0,
        "passive_count": 0,
        "eliminated_count": 0,
        "top_symbol": "",
        "top_decision": "",
        "top_score": 0.0,
        "version": "V27.1",
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


def main() -> None:
    signals = normalize_symbol_column(
        load_csv(V22_FILE)
    )

    live = normalize_symbol_column(
        load_csv(V24_FILE)
    )

    performance = normalize_symbol_column(
        load_csv(V25_FILE)
    )

    portfolio = normalize_symbol_column(
        load_csv(V26_FILE)
    )

    if signals.empty:
        save_empty_status(
            "v22_input_missing"
        )
        return

    merged = signals.copy()

    if not live.empty:
        live_columns = [
            column
            for column in [
                "symbol",
                "v24_state",
                "v24_score",
                "live_confirmation_score",
                "live_price",
            ]
            if column in live.columns
        ]

        live_data = live[
            live_columns
        ].copy()

        if (
            "v24_score"
            not in live_data.columns
            and "live_confirmation_score"
            in live_data.columns
        ):
            live_data["v24_score"] = (
                live_data[
                    "live_confirmation_score"
                ]
            )

        merged = merged.merge(
            live_data,
            on="symbol",
            how="left",
        )

    if not performance.empty:
        performance_columns = [
            column
            for column in [
                "symbol",
                "quality_score",
                "reliability_class",
            ]
            if column in performance.columns
        ]

        merged = merged.merge(
            performance[
                performance_columns
            ],
            on="symbol",
            how="left",
        )

    if not portfolio.empty:
        portfolio_columns = [
            column
            for column in [
                "symbol",
                "optimized_weight_pct",
                "optimizer_score",
                "portfolio_role",
            ]
            if column in portfolio.columns
        ]

        merged = merged.merge(
            portfolio[
                portfolio_columns
            ],
            on="symbol",
            how="left",
        )

    text_defaults = {
        "v22_signal_state": "TEYİT BEKLE",
        "v24_state": "TEYİT BEKLE",
        "reliability_class": "YENİ",
        "risk_class": "ORTA",
        "regime": "",
        "portfolio_role": "",
    }

    for column, default in (
        text_defaults.items()
    ):
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
        "v22_signal_score": 0.0,
        "v24_score": 0.0,
        "quality_score": 50.0,
        "optimized_weight_pct": 0.0,
        "optimizer_score": 0.0,
        "consensus_score": 0.0,
        "risk_score": 100.0,
        "market_percentile": 0.0,
        "best_horizon_days": 1.0,
        "timing_confidence": 0.0,
        "expected_return": 0.0,
        "downside_20pct": 0.0,
        "upside_80pct": 0.0,
        "close": 0.0,
    }

    for column, default in (
        numeric_defaults.items()
    ):
        ensure_column(
            merged,
            column,
            default,
        )

        merged[column] = pd.to_numeric(
            merged[column],
            errors="coerce",
        ).fillna(default)

    expected_return_component = (
        np.clip(
            merged["expected_return"],
            -5,
            10,
        )
        + 5
    ) / 15 * 100

    expected_return_component = (
        expected_return_component.clip(
            0,
            100,
        )
    )

    merged[
        "v27_master_score"
    ] = (
        merged["v22_signal_score"]
        * 0.22
        + merged["v24_score"]
        * 0.22
        + merged["optimizer_score"]
        * 0.16
        + merged["quality_score"]
        * 0.10
        + merged["consensus_score"]
        * 0.12
        + merged["timing_confidence"]
        * 0.08
        + merged["market_percentile"]
        * 0.05
        + expected_return_component
        * 0.05
        - merged["risk_score"]
        * 0.18
        + merged["v24_state"]
        .apply(
            live_state_bonus
        )
    ).clip(
        0,
        100,
    )

    decisions = merged.apply(
        determine_decision,
        axis=1,
    )

    merged["v27_decision"] = [
        item[0]
        for item in decisions
    ]

    merged["v27_reason"] = [
        item[1]
        for item in decisions
    ]

    priority = {
        "ÜST DÜZEY TEYİT": 5,
        "AKTİF İZLEME": 4,
        "TEYİT BEKLE": 3,
        "PASİF İZLEME": 2,
        "ELE": 1,
    }

    merged["_priority"] = (
        merged["v27_decision"]
        .map(priority)
        .fillna(0)
    )

    merged = merged.sort_values(
        [
            "_priority",
            "v27_master_score",
            "optimized_weight_pct",
        ],
        ascending=False,
    ).drop(
        columns="_priority"
    ).reset_index(
        drop=True
    )

    merged.insert(
        0,
        "v27_rank",
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
        "ÜST DÜZEY TEYİT",
        "AKTİF İZLEME",
    }

    status = {
        "status": "ready",
        "candidate_count": int(
            len(result)
        ),
        "approved_count": int(
            result["v27_decision"]
            .isin(
                approved_states
            )
            .sum()
        ),
        "top_level_confirmation_count": int(
            (
                result[
                    "v27_decision"
                ]
                == "ÜST DÜZEY TEYİT"
            ).sum()
        ),
        "active_tracking_count": int(
            (
                result[
                    "v27_decision"
                ]
                == "AKTİF İZLEME"
            ).sum()
        ),
        "waiting_count": int(
            (
                result[
                    "v27_decision"
                ]
                == "TEYİT BEKLE"
            ).sum()
        ),
        "passive_count": int(
            (
                result[
                    "v27_decision"
                ]
                == "PASİF İZLEME"
            ).sum()
        ),
        "eliminated_count": int(
            (
                result[
                    "v27_decision"
                ]
                == "ELE"
            ).sum()
        ),
        "top_symbol": (
            tx(
                result.iloc[0][
                    "symbol"
                ]
            )
            if len(result)
            else ""
        ),
        "top_decision": (
            tx(
                result.iloc[0][
                    "v27_decision"
                ]
            )
            if len(result)
            else ""
        ),
        "top_score": (
            round(
                sf(
                    result.iloc[0][
                        "v27_master_score"
                    ]
                ),
                2,
            )
            if len(result)
            else 0.0
        ),
        "version": "V27.1",
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
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
