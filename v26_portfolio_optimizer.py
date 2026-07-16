from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


V22_FILE = Path("v22_signal_states.csv")
V24_FILE = Path("v24_live_confirmations.csv")
V25_FILE = Path("v25_performance_evaluations.csv")

OUTPUT_FILE = Path("v26_optimized_portfolio.csv")
STATUS_FILE = Path("v26_status.json")


OUTPUT_COLUMNS = [
    "v26_rank",
    "symbol",
    "portfolio_role",
    "optimized_weight_pct",
    "optimizer_score",
    "allocation_cap_pct",
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


def empty_output() -> pd.DataFrame:
    return pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )


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
        .str.replace(".IS", "", regex=False)
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


def live_state_score(
    state: str,
) -> float:
    state = tx(state)

    scores = {
        "CANLI TEYİT GELDİ": 100.0,
        "GÜÇLÜ CANLI TEYİT": 100.0,
        "ERKEN TEYİT": 75.0,
        "TEYİT BEKLE": 35.0,
        "PASİF": 15.0,
        "ŞİŞKİN / RİSKLİ": 0.0,
        "ELE": 0.0,
    }

    return scores.get(
        state,
        20.0,
    )


def allocation_cap(
    row: pd.Series,
) -> float:
    live_state = tx(
        row.get("v24_state")
    )

    v22_state = tx(
        row.get("v22_signal_state")
    )

    risk_score = sf(
        row.get("risk_score"),
        100.0,
    )

    if risk_score > 55:
        return 0.0

    if live_state in {
        "CANLI TEYİT GELDİ",
        "GÜÇLÜ CANLI TEYİT",
    }:
        if (
            v22_state == "GÜÇLÜ TEYİT"
            and risk_score <= 35
        ):
            return 35.0

        return 30.0

    if live_state == "ERKEN TEYİT":
        if risk_score <= 40:
            return 25.0

        return 20.0

    if live_state == "TEYİT BEKLE":
        if risk_score <= 30:
            return 18.0

        return 12.0

    return 0.0


def portfolio_investment_limit(
    frame: pd.DataFrame,
) -> float:
    if frame.empty:
        return 0.0

    confirmed_count = int(
        frame["v24_state"]
        .isin(
            {
                "CANLI TEYİT GELDİ",
                "GÜÇLÜ CANLI TEYİT",
            }
        )
        .sum()
    )

    early_count = int(
        (
            frame["v24_state"]
            == "ERKEN TEYİT"
        ).sum()
    )

    waiting_count = int(
        (
            frame["v24_state"]
            == "TEYİT BEKLE"
        ).sum()
    )

    average_risk = pd.to_numeric(
        frame["risk_score"],
        errors="coerce",
    ).fillna(100.0).mean()

    if (
        confirmed_count >= 2
        and average_risk <= 35
    ):
        return 80.0

    if (
        confirmed_count >= 1
        and average_risk <= 40
    ):
        return 65.0

    if (
        confirmed_count == 0
        and early_count >= 1
    ):
        return 50.0

    if (
        confirmed_count == 0
        and early_count == 0
        and waiting_count > 0
    ):
        return 40.0

    return 25.0


def save_empty_status(
    status_name: str,
) -> None:
    empty_output().to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": status_name,
        "position_count": 0,
        "invested_pct": 0.0,
        "cash_pct": 100.0,
        "investment_limit_pct": 0.0,
        "live_confirmed_count": 0,
        "early_confirmed_count": 0,
        "waiting_count": 0,
        "top_symbol": "",
        "top_weight_pct": 0.0,
        "average_optimizer_score": 0.0,
        "version": "V26.1",
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

    ensure_column(
        merged,
        "v22_signal_state",
        "TEYİT BEKLE",
    )

    ensure_column(
        merged,
        "v22_signal_score",
        0.0,
    )

    ensure_column(
        merged,
        "v24_state",
        "TEYİT BEKLE",
    )

    ensure_column(
        merged,
        "v24_score",
        0.0,
    )

    ensure_column(
        merged,
        "quality_score",
        50.0,
    )

    ensure_column(
        merged,
        "reliability_class",
        "YENİ",
    )

    numeric_defaults = {
        "v22_signal_score": 0.0,
        "v24_score": 0.0,
        "quality_score": 50.0,
        "consensus_score": 0.0,
        "market_percentile": 0.0,
        "risk_score": 100.0,
        "expected_return": 0.0,
        "downside_20pct": 0.0,
        "upside_80pct": 0.0,
        "best_horizon_days": 1.0,
        "timing_confidence": 0.0,
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

    ensure_column(
        merged,
        "risk_class",
        "ORTA",
    )

    ensure_column(
        merged,
        "regime",
        "",
    )

    merged["v24_state"] = (
        merged["v24_state"]
        .fillna("TEYİT BEKLE")
        .astype(str)
        .str.strip()
    )

    merged["live_state_score"] = (
        merged["v24_state"]
        .apply(live_state_score)
    )

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

    merged["optimizer_score"] = (
        merged["v22_signal_score"] * 0.24
        + merged["v24_score"] * 0.20
        + merged["live_state_score"] * 0.18
        + merged["consensus_score"] * 0.12
        + merged["quality_score"] * 0.10
        + merged["market_percentile"] * 0.08
        + expected_return_component * 0.08
        - merged["risk_score"] * 0.18
    ).clip(
        0,
        100,
    )

    merged[
        "allocation_cap_pct"
    ] = merged.apply(
        allocation_cap,
        axis=1,
    )

    eligible = merged[
        (
            merged[
                "optimizer_score"
            ]
            >= 50
        )
        & (
            merged[
                "risk_score"
            ]
            <= 55
        )
        & (
            merged[
                "expected_return"
            ]
            > 0
        )
        & (
            merged[
                "allocation_cap_pct"
            ]
            > 0
        )
        & (
            ~merged[
                "v24_state"
            ].isin(
                {
                    "ŞİŞKİN / RİSKLİ",
                    "ELE",
                }
            )
        )
    ].copy()

    eligible = eligible.sort_values(
        [
            "optimizer_score",
            "v24_score",
            "v22_signal_score",
        ],
        ascending=False,
    ).head(5)

    if eligible.empty:
        save_empty_status(
            "no_eligible_candidate"
        )
        return

    investment_limit = (
        portfolio_investment_limit(
            eligible
        )
    )

    raw_scores = eligible[
        "optimizer_score"
    ].clip(
        lower=1.0
    )

    raw_weights = (
        raw_scores
        / raw_scores.sum()
        * investment_limit
    )

    capped_weights = np.minimum(
        raw_weights.to_numpy(),
        eligible[
            "allocation_cap_pct"
        ].to_numpy(),
    )

    for _ in range(20):
        unused_weight = (
            investment_limit
            - float(
                capped_weights.sum()
            )
        )

        if unused_weight <= 0.01:
            break

        remaining_room = (
            eligible[
                "allocation_cap_pct"
            ].to_numpy()
            - capped_weights
        )

        remaining_room = np.clip(
            remaining_room,
            0,
            None,
        )

        available = (
            remaining_room > 0.01
        )

        if not available.any():
            break

        score_share = (
            raw_scores.to_numpy()
            * available
        )

        score_total = float(
            score_share.sum()
        )

        if score_total <= 0:
            break

        score_share = (
            score_share
            / score_total
        )

        addition = np.minimum(
            unused_weight
            * score_share,
            remaining_room,
        )

        capped_weights += addition

    eligible[
        "optimized_weight_pct"
    ] = np.round(
        capped_weights,
        2,
    )

    eligible = eligible[
        eligible[
            "optimized_weight_pct"
        ]
        >= 5.0
    ].copy()

    if eligible.empty:
        save_empty_status(
            "weights_below_minimum"
        )
        return

    eligible = eligible.sort_values(
        [
            "optimized_weight_pct",
            "optimizer_score",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    eligible[
        "portfolio_role"
    ] = [
        (
            "ANA ADAY"
            if index == 0
            else "DESTEK ADAY"
        )
        for index in range(
            len(eligible)
        )
    ]

    result = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column == "v26_rank":
            continue

        if column in eligible.columns:
            result[column] = (
                eligible[column]
            )
        else:
            result[column] = np.nan

    result.insert(
        0,
        "v26_rank",
        range(
            1,
            len(result) + 1,
        ),
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    invested_pct = round(
        sf(
            result[
                "optimized_weight_pct"
            ].sum()
        ),
        2,
    )

    confirmed_count = int(
        eligible["v24_state"]
        .isin(
            {
                "CANLI TEYİT GELDİ",
                "GÜÇLÜ CANLI TEYİT",
            }
        )
        .sum()
    )

    early_count = int(
        (
            eligible["v24_state"]
            == "ERKEN TEYİT"
        ).sum()
    )

    waiting_count = int(
        (
            eligible["v24_state"]
            == "TEYİT BEKLE"
        ).sum()
    )

    status = {
        "status": "ready",
        "position_count": int(
            len(result)
        ),
        "invested_pct": (
            invested_pct
        ),
        "cash_pct": round(
            max(
                0.0,
                100.0
                - invested_pct,
            ),
            2,
        ),
        "investment_limit_pct": round(
            investment_limit,
            2,
        ),
        "live_confirmed_count": (
            confirmed_count
        ),
        "early_confirmed_count": (
            early_count
        ),
        "waiting_count": (
            waiting_count
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
        "top_weight_pct": (
            round(
                sf(
                    result.iloc[0][
                        "optimized_weight_pct"
                    ]
                ),
                2,
            )
            if len(result)
            else 0.0
        ),
        "average_optimizer_score": round(
            sf(
                result[
                    "optimizer_score"
                ].mean()
            ),
            2,
        ),
        "version": "V26.1",
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
