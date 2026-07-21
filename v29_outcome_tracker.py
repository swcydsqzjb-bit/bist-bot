from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# LARUS V29 — GERÇEKLEŞEN SONUÇ TAKİBİ
#
# V28 gözlem kayıtlarının:
# - 1 işlem günü
# - 3 işlem günü
# - 5 işlem günü
# - 10 işlem günü
# - 15 işlem günü
#
# sonraki sonuçlarını ölçer.
#
# Bu katman henüz karar puanı değiştirmez.
# V30, tamamlanan V29 sonuçlarından örüntü öğrenecektir.
# ============================================================


V28_HISTORY_FILE = Path("v28_observation_history.csv")

OUTPUT_FILE = Path("v29_evaluated_observations.csv")
SUMMARY_FILE = Path("v29_outcome_summary.csv")
MISSED_FILE = Path("v29_missed_opportunities.csv")
STATUS_FILE = Path("v29_status.json")


HORIZONS = (1, 3, 5, 10, 15)

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


# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================


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


def yahoo_symbol(
    value: Any,
) -> str:
    symbol = normalize_symbol(value)

    if not symbol:
        return ""

    return f"{symbol}.IS"


def safe_bool(
    value: Any,
    default: bool = False,
) -> bool:
    if isinstance(value, bool):
        return value

    text = tx(value).lower()

    if text in {
        "true",
        "1",
        "yes",
        "evet",
    }:
        return True

    if text in {
        "false",
        "0",
        "no",
        "hayır",
        "hayir",
    }:
        return False

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


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


def normalize_observations(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()

    if "observation_id" not in result.columns:
        return pd.DataFrame()

    if "symbol" not in result.columns:
        return pd.DataFrame()

    result["observation_id"] = (
        result["observation_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    result["symbol"] = result["symbol"].apply(
        normalize_symbol
    )

    result = result[
        result["observation_id"].ne("")
        & result["symbol"].ne("")
    ].copy()

    result = result.drop_duplicates(
        subset=["observation_id"],
        keep="last",
    )

    return result


def prepare_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    text_defaults = {
        "observation_id": "",
        "observation_date": "",
        "observation_datetime": "",
        "symbol": "",
        "entry_decision": "",
        "tracking_class": "",
        "v27_reason": "",
        "v22_signal_state": "",
        "v24_state": "",
        "portfolio_role": "",
        "reliability_class": "",
        "risk_class": "",
        "regime": "",
        "last_evaluated_date": "",
        "record_status": "OPEN",
        "version": "V29.0",
    }

    numeric_defaults = {
        "reference_price": 0.0,
        "entry_price": 0.0,

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

        "return_1d": np.nan,
        "return_3d": np.nan,
        "return_5d": np.nan,
        "return_10d": np.nan,
        "return_15d": np.nan,

        "max_return_15d": np.nan,
        "max_drawdown_15d": np.nan,
    }

    boolean_defaults = {
        "is_active_observation": False,

        "completed_1d": False,
        "completed_3d": False,
        "completed_5d": False,
        "completed_10d": False,
        "completed_15d": False,

        "missed_opportunity": False,
        "successful_observation": False,
    }

    for column, default in text_defaults.items():
        ensure_column(
            result,
            column,
            default,
        )

        result[column] = (
            result[column]
            .fillna(default)
            .astype(str)
            .str.strip()
        )

    for column, default in numeric_defaults.items():
        ensure_column(
            result,
            column,
            default,
        )

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

        if not pd.isna(default):
            result[column] = result[column].fillna(
                default
            )

    for column, default in boolean_defaults.items():
        ensure_column(
            result,
            column,
            default,
        )

        result[column] = result[column].apply(
            lambda value: safe_bool(
                value,
                default,
            )
        )

    result["reference_price"] = np.where(
        result["reference_price"] > 0,
        result["reference_price"],
        result["entry_price"],
    )

    result["entry_price"] = np.where(
        result["entry_price"] > 0,
        result["entry_price"],
        result["reference_price"],
    )

    return result


# ============================================================
# YENİ V28 KAYITLARINI V29 HAFIZASINA EKLE
# ============================================================


def combine_history(
    v28_history: pd.DataFrame,
    old_v29: pd.DataFrame,
) -> pd.DataFrame:
    if old_v29.empty:
        combined = v28_history.copy()

    else:
        old_ids = set(
            old_v29["observation_id"]
            .fillna("")
            .astype(str)
        )

        additions = v28_history[
            ~v28_history["observation_id"]
            .astype(str)
            .isin(old_ids)
        ].copy()

        combined = pd.concat(
            [
                old_v29,
                additions,
            ],
            ignore_index=True,
        )

    combined = combined.drop_duplicates(
        subset=["observation_id"],
        keep="first",
    )

    return prepare_columns(
        combined
    )


# ============================================================
# FİYAT VERİSİ
# ============================================================


def clean_price_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    result = frame.copy()

    if isinstance(
        result.columns,
        pd.MultiIndex,
    ):
        if len(result.columns.levels) >= 2:
            result.columns = [
                column[0]
                for column in result.columns
            ]

    required = {
        "Close",
        "High",
        "Low",
    }

    if not required.issubset(
        set(result.columns)
    ):
        return pd.DataFrame()

    result = result[
        [
            "Close",
            "High",
            "Low",
        ]
    ].copy()

    for column in [
        "Close",
        "High",
        "Low",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    result = result.dropna(
        subset=["Close"]
    )

    result.index = pd.to_datetime(
        result.index,
        errors="coerce",
    )

    result = result[
        ~result.index.isna()
    ].copy()

    if getattr(
        result.index,
        "tz",
        None,
    ) is not None:
        result.index = (
            result.index
            .tz_convert(None)
        )

    result.index = result.index.normalize()

    result = result[
        ~result.index.duplicated(
            keep="last"
        )
    ]

    result = result.sort_index()

    return result


def download_symbol_history(
    symbol: str,
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    ticker = yahoo_symbol(symbol)

    if not ticker:
        return pd.DataFrame()

    today = pd.Timestamp.today().normalize()

    download_start = (
        start_date
        - pd.Timedelta(days=7)
    ).strftime("%Y-%m-%d")

    download_end = (
        today
        + pd.Timedelta(days=2)
    ).strftime("%Y-%m-%d")

    try:
        data = yf.download(
            ticker,
            start=download_start,
            end=download_end,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
            timeout=30,
        )

        return clean_price_frame(
            data
        )

    except Exception as exc:
        print(
            f"Uyarı: {symbol} fiyat verisi alınamadı: {exc}"
        )
        return pd.DataFrame()


# ============================================================
# SONUÇ HESAPLAMA
# ============================================================


def percentage_return(
    future_price: float,
    entry_price: float,
) -> float:
    if entry_price <= 0:
        return np.nan

    return round(
        (
            future_price
            / entry_price
            - 1.0
        )
        * 100.0,
        2,
    )


def evaluate_row(
    row: pd.Series,
    prices: pd.DataFrame,
    evaluated_date: str,
) -> dict[str, Any]:
    updates: dict[str, Any] = {}

    entry_price = sf(
        row.get("entry_price")
    )

    if entry_price <= 0:
        entry_price = sf(
            row.get("reference_price")
        )

    observation_date = pd.to_datetime(
        row.get("observation_date"),
        errors="coerce",
    )

    if (
        entry_price <= 0
        or pd.isna(observation_date)
        or prices.empty
    ):
        updates["last_evaluated_date"] = (
            evaluated_date
        )

        return updates

    observation_date = (
        observation_date.normalize()
    )

    # Gözlem gününden sonraki işlem günleri.
    future = prices[
        prices.index > observation_date
    ].copy()

    available_days = len(future)

    for horizon in HORIZONS:
        return_column = (
            f"return_{horizon}d"
        )

        completed_column = (
            f"completed_{horizon}d"
        )

        if available_days >= horizon:
            future_close = sf(
                future.iloc[
                    horizon - 1
                ]["Close"]
            )

            updates[return_column] = (
                percentage_return(
                    future_close,
                    entry_price,
                )
            )

            updates[completed_column] = True

    first_15 = future.head(15)

    if not first_15.empty:
        highest_price = pd.to_numeric(
            first_15["High"],
            errors="coerce",
        ).max()

        lowest_price = pd.to_numeric(
            first_15["Low"],
            errors="coerce",
        ).min()

        if pd.notna(highest_price):
            updates[
                "max_return_15d"
            ] = percentage_return(
                float(highest_price),
                entry_price,
            )

        if pd.notna(lowest_price):
            updates[
                "max_drawdown_15d"
            ] = percentage_return(
                float(lowest_price),
                entry_price,
            )

    decision = tx(
        row.get("entry_decision")
    ).upper()

    return_5d = sf(
        updates.get(
            "return_5d",
            row.get("return_5d"),
        ),
        np.nan,
    )

    return_10d = sf(
        updates.get(
            "return_10d",
            row.get("return_10d"),
        ),
        np.nan,
    )

    return_15d = sf(
        updates.get(
            "return_15d",
            row.get("return_15d"),
        ),
        np.nan,
    )

    max_return = sf(
        updates.get(
            "max_return_15d",
            row.get("max_return_15d"),
        ),
        np.nan,
    )

    missed_decisions = {
        "TEYİT BEKLE",
        "PASİF İZLEME",
        "ELE",
        "RİSKLİ - ELE",
    }

    missed_opportunity = False

    if decision in missed_decisions:
        if (
            np.isfinite(return_10d)
            and return_10d >= 5.0
        ):
            missed_opportunity = True

        elif (
            np.isfinite(return_15d)
            and return_15d >= 7.0
        ):
            missed_opportunity = True

        elif (
            np.isfinite(max_return)
            and max_return >= 8.0
        ):
            missed_opportunity = True

    successful_observation = False

    if (
        np.isfinite(return_5d)
        and return_5d >= 3.0
    ):
        successful_observation = True

    elif (
        np.isfinite(return_10d)
        and return_10d >= 5.0
    ):
        successful_observation = True

    elif (
        np.isfinite(return_15d)
        and return_15d >= 7.0
    ):
        successful_observation = True

    updates[
        "missed_opportunity"
    ] = missed_opportunity

    updates[
        "successful_observation"
    ] = successful_observation

    completed_15d = safe_bool(
        updates.get(
            "completed_15d",
            row.get("completed_15d"),
        )
    )

    if completed_15d:
        updates["record_status"] = "COMPLETED"
    elif available_days > 0:
        updates["record_status"] = "IN_PROGRESS"
    else:
        updates["record_status"] = "OPEN"

    updates["last_evaluated_date"] = (
        evaluated_date
    )

    updates["version"] = "V29.0"

    return updates


# ============================================================
# ÖZET RAPOR
# ============================================================


def build_summary(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "entry_decision",
                "observation_count",
                "completed_5d_count",
                "completed_10d_count",
                "completed_15d_count",
                "positive_5d_rate",
                "positive_10d_rate",
                "positive_15d_rate",
                "average_return_5d",
                "average_return_10d",
                "average_return_15d",
                "missed_opportunity_count",
                "successful_observation_count",
            ]
        )

    rows: list[dict[str, Any]] = []

    for decision, group in frame.groupby(
        "entry_decision",
        dropna=False,
    ):
        completed_5 = group[
            group["completed_5d"]
            .fillna(False)
            .astype(bool)
        ]

        completed_10 = group[
            group["completed_10d"]
            .fillna(False)
            .astype(bool)
        ]

        completed_15 = group[
            group["completed_15d"]
            .fillna(False)
            .astype(bool)
        ]

        def positive_rate(
            subset: pd.DataFrame,
            column: str,
        ) -> float:
            if subset.empty:
                return 0.0

            values = pd.to_numeric(
                subset[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                return 0.0

            return round(
                float(
                    (
                        values > 0
                    ).mean()
                    * 100.0
                ),
                2,
            )

        def average_return(
            subset: pd.DataFrame,
            column: str,
        ) -> float:
            if subset.empty:
                return 0.0

            values = pd.to_numeric(
                subset[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                return 0.0

            return round(
                float(values.mean()),
                2,
            )

        rows.append(
            {
                "entry_decision": tx(
                    decision
                ),
                "observation_count": int(
                    len(group)
                ),
                "completed_5d_count": int(
                    len(completed_5)
                ),
                "completed_10d_count": int(
                    len(completed_10)
                ),
                "completed_15d_count": int(
                    len(completed_15)
                ),
                "positive_5d_rate": positive_rate(
                    completed_5,
                    "return_5d",
                ),
                "positive_10d_rate": positive_rate(
                    completed_10,
                    "return_10d",
                ),
                "positive_15d_rate": positive_rate(
                    completed_15,
                    "return_15d",
                ),
                "average_return_5d": average_return(
                    completed_5,
                    "return_5d",
                ),
                "average_return_10d": average_return(
                    completed_10,
                    "return_10d",
                ),
                "average_return_15d": average_return(
                    completed_15,
                    "return_15d",
                ),
                "missed_opportunity_count": int(
                    group[
                        "missed_opportunity"
                    ]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
                "successful_observation_count": int(
                    group[
                        "successful_observation"
                    ]
                    .fillna(False)
                    .astype(bool)
                    .sum()
                ),
            }
        )

    summary = pd.DataFrame(
        rows
    )

    return summary.sort_values(
        [
            "missed_opportunity_count",
            "observation_count",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================
# BOŞ DURUM
# ============================================================


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

    build_summary(
        pd.DataFrame()
    ).to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        columns=OUTPUT_COLUMNS
    ).to_csv(
        MISSED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": status_name,
        "observation_count": 0,
        "symbol_count": 0,
        "completed_1d_count": 0,
        "completed_3d_count": 0,
        "completed_5d_count": 0,
        "completed_10d_count": 0,
        "completed_15d_count": 0,
        "missed_opportunity_count": 0,
        "successful_observation_count": 0,
        "version": "V29.0",
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


# ============================================================
# ANA ÇALIŞMA
# ============================================================


def main() -> None:
    v28_history = normalize_observations(
        load_csv(
            V28_HISTORY_FILE
        )
    )

    if v28_history.empty:
        save_empty_status(
            "v28_history_missing"
        )
        return

    old_v29 = normalize_observations(
        load_csv(
            OUTPUT_FILE
        )
    )

    evaluated = combine_history(
        v28_history,
        old_v29,
    )

    if evaluated.empty:
        save_empty_status(
            "no_observations"
        )
        return

    evaluated_date = (
        pd.Timestamp.today()
        .normalize()
        .strftime("%Y-%m-%d")
    )

    observation_dates = pd.to_datetime(
        evaluated["observation_date"],
        errors="coerce",
    )

    earliest_dates = (
        pd.DataFrame(
            {
                "symbol": evaluated["symbol"],
                "observation_date": observation_dates,
            }
        )
        .dropna(
            subset=["observation_date"]
        )
        .groupby("symbol")[
            "observation_date"
        ]
        .min()
        .to_dict()
    )

    price_cache: dict[
        str,
        pd.DataFrame,
    ] = {}

    symbols = sorted(
        evaluated["symbol"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    for index, symbol in enumerate(
        symbols,
        start=1,
    ):
        start_date = earliest_dates.get(
            symbol
        )

        if start_date is None:
            price_cache[symbol] = (
                pd.DataFrame()
            )
            continue

        print(
            f"V29 fiyat kontrolü "
            f"{index}/{len(symbols)}: "
            f"{symbol}"
        )

        price_cache[symbol] = (
            download_symbol_history(
                symbol,
                pd.Timestamp(start_date),
            )
        )

        time.sleep(0.15)

    for row_index, row in evaluated.iterrows():
        symbol = normalize_symbol(
            row.get("symbol")
        )

        prices = price_cache.get(
            symbol,
            pd.DataFrame(),
        )

        updates = evaluate_row(
            row,
            prices,
            evaluated_date,
        )

        for column, value in updates.items():
            evaluated.at[
                row_index,
                column,
            ] = value

    evaluated = prepare_columns(
        evaluated
    )

    evaluated = evaluated.sort_values(
        [
            "observation_date",
            "symbol",
            "observation_id",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(
        drop=True
    )

    result = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column in evaluated.columns:
            result[column] = (
                evaluated[column]
            )
        else:
            result[column] = np.nan

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    summary = build_summary(
        result
    )

    summary.to_csv(
        SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    missed = result[
        result["missed_opportunity"]
        .fillna(False)
        .astype(bool)
    ].copy()

    missed = missed.sort_values(
        [
            "return_15d",
            "return_10d",
            "max_return_15d",
        ],
        ascending=False,
        na_position="last",
    )

    missed.to_csv(
        MISSED_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": "ready",

        "observation_count": int(
            len(result)
        ),

        "symbol_count": int(
            result["symbol"]
            .nunique()
        ),

        "completed_1d_count": int(
            result["completed_1d"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

        "completed_3d_count": int(
            result["completed_3d"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

        "completed_5d_count": int(
            result["completed_5d"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

        "completed_10d_count": int(
            result["completed_10d"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

        "completed_15d_count": int(
            result["completed_15d"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

        "missed_opportunity_count": int(
            result["missed_opportunity"]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

        "successful_observation_count": int(
            result[
                "successful_observation"
            ]
            .fillna(False)
            .astype(bool)
            .sum()
        ),

        "open_count": int(
            (
                result["record_status"]
                == "OPEN"
            ).sum()
        ),

        "in_progress_count": int(
            (
                result["record_status"]
                == "IN_PROGRESS"
            ).sum()
        ),

        "completed_count": int(
            (
                result["record_status"]
                == "COMPLETED"
            ).sum()
        ),

        "version": "V29.0",
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
        "===== V29 SONUÇ ÖZETİ ====="
    )

    if summary.empty:
        print(
            "Henüz tamamlanmış sonuç bulunmuyor."
        )
    else:
        print(
            summary.to_string(
                index=False
            )
        )

    print(
        "===== V29 KAÇIRILAN FIRSATLAR ====="
    )

    if missed.empty:
        print(
            "Henüz kaçırılmış fırsat oluşmadı."
        )
    else:
        visible_columns = [
            column
            for column in [
                "symbol",
                "observation_date",
                "entry_decision",
                "entry_price",
                "return_5d",
                "return_10d",
                "return_15d",
                "max_return_15d",
                "max_drawdown_15d",
            ]
            if column in missed.columns
        ]

        print(
            missed[
                visible_columns
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
