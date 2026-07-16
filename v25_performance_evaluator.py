from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


POSITIONS_FILE = Path("v23_positions.csv")
HISTORY_FILE = Path("v23_position_history.csv")

OUTPUT_FILE = Path("v25_performance_evaluations.csv")
MEMORY_FILE = Path("v25_performance_memory.csv")
STATUS_FILE = Path("v25_status.json")


OUTPUT_COLUMNS = [
    "symbol",
    "evaluation_state",
    "realized_proxy_return_pct",
    "max_gain_pct",
    "max_drawdown_pct",
    "hit_first_objective",
    "hit_optimistic_objective",
    "hit_invalidation",
    "quality_score",
    "reliability_class",
    "evaluated_at_utc",
    "position_state",
    "action",
    "entry_reference",
    "last_price",
    "latest_v22_score",
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


def normalize_output(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return empty_output()

    result = frame.copy()

    for column in OUTPUT_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    return result[OUTPUT_COLUMNS].copy()


def fetch_daily(
    symbol: str,
) -> pd.DataFrame:
    ticker = (
        symbol
        if symbol.endswith(".IS")
        else f"{symbol}.IS"
    )

    try:
        frame = yf.download(
            ticker,
            period="3mo",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
        )

    except Exception as exc:
        print(
            f"Uyarı: {symbol} günlük verisi alınamadı: {exc}"
        )
        return pd.DataFrame()

    if frame.empty:
        return pd.DataFrame()

    if isinstance(
        frame.columns,
        pd.MultiIndex,
    ):
        frame.columns = [
            column[0]
            if isinstance(column, tuple)
            else column
            for column in frame.columns
        ]

    frame.columns = [
        str(column).title()
        for column in frame.columns
    ]

    required_columns = [
        "Close",
        "High",
        "Low",
    ]

    if not all(
        column in frame.columns
        for column in required_columns
    ):
        return pd.DataFrame()

    frame = frame.dropna(
        subset=required_columns
    )

    return frame


def classify_reliability(
    score: float,
) -> str:
    if score >= 75:
        return "ÇOK İYİ"

    if score >= 60:
        return "İYİ"

    if score >= 45:
        return "ORTA"

    return "ZAYIF"


def determine_evaluation_state(
    return_pct: float,
    hit_first: bool,
    hit_optimistic: bool,
    hit_invalidation: bool,
) -> str:
    if hit_optimistic:
        return "OLUMLU SENARYO GERÇEKLEŞTİ"

    if hit_first:
        return "İLK HEDEF GERÇEKLEŞTİ"

    if hit_invalidation:
        return "GEÇERSİZLİK GÖRÜLDÜ"

    if return_pct > 0.25:
        return "POZİTİF İLERLİYOR"

    if return_pct < -0.25:
        return "NEGATİF İLERLİYOR"

    return "YATAY"


def calculate_quality_score(
    return_pct: float,
    max_gain_pct: float,
    max_drawdown_pct: float,
    hit_first: bool,
    hit_optimistic: bool,
    hit_invalidation: bool,
    latest_v22_score: float,
) -> float:
    score = 50.0

    score += float(
        np.clip(
            return_pct * 4.0,
            -24,
            24,
        )
    )

    if max_gain_pct >= 3:
        score += 8

    elif max_gain_pct >= 1:
        score += 4

    if max_drawdown_pct > -2:
        score += 6

    elif max_drawdown_pct < -5:
        score -= 10

    if hit_first:
        score += 10

    if hit_optimistic:
        score += 12

    if hit_invalidation:
        score -= 18

    if latest_v22_score >= 75:
        score += 6

    elif latest_v22_score >= 65:
        score += 3

    elif latest_v22_score < 50:
        score -= 6

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


def evaluate_position(
    row: pd.Series,
) -> dict[str, Any]:
    symbol = tx(
        row.get("symbol")
    )

    entry_price = sf(
        row.get("entry_reference")
    )

    last_price = sf(
        row.get("last_price"),
        entry_price,
    )

    latest_v22_score = sf(
        row.get("latest_v22_score")
    )

    position_state = tx(
        row.get("position_state")
    )

    action = tx(
        row.get("action")
    )

    days_in_position = max(
        0,
        int(
            sf(
                row.get("days_in_position")
            )
        ),
    )

    invalidation_price = sf(
        row.get(
            "statistical_invalidation_price"
        )
    )

    first_objective_price = sf(
        row.get(
            "first_objective_price"
        )
    )

    optimistic_objective_price = sf(
        row.get(
            "optimistic_objective_price"
        )
    )

    if (
        not symbol
        or entry_price <= 0
    ):
        return {
            "symbol": symbol,
            "evaluation_state": "VERİ YETERSİZ",
            "realized_proxy_return_pct": 0.0,
            "max_gain_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "hit_first_objective": False,
            "hit_optimistic_objective": False,
            "hit_invalidation": False,
            "quality_score": 0.0,
            "reliability_class": "VERİ YETERSİZ",
            "position_state": position_state,
            "action": action,
            "entry_reference": entry_price,
            "last_price": last_price,
            "latest_v22_score": latest_v22_score,
        }

    frame = fetch_daily(
        symbol
    )

    if frame.empty:
        return_pct = (
            last_price / entry_price - 1
        ) * 100

        return {
            "symbol": symbol,
            "evaluation_state": (
                "GÜNCEL VERİ SINIRLI"
            ),
            "realized_proxy_return_pct": round(
                return_pct,
                2,
            ),
            "max_gain_pct": round(
                return_pct,
                2,
            ),
            "max_drawdown_pct": round(
                min(
                    return_pct,
                    0.0,
                ),
                2,
            ),
            "hit_first_objective": False,
            "hit_optimistic_objective": False,
            "hit_invalidation": False,
            "quality_score": 40.0,
            "reliability_class": "DÜŞÜK VERİ",
            "position_state": position_state,
            "action": action,
            "entry_reference": round(
                entry_price,
                4,
            ),
            "last_price": round(
                last_price,
                4,
            ),
            "latest_v22_score": round(
                latest_v22_score,
                2,
            ),
        }

    lookback_days = max(
        2,
        days_in_position + 2,
    )

    recent = frame.tail(
        lookback_days
    )

    highest_price = sf(
        recent["High"].max(),
        last_price,
    )

    lowest_price = sf(
        recent["Low"].min(),
        last_price,
    )

    latest_daily_close = sf(
        recent["Close"].iloc[-1],
        last_price,
    )

    if last_price <= 0:
        last_price = latest_daily_close

    return_pct = (
        last_price / entry_price - 1
    ) * 100

    max_gain_pct = (
        highest_price / entry_price - 1
    ) * 100

    max_drawdown_pct = (
        lowest_price / entry_price - 1
    ) * 100

    hit_first = bool(
        first_objective_price > 0
        and highest_price
        >= first_objective_price
    )

    hit_optimistic = bool(
        optimistic_objective_price > 0
        and highest_price
        >= optimistic_objective_price
    )

    hit_invalidation = bool(
        invalidation_price > 0
        and lowest_price
        <= invalidation_price
    )

    quality_score = calculate_quality_score(
        return_pct=return_pct,
        max_gain_pct=max_gain_pct,
        max_drawdown_pct=max_drawdown_pct,
        hit_first=hit_first,
        hit_optimistic=hit_optimistic,
        hit_invalidation=hit_invalidation,
        latest_v22_score=latest_v22_score,
    )

    reliability_class = (
        classify_reliability(
            quality_score
        )
    )

    evaluation_state = (
        determine_evaluation_state(
            return_pct=return_pct,
            hit_first=hit_first,
            hit_optimistic=hit_optimistic,
            hit_invalidation=hit_invalidation,
        )
    )

    return {
        "symbol": symbol,
        "evaluation_state": (
            evaluation_state
        ),
        "realized_proxy_return_pct": round(
            return_pct,
            2,
        ),
        "max_gain_pct": round(
            max_gain_pct,
            2,
        ),
        "max_drawdown_pct": round(
            max_drawdown_pct,
            2,
        ),
        "hit_first_objective": (
            hit_first
        ),
        "hit_optimistic_objective": (
            hit_optimistic
        ),
        "hit_invalidation": (
            hit_invalidation
        ),
        "quality_score": (
            quality_score
        ),
        "reliability_class": (
            reliability_class
        ),
        "position_state": (
            position_state
        ),
        "action": action,
        "entry_reference": round(
            entry_price,
            4,
        ),
        "last_price": round(
            last_price,
            4,
        ),
        "latest_v22_score": round(
            latest_v22_score,
            2,
        ),
    }


def save_empty_result(
    timestamp: str,
) -> None:
    result = empty_output()

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    existing_memory = load_csv(
        MEMORY_FILE
    )

    if existing_memory.empty:
        result.to_csv(
            MEMORY_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    else:
        normalized_memory = normalize_output(
            existing_memory
        )

        normalized_memory.to_csv(
            MEMORY_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    status = {
        "status": "no_positions",
        "evaluated_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "first_objective_count": 0,
        "optimistic_objective_count": 0,
        "invalidation_count": 0,
        "average_quality_score": 0.0,
        "top_symbol": "",
        "top_quality_score": 0.0,
        "evaluated_at_utc": timestamp,
        "version": "V25.1",
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
        "V25: Değerlendirilecek aktif "
        "pozisyon bulunamadı."
    )


def main() -> None:
    timestamp = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )

    positions = load_csv(
        POSITIONS_FILE
    )

    if (
        positions.empty
        or "symbol" not in positions.columns
    ):
        save_empty_result(
            timestamp
        )
        return

    valid_positions = positions.copy()

    valid_positions = valid_positions[
        valid_positions["symbol"]
        .astype(str)
        .str.strip()
        .ne("")
    ].copy()

    if valid_positions.empty:
        save_empty_result(
            timestamp
        )
        return

    rows: list[
        dict[str, Any]
    ] = []

    for _, position_row in (
        valid_positions.iterrows()
    ):
        try:
            evaluated = evaluate_position(
                position_row
            )

        except Exception as exc:
            symbol = tx(
                position_row.get("symbol")
            )

            print(
                f"Uyarı: {symbol} "
                f"değerlendirilemedi: {exc}"
            )

            evaluated = {
                "symbol": symbol,
                "evaluation_state": (
                    "DEĞERLENDİRME HATASI"
                ),
                "realized_proxy_return_pct": 0.0,
                "max_gain_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "hit_first_objective": False,
                "hit_optimistic_objective": False,
                "hit_invalidation": False,
                "quality_score": 0.0,
                "reliability_class": (
                    "VERİ YETERSİZ"
                ),
                "position_state": tx(
                    position_row.get(
                        "position_state"
                    )
                ),
                "action": tx(
                    position_row.get(
                        "action"
                    )
                ),
                "entry_reference": sf(
                    position_row.get(
                        "entry_reference"
                    )
                ),
                "last_price": sf(
                    position_row.get(
                        "last_price"
                    )
                ),
                "latest_v22_score": sf(
                    position_row.get(
                        "latest_v22_score"
                    )
                ),
            }

        evaluated[
            "evaluated_at_utc"
        ] = timestamp

        rows.append(
            evaluated
        )

    result = pd.DataFrame(
        rows
    )

    result = normalize_output(
        result
    )

    result["quality_score"] = (
        pd.to_numeric(
            result["quality_score"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    result[
        "realized_proxy_return_pct"
    ] = pd.to_numeric(
        result[
            "realized_proxy_return_pct"
        ],
        errors="coerce",
    ).fillna(0.0)

    result = result.sort_values(
        [
            "quality_score",
            "realized_proxy_return_pct",
        ],
        ascending=False,
    ).reset_index(drop=True)

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    existing_memory = load_csv(
        MEMORY_FILE
    )

    if existing_memory.empty:
        complete_memory = result.copy()

    else:
        normalized_memory = normalize_output(
            existing_memory
        )

        complete_memory = pd.concat(
            [
                normalized_memory,
                result,
            ],
            ignore_index=True,
            sort=False,
        )

    complete_memory = normalize_output(
        complete_memory
    )

    complete_memory.to_csv(
        MEMORY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    positive_count = int(
        (
            result[
                "realized_proxy_return_pct"
            ]
            > 0
        ).sum()
    )

    negative_count = int(
        (
            result[
                "realized_proxy_return_pct"
            ]
            < 0
        ).sum()
    )

    first_objective_count = int(
        result[
            "hit_first_objective"
        ]
        .astype(bool)
        .sum()
    )

    optimistic_objective_count = int(
        result[
            "hit_optimistic_objective"
        ]
        .astype(bool)
        .sum()
    )

    invalidation_count = int(
        result[
            "hit_invalidation"
        ]
        .astype(bool)
        .sum()
    )

    average_quality_score = round(
        sf(
            result[
                "quality_score"
            ].mean()
        ),
        2,
    )

    top_symbol = (
        tx(
            result.iloc[0].get(
                "symbol"
            )
        )
        if len(result)
        else ""
    )

    top_quality_score = (
        sf(
            result.iloc[0].get(
                "quality_score"
            )
        )
        if len(result)
        else 0.0
    )

    status = {
        "status": "ready",
        "evaluated_count": int(
            len(result)
        ),
        "positive_count": (
            positive_count
        ),
        "negative_count": (
            negative_count
        ),
        "first_objective_count": (
            first_objective_count
        ),
        "optimistic_objective_count": (
            optimistic_objective_count
        ),
        "invalidation_count": (
            invalidation_count
        ),
        "average_quality_score": (
            average_quality_score
        ),
        "top_symbol": top_symbol,
        "top_quality_score": round(
            top_quality_score,
            2,
        ),
        "evaluated_at_utc": timestamp,
        "version": "V25.1",
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
