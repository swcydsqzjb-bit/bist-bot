from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SIGNALS = Path("v22_signal_states.csv")
POSITIONS = Path("v23_positions.csv")
HISTORY = Path("v23_position_history.csv")
STATUS = Path("v23_status.json")


POSITION_COLUMNS = [
    "symbol",
    "position_state",
    "opened_at",
    "last_updated_at",
    "entry_reference",
    "last_price",
    "highest_price",
    "lowest_price",
    "days_in_position",
    "entry_v22_score",
    "latest_v22_score",
    "entry_signal_state",
    "latest_signal_state",
    "model_weight_pct",
    "best_horizon_days",
    "expected_return",
    "statistical_invalidation_price",
    "first_objective_price",
    "optimistic_objective_price",
    "unrealized_return_pct",
    "max_favorable_excursion_pct",
    "max_adverse_excursion_pct",
    "action",
    "action_reason",
]


HISTORY_COLUMNS = [
    "recorded_at",
    "symbol",
    "event",
    "position_state",
    "action",
    "action_reason",
    "last_price",
    "unrealized_return_pct",
    "latest_v22_score",
    "latest_signal_state",
    "days_in_position",
]


def sf(value: Any, default: float = 0.0) -> float:
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


def load(path: Path) -> pd.DataFrame:
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


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
    )


def empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=POSITION_COLUMNS
    )


def empty_history() -> pd.DataFrame:
    return pd.DataFrame(
        columns=HISTORY_COLUMNS
    )


def normalize_positions(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return empty_positions()

    result = frame.copy()

    for column in POSITION_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    return result[POSITION_COLUMNS].copy()


def normalize_history(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return empty_history()

    result = frame.copy()

    for column in HISTORY_COLUMNS:
        if column not in result.columns:
            result[column] = np.nan

    return result[HISTORY_COLUMNS].copy()


def can_open_position(
    row: pd.Series,
) -> bool:
    signal_state = tx(
        row.get("v22_signal_state")
    )

    signal_score = sf(
        row.get("v22_signal_score")
    )

    risk_score = sf(
        row.get("risk_score"),
        100.0,
    )

    expected_return = sf(
        row.get("expected_return")
    )

    valid_state = signal_state in {
        "GÜÇLÜ TEYİT",
        "İZLEMEYE AL",
    }

    return (
        valid_state
        and signal_score >= 68
        and risk_score <= 45
        and expected_return > 0
    )


def determine_action(
    signal_state: str,
    current_price: float,
    invalidation_price: float,
    first_objective_price: float,
    optimistic_objective_price: float,
    latest_score: float,
    previous_score: float,
) -> tuple[str, str]:
    if signal_state in {
        "RİSKLİ - ELE",
        "ELE",
    }:
        return (
            "ÇIKIŞ İZLE",
            "V22 sinyal durumu zayıfladı",
        )

    if (
        invalidation_price > 0
        and current_price <= invalidation_price
    ):
        return (
            "GEÇERSİZLİK",
            "Fiyat istatistiksel geçersizlik bölgesine ulaştı",
        )

    if (
        optimistic_objective_price > 0
        and current_price >= optimistic_objective_price
    ):
        return (
            "KÂR KORU",
            "Olumlu senaryo bölgesine ulaşıldı",
        )

    if (
        first_objective_price > 0
        and current_price >= first_objective_price
    ):
        return (
            "KISMİ KÂR İZLE",
            "İlk gözlem bölgesine ulaşıldı",
        )

    if signal_state == "GÜÇLÜ TEYİT":
        return (
            "TAŞI",
            "Güçlü teyit devam ediyor",
        )

    if signal_state == "İZLEMEYE AL":
        if latest_score - previous_score >= 3:
            return (
                "TAŞI",
                "V22 skoru güçleniyor",
            )

        return (
            "KORU",
            "İzleme sinyali korunuyor",
        )

    if signal_state == "TEYİT BEKLE":
        if latest_score - previous_score <= -5:
            return (
                "ZAYIFLAMA",
                "V22 skoru belirgin geriledi",
            )

        return (
            "TEYİT BEKLE",
            "Yeni teyit henüz oluşmadı",
        )

    if signal_state == "PASİF İZLEME":
        return (
            "PASİF TAKİP",
            "Sinyal aktif teyit üretmiyor",
        )

    return (
        "İZLE",
        "Belirgin yeni durum oluşmadı",
    )


def create_position(
    signal_row: pd.Series,
    timestamp: str,
) -> dict[str, Any]:
    price = sf(
        signal_row.get("close")
    )

    return {
        "symbol": tx(
            signal_row.get("symbol")
        ),
        "position_state": "AÇIK İZLEME",
        "opened_at": timestamp,
        "last_updated_at": timestamp,
        "entry_reference": round(
            price,
            4,
        ),
        "last_price": round(
            price,
            4,
        ),
        "highest_price": round(
            price,
            4,
        ),
        "lowest_price": round(
            price,
            4,
        ),
        "days_in_position": 0,
        "entry_v22_score": round(
            sf(
                signal_row.get(
                    "v22_signal_score"
                )
            ),
            2,
        ),
        "latest_v22_score": round(
            sf(
                signal_row.get(
                    "v22_signal_score"
                )
            ),
            2,
        ),
        "entry_signal_state": tx(
            signal_row.get(
                "v22_signal_state"
            )
        ),
        "latest_signal_state": tx(
            signal_row.get(
                "v22_signal_state"
            )
        ),
        "model_weight_pct": round(
            sf(
                signal_row.get(
                    "model_weight_pct"
                )
            ),
            2,
        ),
        "best_horizon_days": int(
            sf(
                signal_row.get(
                    "best_horizon_days"
                ),
                1,
            )
        ),
        "expected_return": round(
            sf(
                signal_row.get(
                    "expected_return"
                )
            ),
            2,
        ),
        "statistical_invalidation_price": round(
            sf(
                signal_row.get(
                    "statistical_invalidation_price"
                )
            ),
            4,
        ),
        "first_objective_price": round(
            sf(
                signal_row.get(
                    "first_objective_price"
                )
            ),
            4,
        ),
        "optimistic_objective_price": round(
            sf(
                signal_row.get(
                    "optimistic_objective_price"
                )
            ),
            4,
        ),
        "unrealized_return_pct": 0.0,
        "max_favorable_excursion_pct": 0.0,
        "max_adverse_excursion_pct": 0.0,
        "action": "YENİ İZLEME",
        "action_reason": (
            "V22 aktif izleme koşullarını geçti"
        ),
    }


def update_position(
    old_row: pd.Series,
    signal_row: pd.Series | None,
    timestamp: str,
) -> dict[str, Any]:
    result = old_row.to_dict()

    entry_price = sf(
        old_row.get("entry_reference")
    )

    previous_price = sf(
        old_row.get("last_price"),
        entry_price,
    )

    current_price = previous_price

    previous_score = sf(
        old_row.get("latest_v22_score")
    )

    latest_score = previous_score

    latest_state = tx(
        old_row.get("latest_signal_state")
    )

    if signal_row is not None:
        current_price = sf(
            signal_row.get("close"),
            previous_price,
        )

        latest_state = (
            tx(
                signal_row.get(
                    "v22_signal_state"
                )
            )
            or latest_state
        )

        latest_score = sf(
            signal_row.get(
                "v22_signal_score"
            ),
            previous_score,
        )

        result["model_weight_pct"] = round(
            sf(
                signal_row.get(
                    "model_weight_pct"
                ),
                sf(
                    old_row.get(
                        "model_weight_pct"
                    )
                ),
            ),
            2,
        )

        result["best_horizon_days"] = int(
            sf(
                signal_row.get(
                    "best_horizon_days"
                ),
                sf(
                    old_row.get(
                        "best_horizon_days"
                    ),
                    1,
                ),
            )
        )

        result["expected_return"] = round(
            sf(
                signal_row.get(
                    "expected_return"
                ),
                sf(
                    old_row.get(
                        "expected_return"
                    )
                ),
            ),
            2,
        )

    highest_price = max(
        sf(
            old_row.get("highest_price"),
            entry_price,
        ),
        current_price,
    )

    lowest_price = min(
        sf(
            old_row.get("lowest_price"),
            entry_price,
        ),
        current_price,
    )

    if entry_price > 0:
        unrealized_return = (
            current_price / entry_price - 1
        ) * 100

        max_favorable_excursion = (
            highest_price / entry_price - 1
        ) * 100

        max_adverse_excursion = (
            lowest_price / entry_price - 1
        ) * 100

    else:
        unrealized_return = 0.0
        max_favorable_excursion = 0.0
        max_adverse_excursion = 0.0

    action, action_reason = determine_action(
        signal_state=latest_state,
        current_price=current_price,
        invalidation_price=sf(
            old_row.get(
                "statistical_invalidation_price"
            )
        ),
        first_objective_price=sf(
            old_row.get(
                "first_objective_price"
            )
        ),
        optimistic_objective_price=sf(
            old_row.get(
                "optimistic_objective_price"
            )
        ),
        latest_score=latest_score,
        previous_score=previous_score,
    )

    if action in {
        "GEÇERSİZLİK",
        "ÇIKIŞ İZLE",
    }:
        position_state = "KAPANMA ADAYI"

    elif action in {
        "KÂR KORU",
        "KISMİ KÂR İZLE",
    }:
        position_state = "HEDEF BÖLGESİ"

    else:
        position_state = "AÇIK İZLEME"

    result.update(
        {
            "position_state": position_state,
            "last_updated_at": timestamp,
            "last_price": round(
                current_price,
                4,
            ),
            "highest_price": round(
                highest_price,
                4,
            ),
            "lowest_price": round(
                lowest_price,
                4,
            ),
            "days_in_position": int(
                sf(
                    old_row.get(
                        "days_in_position"
                    )
                )
            )
            + 1,
            "latest_v22_score": round(
                latest_score,
                2,
            ),
            "latest_signal_state": latest_state,
            "unrealized_return_pct": round(
                unrealized_return,
                2,
            ),
            "max_favorable_excursion_pct": round(
                max_favorable_excursion,
                2,
            ),
            "max_adverse_excursion_pct": round(
                max_adverse_excursion,
                2,
            ),
            "action": action,
            "action_reason": action_reason,
        }
    )

    return result


def history_row(
    position: dict[str, Any],
    event: str,
) -> dict[str, Any]:
    return {
        "recorded_at": position[
            "last_updated_at"
        ],
        "symbol": position["symbol"],
        "event": event,
        "position_state": position[
            "position_state"
        ],
        "action": position["action"],
        "action_reason": position[
            "action_reason"
        ],
        "last_price": position[
            "last_price"
        ],
        "unrealized_return_pct": position[
            "unrealized_return_pct"
        ],
        "latest_v22_score": position[
            "latest_v22_score"
        ],
        "latest_signal_state": position[
            "latest_signal_state"
        ],
        "days_in_position": position[
            "days_in_position"
        ],
    }


def save_empty_outputs() -> None:
    empty_positions().to_csv(
        POSITIONS,
        index=False,
        encoding="utf-8-sig",
    )

    current_history = normalize_history(
        load(HISTORY)
    )

    current_history.to_csv(
        HISTORY,
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    signals = load(SIGNALS)

    old_positions = normalize_positions(
        load(POSITIONS)
    )

    timestamp = utc_now()

    signal_map: dict[str, pd.Series] = {}

    if (
        not signals.empty
        and "symbol" in signals.columns
    ):
        for _, signal_row in signals.iterrows():
            symbol = tx(
                signal_row.get("symbol")
            )

            if symbol:
                signal_map[symbol] = signal_row

    output_positions: list[
        dict[str, Any]
    ] = []

    new_history_rows: list[
        dict[str, Any]
    ] = []

    existing_symbols: set[str] = set()

    for _, old_row in old_positions.iterrows():
        symbol = tx(
            old_row.get("symbol")
        )

        if not symbol:
            continue

        existing_symbols.add(symbol)

        updated = update_position(
            old_row=old_row,
            signal_row=signal_map.get(
                symbol
            ),
            timestamp=timestamp,
        )

        output_positions.append(
            updated
        )

        new_history_rows.append(
            history_row(
                updated,
                "GÜNCELLEME",
            )
        )

    if not signals.empty:
        for _, signal_row in signals.iterrows():
            symbol = tx(
                signal_row.get("symbol")
            )

            if not symbol:
                continue

            if symbol in existing_symbols:
                continue

            if not can_open_position(
                signal_row
            ):
                continue

            created = create_position(
                signal_row=signal_row,
                timestamp=timestamp,
            )

            output_positions.append(
                created
            )

            new_history_rows.append(
                history_row(
                    created,
                    "YENİ İZLEME",
                )
            )

            existing_symbols.add(
                symbol
            )

    if output_positions:
        positions_frame = pd.DataFrame(
            output_positions
        )

        positions_frame = (
            normalize_positions(
                positions_frame
            )
        )

        state_priority = {
            "HEDEF BÖLGESİ": 4,
            "AÇIK İZLEME": 3,
            "KAPANMA ADAYI": 2,
        }

        positions_frame["_priority"] = (
            positions_frame[
                "position_state"
            ]
            .map(state_priority)
            .fillna(0)
        )

        positions_frame = (
            positions_frame.sort_values(
                [
                    "_priority",
                    "latest_v22_score",
                ],
                ascending=False,
            )
            .drop(
                columns="_priority"
            )
            .reset_index(drop=True)
        )

    else:
        positions_frame = (
            empty_positions()
        )

    positions_frame.to_csv(
        POSITIONS,
        index=False,
        encoding="utf-8-sig",
    )

    old_history = normalize_history(
        load(HISTORY)
    )

    if new_history_rows:
        new_history = pd.DataFrame(
            new_history_rows
        )

        new_history = normalize_history(
            new_history
        )

        if old_history.empty:
            complete_history = new_history

        else:
            complete_history = pd.concat(
                [
                    old_history,
                    new_history,
                ],
                ignore_index=True,
                sort=False,
            )

    else:
        complete_history = old_history

    complete_history = normalize_history(
        complete_history
    )

    complete_history.to_csv(
        HISTORY,
        index=False,
        encoding="utf-8-sig",
    )

    if positions_frame.empty:
        target_zone_count = 0
        closing_candidate_count = 0
        top_symbol = ""
        top_action = ""

    else:
        target_zone_count = int(
            (
                positions_frame[
                    "position_state"
                ]
                == "HEDEF BÖLGESİ"
            ).sum()
        )

        closing_candidate_count = int(
            (
                positions_frame[
                    "position_state"
                ]
                == "KAPANMA ADAYI"
            ).sum()
        )

        top_symbol = tx(
            positions_frame.iloc[0].get(
                "symbol"
            )
        )

        top_action = tx(
            positions_frame.iloc[0].get(
                "action"
            )
        )

    status = {
        "status": "ready",
        "position_count": int(
            len(positions_frame)
        ),
        "new_position_count": int(
            sum(
                item["event"]
                == "YENİ İZLEME"
                for item in new_history_rows
            )
        ),
        "target_zone_count": (
            target_zone_count
        ),
        "closing_candidate_count": (
            closing_candidate_count
        ),
        "top_symbol": top_symbol,
        "top_action": top_action,
        "version": "V23.1",
    }

    STATUS.write_text(
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

    if positions_frame.empty:
        print(
            "V23: Aktif izleme yaşam döngüsüne "
            "alınan aday bulunamadı."
        )

    else:
        print(
            positions_frame.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
