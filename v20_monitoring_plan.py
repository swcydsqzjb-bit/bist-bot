from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# DOSYALAR
# ---------------------------------------------------------

INPUT_CANDIDATES = [
    Path("v20_3_model_portfolio.csv"),
    Path("v20_model_portfolio.csv"),
    Path("v20_3_portfolio.csv"),
]

OUTPUT_FILE = Path("v20_4_monitoring_plan.csv")
STATUS_FILE = Path("v20_4_status.json")


# ---------------------------------------------------------
# ÇIKTI SÜTUNLARI
# ---------------------------------------------------------

OUTPUT_COLUMNS = [
    "rank",
    "symbol",
    "monitoring_state",
    "status",
    "model_weight_pct",
    "reference_price",
    "review_horizon_days",
    "review_rule",
    "invalidation_price",
    "first_observation_price",
    "optimistic_price",
    "expected_return",
    "downside_scenario",
    "upside_scenario",
    "risk_class",
    "risk_score",
    "top_pick_score",
    "ai_final_score",
    "consensus_score",
    "motor_note",
]


# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------

def sf(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Değeri güvenli biçimde float yapar.
    NaN, sonsuz veya geçersiz değerlerde default döner.
    """

    try:
        number = float(value)

        if np.isfinite(number):
            return number

        return default

    except (TypeError, ValueError):
        return default


def tx(value: Any) -> str:
    """
    Değeri güvenli biçimde temiz metne dönüştürür.
    """

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


def first_value(
    row: pd.Series,
    names: list[str],
    default: Any = None,
) -> Any:
    """
    Birden fazla olası sütun isminden ilk dolu değeri bulur.
    """

    for name in names:
        if name not in row.index:
            continue

        value = row.get(name)

        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        if isinstance(value, str) and not value.strip():
            continue

        return value

    return default


def normalize_symbol(value: Any) -> str:
    """
    BIST sembolünü temizler.
    Örnek: BIMAS.IS -> BIMAS
    """

    symbol = tx(value).upper()

    if symbol.endswith(".IS"):
        symbol = symbol[:-3]

    return symbol


def load_csv(path: Path) -> pd.DataFrame:
    """
    CSV dosyasını güvenli biçimde okur.

    Dosya:
    - yoksa,
    - sıfır baytsa,
    - tamamen boşsa,
    - kodlama sorunu varsa

    boş DataFrame döndürür ve workflow'u durdurmaz.
    """

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
                f"Uyarı: {path} UTF-8 olarak "
                f"okunamadı: {exc}"
            )
            return pd.DataFrame()

    except Exception as exc:
        print(
            f"Uyarı: {path} okunamadı: {exc}"
        )
        return pd.DataFrame()


def locate_input_file() -> Path | None:
    """
    Olası V20.3 giriş dosyalarından mevcut olanı bulur.
    """

    for path in INPUT_CANDIDATES:
        if path.exists():
            return path

    return None


def empty_output() -> pd.DataFrame:
    """
    Başlıklı boş çıktı üretir.
    """

    return pd.DataFrame(
        columns=OUTPUT_COLUMNS
    )


def write_status(
    status_name: str,
    planned_count: int = 0,
    active_count: int = 0,
    waiting_count: int = 0,
    passive_count: int = 0,
    top_symbol: str = "",
) -> None:
    """
    V20.4 durum JSON dosyasını yazar.
    """

    status = {
        "status": status_name,
        "planned_candidate_count": int(
            planned_count
        ),
        "active_confirmation_count": int(
            active_count
        ),
        "waiting_confirmation_count": int(
            waiting_count
        ),
        "passive_count": int(
            passive_count
        ),
        "top_symbol": top_symbol,
        "version": "V20.4.1",
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


def save_empty_result(
    status_name: str,
) -> None:
    """
    Aday olmadığında hata vermeden gerekli dosyaları oluşturur.
    """

    empty_output().to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    write_status(
        status_name=status_name,
    )

    print(
        "V20.4: İzleme planına alınacak "
        "aday bulunamadı."
    )


# ---------------------------------------------------------
# KARAR MANTIĞI
# ---------------------------------------------------------

def determine_monitoring_state(
    portfolio_status: str,
    model_weight: float,
    top_pick_score: float,
    ai_final_score: float,
    risk_score: float,
) -> tuple[str, str]:
    """
    İzleme durumunu ve motor açıklamasını belirler.
    """

    normalized_status = portfolio_status.upper()

    if risk_score >= 65:
        return (
            "PASİF",
            "Risk seviyesi yüksek",
        )

    if normalized_status in {
        "TEYİT GELDİ",
        "GÜÇLÜ TEYİT",
        "AKTİF İZLEME",
        "ÜST DÜZEY TEYİT",
    }:
        return (
            "AKTİF",
            "Ana model tarafından teyit edilen aday",
        )

    if normalized_status in {
        "İZLEMEDE TUT",
        "TEMKİNLİ İZLE",
        "TEYİT BEKLE",
        "İZLEME ADAYI",
    }:
        return (
            "TEYİT BEKLE",
            "Canlı teknik teyit bekleniyor",
        )

    if (
        model_weight >= 20
        and top_pick_score >= 65
        and ai_final_score >= 65
        and risk_score <= 45
    ):
        return (
            "TEYİT BEKLE",
            "Model görünümü olumlu, giriş teyidi bekleniyor",
        )

    return (
        "PASİF",
        "Aktif takip için yeterli ortak güç oluşmadı",
    )


def calculate_monitoring_row(
    row: pd.Series,
    rank: int,
) -> dict[str, Any] | None:
    """
    V20.3 satırından V20.4 izleme planı üretir.
    """

    symbol = normalize_symbol(
        first_value(
            row,
            [
                "symbol",
                "ticker",
                "hisse",
            ],
            "",
        )
    )

    if not symbol:
        return None

    model_weight = sf(
        first_value(
            row,
            [
                "model_weight_pct",
                "optimized_weight_pct",
                "weight_pct",
                "portfolio_weight_pct",
                "model_weight",
            ],
            0.0,
        )
    )

    reference_price = sf(
        first_value(
            row,
            [
                "reference_price",
                "close",
                "price",
                "last_price",
                "entry_reference",
            ],
            0.0,
        )
    )

    horizon = int(
        max(
            1,
            sf(
                first_value(
                    row,
                    [
                        "best_horizon_days",
                        "review_horizon_days",
                        "monitoring_days",
                        "horizon_days",
                    ],
                    5,
                ),
                5,
            ),
        )
    )

    expected_return = sf(
        first_value(
            row,
            [
                "expected_return",
                "expected_average_return",
                "expected_avg_return",
                "mean_return",
            ],
            0.0,
        )
    )

    downside = sf(
        first_value(
            row,
            [
                "downside_20pct",
                "downside_scenario",
                "cautious_scenario",
                "temkinli_senaryo",
            ],
            0.0,
        )
    )

    upside = sf(
        first_value(
            row,
            [
                "upside_80pct",
                "upside_scenario",
                "optimistic_scenario",
                "olumlu_senaryo",
            ],
            0.0,
        )
    )

    risk_score = sf(
        first_value(
            row,
            [
                "risk_score",
                "risk_point",
            ],
            50.0,
        ),
        50.0,
    )

    risk_class = tx(
        first_value(
            row,
            [
                "risk_class",
                "risk_level",
            ],
            "",
        )
    )

    if not risk_class:
        if risk_score <= 25:
            risk_class = "DÜŞÜK"
        elif risk_score <= 55:
            risk_class = "ORTA"
        else:
            risk_class = "YÜKSEK"

    top_pick_score = sf(
        first_value(
            row,
            [
                "top_pick_score",
                "top_pick",
            ],
            0.0,
        )
    )

    ai_final_score = sf(
        first_value(
            row,
            [
                "ai_final_score",
                "final_score",
                "v20_score",
            ],
            0.0,
        )
    )

    consensus_score = sf(
        first_value(
            row,
            [
                "consensus_score",
                "consensus",
            ],
            0.0,
        )
    )

    portfolio_status = tx(
        first_value(
            row,
            [
                "status",
                "portfolio_status",
                "decision",
                "top_pick_status",
                "v20_decision",
            ],
            "",
        )
    )

    monitoring_state, motor_note = (
        determine_monitoring_state(
            portfolio_status=portfolio_status,
            model_weight=model_weight,
            top_pick_score=top_pick_score,
            ai_final_score=ai_final_score,
            risk_score=risk_score,
        )
    )

    if reference_price > 0:
        invalidation_price = (
            reference_price
            * (
                1
                + downside / 100
            )
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
            * (
                1
                + upside / 100
            )
        )

    else:
        invalidation_price = 0.0
        first_observation_price = 0.0
        optimistic_price = 0.0

    review_rule = (
        f"{horizon} işlem günü içinde "
        "sonuç ve risk görünümünü yenile"
    )

    return {
        "rank": rank,
        "symbol": symbol,
        "monitoring_state": monitoring_state,
        "status": monitoring_state,
        "model_weight_pct": round(
            model_weight,
            2,
        ),
        "reference_price": round(
            reference_price,
            4,
        ),
        "review_horizon_days": horizon,
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
        "downside_scenario": round(
            downside,
            2,
        ),
        "upside_scenario": round(
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
        "motor_note": motor_note,
    }


# ---------------------------------------------------------
# ANA ÇALIŞMA
# ---------------------------------------------------------

def main() -> None:
    input_file = locate_input_file()

    if input_file is None:
        save_empty_result(
            "input_file_missing"
        )
        return

    print(
        f"V20.4 giriş dosyası: {input_file}"
    )

    frame = load_csv(
        input_file
    )

    if frame.empty:
        save_empty_result(
            "no_candidates"
        )
        return

    if "symbol" not in frame.columns:
        alternative_symbol_columns = [
            "ticker",
            "hisse",
        ]

        found_symbol_column = None

        for column in alternative_symbol_columns:
            if column in frame.columns:
                found_symbol_column = column
                break

        if found_symbol_column is None:
            save_empty_result(
                "symbol_column_missing"
            )
            return

    rows: list[
        dict[str, Any]
    ] = []

    for index, (_, source_row) in enumerate(
        frame.iterrows(),
        start=1,
    ):
        try:
            monitoring_row = (
                calculate_monitoring_row(
                    row=source_row,
                    rank=index,
                )
            )

            if monitoring_row is not None:
                rows.append(
                    monitoring_row
                )

        except Exception as exc:
            symbol = normalize_symbol(
                first_value(
                    source_row,
                    [
                        "symbol",
                        "ticker",
                        "hisse",
                    ],
                    "",
                )
            )

            print(
                f"Uyarı: {symbol or index} "
                f"izleme planına çevrilemedi: {exc}"
            )

    if not rows:
        save_empty_result(
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

    state_priority = {
        "AKTİF": 3,
        "TEYİT BEKLE": 2,
        "PASİF": 1,
    }

    result["_priority"] = (
        result["monitoring_state"]
        .map(state_priority)
        .fillna(0)
    )

    result = result.sort_values(
        [
            "_priority",
            "model_weight_pct",
            "top_pick_score",
            "ai_final_score",
        ],
        ascending=False,
    ).drop(
        columns="_priority"
    ).reset_index(
        drop=True
    )

    result["rank"] = range(
        1,
        len(result) + 1,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    active_count = int(
        (
            result[
                "monitoring_state"
            ]
            == "AKTİF"
        ).sum()
    )

    waiting_count = int(
        (
            result[
                "monitoring_state"
            ]
            == "TEYİT BEKLE"
        ).sum()
    )

    passive_count = int(
        (
            result[
                "monitoring_state"
            ]
            == "PASİF"
        ).sum()
    )

    top_symbol = tx(
        result.iloc[0].get(
            "symbol"
        )
    )

    write_status(
        status_name="ready",
        planned_count=len(result),
        active_count=active_count,
        waiting_count=waiting_count,
        passive_count=passive_count,
        top_symbol=top_symbol,
    )

    print(
        result.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
