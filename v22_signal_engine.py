from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INPUT_PLAN = Path("v20_monitoring_plan.csv")
INPUT_TOP_PICKS = Path("v20_top_picks.csv")
INPUT_V21 = Path("v21_learned_weights.csv")

OUTPUT_FILE = Path("v22_signal_states.csv")
STATUS_FILE = Path("v22_status.json")


OUTPUT_COLUMNS = [
    "v22_rank",
    "symbol",
    "v22_signal_state",
    "v22_signal_score",
    "learning_bonus",
    "monitoring_state",
    "confirmation_state",
    "model_weight_pct",
    "reference_price",
    "close",
    "top_pick_score",
    "ai_final_score",
    "consensus_score",
    "risk_class",
    "risk_score",
    "market_percentile",
    "regime",
    "best_horizon_days",
    "timing_confidence",
    "expected_return",
    "downside_20pct",
    "upside_80pct",
    "v22_reasons",
    "v22_risks",
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


def normalize_symbol(value: Any) -> str:
    symbol = tx(value).upper()

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
                f"Uyarı: {path} UTF-8 olarak okunamadı: {exc}"
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


def learning_adjustment(
    weights: pd.DataFrame,
) -> float:
    if weights.empty:
        return 0.0

    if "learned_weight" not in weights.columns:
        return 0.0

    values = pd.to_numeric(
        weights["learned_weight"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0

    concentration = float(
        values.max()
    )

    if concentration >= 25:
        return 2.0

    if concentration >= 20:
        return 1.0

    return 0.0


def signal_state(
    row: pd.Series,
) -> str:
    monitoring = tx(
        row.get("monitoring_state")
    ).upper()

    confirmation = tx(
        row.get("confirmation_state")
    ).upper()

    top_pick = sf(
        row.get("top_pick_score")
    )

    ai_score = sf(
        row.get("ai_final_score")
    )

    consensus = sf(
        row.get("consensus_score")
    )

    risk_score = sf(
        row.get("risk_score"),
        50.0,
    )

    expected = sf(
        row.get("expected_return")
    )

    timing = sf(
        row.get("timing_confidence")
    )

    downside = sf(
        row.get("downside_20pct")
    )

    # Önce gerçek eleme şartları kontrol edilir.
    if risk_score >= 60:
        return "RİSKLİ - ELE"

    if downside <= -6:
        return "RİSKLİ - ELE"

    if expected <= 0:
        return "ELE"

    # Güçlü teyit yalnızca gerçek teyit gelmişse üretilebilir.
    if (
        monitoring == "AKTİF"
        and confirmation
        in {
            "TEYİT GELDİ",
            "GÜÇLÜ TEYİT",
            "ÜST DÜZEY TEYİT",
        }
        and top_pick >= 78
        and ai_score >= 70
        and consensus >= 75
        and risk_score <= 30
        and timing >= 75
    ):
        return "GÜÇLÜ TEYİT"

    # İzlemeye alma da gerçek veya erken teyit gerektirir.
    if (
        monitoring == "AKTİF"
        and confirmation
        in {
            "TEYİT GELDİ",
            "GÜÇLÜ TEYİT",
            "ERKEN TEYİT",
            "AKTİF İZLEME",
        }
        and top_pick >= 68
        and ai_score >= 60
        and consensus >= 62
        and risk_score <= 45
        and timing >= 65
    ):
        return "İZLEMEYE AL"

    # V20.3'te İZLEMEDE TUT / TEMKİNLİ İZLE olanlar,
    # canlı teyit gelmeden doğrudan aktif izlemeye alınmaz.
    if (
        monitoring == "TEYİT BEKLE"
        and confirmation
        in {
            "İZLEMEDE TUT",
            "TEMKİNLİ İZLE",
            "TEYİT BEKLE",
            "İZLEME ADAYI",
            "",
        }
        and top_pick >= 60
        and ai_score >= 55
        and consensus >= 55
        and risk_score <= 50
        and downside > -5
    ):
        return "TEYİT BEKLE"

    # Güçlü skorlar var fakat izleme durumu pasifse,
    # yine de teyit bekleme sınıfında tutulabilir.
    if (
        monitoring in {
            "PASİF",
            "PASİF İZLEME",
        }
        and top_pick >= 68
        and ai_score >= 65
        and consensus >= 65
        and risk_score <= 35
        and expected > 0
        and downside > -5
    ):
        return "TEYİT BEKLE"

    if (
        ai_score < 50
        or consensus < 50
        or top_pick < 50
    ):
        return "ELE"

    return "PASİF İZLEME"


def signal_score(
    row: pd.Series,
    learning_bonus: float,
) -> float:
    top_pick = sf(
        row.get("top_pick_score")
    )

    ai_score = sf(
        row.get("ai_final_score")
    )

    consensus = sf(
        row.get("consensus_score")
    )

    timing = sf(
        row.get("timing_confidence")
    )

    expected = sf(
        row.get("expected_return")
    )

    risk_score = sf(
        row.get("risk_score"),
        50.0,
    )

    market_percentile = sf(
        row.get("market_percentile")
    )

    expected_component = float(
        np.clip(
            (expected + 3.0)
            / 10.0
            * 100.0,
            0.0,
            100.0,
        )
    )

    score = (
        top_pick * 0.30
        + ai_score * 0.22
        + consensus * 0.18
        + timing * 0.10
        + expected_component * 0.08
        + market_percentile * 0.07
        - risk_score * 0.15
        + learning_bonus
    )

    return round(
        float(
            np.clip(
                score,
                0.0,
                100.0,
            )
        ),
        2,
    )


def reason_text(
    row: pd.Series,
) -> str:
    reasons: list[str] = []

    if sf(
        row.get("top_pick_score")
    ) >= 70:
        reasons.append(
            "Top Pick skoru güçlü"
        )

    if sf(
        row.get("ai_final_score")
    ) >= 70:
        reasons.append(
            "AI Final skoru güçlü"
        )

    if sf(
        row.get("consensus_score")
    ) >= 70:
        reasons.append(
            "Motor uyumu yüksek"
        )

    if sf(
        row.get("timing_confidence")
    ) >= 75:
        reasons.append(
            "Zamanlama güveni güçlü"
        )

    if sf(
        row.get("expected_return")
    ) >= 3:
        reasons.append(
            "Tarihsel beklenen sonuç güçlü"
        )

    elif sf(
        row.get("expected_return")
    ) > 0:
        reasons.append(
            "Tarihsel beklenen sonuç pozitif"
        )

    if sf(
        row.get("market_percentile")
    ) >= 90:
        reasons.append(
            "Piyasa göreli gücü üst %10 diliminde"
        )

    if sf(
        row.get("risk_score"),
        50.0,
    ) <= 30:
        reasons.append(
            "Risk puanı düşük"
        )

    monitoring = tx(
        row.get("monitoring_state")
    ).upper()

    if monitoring == "TEYİT BEKLE":
        reasons.append(
            "Canlı teknik teyit bekleniyor"
        )

    if not reasons:
        reasons.append(
            "Katmanlar henüz ortak güçlü teyit üretmedi"
        )

    return " | ".join(
        dict.fromkeys(reasons)
    )


def risk_text(
    row: pd.Series,
) -> str:
    notes: list[str] = []

    risk_score = sf(
        row.get("risk_score"),
        50.0,
    )

    if risk_score >= 60:
        notes.append(
            "Risk puanı yüksek"
        )

    elif risk_score >= 35:
        notes.append(
            "Risk puanı orta"
        )

    if sf(
        row.get("consensus_score")
    ) < 55:
        notes.append(
            "Motorlar arasında görüş ayrılığı var"
        )

    if sf(
        row.get("expected_return")
    ) <= 0:
        notes.append(
            "Beklenen istatistiksel sonuç pozitif değil"
        )

    if sf(
        row.get("downside_20pct")
    ) <= -5:
        notes.append(
            "Temkinli senaryo zayıf"
        )

    conflict = tx(
        row.get("conflict_note")
    )

    if (
        conflict
        and conflict
        != "Belirgin motor çelişkisi yok"
    ):
        notes.append(
            conflict
        )

    if not notes:
        notes.append(
            "Belirgin ek risk notu yok"
        )

    return " | ".join(
        dict.fromkeys(notes)
    )


def save_empty(
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
        "actionable_count": 0,
        "strong_confirmation_count": 0,
        "waiting_count": 0,
        "passive_count": 0,
        "eliminated_count": 0,
        "top_symbol": "",
        "top_state": "",
        "top_score": 0.0,
        "learning_bonus": 0.0,
        "version": "V22.1",
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
    plan = normalize_frame(
        load_csv(INPUT_PLAN)
    )

    top = normalize_frame(
        load_csv(INPUT_TOP_PICKS)
    )

    learned = load_csv(
        INPUT_V21
    )

    if plan.empty:
        save_empty(
            "monitoring_plan_missing"
        )
        return

    if top.empty:
        save_empty(
            "top_picks_missing"
        )
        return

    # V20.4 planındaki alanları koruyup,
    # V20.2 Top Picks verilerini aynı sembolle birleştirir.
    merged = plan.merge(
        top,
        on="symbol",
        how="left",
        suffixes=("_plan", ""),
    )

    # Plan tarafında bulunan değerleri,
    # Top Picks tarafı boşsa yedek olarak kullanır.
    fallback_columns = [
        "top_pick_score",
        "ai_final_score",
        "consensus_score",
        "risk_score",
        "risk_class",
        "market_percentile",
        "regime",
        "best_horizon_days",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "close",
        "reference_price",
        "conflict_note",
    ]

    for column in fallback_columns:
        plan_column = f"{column}_plan"

        if column not in merged.columns:
            if plan_column in merged.columns:
                merged[column] = merged[
                    plan_column
                ]
            else:
                merged[column] = np.nan

        elif plan_column in merged.columns:
            merged[column] = merged[
                column
            ].combine_first(
                merged[plan_column]
            )

    text_defaults = {
        "monitoring_state": "PASİF",
        "confirmation_state": "",
        "risk_class": "ORTA",
        "regime": "",
        "conflict_note": "",
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
        "model_weight_pct": 0.0,
        "reference_price": 0.0,
        "close": 0.0,
        "top_pick_score": 0.0,
        "ai_final_score": 0.0,
        "consensus_score": 0.0,
        "risk_score": 50.0,
        "market_percentile": 0.0,
        "best_horizon_days": 5.0,
        "timing_confidence": 0.0,
        "expected_return": 0.0,
        "downside_20pct": 0.0,
        "upside_80pct": 0.0,
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

    # Referans fiyat varsa close alanını da doldur.
    merged["close"] = np.where(
        merged["close"] > 0,
        merged["close"],
        merged["reference_price"],
    )

    merged["reference_price"] = np.where(
        merged["reference_price"] > 0,
        merged["reference_price"],
        merged["close"],
    )

    bonus = learning_adjustment(
        learned
    )

    merged["v22_signal_score"] = (
        merged.apply(
            lambda row: signal_score(
                row,
                bonus,
            ),
            axis=1,
        )
    )

    merged["v22_signal_state"] = (
        merged.apply(
            signal_state,
            axis=1,
        )
    )

    merged["v22_reasons"] = (
        merged.apply(
            reason_text,
            axis=1,
        )
    )

    merged["v22_risks"] = (
        merged.apply(
            risk_text,
            axis=1,
        )
    )

    merged["learning_bonus"] = bonus

    priority = {
        "GÜÇLÜ TEYİT": 6,
        "İZLEMEYE AL": 5,
        "TEYİT BEKLE": 4,
        "PASİF İZLEME": 3,
        "ELE": 2,
        "RİSKLİ - ELE": 1,
    }

    merged["_priority"] = (
        merged["v22_signal_state"]
        .map(priority)
        .fillna(0)
    )

    merged = (
        merged.sort_values(
            [
                "_priority",
                "v22_signal_score",
            ],
            ascending=False,
        )
        .drop(
            columns="_priority"
        )
        .reset_index(
            drop=True
        )
    )

    merged.insert(
        0,
        "v22_rank",
        range(
            1,
            len(merged) + 1,
        ),
    )

    result = pd.DataFrame()

    for column in OUTPUT_COLUMNS:
        if column in merged.columns:
            result[column] = merged[
                column
            ]
        else:
            result[column] = np.nan

    result.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    actionable_states = {
        "GÜÇLÜ TEYİT",
        "İZLEMEYE AL",
    }

    eliminated_states = {
        "ELE",
        "RİSKLİ - ELE",
    }

    status = {
        "status": "ready",
        "candidate_count": int(
            len(result)
        ),
        "actionable_count": int(
            result["v22_signal_state"]
            .isin(actionable_states)
            .sum()
        ),
        "strong_confirmation_count": int(
            (
                result[
                    "v22_signal_state"
                ]
                == "GÜÇLÜ TEYİT"
            ).sum()
        ),
        "waiting_count": int(
            (
                result[
                    "v22_signal_state"
                ]
                == "TEYİT BEKLE"
            ).sum()
        ),
        "passive_count": int(
            (
                result[
                    "v22_signal_state"
                ]
                == "PASİF İZLEME"
            ).sum()
        ),
        "eliminated_count": int(
            result["v22_signal_state"]
            .isin(eliminated_states)
            .sum()
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
        "top_state": (
            tx(
                result.iloc[0][
                    "v22_signal_state"
                ]
            )
            if len(result)
            else ""
        ),
        "top_score": (
            sf(
                result.iloc[0][
                    "v22_signal_score"
                ]
            )
            if len(result)
            else 0.0
        ),
        "learning_bonus": bonus,
        "version": "V22.1",
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
