from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# LARUS V30 — SONUÇTABANLI ÖRÜNTÜ ÖĞRENME MOTORU
#
# Amaç:
# - V29 tamamlanmış sonuçlarını inceler.
# - Düşük karar alıp sonradan yükselen örnekleri bulur.
# - Benzer yeni adaylara uygulanabilecek kontrollü bonus üretir.
#
# Güvenlik:
# - Yeterli tamamlanmış örnek yoksa öğrenme PASİF kalır.
# - Tek örnekten öğrenme yapılmaz.
# - Bonus üst sınırı vardır.
# - Doğrudan alım-satım emri üretmez.
# ============================================================


V29_FILE = Path("v29_evaluated_observations.csv")

OUTPUT_RULES_FILE = Path("v30_learned_patterns.csv")
OUTPUT_FEATURE_FILE = Path("v30_feature_adjustments.csv")
STATUS_FILE = Path("v30_status.json")


MIN_COMPLETED_OBSERVATIONS = 20
MIN_GROUP_OBSERVATIONS = 5
MAX_FEATURE_BONUS = 8.0
MAX_TOTAL_BONUS = 12.0


FEATURES = [
    "v27_master_score",
    "v22_signal_score",
    "v24_score",
    "optimized_weight_pct",
    "optimizer_score",
    "top_pick_score",
    "ai_final_score",
    "consensus_score",
    "quality_score",
    "risk_score",
    "market_percentile",
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
]


PATTERN_COLUMNS = [
    "pattern_id",
    "source_decision",
    "feature",
    "direction",
    "threshold_low",
    "threshold_high",
    "sample_count",
    "success_count",
    "success_rate",
    "average_return_5d",
    "average_return_10d",
    "average_return_15d",
    "average_max_return_15d",
    "recommended_bonus",
    "confidence_class",
    "active",
    "version",
]


ADJUSTMENT_COLUMNS = [
    "feature",
    "direction",
    "threshold_low",
    "threshold_high",
    "sample_count",
    "success_rate",
    "average_forward_return",
    "raw_bonus",
    "normalized_bonus",
    "confidence_class",
    "active",
    "version",
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


def ensure_numeric_column(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> None:
    if column not in frame.columns:
        frame[column] = default

    frame[column] = pd.to_numeric(
        frame[column],
        errors="coerce",
    )


def prepare_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    result = frame.copy()

    text_defaults = {
        "entry_decision": "",
        "tracking_class": "",
        "record_status": "",
        "symbol": "",
    }

    for column, default in text_defaults.items():
        if column not in result.columns:
            result[column] = default

        result[column] = (
            result[column]
            .fillna(default)
            .astype(str)
            .str.strip()
            .str.upper()
        )

    boolean_columns = [
        "completed_5d",
        "completed_10d",
        "completed_15d",
        "missed_opportunity",
        "successful_observation",
    ]

    for column in boolean_columns:
        if column not in result.columns:
            result[column] = False

        result[column] = result[column].apply(
            safe_bool
        )

    numeric_columns = list(
        dict.fromkeys(
            FEATURES
            + [
                "return_5d",
                "return_10d",
                "return_15d",
                "max_return_15d",
                "max_drawdown_15d",
            ]
        )
    )

    for column in numeric_columns:
        ensure_numeric_column(
            result,
            column,
        )

    return result


def completed_sample(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    completed = frame[
        frame["completed_5d"]
        | frame["completed_10d"]
        | frame["completed_15d"]
    ].copy()

    return completed


def forward_return(
    row: pd.Series,
) -> float:
    return_15 = sf(
        row.get("return_15d"),
        np.nan,
    )

    return_10 = sf(
        row.get("return_10d"),
        np.nan,
    )

    return_5 = sf(
        row.get("return_5d"),
        np.nan,
    )

    if np.isfinite(return_15):
        return return_15

    if np.isfinite(return_10):
        return return_10

    if np.isfinite(return_5):
        return return_5

    return np.nan


def success_label(
    row: pd.Series,
) -> bool:
    return_5 = sf(
        row.get("return_5d"),
        np.nan,
    )

    return_10 = sf(
        row.get("return_10d"),
        np.nan,
    )

    return_15 = sf(
        row.get("return_15d"),
        np.nan,
    )

    max_return = sf(
        row.get("max_return_15d"),
        np.nan,
    )

    if (
        np.isfinite(return_5)
        and return_5 >= 3.0
    ):
        return True

    if (
        np.isfinite(return_10)
        and return_10 >= 5.0
    ):
        return True

    if (
        np.isfinite(return_15)
        and return_15 >= 7.0
    ):
        return True

    if (
        np.isfinite(max_return)
        and max_return >= 8.0
    ):
        return True

    return False


def confidence_class(
    sample_count: int,
    success_rate: float,
) -> str:
    if (
        sample_count >= 30
        and success_rate >= 70
    ):
        return "YÜKSEK"

    if (
        sample_count >= 15
        and success_rate >= 60
    ):
        return "ORTA"

    return "DÜŞÜK"


def calculate_bonus(
    sample_count: int,
    success_rate: float,
    average_return: float,
) -> float:
    if sample_count < MIN_GROUP_OBSERVATIONS:
        return 0.0

    success_component = np.clip(
        (
            success_rate - 50.0
        )
        / 30.0,
        0.0,
        1.0,
    )

    return_component = np.clip(
        average_return / 8.0,
        0.0,
        1.0,
    )

    sample_component = np.clip(
        sample_count / 30.0,
        0.0,
        1.0,
    )

    raw_bonus = (
        success_component * 4.0
        + return_component * 2.5
        + sample_component * 1.5
    )

    return round(
        float(
            np.clip(
                raw_bonus,
                0.0,
                MAX_FEATURE_BONUS,
            )
        ),
        2,
    )


def feature_patterns(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern_rows: list[dict[str, Any]] = []
    adjustment_rows: list[dict[str, Any]] = []

    target_decisions = {
        "TEYİT BEKLE",
        "PASİF İZLEME",
        "ELE",
        "RİSKLİ - ELE",
    }

    learning_frame = frame[
        frame["entry_decision"].isin(
            target_decisions
        )
    ].copy()

    if learning_frame.empty:
        return (
            pd.DataFrame(
                columns=PATTERN_COLUMNS
            ),
            pd.DataFrame(
                columns=ADJUSTMENT_COLUMNS
            ),
        )

    learning_frame["forward_return"] = (
        learning_frame.apply(
            forward_return,
            axis=1,
        )
    )

    learning_frame["success"] = (
        learning_frame.apply(
            success_label,
            axis=1,
        )
    )

    pattern_number = 1

    for decision in sorted(
        learning_frame[
            "entry_decision"
        ].dropna().unique()
    ):
        decision_frame = learning_frame[
            learning_frame[
                "entry_decision"
            ]
            == decision
        ].copy()

        if len(decision_frame) < MIN_GROUP_OBSERVATIONS:
            continue

        for feature in FEATURES:
            if feature not in decision_frame.columns:
                continue

            feature_data = decision_frame[
                [
                    feature,
                    "forward_return",
                    "success",
                    "return_5d",
                    "return_10d",
                    "return_15d",
                    "max_return_15d",
                ]
            ].dropna(
                subset=[
                    feature,
                    "forward_return",
                ]
            )

            if len(feature_data) < MIN_GROUP_OBSERVATIONS:
                continue

            try:
                median_value = float(
                    feature_data[
                        feature
                    ].median()
                )
            except Exception:
                continue

            groups = [
                (
                    "LOW",
                    feature_data[
                        feature_data[
                            feature
                        ]
                        <= median_value
                    ],
                    -np.inf,
                    median_value,
                ),
                (
                    "HIGH",
                    feature_data[
                        feature_data[
                            feature
                        ]
                        > median_value
                    ],
                    median_value,
                    np.inf,
                ),
            ]

            for (
                direction,
                group,
                threshold_low,
                threshold_high,
            ) in groups:
                sample_count = int(
                    len(group)
                )

                if (
                    sample_count
                    < MIN_GROUP_OBSERVATIONS
                ):
                    continue

                success_count = int(
                    group["success"].sum()
                )

                success_rate = round(
                    success_count
                    / sample_count
                    * 100.0,
                    2,
                )

                average_forward = round(
                    float(
                        group[
                            "forward_return"
                        ].mean()
                    ),
                    2,
                )

                average_return_5d = round(
                    float(
                        group[
                            "return_5d"
                        ].dropna().mean()
                    )
                    if group[
                        "return_5d"
                    ].notna().any()
                    else 0.0,
                    2,
                )

                average_return_10d = round(
                    float(
                        group[
                            "return_10d"
                        ].dropna().mean()
                    )
                    if group[
                        "return_10d"
                    ].notna().any()
                    else 0.0,
                    2,
                )

                average_return_15d = round(
                    float(
                        group[
                            "return_15d"
                        ].dropna().mean()
                    )
                    if group[
                        "return_15d"
                    ].notna().any()
                    else 0.0,
                    2,
                )

                average_max_return = round(
                    float(
                        group[
                            "max_return_15d"
                        ].dropna().mean()
                    )
                    if group[
                        "max_return_15d"
                    ].notna().any()
                    else 0.0,
                    2,
                )

                bonus = calculate_bonus(
                    sample_count,
                    success_rate,
                    average_forward,
                )

                confidence = confidence_class(
                    sample_count,
                    success_rate,
                )

                active = bool(
                    bonus > 0
                    and success_rate >= 60
                    and average_forward > 0
                    and confidence
                    in {
                        "ORTA",
                        "YÜKSEK",
                    }
                )

                pattern_rows.append(
                    {
                        "pattern_id": (
                            f"V30-{pattern_number:04d}"
                        ),
                        "source_decision": decision,
                        "feature": feature,
                        "direction": direction,
                        "threshold_low": (
                            ""
                            if not np.isfinite(
                                threshold_low
                            )
                            else round(
                                threshold_low,
                                4,
                            )
                        ),
                        "threshold_high": (
                            ""
                            if not np.isfinite(
                                threshold_high
                            )
                            else round(
                                threshold_high,
                                4,
                            )
                        ),
                        "sample_count": sample_count,
                        "success_count": success_count,
                        "success_rate": success_rate,
                        "average_return_5d": (
                            average_return_5d
                        ),
                        "average_return_10d": (
                            average_return_10d
                        ),
                        "average_return_15d": (
                            average_return_15d
                        ),
                        "average_max_return_15d": (
                            average_max_return
                        ),
                        "recommended_bonus": bonus,
                        "confidence_class": confidence,
                        "active": active,
                        "version": "V30.0",
                    }
                )

                adjustment_rows.append(
                    {
                        "feature": feature,
                        "direction": direction,
                        "threshold_low": (
                            ""
                            if not np.isfinite(
                                threshold_low
                            )
                            else round(
                                threshold_low,
                                4,
                            )
                        ),
                        "threshold_high": (
                            ""
                            if not np.isfinite(
                                threshold_high
                            )
                            else round(
                                threshold_high,
                                4,
                            )
                        ),
                        "sample_count": sample_count,
                        "success_rate": success_rate,
                        "average_forward_return": (
                            average_forward
                        ),
                        "raw_bonus": bonus,
                        "normalized_bonus": bonus,
                        "confidence_class": confidence,
                        "active": active,
                        "version": "V30.0",
                    }
                )

                pattern_number += 1

    patterns = pd.DataFrame(
        pattern_rows,
        columns=PATTERN_COLUMNS,
    )

    adjustments = pd.DataFrame(
        adjustment_rows,
        columns=ADJUSTMENT_COLUMNS,
    )

    if adjustments.empty:
        return patterns, adjustments

    adjustments = adjustments.sort_values(
        [
            "active",
            "normalized_bonus",
            "success_rate",
            "sample_count",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    active_adjustments = adjustments[
        adjustments["active"]
    ].copy()

    if not active_adjustments.empty:
        active_total = float(
            active_adjustments[
                "normalized_bonus"
            ].sum()
        )

        if active_total > MAX_TOTAL_BONUS:
            scale = (
                MAX_TOTAL_BONUS
                / active_total
            )

            adjustments.loc[
                adjustments["active"],
                "normalized_bonus",
            ] = (
                adjustments.loc[
                    adjustments["active"],
                    "normalized_bonus",
                ]
                * scale
            ).round(2)

    return patterns, adjustments


def save_empty(
    status_name: str,
    completed_count: int = 0,
) -> None:
    pd.DataFrame(
        columns=PATTERN_COLUMNS
    ).to_csv(
        OUTPUT_RULES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        columns=ADJUSTMENT_COLUMNS
    ).to_csv(
        OUTPUT_FEATURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": status_name,
        "learning_active": False,
        "completed_observation_count": int(
            completed_count
        ),
        "minimum_required": int(
            MIN_COMPLETED_OBSERVATIONS
        ),
        "pattern_count": 0,
        "active_pattern_count": 0,
        "total_available_bonus": 0.0,
        "max_total_bonus": MAX_TOTAL_BONUS,
        "version": "V30.0",
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
    observations = prepare_frame(
        load_csv(
            V29_FILE
        )
    )

    if observations.empty:
        save_empty(
            "v29_input_missing",
            0,
        )
        return

    completed = completed_sample(
        observations
    )

    completed_count = int(
        len(completed)
    )

    if (
        completed_count
        < MIN_COMPLETED_OBSERVATIONS
    ):
        save_empty(
            "insufficient_completed_observations",
            completed_count,
        )
        return

    patterns, adjustments = (
        feature_patterns(
            completed
        )
    )

    patterns.to_csv(
        OUTPUT_RULES_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    adjustments.to_csv(
        OUTPUT_FEATURE_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    active_patterns = (
        patterns[
            patterns["active"]
        ]
        if not patterns.empty
        else pd.DataFrame()
    )

    active_adjustments = (
        adjustments[
            adjustments["active"]
        ]
        if not adjustments.empty
        else pd.DataFrame()
    )

    total_available_bonus = (
        round(
            float(
                active_adjustments[
                    "normalized_bonus"
                ].sum()
            ),
            2,
        )
        if not active_adjustments.empty
        else 0.0
    )

    strongest_feature = ""

    if not active_adjustments.empty:
        strongest_feature = tx(
            active_adjustments.iloc[0][
                "feature"
            ]
        )

    status = {
        "status": "ready",
        "learning_active": bool(
            len(active_patterns) > 0
        ),
        "completed_observation_count": (
            completed_count
        ),
        "minimum_required": int(
            MIN_COMPLETED_OBSERVATIONS
        ),
        "pattern_count": int(
            len(patterns)
        ),
        "active_pattern_count": int(
            len(active_patterns)
        ),
        "active_adjustment_count": int(
            len(active_adjustments)
        ),
        "total_available_bonus": (
            total_available_bonus
        ),
        "max_total_bonus": (
            MAX_TOTAL_BONUS
        ),
        "strongest_feature": (
            strongest_feature
        ),
        "version": "V30.0",
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
        "===== V30 AKTİF ÖRÜNTÜLER ====="
    )

    if active_patterns.empty:
        print(
            "Henüz güvenilir aktif örüntü oluşmadı."
        )
    else:
        visible_columns = [
            "pattern_id",
            "source_decision",
            "feature",
            "direction",
            "sample_count",
            "success_rate",
            "average_return_10d",
            "average_return_15d",
            "recommended_bonus",
            "confidence_class",
        ]

        print(
            active_patterns[
                visible_columns
            ].to_string(
                index=False
            )
        )

    print(
        "===== V30 ÖZELLİK BONUSLARI ====="
    )

    if active_adjustments.empty:
        print(
            "Uygulanabilir bonus bulunmadı."
        )
    else:
        print(
            active_adjustments.to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
