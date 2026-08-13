from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# V33.4 - ADAY PERFORMANS TAKIP / SONUC DOGRULAMA
# ============================================================

VERSION = "V33.4"

INPUT_FILE = Path("v33_3_confirmed_candidates.csv")

TRACKING_FILE = Path("v33_4_candidate_history.csv")
STATUS_FILE = Path("v33_4_status.json")

RSI_USAGE = "DISABLED"


OUTPUT_COLUMNS = [
    "tracking_date",
    "symbol",
    "v33_3_rank",
    "v33_3_decision",
    "v33_3_score",
    "v33_3_confidence",
    "reference_price",

    "price_d1",
    "return_d1",

    "price_d2",
    "return_d2",

    "price_d3",
    "return_d3",

    "price_d5",
    "return_d5",

    "max_return",
    "min_return",

    "hit_3pct",
    "hit_5pct",
    "hit_7pct",
    "hit_9pct",

    "result_class",
    "tracking_status",

    "prescan_score",
    "v33_score",
    "positive_5d_rate",
    "average_return_5d",
    "median_return_5d",
    "worst_return_5d",

    "current_technical_score",
    "historical_quality_score",
    "consistency_score",

    "risk_score",
    "risk_class",
    "regime",
    "market_percentile",
    "timing_confidence",

    "rsi_usage",
    "tracker_version",
]


# ============================================================
# YARDIMCILAR
# ============================================================

def text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    return str(value).strip()


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


def normalize_symbol(
    value: Any,
) -> str:
    symbol = (
        text(value)
        .upper()
        .replace(" ", "")
    )

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

    for encoding in (
        "utf-8-sig",
        "utf-8",
        "latin-1",
    ):
        try:
            return pd.read_csv(
                path,
                encoding=encoding,
            )

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

        except UnicodeDecodeError:
            continue

        except Exception as exc:
            print(
                f"UYARI: {path} okunamadi: {exc}"
            )
            return pd.DataFrame()

    return pd.DataFrame()


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


# ============================================================
# BUGUNUN ADAYLARINI HAZIRLA
# ============================================================

def prepare_today_candidates() -> pd.DataFrame:
    frame = load_csv(
        INPUT_FILE
    )

    if frame.empty:
        return pd.DataFrame()

    if "symbol" not in frame.columns:
        return pd.DataFrame()

    frame = frame.copy()

    frame["symbol"] = frame[
        "symbol"
    ].apply(
        normalize_symbol
    )

    frame = frame[
        frame["symbol"].ne("")
    ].copy()

    frame = frame.drop_duplicates(
        subset=["symbol"],
        keep="first",
    )

    return frame.reset_index(
        drop=True
    )


# ============================================================
# YENI TAKIP KAYDI OLUSTUR
# ============================================================

def create_tracking_rows(
    candidates: pd.DataFrame,
) -> pd.DataFrame:

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    rows = []

    for _, row in candidates.iterrows():

        symbol = normalize_symbol(
            row.get("symbol")
        )

        reference_price = number(
            row.get("close")
        )

        tracking_row = {
            "tracking_date": today,

            "symbol": symbol,

            "v33_3_rank": number(
                row.get("v33_3_rank")
            ),

            "v33_3_decision": text(
                row.get("v33_3_decision")
            ),

            "v33_3_score": number(
                row.get("v33_3_score")
            ),

            "v33_3_confidence": number(
                row.get("v33_3_confidence")
            ),

            "reference_price": reference_price,

            "price_d1": np.nan,
            "return_d1": np.nan,

            "price_d2": np.nan,
            "return_d2": np.nan,

            "price_d3": np.nan,
            "return_d3": np.nan,

            "price_d5": np.nan,
            "return_d5": np.nan,

            "max_return": np.nan,
            "min_return": np.nan,

            "hit_3pct": False,
            "hit_5pct": False,
            "hit_7pct": False,
            "hit_9pct": False,

            "result_class": "BEKLIYOR",
            "tracking_status": "AKTIF",

            "prescan_score": number(
                row.get("prescan_score")
            ),

            "v33_score": number(
                row.get("v33_score")
            ),

            "positive_5d_rate": number(
                row.get("positive_5d_rate")
            ),

            "average_return_5d": number(
                row.get("average_return_5d")
            ),

            "median_return_5d": number(
                row.get("median_return_5d")
            ),

            "worst_return_5d": number(
                row.get("worst_return_5d")
            ),

            "current_technical_score": number(
                row.get(
                    "current_technical_score"
                )
            ),

            "historical_quality_score": number(
                row.get(
                    "historical_quality_score"
                )
            ),

            "consistency_score": number(
                row.get("consistency_score")
            ),

            "risk_score": number(
                row.get("risk_score"),
                np.nan,
            ),

            "risk_class": text(
                row.get("risk_class")
            ),

            "regime": text(
                row.get("regime")
            ),

            "market_percentile": number(
                row.get(
                    "market_percentile"
                )
            ),

            "timing_confidence": number(
                row.get(
                    "timing_confidence"
                ),
                np.nan,
            ),

            "rsi_usage": RSI_USAGE,

            "tracker_version": VERSION,
        }

        rows.append(
            tracking_row
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# ESKI TAKIP DOSYASI
# ============================================================

def prepare_existing_history() -> pd.DataFrame:

    history = load_csv(
        TRACKING_FILE
    )

    if history.empty:
        return pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    for column in OUTPUT_COLUMNS:

        if column not in history.columns:
            history[column] = np.nan

    history = history[
        OUTPUT_COLUMNS
    ].copy()

    history["symbol"] = history[
        "symbol"
    ].apply(
        normalize_symbol
    )

    return history


# ============================================================
# SONUC SINIFLANDIRMA
# ============================================================

def classify_result(
    max_return: float,
    final_return: float,
) -> str:

    if pd.isna(max_return):
        return "BEKLIYOR"

    if max_return >= 9:
        return "9%+ BASARI"

    if max_return >= 7:
        return "7%+ GUCLU"

    if max_return >= 5:
        return "5%+ BASARILI"

    if max_return >= 3:
        return "3%+ POZITIF"

    if final_return >= 0:
        return "POZITIF"

    if final_return > -3:
        return "ZAYIF"

    return "BASARISIZ"


# ============================================================
# MEVCUT KAYITLARI GUNCELLE
# ============================================================

def refresh_tracking_flags(
    history: pd.DataFrame,
) -> pd.DataFrame:

    if history.empty:
        return history

    history = history.copy()

    return_columns = [
        "return_d1",
        "return_d2",
        "return_d3",
        "return_d5",
    ]

    for column in return_columns:
        history[column] = pd.to_numeric(
            history[column],
            errors="coerce",
        )

    for index, row in history.iterrows():

        returns = []

        for column in return_columns:

            value = row.get(
                column
            )

            if pd.notna(value):
                returns.append(
                    float(value)
                )

        if not returns:
            continue

        max_return = max(
            returns
        )

        min_return = min(
            returns
        )

        history.at[
            index,
            "max_return",
        ] = round(
            max_return,
            2,
        )

        history.at[
            index,
            "min_return",
        ] = round(
            min_return,
            2,
        )

        history.at[
            index,
            "hit_3pct",
        ] = bool(
            max_return >= 3
        )

        history.at[
            index,
            "hit_5pct",
        ] = bool(
            max_return >= 5
        )

        history.at[
            index,
            "hit_7pct",
        ] = bool(
            max_return >= 7
        )

        history.at[
            index,
            "hit_9pct",
        ] = bool(
            max_return >= 9
        )

        final_return = returns[
            -1
        ]

        history.at[
            index,
            "result_class",
        ] = classify_result(
            max_return,
            final_return,
        )

        if pd.notna(
            row.get("return_d5")
        ):
            history.at[
                index,
                "tracking_status",
            ] = "TAMAMLANDI"

    return history


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    candidates = prepare_today_candidates()

    existing = prepare_existing_history()

    if candidates.empty:

        status = {
            "status": "no_candidates",
            "new_candidate_count": 0,
            "total_tracking_count": int(
                len(existing)
            ),
            "rsi_usage": RSI_USAGE,
            "version": VERSION,
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

        return

    new_rows = create_tracking_rows(
        candidates
    )

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # AYNI GUN + AYNI HISSE TEKRAR KAYDEDILMESIN
    # --------------------------------------------------------

    if not existing.empty:

        existing_keys = set(
            zip(
                existing[
                    "tracking_date"
                ].astype(str),

                existing[
                    "symbol"
                ].astype(str),
            )
        )

        new_rows = new_rows[
            ~new_rows.apply(
                lambda row: (
                    str(
                        row[
                            "tracking_date"
                        ]
                    ),
                    str(
                        row[
                            "symbol"
                        ]
                    ),
                )
                in existing_keys,
                axis=1,
            )
        ].copy()

    # --------------------------------------------------------
    # ESKI + YENI
    # --------------------------------------------------------

    frames = [
        frame
        for frame in (
            existing,
            new_rows,
        )
        if not frame.empty
    ]

    if frames:
        history = pd.concat(
            frames,
            ignore_index=True,
        )
    else:
        history = pd.DataFrame(
            columns=OUTPUT_COLUMNS
        )

    history = refresh_tracking_flags(
        history
    )

    # --------------------------------------------------------
    # KOLON SIRASI
    # --------------------------------------------------------

    for column in OUTPUT_COLUMNS:

        if column not in history.columns:
            history[column] = np.nan

    history = history[
        OUTPUT_COLUMNS
    ]

    # --------------------------------------------------------
    # KAYDET
    # --------------------------------------------------------

    history.to_csv(
        TRACKING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # ISTATISTIKLER
    # --------------------------------------------------------

    completed = history[
        history[
            "tracking_status"
        ]
        == "TAMAMLANDI"
    ]

    hit_3_count = int(
        history[
            "hit_3pct"
        ].fillna(False).sum()
    )

    hit_5_count = int(
        history[
            "hit_5pct"
        ].fillna(False).sum()
    )

    hit_7_count = int(
        history[
            "hit_7pct"
        ].fillna(False).sum()
    )

    hit_9_count = int(
        history[
            "hit_9pct"
        ].fillna(False).sum()
    )

    status = {
        "status": "ready",

        "tracking_date": today,

        "new_candidate_count": int(
            len(new_rows)
        ),

        "total_tracking_count": int(
            len(history)
        ),

        "completed_count": int(
            len(completed)
        ),

        "active_count": int(
            (
                history[
                    "tracking_status"
                ]
                == "AKTIF"
            ).sum()
        ),

        "hit_3pct_count": hit_3_count,
        "hit_5pct_count": hit_5_count,
        "hit_7pct_count": hit_7_count,
        "hit_9pct_count": hit_9_count,

        "rsi_usage": RSI_USAGE,

        "version": VERSION,
    }

    STATUS_FILE.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # TERMINAL
    # --------------------------------------------------------

    print(
        "===== V33.4 PERFORMANCE TRACKER ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "\n===== BUGUN EKLENEN ADAYLAR ====="
    )

    if new_rows.empty:

        print(
            "Bugunun adaylari zaten kayitli."
        )

    else:

        print(
            new_rows[
                [
                    "symbol",
                    "v33_3_rank",
                    "v33_3_decision",
                    "v33_3_score",
                    "v33_3_confidence",
                    "reference_price",
                ]
            ].to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
