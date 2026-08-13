from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# V33.4 - GERCEK PERFORMANS TAKIP MOTORU
# ============================================================

VERSION = "V33.4"

INPUT_FILE = Path("v33_3_confirmed_candidates.csv")

TRACKING_FILE = Path("v33_4_candidate_history.csv")
STATUS_FILE = Path("v33_4_status.json")

RSI_USAGE = "DISABLED"

YF_PERIOD = "6mo"


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
    "regime_confidence",

    "market_percentile",

    "timing_confidence",

    "rsi_usage",
    "tracker_version",
]


# ============================================================
# YARDIMCILAR
# ============================================================

def text(
    value: Any,
) -> str:

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


def optional_number(
    value: Any,
) -> float:

    try:
        result = float(value)

        if np.isfinite(result):
            return result

    except (TypeError, ValueError):
        pass

    return np.nan


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


def yahoo_symbol(
    value: Any,
) -> str:

    symbol = normalize_symbol(
        value
    )

    if not symbol:
        return ""

    return f"{symbol}.IS"


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
# BUGUNUN ADAYLARI
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

    frame["symbol"] = (
        frame["symbol"]
        .apply(
            normalize_symbol
        )
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

    history = history.copy()

    for column in OUTPUT_COLUMNS:

        ensure_column(
            history,
            column,
            np.nan,
        )

    history["symbol"] = (
        history["symbol"]
        .apply(
            normalize_symbol
        )
    )

    history["tracking_date"] = (
        history["tracking_date"]
        .astype(str)
    )

    return history[
        OUTPUT_COLUMNS
    ].copy()


# ============================================================
# YENI ADAY KAYDI
# ============================================================

def create_tracking_rows(
    candidates: pd.DataFrame,
) -> pd.DataFrame:

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    rows: list[
        dict[str, Any]
    ] = []

    for _, row in candidates.iterrows():

        symbol = normalize_symbol(
            row.get("symbol")
        )

        reference_price = number(
            row.get("close")
        )

        if (
            not symbol
            or reference_price <= 0
        ):
            continue

        rows.append(
            {
                "tracking_date": today,

                "symbol": symbol,

                "v33_3_rank": number(
                    row.get(
                        "v33_3_rank"
                    )
                ),

                "v33_3_decision": text(
                    row.get(
                        "v33_3_decision"
                    )
                ),

                "v33_3_score": number(
                    row.get(
                        "v33_3_score"
                    )
                ),

                "v33_3_confidence": number(
                    row.get(
                        "v33_3_confidence"
                    )
                ),

                "reference_price": (
                    reference_price
                ),

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
                    row.get(
                        "prescan_score"
                    )
                ),

                "v33_score": number(
                    row.get(
                        "v33_score"
                    )
                ),

                "positive_5d_rate": number(
                    row.get(
                        "positive_5d_rate"
                    )
                ),

                "average_return_5d": number(
                    row.get(
                        "average_return_5d"
                    )
                ),

                "median_return_5d": number(
                    row.get(
                        "median_return_5d"
                    )
                ),

                "worst_return_5d": number(
                    row.get(
                        "worst_return_5d"
                    )
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
                    row.get(
                        "consistency_score"
                    )
                ),

                "risk_score": optional_number(
                    row.get(
                        "risk_score"
                    )
                ),

                "risk_class": text(
                    row.get(
                        "risk_class"
                    )
                ),

                "regime": text(
                    row.get(
                        "regime"
                    )
                ),

                "regime_confidence": optional_number(
                    row.get(
                        "regime_confidence"
                    )
                ),

                "market_percentile": number(
                    row.get(
                        "market_percentile"
                    )
                ),

                "timing_confidence": optional_number(
                    row.get(
                        "timing_confidence"
                    )
                ),

                "rsi_usage": RSI_USAGE,

                "tracker_version": VERSION,
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# YAHOO FINANCE
# ============================================================

def download_prices(
    symbols: list[str],
) -> pd.DataFrame:

    tickers = [
        yahoo_symbol(symbol)
        for symbol in symbols
        if yahoo_symbol(symbol)
    ]

    if not tickers:
        return pd.DataFrame()

    print(
        f"V33.4 fiyat indiriliyor: "
        f"{len(tickers)} hisse"
    )

    try:

        data = yf.download(
            tickers=tickers,
            period=YF_PERIOD,
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            progress=False,
            threads=True,
            timeout=45,
        )

        return data

    except Exception as exc:

        print(
            f"V33.4 Yahoo Finance hatasi: {exc}"
        )

        return pd.DataFrame()


def extract_close_series(
    data: pd.DataFrame,
    symbol: str,
    total_symbols: int,
) -> pd.Series:

    ticker = yahoo_symbol(
        symbol
    )

    if data.empty:
        return pd.Series(
            dtype=float
        )

    try:

        if total_symbols == 1:

            frame = data.copy()

        else:

            if isinstance(
                data.columns,
                pd.MultiIndex,
            ):

                level_zero = (
                    data.columns
                    .get_level_values(0)
                    .astype(str)
                )

                level_one = (
                    data.columns
                    .get_level_values(1)
                    .astype(str)
                )

                if ticker in set(
                    level_zero
                ):

                    frame = data[
                        ticker
                    ].copy()

                elif ticker in set(
                    level_one
                ):

                    frame = data.xs(
                        ticker,
                        axis=1,
                        level=1,
                    ).copy()

                else:
                    return pd.Series(
                        dtype=float
                    )

            else:

                frame = data.copy()

        frame.columns = [
            str(column)
            .lower()
            .strip()
            for column in frame.columns
        ]

        if "close" not in frame.columns:

            return pd.Series(
                dtype=float
            )

        close = pd.to_numeric(
            frame["close"],
            errors="coerce",
        ).dropna()

        close.index = pd.to_datetime(
            close.index,
            errors="coerce",
        )

        close = close[
            ~close.index.isna()
        ]

        try:

            close.index = (
                close.index
                .tz_localize(None)
                .normalize()
            )

        except TypeError:

            close.index = (
                close.index
                .normalize()
            )

        close = close[
            close > 0
        ]

        return close.sort_index()

    except Exception as exc:

        print(
            f"{symbol} fiyat verisi ayrıştırılamadı: {exc}"
        )

        return pd.Series(
            dtype=float
        )


# ============================================================
# GETIRI
# ============================================================

def calculate_return(
    reference_price: float,
    future_price: float,
) -> float:

    if (
        reference_price <= 0
        or future_price <= 0
    ):
        return np.nan

    return round(
        (
            future_price
            / reference_price
            - 1.0
        )
        * 100.0,
        2,
    )


# ============================================================
# D+1 D+2 D+3 D+5
# ============================================================

def update_historical_prices(
    history: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    int,
]:

    if history.empty:
        return history, 0

    history = history.copy()

    active = history[
        history[
            "tracking_status"
        ].astype(str).ne(
            "TAMAMLANDI"
        )
    ]

    symbols = sorted(
        {
            normalize_symbol(
                symbol
            )
            for symbol
            in active[
                "symbol"
            ].dropna()
            if normalize_symbol(
                symbol
            )
        }
    )

    if not symbols:

        return history, 0

    downloaded = download_prices(
        symbols
    )

    if downloaded.empty:

        return history, 0

    price_map: dict[
        str,
        pd.Series
    ] = {}

    for symbol in symbols:

        price_map[
            symbol
        ] = extract_close_series(
            downloaded,
            symbol,
            len(symbols),
        )

    updated_cells = 0

    day_targets = {
        1: (
            "price_d1",
            "return_d1",
        ),

        2: (
            "price_d2",
            "return_d2",
        ),

        3: (
            "price_d3",
            "return_d3",
        ),

        5: (
            "price_d5",
            "return_d5",
        ),
    }

    for index, row in history.iterrows():

        if (
            text(
                row.get(
                    "tracking_status"
                )
            )
            == "TAMAMLANDI"
        ):
            continue

        symbol = normalize_symbol(
            row.get("symbol")
        )

        close_series = price_map.get(
            symbol
        )

        if (
            close_series is None
            or close_series.empty
        ):
            continue

        tracking_date = pd.to_datetime(
            row.get(
                "tracking_date"
            ),
            errors="coerce",
        )

        if pd.isna(
            tracking_date
        ):
            continue

        tracking_date = (
            tracking_date.normalize()
        )

        # SADECE takip tarihinden SONRAKI
        # GERCEK BIST işlem günleri.
        future_prices = close_series[
            close_series.index
            > tracking_date
        ]

        if future_prices.empty:
            continue

        reference_price = number(
            row.get(
                "reference_price"
            )
        )

        if reference_price <= 0:
            continue

        for trading_day, (
            price_column,
            return_column,
        ) in day_targets.items():

            existing_price = optional_number(
                row.get(
                    price_column
                )
            )

            if np.isfinite(
                existing_price
            ):
                continue

            # D+1 için index 0
            # D+2 için index 1
            # D+3 için index 2
            # D+5 için index 4
            required_index = (
                trading_day - 1
            )

            if (
                len(future_prices)
                <= required_index
            ):
                continue

            future_price = float(
                future_prices.iloc[
                    required_index
                ]
            )

            future_return = calculate_return(
                reference_price,
                future_price,
            )

            history.at[
                index,
                price_column,
            ] = round(
                future_price,
                4,
            )

            history.at[
                index,
                return_column,
            ] = future_return

            updated_cells += 1

    return (
        history,
        updated_cells,
    )


# ============================================================
# SONUC SINIFLANDIRMA
# ============================================================

def classify_result(
    max_return: float,
    final_return: float,
) -> str:

    if not np.isfinite(
        max_return
    ):
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
# PERFORMANS BAYRAKLARI
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

        available_returns: list[
            float
        ] = []

        for column in return_columns:

            value = optional_number(
                row.get(
                    column
                )
            )

            if np.isfinite(
                value
            ):

                available_returns.append(
                    value
                )

        if not available_returns:
            continue

        max_return = max(
            available_returns
        )

        min_return = min(
            available_returns
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

        final_return = (
            available_returns[-1]
        )

        history.at[
            index,
            "result_class",
        ] = classify_result(
            max_return,
            final_return,
        )

        return_d5 = optional_number(
            row.get(
                "return_d5"
            )
        )

        if np.isfinite(
            return_d5
        ):

            history.at[
                index,
                "tracking_status",
            ] = "TAMAMLANDI"

        else:

            history.at[
                index,
                "tracking_status",
            ] = "AKTIF"

    return history


# ============================================================
# BUGUN AYNI ADAYI TEKRAR EKLEME
# ============================================================

def remove_duplicate_today_rows(
    existing: pd.DataFrame,
    new_rows: pd.DataFrame,
) -> pd.DataFrame:

    if (
        existing.empty
        or new_rows.empty
    ):
        return new_rows

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

    keep_mask = []

    for _, row in new_rows.iterrows():

        key = (
            str(
                row.get(
                    "tracking_date"
                )
            ),

            str(
                row.get(
                    "symbol"
                )
            ),
        )

        keep_mask.append(
            key not in existing_keys
        )

    return new_rows[
        keep_mask
    ].copy()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    candidates = (
        prepare_today_candidates()
    )

    existing = (
        prepare_existing_history()
    )

    # --------------------------------------------------------
    # ÖNCE ESKI ADAYLARIN GERCEK FIYATLARINI GUNCELLE
    # --------------------------------------------------------

    existing, price_updates = (
        update_historical_prices(
            existing
        )
    )

    existing = refresh_tracking_flags(
        existing
    )

    # --------------------------------------------------------
    # BUGUNUN YENI ADAYLARI
    # --------------------------------------------------------

    new_rows = create_tracking_rows(
        candidates
    )

    new_rows = remove_duplicate_today_rows(
        existing,
        new_rows,
    )

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

    # --------------------------------------------------------
    # KOLONLARI GARANTI ET
    # --------------------------------------------------------

    for column in OUTPUT_COLUMNS:

        ensure_column(
            history,
            column,
            np.nan,
        )

    history = history[
        OUTPUT_COLUMNS
    ].copy()

    # --------------------------------------------------------
    # TEKRAR BAYRAKLARI HESAPLA
    # --------------------------------------------------------

    history = refresh_tracking_flags(
        history
    )

    # --------------------------------------------------------
    # SIRALA
    # --------------------------------------------------------

    if not history.empty:

        history = history.sort_values(
            [
                "tracking_date",
                "v33_3_rank",
            ],
            ascending=[
                False,
                True,
            ],
        ).reset_index(
            drop=True
        )

    # --------------------------------------------------------
    # KAYDET
    # --------------------------------------------------------

    history.to_csv(
        TRACKING_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------
    # ISTATISTIK
    # --------------------------------------------------------

    completed_count = int(
        (
            history[
                "tracking_status"
            ]
            == "TAMAMLANDI"
        ).sum()
    ) if not history.empty else 0

    active_count = int(
        (
            history[
                "tracking_status"
            ]
            == "AKTIF"
        ).sum()
    ) if not history.empty else 0

    hit_3 = int(
        history[
            "hit_3pct"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ) if not history.empty else 0

    hit_5 = int(
        history[
            "hit_5pct"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ) if not history.empty else 0

    hit_7 = int(
        history[
            "hit_7pct"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ) if not history.empty else 0

    hit_9 = int(
        history[
            "hit_9pct"
        ]
        .fillna(False)
        .astype(bool)
        .sum()
    ) if not history.empty else 0

    # --------------------------------------------------------
    # D+1 / D+3 / D+5 ORTALAMALARI
    # --------------------------------------------------------

    def column_average(
        column: str,
    ) -> float | None:

        if (
            history.empty
            or column not in history.columns
        ):
            return None

        values = pd.to_numeric(
            history[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            return None

        return round(
            float(
                values.mean()
            ),
            2,
        )

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    status = {
        "status": "ready",

        "tracking_date": (
            datetime.now().strftime(
                "%Y-%m-%d"
            )
        ),

        "new_candidate_count": int(
            len(new_rows)
        ),

        "total_tracking_count": int(
            len(history)
        ),

        "completed_count": (
            completed_count
        ),

        "active_count": active_count,

        "price_update_count": int(
            price_updates
        ),

        "hit_3pct_count": hit_3,
        "hit_5pct_count": hit_5,
        "hit_7pct_count": hit_7,
        "hit_9pct_count": hit_9,

        "average_return_d1": (
            column_average(
                "return_d1"
            )
        ),

        "average_return_d3": (
            column_average(
                "return_d3"
            )
        ),

        "average_return_d5": (
            column_average(
                "return_d5"
            )
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

    # --------------------------------------------------------
    # TERMINAL
    # --------------------------------------------------------

    print(
        "===== V33.4 GERCEK PERFORMANS TAKIP ====="
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
            "Bugunun adaylari zaten kayitli "
            "veya yeni aday bulunmadi."
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

    print(
        "\n===== TAKIPTEKI SON KAYITLAR ====="
    )

    if not history.empty:

        print(
            history[
                [
                    "tracking_date",
                    "symbol",
                    "v33_3_decision",
                    "reference_price",

                    "return_d1",
                    "return_d2",
                    "return_d3",
                    "return_d5",

                    "max_return",
                    "result_class",
                    "tracking_status",
                ]
            ].head(
                20
            ).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()
