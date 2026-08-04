from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# DOSYA YOLLARI
# ============================================================

MARKET_SNAPSHOT_FILE = Path("v16_full_market_snapshot.csv")
CANDIDATE_FILE = Path("v32_adaptive_decisions.csv")

SIMILAR_DAYS_FILE = Path("v33_similar_days.csv")
CANDIDATE_RESULTS_FILE = Path("v33_similar_day_candidates.csv")
STATUS_FILE = Path("v33_status.json")


# ============================================================
# TEMEL AYARLAR
# ============================================================

VERSION = "V33.0"

LOOKBACK_PERIOD = "2y"
MINIMUM_MARKET_SYMBOLS = 80
MINIMUM_HISTORY_DAYS = 120
MINIMUM_SIMILAR_DAYS = 5

SIMILAR_DAY_COUNT = 12
MAX_CANDIDATES = 10

# Çok uzak günlerin kullanılmasını önler.
MAX_SIMILARITY_DISTANCE = 3.50

# Aşırı sonuçların ortalamayı bozmasını azaltır.
RETURN_CLIP_LIMIT = 30.0

# RSI kesinlikle kullanılmaz.
RSI_USAGE = "DISABLED"


# ============================================================
# PİYASA BAĞLAMINDA KULLANILAN ÖZELLİKLER
# ============================================================

CONTEXT_FEATURES = [
    "breadth_1d",
    "breadth_5d",
    "breadth_20d",
    "above_ema20",
    "median_return_1d",
    "median_return_5d",
    "median_return_20d",
    "median_volume_ratio",
    "median_volatility_20d",
    "dispersion_1d",
]


SIMILAR_DAY_COLUMNS = [
    "similarity_rank",
    "target_date",
    "similar_date",
    "similarity_distance",
    "similarity_score",
    "breadth_1d",
    "breadth_5d",
    "breadth_20d",
    "above_ema20",
    "median_return_1d",
    "median_return_5d",
    "median_return_20d",
    "median_volume_ratio",
    "median_volatility_20d",
    "dispersion_1d",
    "market_next_1d_return",
    "market_next_3d_return",
    "market_next_5d_return",
]


CANDIDATE_RESULT_COLUMNS = [
    "v33_rank",
    "symbol",
    "v33_decision",
    "v33_score",
    "similar_day_count",
    "positive_1d_rate",
    "positive_3d_rate",
    "positive_5d_rate",
    "average_return_1d",
    "average_return_3d",
    "average_return_5d",
    "median_return_1d",
    "median_return_3d",
    "median_return_5d",
    "best_return_5d",
    "worst_return_5d",
    "v32_decision",
    "v32_score",
    "v32_confidence",
    "v32_ai_adjustment",
    "risk_class",
    "risk_score",
    "regime",
    "expected_return",
    "downside_20pct",
    "upside_80pct",
    "close",
    "v33_reason",
    "rsi_usage",
]


# ============================================================
# YARDIMCI FONKSİYONLAR
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


def integer(
    value: Any,
    default: int = 0,
) -> int:
    try:
        result = float(value)

        if np.isfinite(result):
            return int(result)

        return default

    except (TypeError, ValueError):
        return default


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"UYARI: {path} bulunamadı.")
        return pd.DataFrame()

    try:
        if path.stat().st_size == 0:
            print(f"UYARI: {path} boş.")
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
                f"UYARI: {path} okunamadı: {exc}"
            )
            return pd.DataFrame()

    return pd.DataFrame()


def normalize_symbol(value: Any) -> str:
    symbol = (
        text(value)
        .upper()
        .replace(" ", "")
    )

    if symbol.endswith(".IS"):
        symbol = symbol[:-3]

    return symbol


def yahoo_symbol(value: Any) -> str:
    symbol = normalize_symbol(value)

    if not symbol:
        return ""

    return f"{symbol}.IS"


def normalize_symbol_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    if frame.empty:
        return frame

    if "symbol" not in frame.columns:
        return pd.DataFrame()

    result = frame.copy()

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

    return result.reset_index(
        drop=True
    )


def write_status(
    status: dict[str, Any],
) -> None:
    STATUS_FILE.write_text(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_empty(
    status_name: str,
    message: str,
) -> None:
    pd.DataFrame(
        columns=SIMILAR_DAY_COLUMNS
    ).to_csv(
        SIMILAR_DAYS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        columns=CANDIDATE_RESULT_COLUMNS
    ).to_csv(
        CANDIDATE_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status = {
        "status": status_name,
        "message": message,
        "symbol_count": 0,
        "downloaded_symbol_count": 0,
        "history_day_count": 0,
        "similar_day_count": 0,
        "candidate_count": 0,
        "approved_count": 0,
        "shadow_mode": True,
        "top_symbol": "",
        "top_decision": "",
        "top_score": 0.0,
        "rsi_usage": RSI_USAGE,
        "version": VERSION,
    }

    write_status(status)

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# YAHOO FINANCE VERİSİNİ DÜZENLEME
# ============================================================

def extract_symbol_frame(
    downloaded: pd.DataFrame,
    ticker: str,
) -> pd.DataFrame:
    if downloaded.empty:
        return pd.DataFrame()

    try:
        if isinstance(
            downloaded.columns,
            pd.MultiIndex,
        ):
            level_zero = (
                downloaded.columns
                .get_level_values(0)
                .astype(str)
            )

            level_one = (
                downloaded.columns
                .get_level_values(1)
                .astype(str)
            )

            if ticker in set(level_zero):
                result = downloaded[
                    ticker
                ].copy()

            elif ticker in set(level_one):
                result = downloaded.xs(
                    ticker,
                    axis=1,
                    level=1,
                ).copy()

            else:
                return pd.DataFrame()

        else:
            result = downloaded.copy()

        result.columns = [
            str(column).strip().lower()
            for column in result.columns
        ]

        if "close" not in result.columns:
            return pd.DataFrame()

        if "volume" not in result.columns:
            result["volume"] = np.nan

        result = result[
            [
                "close",
                "volume",
            ]
        ].copy()

        result["close"] = pd.to_numeric(
            result["close"],
            errors="coerce",
        )

        result["volume"] = pd.to_numeric(
            result["volume"],
            errors="coerce",
        )

        result = result.dropna(
            subset=["close"]
        )

        result = result[
            result["close"] > 0
        ].copy()

        result.index = pd.to_datetime(
            result.index,
            errors="coerce",
        )

        result = result[
            ~result.index.isna()
        ].copy()

        result.index = (
            result.index
            .tz_localize(None)
            .normalize()
        )

        return result.sort_index()

    except Exception as exc:
        print(
            f"UYARI: {ticker} verisi ayrıştırılamadı: {exc}"
        )
        return pd.DataFrame()


def download_market_data(
    symbols: list[str],
) -> dict[str, pd.DataFrame]:
    tickers = [
        yahoo_symbol(symbol)
        for symbol in symbols
        if yahoo_symbol(symbol)
    ]

    if not tickers:
        return {}

    print(
        f"V33 veri indirme başladı. "
        f"Toplam sembol: {len(tickers)}"
    )

    downloaded = yf.download(
        tickers=tickers,
        period=LOOKBACK_PERIOD,
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        actions=False,
    )

    result: dict[str, pd.DataFrame] = {}

    for index, ticker in enumerate(
        tickers,
        start=1,
    ):
        frame = extract_symbol_frame(
            downloaded,
            ticker,
        )

        if len(frame) >= 40:
            symbol = normalize_symbol(
                ticker
            )

            result[symbol] = frame

        if (
            index % 50 == 0
            or index == len(tickers)
        ):
            print(
                f"V33 veri düzenleme "
                f"{index}/{len(tickers)}"
            )

    return result


# ============================================================
# HER HİSSE İÇİN GÜNLÜK ÖZELLİKLER
# ============================================================

def calculate_symbol_features(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()

    result["return_1d"] = (
        result["close"]
        .pct_change(1)
        * 100.0
    )

    result["return_5d"] = (
        result["close"]
        .pct_change(5)
        * 100.0
    )

    result["return_20d"] = (
        result["close"]
        .pct_change(20)
        * 100.0
    )

    result["ema20"] = (
        result["close"]
        .ewm(
            span=20,
            adjust=False,
        )
        .mean()
    )

    result["ema20_distance"] = (
        (
            result["close"]
            / result["ema20"]
        )
        - 1.0
    ) * 100.0

    result["volume_average_20"] = (
        result["volume"]
        .rolling(
            20,
            min_periods=5,
        )
        .mean()
    )

    result["volume_ratio"] = (
        result["volume"]
        / result["volume_average_20"]
    )

    result["volatility_20d"] = (
        result["return_1d"]
        .rolling(
            20,
            min_periods=10,
        )
        .std()
    )

    result["forward_return_1d"] = (
        result["close"]
        .shift(-1)
        / result["close"]
        - 1.0
    ) * 100.0

    result["forward_return_3d"] = (
        result["close"]
        .shift(-3)
        / result["close"]
        - 1.0
    ) * 100.0

    result["forward_return_5d"] = (
        result["close"]
        .shift(-5)
        / result["close"]
        - 1.0
    ) * 100.0

    return result


# ============================================================
# TAM PİYASA GÜNLÜK BAĞLAMI
# ============================================================

def build_market_context(
    market_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    featured: dict[str, pd.DataFrame] = {}

    rows: list[pd.DataFrame] = []

    for symbol, frame in market_data.items():
        symbol_features = calculate_symbol_features(
            frame
        )

        featured[symbol] = symbol_features

        temporary = symbol_features[
            [
                "return_1d",
                "return_5d",
                "return_20d",
                "ema20_distance",
                "volume_ratio",
                "volatility_20d",
                "forward_return_1d",
                "forward_return_3d",
                "forward_return_5d",
            ]
        ].copy()

        temporary["symbol"] = symbol
        temporary["date"] = temporary.index

        rows.append(
            temporary.reset_index(
                drop=True
            )
        )

    if not rows:
        return (
            pd.DataFrame(),
            featured,
        )

    long_frame = pd.concat(
        rows,
        ignore_index=True,
    )

    context_rows: list[dict[str, Any]] = []

    for date, group in long_frame.groupby(
        "date"
    ):
        valid_1d = pd.to_numeric(
            group["return_1d"],
            errors="coerce",
        ).dropna()

        valid_5d = pd.to_numeric(
            group["return_5d"],
            errors="coerce",
        ).dropna()

        valid_20d = pd.to_numeric(
            group["return_20d"],
            errors="coerce",
        ).dropna()

        ema_distance = pd.to_numeric(
            group["ema20_distance"],
            errors="coerce",
        ).dropna()

        volume_ratio = pd.to_numeric(
            group["volume_ratio"],
            errors="coerce",
        ).replace(
            [np.inf, -np.inf],
            np.nan,
        ).dropna()

        volatility = pd.to_numeric(
            group["volatility_20d"],
            errors="coerce",
        ).dropna()

        next_1d = pd.to_numeric(
            group["forward_return_1d"],
            errors="coerce",
        ).dropna()

        next_3d = pd.to_numeric(
            group["forward_return_3d"],
            errors="coerce",
        ).dropna()

        next_5d = pd.to_numeric(
            group["forward_return_5d"],
            errors="coerce",
        ).dropna()

        symbol_count = int(
            valid_1d.count()
        )

        if symbol_count < MINIMUM_MARKET_SYMBOLS:
            continue

        context_rows.append(
            {
                "date": pd.Timestamp(date),
                "symbol_count": symbol_count,
                "breadth_1d": (
                    float(
                        (valid_1d > 0).mean()
                        * 100.0
                    )
                    if not valid_1d.empty
                    else np.nan
                ),
                "breadth_5d": (
                    float(
                        (valid_5d > 0).mean()
                        * 100.0
                    )
                    if not valid_5d.empty
                    else np.nan
                ),
                "breadth_20d": (
                    float(
                        (valid_20d > 0).mean()
                        * 100.0
                    )
                    if not valid_20d.empty
                    else np.nan
                ),
                "above_ema20": (
                    float(
                        (ema_distance > 0).mean()
                        * 100.0
                    )
                    if not ema_distance.empty
                    else np.nan
                ),
                "median_return_1d": (
                    float(valid_1d.median())
                    if not valid_1d.empty
                    else np.nan
                ),
                "median_return_5d": (
                    float(valid_5d.median())
                    if not valid_5d.empty
                    else np.nan
                ),
                "median_return_20d": (
                    float(valid_20d.median())
                    if not valid_20d.empty
                    else np.nan
                ),
                "median_volume_ratio": (
                    float(volume_ratio.median())
                    if not volume_ratio.empty
                    else np.nan
                ),
                "median_volatility_20d": (
                    float(volatility.median())
                    if not volatility.empty
                    else np.nan
                ),
                "dispersion_1d": (
                    float(valid_1d.std())
                    if len(valid_1d) >= 2
                    else np.nan
                ),
                "market_next_1d_return": (
                    float(next_1d.median())
                    if not next_1d.empty
                    else np.nan
                ),
                "market_next_3d_return": (
                    float(next_3d.median())
                    if not next_3d.empty
                    else np.nan
                ),
                "market_next_5d_return": (
                    float(next_5d.median())
                    if not next_5d.empty
                    else np.nan
                ),
            }
        )

    context = pd.DataFrame(
        context_rows
    )

    if context.empty:
        return (
            context,
            featured,
        )

    context = (
        context.sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return (
        context,
        featured,
    )


# ============================================================
# BENZER GÜN HESABI
# ============================================================

def find_similar_days(
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    if context.empty:
        return (
            pd.DataFrame(),
            None,
        )

    usable = context.dropna(
        subset=CONTEXT_FEATURES
    ).copy()

    if len(usable) < MINIMUM_HISTORY_DAYS:
        return (
            pd.DataFrame(),
            None,
        )

    target_row = usable.iloc[-1].copy()
    target_date = pd.Timestamp(
        target_row["date"]
    )

    historical = usable.iloc[:-5].copy()

    if historical.empty:
        return (
            pd.DataFrame(),
            target_date,
        )

    feature_matrix = historical[
        CONTEXT_FEATURES
    ].astype(float)

    target_values = target_row[
        CONTEXT_FEATURES
    ].astype(float)

    medians = feature_matrix.median()
    deviations = (
        feature_matrix
        .sub(medians)
        .abs()
        .median()
    )

    deviations = deviations.replace(
        0,
        np.nan,
    )

    standard_deviation = (
        feature_matrix.std()
    )

    scale = deviations * 1.4826

    scale = scale.fillna(
        standard_deviation
    )

    scale = scale.replace(
        0,
        1.0,
    ).fillna(
        1.0
    )

    standardized_history = (
        feature_matrix
        - target_values
    ) / scale

    distances = np.sqrt(
        (
            standardized_history
            ** 2
        ).mean(
            axis=1
        )
    )

    historical["similarity_distance"] = (
        distances
    )

    historical["similarity_score"] = (
        100.0
        * np.exp(
            -historical[
                "similarity_distance"
            ]
        )
    ).clip(
        0,
        100,
    )

    historical = historical[
        historical[
            "similarity_distance"
        ]
        <= MAX_SIMILARITY_DISTANCE
    ].copy()

    historical = historical.sort_values(
        [
            "similarity_distance",
            "date",
        ],
        ascending=[
            True,
            False,
        ],
    ).head(
        SIMILAR_DAY_COUNT
    )

    if historical.empty:
        return (
            historical,
            target_date,
        )

    historical.insert(
        0,
        "similarity_rank",
        range(
            1,
            len(historical) + 1,
        ),
    )

    historical.insert(
        1,
        "target_date",
        target_date.strftime(
            "%Y-%m-%d"
        ),
    )

    historical = historical.rename(
        columns={
            "date": "similar_date",
        }
    )

    historical["similar_date"] = (
        pd.to_datetime(
            historical["similar_date"]
        )
        .dt.strftime("%Y-%m-%d")
    )

    output = pd.DataFrame()

    for column in SIMILAR_DAY_COLUMNS:
        if column in historical.columns:
            output[column] = (
                historical[column]
            )
        else:
            output[column] = np.nan

    return (
        output,
        target_date,
    )


# ============================================================
# ADAYLARIN BENZER GÜN SONRASI DAVRANIŞI
# ============================================================

def candidate_forward_returns(
    symbol_frame: pd.DataFrame,
    similar_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[dict[str, float]] = []

    for similar_date in similar_dates:
        if similar_date not in symbol_frame.index:
            continue

        row = symbol_frame.loc[
            similar_date
        ]

        return_1d = number(
            row.get("forward_return_1d"),
            np.nan,
        )

        return_3d = number(
            row.get("forward_return_3d"),
            np.nan,
        )

        return_5d = number(
            row.get("forward_return_5d"),
            np.nan,
        )

        if not any(
            np.isfinite(value)
            for value in [
                return_1d,
                return_3d,
                return_5d,
            ]
        ):
            continue

        rows.append(
            {
                "return_1d": return_1d,
                "return_3d": return_3d,
                "return_5d": return_5d,
            }
        )

    return pd.DataFrame(
        rows
    )


def clipped_series(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().clip(
        -RETURN_CLIP_LIMIT,
        RETURN_CLIP_LIMIT,
    )


def calculate_candidate_similarity_score(
    stats: dict[str, float],
    v32_score: float,
    risk_score: float,
) -> float:
    positive_1d = stats[
        "positive_1d_rate"
    ]

    positive_3d = stats[
        "positive_3d_rate"
    ]

    positive_5d = stats[
        "positive_5d_rate"
    ]

    average_1d = stats[
        "average_return_1d"
    ]

    average_3d = stats[
        "average_return_3d"
    ]

    average_5d = stats[
        "average_return_5d"
    ]

    historical_component = (
        positive_1d * 0.12
        + positive_3d * 0.16
        + positive_5d * 0.22
        + np.clip(
            average_1d * 10.0 + 50.0,
            0.0,
            100.0,
        ) * 0.10
        + np.clip(
            average_3d * 7.0 + 50.0,
            0.0,
            100.0,
        ) * 0.14
        + np.clip(
            average_5d * 5.0 + 50.0,
            0.0,
            100.0,
        ) * 0.16
    )

    combined = (
        historical_component * 0.70
        + v32_score * 0.30
        - max(
            risk_score - 35.0,
            0.0,
        ) * 0.20
    )

    return round(
        float(
            np.clip(
                combined,
                0.0,
                100.0,
            )
        ),
        2,
    )


def determine_v33_decision(
    score: float,
    similar_day_count: int,
    positive_5d_rate: float,
    average_return_5d: float,
    risk_score: float,
) -> tuple[str, str]:
    if similar_day_count < MINIMUM_SIMILAR_DAYS:
        return (
            "YETERSİZ BENZER GÜN",
            (
                "Güvenilir değerlendirme için "
                "yeterli benzer piyasa günü bulunamadı"
            ),
        )

    if risk_score >= 65:
        return (
            "ELE",
            "Risk puanı kabul edilebilir seviyenin üzerinde",
        )

    if (
        score >= 78
        and positive_5d_rate >= 70
        and average_return_5d >= 2.0
        and risk_score <= 40
    ):
        return (
            "BENZER GÜN GÜÇLÜ TEYİT",
            (
                "Benzer piyasa günlerinin çoğunda "
                "aday sonraki beş günde pozitif ve güçlü"
            ),
        )

    if (
        score >= 66
        and positive_5d_rate >= 60
        and average_return_5d > 0
        and risk_score <= 50
    ):
        return (
            "BENZER GÜN AKTİF İZLEME",
            (
                "Geçmiş benzer piyasa günleri "
                "adayı istatistiksel olarak destekliyor"
            ),
        )

    if (
        score >= 54
        and positive_5d_rate >= 50
        and risk_score <= 55
    ):
        return (
            "BENZER GÜN TEYİT BEKLE",
            (
                "Benzer gün görünümü kısmen olumlu "
                "fakat güçlü ortak sonuç oluşmadı"
            ),
        )

    return (
        "BENZER GÜN PASİF",
        (
            "Geçmiş benzer piyasa günlerinde "
            "aday yeterince güvenilir performans göstermedi"
        ),
    )


def evaluate_candidates(
    candidates: pd.DataFrame,
    featured_data: dict[str, pd.DataFrame],
    similar_days: pd.DataFrame,
) -> pd.DataFrame:
    if (
        candidates.empty
        or similar_days.empty
    ):
        return pd.DataFrame(
            columns=CANDIDATE_RESULT_COLUMNS
        )

    similar_dates = (
        pd.to_datetime(
            similar_days["similar_date"],
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
        .tolist()
    )

    rows: list[dict[str, Any]] = []

    limited_candidates = candidates.head(
        MAX_CANDIDATES
    ).copy()

    for _, candidate in limited_candidates.iterrows():
        symbol = normalize_symbol(
            candidate.get("symbol")
        )

        if symbol not in featured_data:
            continue

        outcomes = candidate_forward_returns(
            featured_data[symbol],
            similar_dates,
        )

        if outcomes.empty:
            continue

        return_1d = clipped_series(
            outcomes["return_1d"]
        )

        return_3d = clipped_series(
            outcomes["return_3d"]
        )

        return_5d = clipped_series(
            outcomes["return_5d"]
        )

        similar_count = int(
            max(
                len(return_1d),
                len(return_3d),
                len(return_5d),
            )
        )

        stats = {
            "positive_1d_rate": (
                float(
                    (return_1d > 0).mean()
                    * 100.0
                )
                if not return_1d.empty
                else 0.0
            ),
            "positive_3d_rate": (
                float(
                    (return_3d > 0).mean()
                    * 100.0
                )
                if not return_3d.empty
                else 0.0
            ),
            "positive_5d_rate": (
                float(
                    (return_5d > 0).mean()
                    * 100.0
                )
                if not return_5d.empty
                else 0.0
            ),
            "average_return_1d": (
                float(return_1d.mean())
                if not return_1d.empty
                else 0.0
            ),
            "average_return_3d": (
                float(return_3d.mean())
                if not return_3d.empty
                else 0.0
            ),
            "average_return_5d": (
                float(return_5d.mean())
                if not return_5d.empty
                else 0.0
            ),
            "median_return_1d": (
                float(return_1d.median())
                if not return_1d.empty
                else 0.0
            ),
            "median_return_3d": (
                float(return_3d.median())
                if not return_3d.empty
                else 0.0
            ),
            "median_return_5d": (
                float(return_5d.median())
                if not return_5d.empty
                else 0.0
            ),
            "best_return_5d": (
                float(return_5d.max())
                if not return_5d.empty
                else 0.0
            ),
            "worst_return_5d": (
                float(return_5d.min())
                if not return_5d.empty
                else 0.0
            ),
        }

        v32_score = number(
            candidate.get("v32_score")
        )

        risk_score = number(
            candidate.get("risk_score"),
            100.0,
        )

        v33_score = (
            calculate_candidate_similarity_score(
                stats=stats,
                v32_score=v32_score,
                risk_score=risk_score,
            )
        )

        decision, reason = determine_v33_decision(
            score=v33_score,
            similar_day_count=similar_count,
            positive_5d_rate=stats[
                "positive_5d_rate"
            ],
            average_return_5d=stats[
                "average_return_5d"
            ],
            risk_score=risk_score,
        )

        rows.append(
            {
                "symbol": symbol,
                "v33_decision": decision,
                "v33_score": v33_score,
                "similar_day_count": similar_count,
                **{
                    key: round(
                        value,
                        2,
                    )
                    for key, value
                    in stats.items()
                },
                "v32_decision": text(
                    candidate.get(
                        "v32_decision"
                    )
                ),
                "v32_score": v32_score,
                "v32_confidence": number(
                    candidate.get(
                        "v32_confidence"
                    )
                ),
                "v32_ai_adjustment": number(
                    candidate.get(
                        "v32_ai_adjustment"
                    )
                ),
                "risk_class": text(
                    candidate.get(
                        "risk_class"
                    )
                ),
                "risk_score": risk_score,
                "regime": text(
                    candidate.get("regime")
                ),
                "expected_return": number(
                    candidate.get(
                        "expected_return"
                    )
                ),
                "downside_20pct": number(
                    candidate.get(
                        "downside_20pct"
                    )
                ),
                "upside_80pct": number(
                    candidate.get(
                        "upside_80pct"
                    )
                ),
                "close": number(
                    candidate.get("close")
                ),
                "v33_reason": reason,
                "rsi_usage": RSI_USAGE,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return pd.DataFrame(
            columns=CANDIDATE_RESULT_COLUMNS
        )

    priority = {
        "BENZER GÜN GÜÇLÜ TEYİT": 6,
        "BENZER GÜN AKTİF İZLEME": 5,
        "BENZER GÜN TEYİT BEKLE": 4,
        "BENZER GÜN PASİF": 3,
        "YETERSİZ BENZER GÜN": 2,
        "ELE": 1,
    }

    result["_priority"] = (
        result["v33_decision"]
        .map(priority)
        .fillna(0)
    )

    result = (
        result.sort_values(
            [
                "_priority",
                "v33_score",
                "positive_5d_rate",
                "average_return_5d",
            ],
            ascending=[
                False,
                False,
                False,
                False,
            ],
        )
        .drop(columns="_priority")
        .reset_index(drop=True)
    )

    result.insert(
        0,
        "v33_rank",
        range(
            1,
            len(result) + 1,
        ),
    )

    output = pd.DataFrame()

    for column in CANDIDATE_RESULT_COLUMNS:
        if column in result.columns:
            output[column] = result[column]
        else:
            output[column] = np.nan

    return output


# ============================================================
# ANA MOTOR
# ============================================================

def main() -> None:
    started_at = time.time()

    print(
        "===== V33 BENZER PİYASA GÜNLERİ MOTORU BAŞLADI ====="
    )

    market_snapshot = normalize_symbol_frame(
        load_csv(
            MARKET_SNAPSHOT_FILE
        )
    )

    candidates = normalize_symbol_frame(
        load_csv(
            CANDIDATE_FILE
        )
    )

    if market_snapshot.empty:
        save_empty(
            status_name="market_snapshot_missing",
            message=(
                "v16_full_market_snapshot.csv "
                "bulunamadı veya geçerli sembol içermiyor."
            ),
        )
        return

    if candidates.empty:
        save_empty(
            status_name="candidate_input_missing",
            message=(
                "v32_adaptive_decisions.csv "
                "bulunamadı veya geçerli aday içermiyor."
            ),
        )
        return

    symbols = (
        market_snapshot["symbol"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    market_data = download_market_data(
        symbols
    )

    if len(market_data) < MINIMUM_MARKET_SYMBOLS:
        save_empty(
            status_name="insufficient_market_data",
            message=(
                "Benzer gün analizi için yeterli "
                "sayıda hisse verisi indirilemedi."
            ),
        )
        return

    context, featured_data = build_market_context(
        market_data
    )

    if len(context) < MINIMUM_HISTORY_DAYS:
        save_empty(
            status_name="insufficient_history",
            message=(
                "Benzer gün analizi için yeterli "
                "geçmiş işlem günü oluşturulamadı."
            ),
        )
        return

    similar_days, target_date = find_similar_days(
        context
    )

    similar_days.to_csv(
        SIMILAR_DAYS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    candidate_results = evaluate_candidates(
        candidates=candidates,
        featured_data=featured_data,
        similar_days=similar_days,
    )

    candidate_results.to_csv(
        CANDIDATE_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    approved_states = {
        "BENZER GÜN GÜÇLÜ TEYİT",
        "BENZER GÜN AKTİF İZLEME",
    }

    approved_count = int(
        candidate_results[
            "v33_decision"
        ].isin(
            approved_states
        ).sum()
    ) if not candidate_results.empty else 0

    status = {
        "status": "ready",
        "target_date": (
            target_date.strftime("%Y-%m-%d")
            if target_date is not None
            else ""
        ),
        "symbol_count": int(
            len(symbols)
        ),
        "downloaded_symbol_count": int(
            len(market_data)
        ),
        "history_day_count": int(
            len(context)
        ),
        "similar_day_count": int(
            len(similar_days)
        ),
        "candidate_count": int(
            len(candidate_results)
        ),
        "approved_count": approved_count,
        "strong_confirmation_count": int(
            (
                candidate_results[
                    "v33_decision"
                ]
                == "BENZER GÜN GÜÇLÜ TEYİT"
            ).sum()
        ) if not candidate_results.empty else 0,
        "active_tracking_count": int(
            (
                candidate_results[
                    "v33_decision"
                ]
                == "BENZER GÜN AKTİF İZLEME"
            ).sum()
        ) if not candidate_results.empty else 0,
        "waiting_count": int(
            (
                candidate_results[
                    "v33_decision"
                ]
                == "BENZER GÜN TEYİT BEKLE"
            ).sum()
        ) if not candidate_results.empty else 0,
        "passive_count": int(
            (
                candidate_results[
                    "v33_decision"
                ]
                == "BENZER GÜN PASİF"
            ).sum()
        ) if not candidate_results.empty else 0,
        "shadow_mode": True,
        "top_symbol": (
            text(
                candidate_results.iloc[0][
                    "symbol"
                ]
            )
            if len(candidate_results)
            else ""
        ),
        "top_decision": (
            text(
                candidate_results.iloc[0][
                    "v33_decision"
                ]
            )
            if len(candidate_results)
            else ""
        ),
        "top_score": (
            number(
                candidate_results.iloc[0][
                    "v33_score"
                ]
            )
            if len(candidate_results)
            else 0.0
        ),
        "runtime_seconds": round(
            time.time() - started_at,
            2,
        ),
        "rsi_usage": RSI_USAGE,
        "version": VERSION,
    }

    write_status(
        status
    )

    print(
        "===== V33 BENZER GÜNLER ====="
    )

    print(
        similar_days.to_string(
            index=False
        )
        if not similar_days.empty
        else "Uygun benzer gün bulunamadı."
    )

    print(
        "===== V33 ADAY SONUÇLARI ====="
    )

    print(
        candidate_results.to_string(
            index=False
        )
        if not candidate_results.empty
        else "Değerlendirilebilecek aday sonucu oluşmadı."
    )

    print(
        "===== V33 STATUS ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "===== V33 BENZER PİYASA GÜNLERİ MOTORU TAMAMLANDI ====="
    )


if __name__ == "__main__":
    main()
