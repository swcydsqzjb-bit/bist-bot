from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# V33 - BENZER PİYASA GÜNLERİ / GENİŞ ADAY HAVUZU
# ============================================================

VERSION = "V33.1"

MARKET_SNAPSHOT_FILE = Path("v16_full_market_snapshot.csv")

V22_FILE = Path("v22_signal_states.csv")
V27_FILE = Path("v27_master_decisions.csv")
V32_FILE = Path("v32_adaptive_decisions.csv")

SIMILAR_DAYS_FILE = Path("v33_similar_days.csv")
CANDIDATE_RESULTS_FILE = Path("v33_similar_day_candidates.csv")
CANDIDATE_POOL_FILE = Path("v33_candidate_pool.csv")
STATUS_FILE = Path("v33_status.json")


# ============================================================
# AYARLAR
# ============================================================

LOOKBACK_PERIOD = "2y"

MINIMUM_MARKET_SYMBOLS = 80
MINIMUM_HISTORY_DAYS = 120
MINIMUM_SIMILAR_DAYS = 5

SIMILAR_DAY_COUNT = 12

# Önceden 1-2 aday kalıyordu.
# Artık V22 + V27 + V32 birleşiminden en iyi 15 aday.
MAX_CANDIDATES = 15

MAX_SIMILARITY_DISTANCE = 3.50
RETURN_CLIP_LIMIT = 30.0

RSI_USAGE = "DISABLED"


# ============================================================
# YENİ V33 KARAR EŞİKLERİ
# ============================================================

STRONG_SCORE = 72.0
STRONG_POSITIVE_5D = 65.0
STRONG_AVERAGE_5D = 1.50

ACTIVE_SCORE = 58.0
ACTIVE_POSITIVE_5D = 55.0
ACTIVE_AVERAGE_5D = 0.0

WAIT_SCORE = 48.0

MAX_ACTIVE_RISK = 55.0
MAX_STRONG_RISK = 45.0
HARD_RISK_LIMIT = 70.0


# ============================================================
# PİYASA BENZERLİK ÖZELLİKLERİ
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
    "candidate_pool_score",
    "candidate_sources",
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
    "v22_signal_state",
    "v22_signal_score",
    "v27_decision",
    "v27_master_score",
    "v32_decision",
    "v32_score",
    "v32_confidence",
    "risk_class",
    "risk_score",
    "regime",
    "market_percentile",
    "timing_confidence",
    "expected_return",
    "downside_20pct",
    "upside_80pct",
    "close",
    "v33_reason",
    "rsi_usage",
]


# ============================================================
# TEMEL YARDIMCILAR
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


def load_csv(path: Path) -> pd.DataFrame:
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

    return result.reset_index(drop=True)


def yahoo_symbol(symbol: str) -> str:
    normalized = normalize_symbol(symbol)

    if not normalized:
        return ""

    return f"{normalized}.IS"


def ensure_column(
    frame: pd.DataFrame,
    column: str,
    default: Any,
) -> None:
    if column not in frame.columns:
        frame[column] = default


def save_status(
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


# ============================================================
# ADAY HAVUZU
# ============================================================

def prepare_v22(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = normalize_symbol_frame(frame)

    if frame.empty:
        return pd.DataFrame()

    wanted = [
        "symbol",
        "v22_signal_state",
        "v22_signal_score",
        "risk_class",
        "risk_score",
        "regime",
        "market_percentile",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "close",
    ]

    for column in wanted:
        ensure_column(
            frame,
            column,
            np.nan,
        )

    return frame[wanted].copy()


def prepare_v27(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = normalize_symbol_frame(frame)

    if frame.empty:
        return pd.DataFrame()

    wanted = [
        "symbol",
        "v27_decision",
        "v27_master_score",
        "risk_class",
        "risk_score",
        "regime",
        "market_percentile",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "close",
    ]

    for column in wanted:
        ensure_column(
            frame,
            column,
            np.nan,
        )

    result = frame[wanted].copy()

    rename_map = {
        "risk_class": "risk_class_v27",
        "risk_score": "risk_score_v27",
        "regime": "regime_v27",
        "market_percentile": "market_percentile_v27",
        "timing_confidence": "timing_confidence_v27",
        "expected_return": "expected_return_v27",
        "downside_20pct": "downside_20pct_v27",
        "upside_80pct": "upside_80pct_v27",
        "close": "close_v27",
    }

    return result.rename(
        columns=rename_map
    )


def prepare_v32(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = normalize_symbol_frame(frame)

    if frame.empty:
        return pd.DataFrame()

    wanted = [
        "symbol",
        "v32_decision",
        "v32_score",
        "v32_confidence",
        "risk_class",
        "risk_score",
        "regime",
        "market_percentile",
        "timing_confidence",
        "expected_return",
        "downside_20pct",
        "upside_80pct",
        "close",
    ]

    for column in wanted:
        ensure_column(
            frame,
            column,
            np.nan,
        )

    result = frame[wanted].copy()

    rename_map = {
        "risk_class": "risk_class_v32",
        "risk_score": "risk_score_v32",
        "regime": "regime_v32",
        "market_percentile": "market_percentile_v32",
        "timing_confidence": "timing_confidence_v32",
        "expected_return": "expected_return_v32",
        "downside_20pct": "downside_20pct_v32",
        "upside_80pct": "upside_80pct_v32",
        "close": "close_v32",
    }

    return result.rename(
        columns=rename_map
    )


def build_candidate_pool() -> pd.DataFrame:
    v22 = prepare_v22(
        load_csv(V22_FILE)
    )

    v27 = prepare_v27(
        load_csv(V27_FILE)
    )

    v32 = prepare_v32(
        load_csv(V32_FILE)
    )

    symbols: set[str] = set()

    for frame in (
        v22,
        v27,
        v32,
    ):
        if not frame.empty:
            symbols.update(
                frame["symbol"].tolist()
            )

    if not symbols:
        return pd.DataFrame()

    pool = pd.DataFrame(
        {
            "symbol": sorted(symbols)
        }
    )

    if not v22.empty:
        pool = pool.merge(
            v22,
            on="symbol",
            how="left",
        )

    if not v27.empty:
        pool = pool.merge(
            v27,
            on="symbol",
            how="left",
        )

    if not v32.empty:
        pool = pool.merge(
            v32,
            on="symbol",
            how="left",
        )

    numeric_columns = [
        "v22_signal_score",
        "v27_master_score",
        "v32_score",
        "v32_confidence",
    ]

    for column in numeric_columns:
        ensure_column(
            pool,
            column,
            0.0,
        )

        pool[column] = pd.to_numeric(
            pool[column],
            errors="coerce",
        ).fillna(0.0)

    # --------------------------------------------------------
    # ORTAK RİSK / REJİM / DİĞER ALANLAR
    # Öncelik: V32 -> V27 -> V22
    # --------------------------------------------------------

    def first_available(
        row: pd.Series,
        columns: list[str],
        default: Any,
    ) -> Any:
        for column in columns:
            value = row.get(column)

            if text(value):
                try:
                    if pd.isna(value):
                        continue
                except Exception:
                    pass

                return value

        return default

    pool["risk_class"] = pool.apply(
        lambda row: first_available(
            row,
            [
                "risk_class_v32",
                "risk_class_v27",
                "risk_class",
            ],
            "ORTA",
        ),
        axis=1,
    )

    pool["risk_score_final"] = pool.apply(
        lambda row: number(
            first_available(
                row,
                [
                    "risk_score_v32",
                    "risk_score_v27",
                    "risk_score",
                ],
                50.0,
            ),
            50.0,
        ),
        axis=1,
    )

    pool["regime_final"] = pool.apply(
        lambda row: first_available(
            row,
            [
                "regime_v32",
                "regime_v27",
                "regime",
            ],
            "",
        ),
        axis=1,
    )

    pool["market_percentile_final"] = pool.apply(
        lambda row: number(
            first_available(
                row,
                [
                    "market_percentile_v32",
                    "market_percentile_v27",
                    "market_percentile",
                ],
                0.0,
            )
        ),
        axis=1,
    )

    pool["timing_confidence_final"] = pool.apply(
        lambda row: number(
            first_available(
                row,
                [
                    "timing_confidence_v32",
                    "timing_confidence_v27",
                    "timing_confidence",
                ],
                0.0,
            )
        ),
        axis=1,
    )

    pool["expected_return_final"] = pool.apply(
        lambda row: number(
            first_available(
                row,
                [
                    "expected_return_v32",
                    "expected_return_v27",
                    "expected_return",
                ],
                0.0,
            )
        ),
        axis=1,
    )

    pool["downside_20pct_final"] = pool.apply(
        lambda row: number(
            first_available(
                row,
                [
                    "downside_20pct_v32",
                    "downside_20pct_v27",
                    "downside_20pct",
                ],
                0.0,
            )
        ),
        axis=1,
    )

    pool["upside_80pct_final"] = pool.apply(
        lambda row: number(
            first_available(
                row,
                [
                    "upside_80pct_v32",
                    "upside_80pct_v27",
                    "upside_80pct",
                ],
                0.0,
            )
        ),
        axis=1,
    )

    pool["close_final"] = pool.apply(
        lambda row: number(
            first_available(
                row,
                [
                    "close_v32",
                    "close_v27",
                    "close",
                ],
                0.0,
            )
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # HANGİ MOTORLARDA GÖRÜLDÜ?
    # --------------------------------------------------------

    def source_text(row: pd.Series) -> str:
        sources: list[str] = []

        if number(
            row.get("v22_signal_score")
        ) > 0:
            sources.append("V22")

        if number(
            row.get("v27_master_score")
        ) > 0:
            sources.append("V27")

        if number(
            row.get("v32_score")
        ) > 0:
            sources.append("V32")

        return "+".join(
            sources
        )

    pool["candidate_sources"] = pool.apply(
        source_text,
        axis=1,
    )

    # --------------------------------------------------------
    # GENİŞ HAVUZ PUANI
    #
    # Burada V32 zorunlu değil.
    # V22 güçlü olup V27/V32'de elenmiş hisseler de incelenebilir.
    # --------------------------------------------------------

    pool["candidate_pool_score"] = (
        pool["v22_signal_score"] * 0.42
        + pool["v27_master_score"] * 0.33
        + pool["v32_score"] * 0.25
    )

    # Eğer motorlardan biri çok güçlüyse tamamen ezilmesin.
    best_layer_score = pool[
        [
            "v22_signal_score",
            "v27_master_score",
            "v32_score",
        ]
    ].max(axis=1)

    pool["candidate_pool_score"] = (
        pool["candidate_pool_score"] * 0.70
        + best_layer_score * 0.30
    )

    # Göreceli güç katkısı
    pool["candidate_pool_score"] += (
        np.clip(
            pool["market_percentile_final"] - 70.0,
            0.0,
            30.0,
        )
        * 0.08
    )

    # Zamanlama katkısı
    pool["candidate_pool_score"] += (
        np.clip(
            pool["timing_confidence_final"] - 65.0,
            0.0,
            35.0,
        )
        * 0.05
    )

    # Risk kesintisi
    pool["candidate_pool_score"] -= (
        np.clip(
            pool["risk_score_final"] - 40.0,
            0.0,
            60.0,
        )
        * 0.12
    )

    pool["candidate_pool_score"] = (
        pool["candidate_pool_score"]
        .clip(
            0.0,
            100.0,
        )
        .round(2)
    )

    # --------------------------------------------------------
    # ÇOK KÖTÜ ADAYLARI BAŞTAN ÇIKAR
    # Ama ELE kararını tek başına dışlama!
    # Çünkü amaç kaçırılan fırsatları da bulmak.
    # --------------------------------------------------------

    pool = pool[
        pool["risk_score_final"]
        < HARD_RISK_LIMIT
    ].copy()

    # En az bir katmanda anlamlı skor üretmiş olsun.
    pool = pool[
        best_layer_score >= 35.0
    ].copy()

    pool = pool.sort_values(
        [
            "candidate_pool_score",
            "market_percentile_final",
            "timing_confidence_final",
        ],
        ascending=False,
    )

    pool = pool.head(
        MAX_CANDIDATES
    ).reset_index(
        drop=True
    )

    pool.insert(
        0,
        "pool_rank",
        range(
            1,
            len(pool) + 1,
        ),
    )

    pool.to_csv(
        CANDIDATE_POOL_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    return pool


# ============================================================
# PİYASA VERİSİ
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
            str(column)
            .strip()
            .lower()
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
            f"UYARI: {ticker} ayrıştırılamadı: {exc}"
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
        f"V33 piyasa verisi indiriliyor: "
        f"{len(tickers)} sembol"
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
            result[
                normalize_symbol(ticker)
            ] = frame

        if (
            index % 50 == 0
            or index == len(tickers)
        ):
            print(
                f"V33 veri hazırlama "
                f"{index}/{len(tickers)}"
            )

    return result


# ============================================================
# HİSSE ÖZELLİKLERİ
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
# TAM PİYASA BAĞLAMI
# ============================================================

def build_market_context(
    market_data: dict[str, pd.DataFrame],
) -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
]:
    featured: dict[str, pd.DataFrame] = {}
    rows: list[pd.DataFrame] = []

    for symbol, frame in market_data.items():
        features = calculate_symbol_features(
            frame
        )

        featured[symbol] = features

        temporary = features[
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

    context_rows: list[
        dict[str, Any]
    ] = []

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

        if len(valid_1d) < MINIMUM_MARKET_SYMBOLS:
            continue

        context_rows.append(
            {
                "date": pd.Timestamp(date),
                "symbol_count": len(
                    valid_1d
                ),
                "breadth_1d": (
                    (valid_1d > 0)
                    .mean()
                    * 100.0
                ),
                "breadth_5d": (
                    (valid_5d > 0)
                    .mean()
                    * 100.0
                    if len(valid_5d)
                    else np.nan
                ),
                "breadth_20d": (
                    (valid_20d > 0)
                    .mean()
                    * 100.0
                    if len(valid_20d)
                    else np.nan
                ),
                "above_ema20": (
                    (ema_distance > 0)
                    .mean()
                    * 100.0
                    if len(ema_distance)
                    else np.nan
                ),
                "median_return_1d": (
                    valid_1d.median()
                ),
                "median_return_5d": (
                    valid_5d.median()
                    if len(valid_5d)
                    else np.nan
                ),
                "median_return_20d": (
                    valid_20d.median()
                    if len(valid_20d)
                    else np.nan
                ),
                "median_volume_ratio": (
                    volume_ratio.median()
                    if len(volume_ratio)
                    else np.nan
                ),
                "median_volatility_20d": (
                    volatility.median()
                    if len(volatility)
                    else np.nan
                ),
                "dispersion_1d": (
                    valid_1d.std()
                    if len(valid_1d) >= 2
                    else np.nan
                ),
                "market_next_1d_return": (
                    next_1d.median()
                    if len(next_1d)
                    else np.nan
                ),
                "market_next_3d_return": (
                    next_3d.median()
                    if len(next_3d)
                    else np.nan
                ),
                "market_next_5d_return": (
                    next_5d.median()
                    if len(next_5d)
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
        context
        .sort_values("date")
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
# BENZER GÜNLERİ BUL
# ============================================================

def find_similar_days(
    context: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Timestamp | None,
]:
    usable = context.dropna(
        subset=CONTEXT_FEATURES
    ).copy()

    if len(usable) < MINIMUM_HISTORY_DAYS:
        return (
            pd.DataFrame(),
            None,
        )

    target_row = usable.iloc[-1]
    target_date = pd.Timestamp(
        target_row["date"]
    )

    # Son 5 gün kullanılmaz.
    historical = usable.iloc[:-5].copy()

    if historical.empty:
        return (
            pd.DataFrame(),
            target_date,
        )

    matrix = historical[
        CONTEXT_FEATURES
    ].astype(float)

    target = target_row[
        CONTEXT_FEATURES
    ].astype(float)

    medians = matrix.median()

    mad = (
        matrix
        .sub(medians)
        .abs()
        .median()
    )

    scale = (
        mad * 1.4826
    ).replace(
        0,
        np.nan,
    )

    scale = scale.fillna(
        matrix.std()
    )

    scale = scale.replace(
        0,
        1.0,
    ).fillna(
        1.0
    )

    standardized = (
        matrix - target
    ) / scale

    historical[
        "similarity_distance"
    ] = np.sqrt(
        (
            standardized ** 2
        ).mean(axis=1)
    )

    historical[
        "similarity_score"
    ] = (
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

    historical[
        "similar_date"
    ] = pd.to_datetime(
        historical[
            "similar_date"
        ]
    ).dt.strftime(
        "%Y-%m-%d"
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
# ADAYIN BENZER GÜNLERDEKİ SONUÇLARI
# ============================================================

def candidate_forward_returns(
    frame: pd.DataFrame,
    similar_dates: list[pd.Timestamp],
) -> pd.DataFrame:
    rows: list[
        dict[str, float]
    ] = []

    for similar_date in similar_dates:
        if similar_date not in frame.index:
            continue

        row = frame.loc[
            similar_date
        ]

        rows.append(
            {
                "return_1d": number(
                    row.get(
                        "forward_return_1d"
                    ),
                    np.nan,
                ),
                "return_3d": number(
                    row.get(
                        "forward_return_3d"
                    ),
                    np.nan,
                ),
                "return_5d": number(
                    row.get(
                        "forward_return_5d"
                    ),
                    np.nan,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def clean_returns(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    ).dropna().clip(
        -RETURN_CLIP_LIMIT,
        RETURN_CLIP_LIMIT,
    )


# ============================================================
# V33 PUAN
# ============================================================

def calculate_v33_score(
    positive_1d: float,
    positive_3d: float,
    positive_5d: float,
    average_1d: float,
    average_3d: float,
    average_5d: float,
    candidate_pool_score: float,
    risk_score: float,
) -> float:
    historical_score = (
        positive_1d * 0.10
        + positive_3d * 0.15
        + positive_5d * 0.25
        + np.clip(
            50 + average_1d * 9,
            0,
            100,
        ) * 0.08
        + np.clip(
            50 + average_3d * 6,
            0,
            100,
        ) * 0.12
        + np.clip(
            50 + average_5d * 5,
            0,
            100,
        ) * 0.15
    )

    score = (
        historical_score * 0.72
        + candidate_pool_score * 0.28
    )

    if risk_score > 50:
        score -= (
            risk_score - 50
        ) * 0.20

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


# ============================================================
# YENİ KARAR MOTORU
# ============================================================

def determine_v33_decision(
    score: float,
    similar_day_count: int,
    positive_5d_rate: float,
    average_return_5d: float,
    median_return_5d: float,
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

    if risk_score >= HARD_RISK_LIMIT:
        return (
            "ELE",
            "Risk puanı çok yüksek",
        )

    # --------------------------------------------------------
    # GÜÇLÜ
    # --------------------------------------------------------

    if (
        score >= STRONG_SCORE
        and positive_5d_rate >= STRONG_POSITIVE_5D
        and average_return_5d >= STRONG_AVERAGE_5D
        and median_return_5d > 0
        and risk_score <= MAX_STRONG_RISK
    ):
        return (
            "BENZER GÜN GÜÇLÜ TEYİT",
            (
                "Benzer piyasa günlerinin büyük bölümünde "
                "pozitif sonuç oluşmuş ve ortalama getiri güçlü"
            ),
        )

    # --------------------------------------------------------
    # AKTİF İZLEME
    # --------------------------------------------------------

    if (
        score >= ACTIVE_SCORE
        and positive_5d_rate >= ACTIVE_POSITIVE_5D
        and average_return_5d > ACTIVE_AVERAGE_5D
        and risk_score <= MAX_ACTIVE_RISK
    ):
        return (
            "BENZER GÜN AKTİF İZLEME",
            (
                "Benzer geçmiş piyasa günleri adayı "
                "istatistiksel olarak destekliyor"
            ),
        )

    # Alternatif aktif izleme:
    # Ortalama çok kuvvetliyse pozitif oran %55 altında olsa bile
    # değerlendirmeye alınabilir.
    if (
        score >= 60
        and average_return_5d >= 2.50
        and median_return_5d > 0
        and positive_5d_rate >= 50
        and risk_score <= 50
    ):
        return (
            "BENZER GÜN AKTİF İZLEME",
            (
                "Pozitif gün oranı sınırlı olsa da "
                "benzer günlerde ortalama ve medyan getiri güçlü"
            ),
        )

    # --------------------------------------------------------
    # TEYİT BEKLE
    # --------------------------------------------------------

    if (
        score >= WAIT_SCORE
        and risk_score <= 60
    ):
        return (
            "BENZER GÜN TEYİT BEKLE",
            (
                "Benzer gün görünümü kısmen olumlu "
                "fakat aktif izleme için yeterli güven oluşmadı"
            ),
        )

    return (
        "BENZER GÜN PASİF",
        (
            "Benzer gün istatistikleri aktif takip "
            "için yeterli güç üretmedi"
        ),
    )


# ============================================================
# ADAY DEĞERLENDİRMESİ
# ============================================================

def evaluate_candidates(
    candidate_pool: pd.DataFrame,
    featured_data: dict[str, pd.DataFrame],
    similar_days: pd.DataFrame,
) -> pd.DataFrame:
    if (
        candidate_pool.empty
        or similar_days.empty
    ):
        return pd.DataFrame(
            columns=CANDIDATE_RESULT_COLUMNS
        )

    similar_dates = (
        pd.to_datetime(
            similar_days[
                "similar_date"
            ],
            errors="coerce",
        )
        .dropna()
        .dt.normalize()
        .tolist()
    )

    rows: list[
        dict[str, Any]
    ] = []

    for _, candidate in candidate_pool.iterrows():
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

        return_1d = clean_returns(
            outcomes["return_1d"]
        )

        return_3d = clean_returns(
            outcomes["return_3d"]
        )

        return_5d = clean_returns(
            outcomes["return_5d"]
        )

        similar_count = max(
            len(return_1d),
            len(return_3d),
            len(return_5d),
        )

        positive_1d = (
            float(
                (return_1d > 0)
                .mean()
                * 100
            )
            if len(return_1d)
            else 0.0
        )

        positive_3d = (
            float(
                (return_3d > 0)
                .mean()
                * 100
            )
            if len(return_3d)
            else 0.0
        )

        positive_5d = (
            float(
                (return_5d > 0)
                .mean()
                * 100
            )
            if len(return_5d)
            else 0.0
        )

        average_1d = (
            float(return_1d.mean())
            if len(return_1d)
            else 0.0
        )

        average_3d = (
            float(return_3d.mean())
            if len(return_3d)
            else 0.0
        )

        average_5d = (
            float(return_5d.mean())
            if len(return_5d)
            else 0.0
        )

        median_1d = (
            float(return_1d.median())
            if len(return_1d)
            else 0.0
        )

        median_3d = (
            float(return_3d.median())
            if len(return_3d)
            else 0.0
        )

        median_5d = (
            float(return_5d.median())
            if len(return_5d)
            else 0.0
        )

        best_5d = (
            float(return_5d.max())
            if len(return_5d)
            else 0.0
        )

        worst_5d = (
            float(return_5d.min())
            if len(return_5d)
            else 0.0
        )

        candidate_pool_score = number(
            candidate.get(
                "candidate_pool_score"
            )
        )

        risk_score = number(
            candidate.get(
                "risk_score_final"
            ),
            50.0,
        )

        v33_score = calculate_v33_score(
            positive_1d=positive_1d,
            positive_3d=positive_3d,
            positive_5d=positive_5d,
            average_1d=average_1d,
            average_3d=average_3d,
            average_5d=average_5d,
            candidate_pool_score=candidate_pool_score,
            risk_score=risk_score,
        )

        decision, reason = determine_v33_decision(
            score=v33_score,
            similar_day_count=similar_count,
            positive_5d_rate=positive_5d,
            average_return_5d=average_5d,
            median_return_5d=median_5d,
            risk_score=risk_score,
        )

        rows.append(
            {
                "symbol": symbol,
                "v33_decision": decision,
                "v33_score": v33_score,
                "candidate_pool_score": candidate_pool_score,
                "candidate_sources": text(
                    candidate.get(
                        "candidate_sources"
                    )
                ),
                "similar_day_count": similar_count,
                "positive_1d_rate": round(
                    positive_1d,
                    2,
                ),
                "positive_3d_rate": round(
                    positive_3d,
                    2,
                ),
                "positive_5d_rate": round(
                    positive_5d,
                    2,
                ),
                "average_return_1d": round(
                    average_1d,
                    2,
                ),
                "average_return_3d": round(
                    average_3d,
                    2,
                ),
                "average_return_5d": round(
                    average_5d,
                    2,
                ),
                "median_return_1d": round(
                    median_1d,
                    2,
                ),
                "median_return_3d": round(
                    median_3d,
                    2,
                ),
                "median_return_5d": round(
                    median_5d,
                    2,
                ),
                "best_return_5d": round(
                    best_5d,
                    2,
                ),
                "worst_return_5d": round(
                    worst_5d,
                    2,
                ),
                "v22_signal_state": text(
                    candidate.get(
                        "v22_signal_state"
                    )
                ),
                "v22_signal_score": number(
                    candidate.get(
                        "v22_signal_score"
                    )
                ),
                "v27_decision": text(
                    candidate.get(
                        "v27_decision"
                    )
                ),
                "v27_master_score": number(
                    candidate.get(
                        "v27_master_score"
                    )
                ),
                "v32_decision": text(
                    candidate.get(
                        "v32_decision"
                    )
                ),
                "v32_score": number(
                    candidate.get(
                        "v32_score"
                    )
                ),
                "v32_confidence": number(
                    candidate.get(
                        "v32_confidence"
                    )
                ),
                "risk_class": text(
                    candidate.get(
                        "risk_class"
                    )
                ),
                "risk_score": risk_score,
                "regime": text(
                    candidate.get(
                        "regime_final"
                    )
                ),
                "market_percentile": number(
                    candidate.get(
                        "market_percentile_final"
                    )
                ),
                "timing_confidence": number(
                    candidate.get(
                        "timing_confidence_final"
                    )
                ),
                "expected_return": number(
                    candidate.get(
                        "expected_return_final"
                    )
                ),
                "downside_20pct": number(
                    candidate.get(
                        "downside_20pct_final"
                    )
                ),
                "upside_80pct": number(
                    candidate.get(
                        "upside_80pct_final"
                    )
                ),
                "close": number(
                    candidate.get(
                        "close_final"
                    )
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
        result
        .sort_values(
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
        .drop(
            columns="_priority"
        )
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
            output[column] = (
                result[column]
            )
        else:
            output[column] = np.nan

    return output


# ============================================================
# BOŞ SONUÇ
# ============================================================

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
        "candidate_pool_count": 0,
        "candidate_count": 0,
        "strong_confirmation_count": 0,
        "active_tracking_count": 0,
        "waiting_count": 0,
        "passive_count": 0,
        "approved_count": 0,
        "rsi_usage": RSI_USAGE,
        "version": VERSION,
    }

    save_status(
        status
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )


# ============================================================
# ANA MOTOR
# ============================================================

def main() -> None:
    started_at = time.time()

    print(
        "===== V33.1 GENIS ADAY + BENZER GUN MOTORU ====="
    )

    market_snapshot = normalize_symbol_frame(
        load_csv(
            MARKET_SNAPSHOT_FILE
        )
    )

    if market_snapshot.empty:
        save_empty(
            "market_snapshot_missing",
            "V16 tam piyasa dosyası bulunamadı.",
        )
        return

    candidate_pool = build_candidate_pool()

    if candidate_pool.empty:
        save_empty(
            "candidate_pool_empty",
            (
                "V22, V27 ve V32 kaynaklarından "
                "uygun aday havuzu oluşturulamadı."
            ),
        )
        return

    print(
        f"V33 aday havuzu: "
        f"{len(candidate_pool)} hisse"
    )

    print(
        candidate_pool[
            [
                "pool_rank",
                "symbol",
                "candidate_pool_score",
                "candidate_sources",
            ]
        ].to_string(
            index=False
        )
    )

    symbols = (
        market_snapshot[
            "symbol"
        ]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    market_data = download_market_data(
        symbols
    )

    if len(
        market_data
    ) < MINIMUM_MARKET_SYMBOLS:
        save_empty(
            "insufficient_market_data",
            (
                "Benzer gün analizi için "
                "yeterli piyasa verisi indirilemedi."
            ),
        )
        return

    context, featured_data = build_market_context(
        market_data
    )

    if len(
        context
    ) < MINIMUM_HISTORY_DAYS:
        save_empty(
            "insufficient_history",
            (
                "Benzer gün analizi için "
                "yeterli geçmiş oluşmadı."
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

    result = evaluate_candidates(
        candidate_pool=candidate_pool,
        featured_data=featured_data,
        similar_days=similar_days,
    )

    result.to_csv(
        CANDIDATE_RESULTS_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    if result.empty:
        top_symbol = ""
        top_decision = ""
        top_score = 0.0

    else:
        top_symbol = text(
            result.iloc[0]["symbol"]
        )

        top_decision = text(
            result.iloc[0][
                "v33_decision"
            ]
        )

        top_score = number(
            result.iloc[0][
                "v33_score"
            ]
        )

    strong_count = int(
        (
            result["v33_decision"]
            == "BENZER GÜN GÜÇLÜ TEYİT"
        ).sum()
    ) if not result.empty else 0

    active_count = int(
        (
            result["v33_decision"]
            == "BENZER GÜN AKTİF İZLEME"
        ).sum()
    ) if not result.empty else 0

    waiting_count = int(
        (
            result["v33_decision"]
            == "BENZER GÜN TEYİT BEKLE"
        ).sum()
    ) if not result.empty else 0

    passive_count = int(
        (
            result["v33_decision"]
            == "BENZER GÜN PASİF"
        ).sum()
    ) if not result.empty else 0

    approved_count = (
        strong_count
        + active_count
    )

    status = {
        "status": "ready",
        "target_date": (
            target_date.strftime(
                "%Y-%m-%d"
            )
            if target_date is not None
            else ""
        ),
        "market_symbol_count": int(
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
        "candidate_pool_count": int(
            len(candidate_pool)
        ),
        "candidate_count": int(
            len(result)
        ),
        "strong_confirmation_count": strong_count,
        "active_tracking_count": active_count,
        "waiting_count": waiting_count,
        "passive_count": passive_count,
        "approved_count": approved_count,
        "top_symbol": top_symbol,
        "top_decision": top_decision,
        "top_score": round(
            top_score,
            2,
        ),
        "shadow_mode": True,
        "rsi_usage": RSI_USAGE,
        "runtime_seconds": round(
            time.time()
            - started_at,
            2,
        ),
        "version": VERSION,
    }

    save_status(
        status
    )

    print(
        "===== V33.1 SONUCLAR ====="
    )

    if result.empty:
        print(
            "Değerlendirilebilir aday bulunamadı."
        )
    else:
        print(
            result[
                [
                    "v33_rank",
                    "symbol",
                    "v33_decision",
                    "v33_score",
                    "candidate_pool_score",
                    "positive_5d_rate",
                    "average_return_5d",
                    "risk_score",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "===== V33.1 STATUS ====="
    )

    print(
        json.dumps(
            status,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
