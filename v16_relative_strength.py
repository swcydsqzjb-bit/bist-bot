from __future__ import annotations

import json
import os
import time
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
import yfinance as yf


RESULT_FILE = "v16_relative_strength.csv"
MARKET_FILE = "v16_full_market_snapshot.csv"
STATUS_FILE = "v16_status.json"
V15_FILE = "v15_final_decisions.csv"

SYMBOL_LIMIT = int(os.getenv("V16_SYMBOL_LIMIT", "0"))
MIN_HISTORY = 70


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if np.isnan(number) or np.isinf(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_symbol(value: Any) -> str:
    return clean_text(value).upper().replace(".IS", "")


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        return pd.DataFrame()


def discover_symbols() -> list[str]:
    module_candidates = [
        ("v3_data", "get_symbols"),
        ("v3_scanner", "get_symbols"),
        ("v8_main", "get_symbols"),
        ("v8_fusion", "get_symbols"),
    ]

    for module_name, function_name in module_candidates:
        try:
            module = __import__(module_name, fromlist=[function_name])
            function = getattr(module, function_name)
            raw = function()
            symbols = sorted({
                normalize_symbol(item)
                for item in raw
                if normalize_symbol(item)
            })
            if len(symbols) >= 100:
                print(f"Sembol listesi {module_name}.{function_name} Ã¼zerinden alÄ±ndÄ±: {len(symbols)}")
                return symbols
        except Exception as exc:
            print(f"{module_name}.{function_name} kullanÄ±lamadÄ±: {exc}")

    urls = [
        "https://stockanalysis.com/list/borsa-istanbul/",
        "https://www.kap.org.tr/tr/bist-sirketler",
    ]

    for url in urls:
        try:
            tables = pd.read_html(url)
            for table in tables:
                for column in table.columns:
                    values = table[column].astype(str).str.upper().str.strip()
                    candidates = values[
                        values.str.fullmatch(r"[A-Z0-9]{4,6}", na=False)
                    ].tolist()
                    symbols = sorted(set(candidates))
                    if len(symbols) >= 100:
                        print(f"Sembol listesi web tablosundan alÄ±ndÄ±: {len(symbols)}")
                        return symbols
        except Exception as exc:
            print(f"{url} okunamadÄ±: {exc}")

    fallback_files = [
        "v9_leader_lag_results.csv",
        "v5_backfill_history.csv",
        "v3_signals_history.csv",
        "signals_history.csv",
    ]

    symbols: set[str] = set()
    for path in fallback_files:
        frame = load_csv(path)
        for column in ["symbol", "leader", "follower"]:
            if column in frame.columns:
                symbols.update(
                    normalize_symbol(item)
                    for item in frame[column].dropna()
                    if normalize_symbol(item)
                )

    if len(symbols) < 50:
        raise RuntimeError("Yeterli BIST sembolÃ¼ bulunamadÄ±.")

    print(f"Sembol listesi hafÄ±za dosyalarÄ±ndan oluÅturuldu: {len(symbols)}")
    return sorted(symbols)


def download_batch(symbols: list[str]) -> pd.DataFrame:
    tickers = [f"{symbol}.IS" for symbol in symbols]
    data = yf.download(
        tickers=tickers,
        period="8mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        timeout=45,
    )
    return data


def ticker_frame(data: pd.DataFrame, ticker: str, total_tickers: int) -> pd.DataFrame:
    try:
        if total_tickers == 1:
            frame = data.copy()
        else:
            frame = data[ticker].copy()
    except Exception:
        return pd.DataFrame()

    frame.columns = [str(column).lower() for column in frame.columns]
    required = {"open", "high", "low", "close", "volume"}
    if not required.issubset(frame.columns):
        return pd.DataFrame()

    frame = frame.dropna(subset=["close", "volume"])
    return frame


def percentile(series: pd.Series, inverse: bool = False) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() <= 1:
        result = pd.Series(50.0, index=series.index)
    else:
        result = values.rank(method="average", pct=True).fillna(0.5) * 100.0
    return 100.0 - result if inverse else result


def calculate_snapshot(symbols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    chunk_size = 120
    for chunk_start in range(0, len(symbols), chunk_size):
        chunk = symbols[chunk_start:chunk_start + chunk_size]
        print(f"V16 veri indiriliyor: {chunk_start + 1}-{chunk_start + len(chunk)}/{len(symbols)}")

        try:
            data = download_batch(chunk)
        except Exception as exc:
            print(f"Toplu indirme hatasÄ±: {exc}")
            continue

        for symbol in chunk:
            frame = ticker_frame(data, f"{symbol}.IS", len(chunk))
            if len(frame) < MIN_HISTORY:
                continue

            close = frame["close"].astype(float)
            high = frame["high"].astype(float)
            low = frame["low"].astype(float)
            volume = frame["volume"].astype(float)

            last_close = safe_float(close.iloc[-1])
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()

            delta = close.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))

            volume_sma20 = volume.rolling(20).mean()
            volume_ratio = safe_float(volume.iloc[-1] / volume_sma20.iloc[-1], 0.0)
            volume_acc = safe_float(volume.tail(10).mean() / volume.tail(30).mean(), 0.0)

            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr_pct = safe_float(tr.rolling(14).mean().iloc[-1] / last_close * 100, 0.0)

            return_1d = safe_float((last_close / close.iloc[-2] - 1) * 100)
            return_5d = safe_float((last_close / close.iloc[-6] - 1) * 100)
            return_20d = safe_float((last_close / close.iloc[-21] - 1) * 100)

            range_20 = safe_float((high.tail(20).max() / low.tail(20).min() - 1) * 100)
            candle_range = max(safe_float(high.iloc[-1] - low.iloc[-1]), 1e-9)
            close_position = safe_float((last_close - low.iloc[-1]) / candle_range)
            upper_wick = safe_float((high.iloc[-1] - max(last_close, frame["open"].iloc[-1])) / candle_range)

            up_volume = volume[close.diff() > 0].tail(20).sum()
            down_volume = volume[close.diff() < 0].tail(20).sum()
            up_down_ratio = safe_float(up_volume / max(down_volume, 1.0))

            rows.append({
                "symbol": symbol,
                "close": round(last_close, 4),
                "return_1d": round(return_1d, 4),
                "return_5d": round(return_5d, 4),
                "return_20d": round(return_20d, 4),
                "rsi": round(safe_float(rsi.iloc[-1], 50.0), 4),
                "ema20_distance": round(safe_float((last_close / ema20.iloc[-1] - 1) * 100), 4),
                "ema50_distance": round(safe_float((last_close / ema50.iloc[-1] - 1) * 100), 4),
                "volume_ratio": round(volume_ratio, 4),
                "volume_accumulation_ratio": round(volume_acc, 4),
                "up_down_volume_ratio": round(up_down_ratio, 4),
                "atr_pct": round(atr_pct, 4),
                "range_20_pct": round(range_20, 4),
                "close_position": round(close_position, 4),
                "upper_wick_ratio": round(upper_wick, 4),
            })

        time.sleep(1)

    return pd.DataFrame(rows)


def score_market(snapshot: pd.DataFrame) -> pd.DataFrame:
    result = snapshot.copy()

    result["momentum_percentile"] = (
        percentile(result["return_5d"]) * 0.45
        + percentile(result["return_20d"]) * 0.40
        + percentile(result["return_1d"]) * 0.15
    )

    rsi_health = (100 - (result["rsi"] - 60).abs() * 3).clip(0, 100)
    ema_health = (
        percentile(result["ema20_distance"]) * 0.55
        + percentile(result["ema50_distance"]) * 0.45
    )

    result["trend_percentile"] = (
        ema_health * 0.70
        + rsi_health * 0.30
    )

    result["volume_percentile"] = (
        percentile(result["volume_ratio"]) * 0.40
        + percentile(result["volume_accumulation_ratio"]) * 0.35
        + percentile(result["up_down_volume_ratio"]) * 0.25
    )

    result["quality_percentile"] = (
        percentile(result["close_position"]) * 0.30
        + percentile(result["upper_wick_ratio"], inverse=True) * 0.25
        + percentile(result["atr_pct"], inverse=True) * 0.20
        + percentile(result["range_20_pct"], inverse=True) * 0.25
    )

    result["relative_strength_score"] = (
        result["momentum_percentile"] * 0.32
        + result["trend_percentile"] * 0.30
        + result["volume_percentile"] * 0.20
        + result["quality_percentile"] * 0.18
    ).clip(0, 100)

    result["market_percentile"] = percentile(result["relative_strength_score"])
    result["relative_class"] = np.select(
        [
            result["market_percentile"] >= 90,
            result["market_percentile"] >= 75,
            result["market_percentile"] >= 55,
        ],
        ["PÄ°YASA LÄ°DERÄ°", "GÃÃLÃ", "ORTA"],
        default="ZAYIF",
    )

    result = result.sort_values(
        ["market_percentile", "relative_strength_score"],
        ascending=False,
    ).reset_index(drop=True)
    result.insert(0, "market_rank", range(1, len(result) + 1))
    return result


def merge_candidates(market: pd.DataFrame, v15: pd.DataFrame) -> pd.DataFrame:
    if v15.empty or "symbol" not in v15.columns:
        return pd.DataFrame()

    candidates = v15.copy()
    candidates["symbol"] = candidates["symbol"].map(normalize_symbol)

    columns = [
        "symbol", "rank", "close", "v15_score", "v15_decision",
        "v14_score", "v14_decision", "dna_classification", "dna_confidence",
    ]
    candidates = candidates[[column for column in columns if column in candidates.columns]]

    merged = candidates.merge(
        market,
        on="symbol",
        how="left",
        suffixes=("_v15", ""),
    )

    if "close_v15" in merged.columns:
        merged["close"] = pd.to_numeric(
            merged["close_v15"], errors="coerce"
        ).fillna(pd.to_numeric(merged.get("close"), errors="coerce"))

    output_columns = [
        "rank", "symbol", "close", "v15_score", "v15_decision",
        "v14_score", "v14_decision", "dna_classification", "dna_confidence",
        "market_rank", "market_percentile", "relative_strength_score",
        "relative_class", "momentum_percentile", "trend_percentile",
        "volume_percentile", "quality_percentile", "return_1d", "return_5d",
        "return_20d", "rsi", "volume_ratio", "volume_accumulation_ratio",
        "ema20_distance",
    ]

    for column in output_columns:
        if column not in merged.columns:
            merged[column] = np.nan

    return merged[output_columns].sort_values(
        ["market_percentile", "v15_score"],
        ascending=False,
    ).reset_index(drop=True)


def main() -> None:
    print("V16 Full Market Relative Strength Engine baÅladÄ±.")

    symbols = discover_symbols()
    if SYMBOL_LIMIT > 0:
        symbols = symbols[:SYMBOL_LIMIT]

    snapshot = calculate_snapshot(symbols)
    if snapshot.empty:
        raise RuntimeError("V16 tam piyasa verisi oluÅturulamadÄ±.")

    market = score_market(snapshot)
    market.to_csv(MARKET_FILE, index=False, encoding="utf-8-sig")

    v15 = load_csv(V15_FILE)
    candidates = merge_candidates(market, v15)
    candidates.to_csv(RESULT_FILE, index=False, encoding="utf-8-sig")

    status = {
        "status": "ready",
        "requested_symbol_count": len(symbols),
        "market_count": len(market),
        "candidate_count": len(candidates),
        "leader_count": int((market["market_percentile"] >= 90).sum()),
        "comparison_scope": "FULL_MARKET",
    }

    with open(STATUS_FILE, "w", encoding="utf-8") as file:
        json.dump(status, file, ensure_ascii=False, indent=2)

    print(json.dumps(status, ensure_ascii=False, indent=2))
    print(candidates.to_string(index=False))


if __name__ == "__main__":
    main()
