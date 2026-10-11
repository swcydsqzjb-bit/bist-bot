"""LARUS haftalık RSI + F/K taraması."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "weekly_rsi_fk" / "state.json"
OUT = ROOT / "weekly_rsi_fk_output"


def finite_positive(value):
    try:
        value = float(value)
        return value if math.isfinite(value) and value > 0 else None
    except (ValueError, TypeError):
        return None


def cutoff_friday(now=None):
    now = now or datetime.now(ZoneInfo("Europe/Istanbul"))
    day = now.date()

    if day.weekday() == 4 and now.hour < 19:
        day -= timedelta(days=7)

    return pd.Timestamp(
        day - timedelta(days=(day.weekday() - 4) % 7)
    )


def wilder_rsi(close, period=14):
    values = np.asarray(close, dtype=float)
    result = np.full(len(values), np.nan)

    if len(values) <= period:
        return pd.Series(result, index=close.index)

    diff = np.diff(values)
    gains = np.maximum(diff, 0)
    losses = np.maximum(-diff, 0)
    gain = gains[:period].mean()
    loss = losses[:period].mean()

    for i in range(period, len(values)):
        if i > period:
            gain = (gain * (period - 1) + gains[i - 1]) / period
            loss = (loss * (period - 1) + losses[i - 1]) / period

        if gain == loss == 0:
            result[i] = 50
        elif loss == 0:
            result[i] = 100
        else:
            result[i] = 100 - 100 / (1 + gain / loss)

    return pd.Series(result, index=close.index)


def weekly_frame(daily, cutoff):
    d = daily.copy()
    index = pd.DatetimeIndex(d.index)

    if index.tz is not None:
        index = index.tz_convert(
            "Europe/Istanbul"
        ).tz_localize(None)

    d.index = index.normalize()
    d = d.loc[d.index <= cutoff].sort_index()
    d = d[~d.index.duplicated(keep="last")]
    d = d.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    if d.empty:
        return pd.DataFrame()

    d["RawClose"] = d["Close"]
    factor = d["Adj Close"] / d["Close"]

    for col in ["Open", "High", "Low", "Close"]:
        d[col] = d[col] * factor

    w = d.resample("W-FRI").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "RawClose": "last",
        "Volume": "sum",
    })

    w["Sessions"] = d["Close"].resample("W-FRI").count()
    w["AvgDailyVolume"] = (
        w["Volume"] / w["Sessions"].replace(0, np.nan)
    )

    return w.loc[w.index <= cutoff].dropna(subset=["Close"])


def features(daily, cutoff):
    w = weekly_frame(daily, cutoff)

    if (
        len(w) < 60
        or w.index[-1] != cutoff
        or w.iloc[-1]["Volume"] <= 0
    ):
        return None

    rsi = wilder_rsi(w["Close"])
    a = w.iloc[-1]
    spread = a["High"] - a["Low"]

    strength = (
        (a["Close"] - a["Low"]) / spread
        if spread > 0 else 0
    )

    baseline = w["AvgDailyVolume"].iloc[-21:-1].mean()
    ratio = (
        a["AvgDailyVolume"] / baseline
        if baseline > 0 else 0
    )

    rising = bool(rsi.iloc[-1] > rsi.iloc[-2])
    strong = bool(strength >= 0.60)
    volume = bool(
        a["Close"] > w.iloc[-2]["Close"]
        and ratio >= 1.2
    )

    reasons = []

    if rising:
        reasons.append("RSI yükseliyor")
    if strong:
        reasons.append("Güçlü haftalık kapanış")
    if volume:
        reasons.append("Hacimli yükseliş")

    return {
        "rsi": float(rsi.iloc[-1]),
        "pe": None,
        "price": float(a["RawClose"]),
        "volume_ratio": float(ratio),
        "score": int(rising) + int(strong) + int(volume),
        "reasons": reasons,
    }


def retry(fn):
    for attempt in range(3):
        try:
            return fn()
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))


def universe():
    manual = ROOT / "weekly_symbols.txt"

    if manual.exists():
        symbols = [
            s.strip().upper().removesuffix(".IS")
            for s in manual.read_text().splitlines()
            if s.strip() and not s.startswith("#")
        ]
        source = "weekly_symbols.txt"
    else:
        def get_page(page):
            url = "https://stockanalysis.com/list/borsa-istanbul/"

            if page > 1:
                url += f"?page={page}"

            req = Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )

            with urlopen(req, timeout=40) as response:
                return response.read().decode()

        html = retry(lambda: get_page(1))
        count = re.search(r"(\d[\d,]*)\s+Stocks", html)

        if count is None:
            raise RuntimeError(
                "Hisse listesinin toplam sayısı doğrulanamadı."
            )

        expected = int(count.group(1).replace(",", ""))
        pattern = r'href=[\"\x27]/quote/ist/([A-Za-z0-9]+)/?'
        symbols = set(re.findall(pattern, html))

        for page in range(2, 11):
            if len(symbols) >= expected:
                break

            html = retry(lambda: get_page(page))
            additional = set(re.findall(pattern, html))

            if not additional - symbols:
                break

            symbols.update(additional)

        if len(symbols) < expected:
            raise RuntimeError(
                f"Hisse listesi eksik: "
                f"{len(symbols)}/{expected}."
            )

        source = "StockAnalysis"

    symbols = sorted({
        s.upper()
        for s in symbols
        if re.fullmatch(r"[A-Za-z0-9]{3,8}", s)
    })

    if len(symbols) < 500 and not manual.exists():
        raise RuntimeError(
            "Otomatik hisse listesi 500 altında."
        )

    if not symbols:
        raise RuntimeError("Hisse listesi boş.")

    return symbols, source


def fetch_one(symbol, cutoff):
    import yfinance as yf

    ticker = yf.Ticker(symbol + ".IS")

    try:
        daily = retry(lambda: ticker.history(
            period="5y",
            interval="1d",
            auto_adjust=False,
            raise_errors=True,
        ))

        if daily.empty or "Adj Close" not in daily:
            return symbol, None, None, "Fiyat verisi eksik"

        f = features(daily, cutoff)

        if f is None:
            return (
                symbol, daily, None,
                "Güncel/yeterli haftalık veri yok"
            )

        if not f["rsi"] < 40:
            return symbol, daily, None, "RSI uygun değil"

        info = retry(ticker.get_info)
        eps = finite_positive(info.get("trailingEps"))
        quarter = finite_positive(info.get("mostRecentQuarter"))

        if eps is None or quarter is None:
            return (
                symbol, daily, None,
                "F/K için EPS/bilanço tarihi eksik"
            )

        age = (
            datetime.now(ZoneInfo("Europe/Istanbul")).timestamp()
            - quarter
        )

        if age < 0 or age > 210 * 86400:
            return (
                symbol, daily, None,
                "Bilanço verisi eski/tutarsız"
            )

        f["pe"] = f["price"] / eps

        if not 5 <= f["pe"] <= 15:
            return symbol, daily, None, "F/K uygun değil"

        f["symbol"] = symbol
        f["earnings_quarter"] = datetime.fromtimestamp(
            quarter, ZoneInfo("Europe/Istanbul")
        ).date().isoformat()

        return symbol, daily, f, "Aday"

    except Exception as exc:
        return (
            symbol, None, None,
            "Veri hatası: " + type(exc).__name__
        )


def update_tracking(state, prices, cutoff):
    results = []

    for entry in state["selections"]:
        origin = pd.Timestamp(entry["week"])
        target = origin + pd.Timedelta(days=7)

        if target > cutoff:
            continue

        for item in entry["candidates"]:
            if item.get("outcome") is not None:
                continue

            d = prices.get(item["symbol"])

            if d is None:
                continue

            w = weekly_frame(d, cutoff)

            if origin not in w.index or target not in w.index:
                continue

            base = w.loc[origin, "Close"]
            end = w.loc[target, "Close"]

            if base > 0:
                item["outcome"] = {
                    "week": target.date().isoformat(),
                    "return_pct": float((end / base - 1) * 100),
                }

                results.append({
                    "symbol": item["symbol"],
                    "origin": entry["week"],
                    **item["outcome"],
                })

    return results


def telegram(message):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not token or not chat:
        raise RuntimeError(
            "Telegram ortam değişkenleri eksik."
        )

    data = urlencode({
        "chat_id": chat,
        "text": message,
    }).encode()

    try:
        req = Request(
            "https://api.telegram.org/bot" + token + "/sendMessage",
            data=data,
        )

        with urlopen(req, timeout=40) as response:
            result = json.load(response)

        if not result.get("ok"):
            raise RuntimeError("Telegram mesajı kabul etmedi.")

    except Exception:
        raise RuntimeError(
            "Telegram gönderimi başarısız; "
            "bot yetkisini ve Secrets değerlerini kontrol edin."
        ) from None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Telegram göndermez ve takip kaydını değiştirmez",
    )
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    cutoff = cutoff_friday()

    state = (
        json.loads(STATE.read_text())
        if STATE.exists()
        else {"selections": []}
    )

    # Web sitesi erişimi engellerse depodaki kod listesi kullanılır.
    try:
        symbols, source = universe()
    except Exception as exc:
        print(
            f"Web listesi alınamadı: {type(exc).__name__}; "
            "yedek liste deneniyor.",
            flush=True,
        )

        snapshot = ROOT / "v16_full_market_snapshot.csv"

        if not snapshot.exists():
            raise RuntimeError(
                "Yedek hisse listesi bulunamadı."
            ) from None

        table = pd.read_csv(
            snapshot, encoding="utf-8-sig"
        )

        if "symbol" not in table.columns:
            raise RuntimeError(
                "Yedek listede symbol sütunu yok."
            )

        codes = (
            table["symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
            .str.replace(r"\.IS$", "", regex=True)
        )

        symbols = sorted({
            code for code in codes
            if re.fullmatch(r"[A-Z0-9]{3,8}", code)
        })

        if len(symbols) < 500:
            raise RuntimeError(
                f"Yedek liste yetersiz: {len(symbols)} hisse."
            )

        source = "V16 yedek hisse kodları"

    print(
        f"Hisse listesi: {source} | {len(symbols)} hisse",
        flush=True,
    )

    tracked = {
        i["symbol"]
        for s in state["selections"]
        for i in s["candidates"]
        if i.get("outcome") is None
    }

    prices = {}
    candidates = []
    diagnostics = []

    workers = max(
        1, min(4, int(os.environ.get("SCAN_WORKERS", "2")))
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_one, s, cutoff)
            for s in sorted(set(symbols) | tracked)
        ]

        for n, future in enumerate(as_completed(futures), 1):
            s, d, f, reason = future.result()

            if d is not None:
                prices[s] = d

            if f and s in symbols:
                candidates.append(f)

            diagnostics.append({
                "symbol": s,
                "status": reason,
            })

            if n % 50 == 0:
                print(
                    f"İşlenen: {n}/{len(futures)}",
                    flush=True,
                )

    update_tracking(state, prices, cutoff)

    invalid_statuses = (
        "Fiyat verisi eksik",
        "Güncel/yeterli haftalık veri yok",
    )

    valid = sum(
        x["symbol"] in symbols
        and x["status"] not in invalid_statuses
        and not x["status"].startswith("Veri hatası")
        for x in diagnostics
    )

    if valid < 0.70 * len(symbols):
        pd.DataFrame(diagnostics).to_csv(
            OUT / "diagnostics.csv", index=False
        )

        raise RuntimeError(
            f"Yetersiz fiyat verisi kapsamı "
            f"({valid}/{len(symbols)}): "
            "liste ve takip kaydı yayımlanmadı."
        )

    candidates.sort(
        key=lambda x: (
            -x["score"], x["rsi"], x["pe"], x["symbol"]
        )
    )

    top = candidates[:5]
    week = cutoff.date().isoformat()

    existing = next(
        (
            s for s in state["selections"]
            if s["week"] == week
        ),
        None,
    )

    if existing is None:
        state["selections"].append({
            "week": week,
            "candidates": top,
            "created_at": datetime.now(
                ZoneInfo("Europe/Istanbul")
            ).isoformat(),
        })
    else:
        top = existing["candidates"]

    lines = [
        "LARUS — HAFTALIK RSI + F/K",
        f"Tamamlanan hafta: {week}",
        f"Evren: {len(symbols)} | Güncel fiyat verisi: {valid}",
        f"Liste kaynağı: {source}",
        "Koşullar: haftalık RSI(14)<40; F/K 5–15",
        "",
    ]

    for n, c in enumerate(top, 1):
        lines.extend([
            (
                f"{n}. {c['symbol']} | "
                f"RSI {c['rsi']:.1f} | F/K {c['pe']:.2f}"
            ),
            (
                f"Cuma/son seans: {c['price']:.2f} TL | "
                f"Puan {c['score']}/3"
            ),
            (
                " · ".join(c["reasons"])
                or "Ek toparlanma koşulu henüz yok"
            ),
            "",
        ])

    if not top:
        lines.append(
            "Koşulları sağlayan aday bulunamadı; "
            "filtre gevşetilmedi."
        )

    recent = [
        {"symbol": i["symbol"], **i["outcome"]}
        for s in state["selections"]
        for i in s["candidates"]
        if (i.get("outcome") or {}).get("week") == week
    ]

    if recent:
        lines.append("ÖNCEKİ LİSTENİN 1 HAFTALIK SONUCU")

        lines.extend(
            f"{r['symbol']}: %{r['return_pct']:+.2f}"
            for r in recent
        )

    unavailable = sum(
        x["symbol"] in symbols
        and (
            "F/K için" in x["status"]
            or "Bilanço verisi" in x["status"]
        )
        for x in diagnostics
    )

    errors = sum(
        x["status"].startswith("Veri hatası")
        for x in diagnostics
    )

    lines.extend([
        "",
        (
            "RSI uygun fakat EPS/bilanço verisi eksik/eski: "
            f"{unavailable} | Veri hatası: {errors}"
        ),
        "F/K: Yahoo geçmiş EPS verisi; bilanço doğrulaması gerekebilir.",
        "Takip listesi; puan yükseliş olasılığı değildir.",
    ])

    message = "\n".join(lines)

    (OUT / "message.txt").write_text(
        message, encoding="utf-8"
    )

    pd.DataFrame(
        top,
        columns=[
            "symbol", "rsi", "pe", "price", "score",
            "volume_ratio", "reasons", "earnings_quarter",
        ],
    ).to_csv(OUT / "candidates.csv", index=False)

    pd.DataFrame(diagnostics).to_csv(
        OUT / "diagnostics.csv", index=False
    )

    (OUT / "tracking.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    (OUT / "metadata.json").write_text(
        json.dumps({
            "week": week,
            "universe_source": source,
            "universe_count": len(symbols),
            "valid_price_count": valid,
            "eligible_count": len(candidates),
            "dry_run": args.dry_run,
        }, indent=2),
        encoding="utf-8",
    )

    print(message)

    if not args.dry_run:
        STATE.parent.mkdir(exist_ok=True)
        temporary = STATE.with_suffix(".tmp")

        temporary.write_text(
            json.dumps(state, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        temporary.replace(STATE)
        telegram(message)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Tarama durdu: {exc}")

        if (
            "--dry-run" not in sys.argv
            and os.environ.get("TELEGRAM_BOT_TOKEN")
            and os.environ.get("TELEGRAM_CHAT_ID")
        ):
            try:
                telegram(
                    "LARUS: Haftalık tarama veya gönderim "
                    "tamamlanamadı. Veri erişimi ya da kurulum "
                    "hatası var; GitHub Actions kaydını kontrol edin."
                )
            except Exception:
                pass

        raise SystemExit(1)
