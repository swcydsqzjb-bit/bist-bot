"""LARUS deneysel haftalık momentum taraması."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import weekly_scan as common

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "weekly_momentum" / "state.json"
OUT = ROOT / "weekly_momentum_output"


def technical(daily, cutoff):
    w = common.weekly_frame(daily, cutoff)

    if len(w) < 60 or w.index[-1] != cutoff:
        return None, "Fiyat geçmişi eksik/eski"

    if w.iloc[-1]["Volume"] <= 0:
        return None, "İşlem hacmi yok"

    rsi = common.wilder_rsi(w["Close"])
    ema = w["Close"].ewm(span=20, adjust=False).mean()
    a = w.iloc[-1]

    if not 50 <= rsi.iloc[-1] <= 65:
        return None, "RSI 50-65 dışında"

    if rsi.iloc[-1] <= rsi.iloc[-2]:
        return None, "RSI yükselmiyor"

    if (
        a["Close"] <= ema.iloc[-1]
        or ema.iloc[-1] <= ema.iloc[-2]
    ):
        return None, "Yükselen EMA20 trendi yok"

    f = common.features(daily, cutoff)

    if f is None or f["score"] < 2:
        return None, "Teyit puanı 2 altında"

    f["ema_distance"] = float(
        (a["Close"] / ema.iloc[-1] - 1) * 100
    )

    return f, "Teknik uygun"


def symbols():
    try:
        return common.universe()
    except Exception:
        table = pd.read_csv(
            ROOT / "v16_full_market_snapshot.csv",
            encoding="utf-8-sig",
        )

        codes = (
            table["symbol"]
            .dropna()
            .astype(str)
            .str.strip()
            .str.upper()
        )

        codes = codes.str.replace(
            r"\.IS$", "", regex=True
        )

        result = sorted(set(
            codes[codes.str.fullmatch(r"[A-Z0-9]{3,8}")]
        ))

        if len(result) < 500:
            raise RuntimeError(
                "Yedek liste 500 hissenin altında."
            )

        return result, "V16 yedek kod listesi"


def fetch(symbol, cutoff):
    import yfinance as yf

    ticker = yf.Ticker(symbol + ".IS")
    daily = None
    usable = False

    try:
        daily = common.retry(lambda: ticker.history(
            period="5y",
            interval="1d",
            auto_adjust=False,
            raise_errors=True,
            timeout=20,
        ))

        if daily.empty or "Adj Close" not in daily:
            return (
                symbol, None, None, False,
                "Fiyat verisi yok"
            )

        w = common.weekly_frame(daily, cutoff)

        usable = (
            len(w) >= 60
            and w.index[-1] == cutoff
            and w.iloc[-1]["Volume"] > 0
        )

        f, status = technical(daily, cutoff)

        if f is None:
            return symbol, daily, None, bool(usable), status

        info = common.retry(ticker.get_info)
        eps = common.finite_positive(
            info.get("trailingEps")
        )
        quarter = common.finite_positive(
            info.get("mostRecentQuarter")
        )

        if eps is None or quarter is None:
            return (
                symbol, daily, None, True,
                "F/K verisi eksik"
            )

        age = (
            datetime.now(
                ZoneInfo("Europe/Istanbul")
            ).timestamp()
            - quarter
        )

        if not 0 <= age <= 210 * 86400:
            return (
                symbol, daily, None, True,
                "F/K verisi eski"
            )

        f["pe"] = f["price"] / eps

        if not 5 <= f["pe"] <= 15:
            return (
                symbol, daily, None, True,
                "F/K 5-15 dışında"
            )

        f["symbol"] = symbol
        return symbol, daily, f, True, "Aday"

    except Exception as exc:
        label = (
            "F/K erişim hatası"
            if usable else "Fiyat erişim hatası"
        )
        return (
            symbol, daily, None, bool(usable),
            label + ": " + type(exc).__name__
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cutoff = common.cutoff_friday()
    week = cutoff.date().isoformat()
    OUT.mkdir(exist_ok=True)

    state = (
        json.loads(STATE.read_text(encoding="utf-8"))
        if STATE.exists()
        else {"selections": []}
    )

    universe, source = symbols()

    pending = {
        c["symbol"]
        for s in state["selections"]
        for c in s["candidates"]
        if not c.get("outcome")
    }

    prices = {}
    candidates = []
    diagnostics = []

    workers = max(
        1, min(4, int(os.environ.get("SCAN_WORKERS", "2")))
    )

    print(
        f"Liste: {source} | {len(universe)} hisse",
        flush=True,
    )

    all_symbols = sorted(set(universe) | pending)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = pool.map(
            lambda s: fetch(s, cutoff),
            all_symbols,
        )

        for n, result in enumerate(results, 1):
            symbol, daily, f, usable, status = result

            if daily is not None:
                prices[symbol] = daily

            if f and symbol in universe:
                candidates.append(f)

            diagnostics.append({
                "symbol": symbol,
                "price_usable": usable,
                "status": status,
            })

            if n % 50 == 0:
                print(
                    f"İşlenen: {n}/{len(all_symbols)}",
                    flush=True,
                )

    pd.DataFrame(diagnostics).to_csv(
        OUT / "diagnostics.csv",
        index=False,
    )

    valid = sum(
        d["price_usable"]
        for d in diagnostics
        if d["symbol"] in universe
    )

    if valid < 0.70 * len(universe):
        raise RuntimeError(
            f"Fiyat kapsamı yetersiz: "
            f"{valid}/{len(universe)}; liste gönderilmedi."
        )

    common.update_tracking(state, prices, cutoff)

    # Önce teyit puanı; eşitlikte EMA20'ye yakınlık.
    # Sıralama yükseliş olasılığı hesabı değildir.
    candidates.sort(key=lambda c: (
        -c["score"],
        c["ema_distance"],
        c["symbol"],
    ))

    existing = next(
        (
            s for s in state["selections"]
            if s["week"] == week
        ),
        None,
    )

    if existing is None:
        top = candidates[:5]
        state["selections"].append({
            "week": week,
            "candidates": top,
        })
    else:
        top = existing["candidates"]

    lines = [
        "LARUS — HAFTALIK MOMENTUM DENEMESİ",
        f"Tamamlanan hafta: {week}",
        f"Liste: {source}",
        (
            f"Evren: {len(universe)} | "
            f"Güncel/yeterli fiyat: {valid}"
        ),
        "Haftalık RSI 50-65 ve yükseliyor.",
        "Fiyat yükselen 20 haftalık EMA üzerinde.",
        "F/K 5-15; teyit puanı en az 2/3.",
        "",
    ]

    for i, c in enumerate(top, 1):
        lines.extend([
            (
                f"{i}. {c['symbol']} | "
                f"RSI {c['rsi']:.1f} | F/K {c['pe']:.2f}"
            ),
            (
                f"Son seans: {c['price']:.2f} TL | "
                f"Puan {c['score']}/3"
            ),
            f"EMA20 mesafesi: %{c['ema_distance']:.1f}",
            " · ".join(c["reasons"]),
            "",
        ])

    if not top:
        lines.append(
            "Koşulları sağlayan aday yok; "
            "liste doldurulmadı."
        )

    for entry in state["selections"]:
        for c in entry["candidates"]:
            outcome = c.get("outcome") or {}

            if outcome.get("week") == week:
                lines.append(
                    f"Önceki aday {c['symbol']}: "
                    f"%{outcome['return_pct']:+.2f}"
                )

    missing = sum(
        d["status"].startswith((
            "F/K verisi", "F/K erişim"
        ))
        for d in diagnostics
        if d["symbol"] in universe
    )

    lines.extend([
        "",
        (
            "Teknik uygun fakat F/K "
            f"eksik/eski/erişilemeyen: {missing}"
        ),
        (
            "Deneysel takip listesi; yükseliş garantisi "
            "veya alım talimatı değildir."
        ),
        (
            "F/K Yahoo EPS verisidir; "
            "eksikler nedeniyle kapsam sınırlıdır."
        ),
    ])

    message = "\n".join(lines)

    (OUT / "message.txt").write_text(
        message,
        encoding="utf-8",
    )

    pd.DataFrame(top).to_csv(
        OUT / "candidates.csv",
        index=False,
    )

    (OUT / "tracking.json").write_text(
        json.dumps(
            state, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    print(message)

    if not args.dry_run:
        STATE.parent.mkdir(exist_ok=True)
        temp = STATE.with_suffix(".tmp")

        temp.write_text(
            json.dumps(
                state, ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )

        temp.replace(STATE)
        common.telegram(message)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Momentum taraması durdu:", exc)
        raise SystemExit(1)
