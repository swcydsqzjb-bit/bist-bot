"""Download current prices using the bot's universe; gate candidates on verified financials."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import engine

ROOT = Path(__file__).resolve().parents[1]
PRICE_COLUMNS = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']


def completed_day(now=None):
    now = now or datetime.now(ZoneInfo('Europe/Istanbul'))
    now = now.astimezone(ZoneInfo('Europe/Istanbul'))
    day = now.date()
    # 19:00 provides a conservative buffer for end-of-session data delivery.
    if now.hour < 19:
        day -= timedelta(days=1)
    while day.weekday() > 4:
        day -= timedelta(days=1)
    return pd.Timestamp(day)


def normalize_history(history, symbol, cutoff):
    if history is None or history.empty:
        raise ValueError('empty_history')
    d = history.copy()
    index = pd.DatetimeIndex(d.index)
    if index.tz is not None:
        index = index.tz_convert('Europe/Istanbul').tz_localize(None)
    d.index = index.normalize()
    d = d[d.index <= cutoff]
    if d.empty or 'Adj Close' not in d:
        raise ValueError('adjustment_data_missing')
    # Yahoo OHLC/volume already account for splits. Apply the common dividend
    # adjustment factor to every OHLC column, never just the closing price.
    factor = d['Adj Close'] / d['Close']
    if not np.isfinite(factor.to_numpy()).all() or (factor <= 0).any():
        raise ValueError('invalid_adjustment_factor')
    result = pd.DataFrame({'symbol': symbol, 'date': d.index})
    for col in ['Open', 'High', 'Low', 'Close']:
        result[col.lower()] = (d[col] * factor).to_numpy()
    result['volume'] = d.Volume.to_numpy()
    return engine.validate_prices(result[PRICE_COLUMNS])


def download_one(symbol, cutoff):
    import yfinance as yf
    ticker = 'XU100.IS' if symbol == 'XU100' else symbol + '.IS'
    try:
        history = yf.Ticker(ticker).history(
            start=(cutoff - pd.Timedelta(days=5*366)).date().isoformat(),
            end=(cutoff + pd.Timedelta(days=1)).date().isoformat(),
            interval='1d', auto_adjust=False, actions=False, timeout=25,
        )
        frame = normalize_history(history, symbol, cutoff)
        return symbol, frame, ''
    except Exception as exc:
        # Never print request headers, account data or tokens in logs.
        return symbol, pd.DataFrame(columns=PRICE_COLUMNS), type(exc).__name__


def finish_report(output, asof, source, symbols, prices, diagnostics, financials_present):
    result = pd.read_csv(output / 'all_results.csv')
    benchmark = prices[prices.symbol == 'XU100']
    preview = []
    usable = 0
    for symbol in symbols:
        status, reasons, metrics = engine.technical(prices[prices.symbol == symbol], benchmark, asof)
        if status != 'DATA_MISSING':
            usable += 1
        preview.append(dict(symbol=symbol, technical_status=status, reasons=';'.join(reasons), **metrics))
    pd.DataFrame(preview).to_csv(output / 'technical_preview.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output / 'download_diagnostics.csv', index=False)
    good_prices = usable >= .7 * len(symbols)
    verified = int((result.fundamental_status == 'PASS').sum())
    evaluated = int(result.fundamental_status.isin(['PASS', 'REJECTED', 'SEPARATE_MODEL', 'REVIEW_REQUIRED']).sum())
    good_financials = evaluated >= .7 * len(symbols)
    readiness = 'READY_WITH_REPORTED_COVERAGE' if good_prices and good_financials else 'INCOMPLETE_DATA'
    # Do not confuse missing coverage with a completed zero-candidate scan.
    if not good_prices or not good_financials:
        result.loc[result.status == 'TRIGGERED', 'status'] = 'COVERAGE_BLOCKED'
        result.to_csv(output / 'all_results.csv', index=False)
        result.iloc[:0].to_csv(output / 'candidates.csv', index=False)
    summary = dict(asof=str(asof.date()), mode='LIVE_PRICE_VERIFIED_FINANCIAL_CSV',
                   readiness=readiness, universe_source=source, expected=len(symbols),
                   reported=len(result), usable_prices=usable, price_coverage=usable/len(symbols),
                   fundamental_pass=verified, financials_evaluated=evaluated,
                   financial_coverage=evaluated/len(symbols), financials_file_present=financials_present,
                   status_counts=result.status.value_counts().to_dict(),
                   note='Technical preview does not certify PEG. No messages/orders sent. No backtest.')
    (output / 'status.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    text = ['# LARUS PEG Momentum', '', f'Tarih: {asof.date()}',
            f'Evren: {len(symbols)} | Raporlanan: {len(result)} | Yeterli fiyat: {usable}',
            f'Temel filtreyi geçen: {verified}', f'Durum: {readiness}', '',
            '**Teknik ön izleme, PEG onaylı aday listesi değildir.**', '',
            'Doğrulanmış bilanço verisi olmayan hisseler aday sayılmaz. Veri eksikliği, şartları sağlayan hisse olmadığı anlamına gelmez.',
            '', '| Durum | Adet |', '|---|---:|']
    text += [f'| {k} | {v} |' for k, v in summary['status_counts'].items()]
    (output / 'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asof', help='Completed session date; defaults to conservative Istanbul cutoff')
    p.add_argument('--output', default=str(ROOT / 'peg_momentum_output'))
    p.add_argument('--fundamentals', default=str(ROOT / 'peg_momentum/data/fundamentals.csv'))
    p.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    args = p.parse_args()
    asof = pd.Timestamp(args.asof) if args.asof else completed_day()
    if pd.isna(asof) or asof.tzinfo is not None or asof != asof.normalize() or asof > completed_day():
        raise ValueError('asof must be a completed session date')
    # Reuse the existing bot's checked, paginated universe; no silent stale fallback.
    import weekly_scan
    symbols, source = weekly_scan.universe()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'symbol': symbols}).to_csv(out/'universe.csv', index=False)
    # Automatic acquisition is an evidence/diagnostic layer, never a substitute
    # for comparison-basis and publication-time verification.
    from .financial_source import acquire
    acquire(symbols, asof.date(), out, args.workers)
    frames, diagnostics = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, (symbol, frame, error) in enumerate(pool.map(lambda s: download_one(s, asof), symbols + ['XU100']), 1):
            frames.append(frame)
            diagnostics.append(dict(symbol=symbol, bars=len(frame), error=error))
            if n % 50 == 0:
                print(f'PEG fiyat verisi: {n}/{len(symbols)+1}', flush=True)
    prices = pd.concat(frames, ignore_index=True)
    prices.to_csv(out/'prices.csv', index=False)
    path = Path(args.fundamentals)
    financials_present = path.is_file()
    if not financials_present:
        path = ROOT / 'peg_momentum/templates/fundamentals.csv'
    engine.run(SimpleNamespace(asof=str(asof.date()), universe=out/'universe.csv', prices=out/'prices.csv',
                               fundamentals=path, benchmark='XU100', output=out))
    summary = finish_report(out, asof, source, symbols, prices, diagnostics, financials_present)
    print(json.dumps(summary, ensure_ascii=False))
    if summary['readiness'] == 'INCOMPLETE_DATA':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
