"""LARUS PEG Momentum: audited CSV inputs, no orders or messages."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


FIN_FIELDS = ['pe_ttm', 'eps_ttm', 'eps_prior_ttm', 'profit_ttm',
              'profit_prior_ttm', 'operating_profit_q', 'operating_profit_prior_year_q',
              'sales_q', 'sales_prior_year_q', 'operating_cashflow_ttm',
              'net_debt', 'ebitda_ttm']
EXCLUDED = {'bank', 'insurance', 'holding', 'reit', 'other_financial'}


def fundamental(row, asof):
    """Return quality status, reasons and trailing PEG; unknown never passes."""
    if row is None:
        return 'DATA_MISSING', ['financial_row_missing'], {}
    if str(row.get('sector_model', '')).lower() in EXCLUDED:
        return 'SEPARATE_MODEL', ['sector_requires_separate_model'], {}
    if row.get('sector_model') != 'nonfinancial':
        return 'DATA_MISSING', ['sector_model_missing_or_unknown'], {}
    for key in ['source', 'comparison_basis']:
        if not str(row.get(key, '')).strip() or pd.isna(row.get(key)):
            return 'DATA_MISSING', [key + '_missing'], {}
    if str(row.get('comparable', '')).lower() not in {'true', '1', '1.0'}:
        return 'DATA_MISSING', ['comparability_not_verified'], {}
    if str(row.get('recurring_earnings_verified', '')).lower() not in {'true', '1', '1.0'}:
        return 'DATA_MISSING', ['recurring_earnings_not_verified'], {}
    for key in FIN_FIELDS:
        try:
            if not np.isfinite(float(row.get(key, np.nan))):
                raise ValueError()
        except (TypeError, ValueError):
            return 'DATA_MISSING', [key + '_missing_or_invalid'], {}
    try:
        published = pd.Timestamp(row['published_at'])
        period = pd.Timestamp(row['period_end'])
        valued = pd.Timestamp(row['valuation_date'])
        if any(x.tzinfo is not None for x in [published, period, valued]):
            raise ValueError('dates must be plain dates')
        if any(pd.isna(x) for x in [published, period, valued]):
            raise ValueError('date missing')
    except (KeyError, ValueError, TypeError):
        return 'DATA_MISSING', ['financial_dates_invalid'], {}
    if published > asof or period > published or valued > asof:
        return 'DATA_MISSING', ['future_or_inconsistent_financial_dates'], {}
    if (asof - period).days > 200 or (asof - valued).days > 7:
        return 'DATA_MISSING', ['stale_financial_or_valuation_data'], {}
    f = {k: float(row[k]) for k in FIN_FIELDS}
    reasons = []
    positive = ['pe_ttm', 'eps_ttm', 'eps_prior_ttm', 'profit_ttm',
                'profit_prior_ttm', 'operating_profit_q',
                'operating_profit_prior_year_q', 'sales_q', 'sales_prior_year_q',
                'ebitda_ttm']
    if any(f[k] <= 0 for k in positive):
        return 'REJECTED', ['nonpositive_earnings_sales_or_ebitda'], {}
    growth = (f['eps_ttm'] / f['eps_prior_ttm'] - 1) * 100
    peg = f['pe_ttm'] / growth if growth > 0 else np.nan
    op_growth = f['operating_profit_q'] / f['operating_profit_prior_year_q'] - 1
    sales_growth = f['sales_q'] / f['sales_prior_year_q'] - 1
    cash_ratio = f['operating_cashflow_ttm'] / f['profit_ttm']
    leverage = f['net_debt'] / f['ebitda_ttm']
    if not 0 < peg < 1:
        reasons.append('peg_not_between_0_and_1')
    # Extreme growth can be a tiny-base effect. Do not auto-classify as cheap.
    if growth > 100:
        return 'REVIEW_REQUIRED', ['eps_growth_above_100pct_base_effect_review'], {'peg_historical': peg, 'eps_growth_pct': growth}
    if op_growth < .20:
        reasons.append('operating_profit_growth_below_20pct')
    if sales_growth <= 0:
        reasons.append('sales_not_growing')
    if f['operating_cashflow_ttm'] <= 0 or cash_ratio < .8:
        reasons.append('weak_cash_conversion')
    if leverage >= 2.5:
        reasons.append('leverage_at_or_above_2_5')
    metrics = dict(peg_historical=peg, eps_growth_pct=growth,
                   operating_profit_growth_pct=op_growth * 100,
                   cash_conversion=cash_ratio, net_debt_ebitda=leverage)
    return ('REJECTED' if reasons else 'PASS'), reasons, metrics


def rma(s, n):
    """Wilder smoothing seeded with the first n arithmetic observations."""
    out = pd.Series(np.nan, index=s.index)
    if len(s) >= n:
        out.iloc[n - 1] = s.iloc[:n].mean()
        for i in range(n, len(s)):
            out.iloc[i] = (out.iloc[i - 1] * (n - 1) + s.iloc[i]) / n
    return out


def rsi(close, n=14):
    change = close.diff().iloc[1:]
    up, down = rma(change.clip(lower=0), n), rma(-change.clip(upper=0), n)
    value = 100 - 100 / (1 + up / down.replace(0, np.nan))
    value = value.mask((down == 0) & (up > 0), 100)
    value = value.mask((down == 0) & (up == 0), 50)
    return value.reindex(close.index)


def validate_prices(prices):
    needed = {'symbol', 'date', 'open', 'high', 'low', 'close', 'volume'}
    if not needed.issubset(prices.columns):
        raise ValueError('prices.csv missing columns: ' + ','.join(sorted(needed - set(prices.columns))))
    prices = prices.copy()
    prices['date'] = pd.to_datetime(prices['date'], errors='raise')
    if prices['date'].isna().any() or prices['symbol'].isna().any():
        raise ValueError('missing price date/symbol')
    if prices.duplicated(['symbol', 'date']).any():
        raise ValueError('duplicate symbol/date in prices')
    cols = ['open', 'high', 'low', 'close', 'volume']
    prices[cols] = prices[cols].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(prices[cols].to_numpy()).all():
        raise ValueError('nonfinite prices')
    if (prices[['open', 'high', 'low', 'close']] <= 0).any().any() or (prices.volume < 0).any():
        raise ValueError('nonpositive OHLC or negative volume')
    if ((prices.high < prices[['open', 'low', 'close']].max(axis=1)) |
        (prices.low > prices[['open', 'high', 'close']].min(axis=1))).any():
        raise ValueError('inconsistent OHLC')
    return prices.sort_values(['symbol', 'date'])


def technical(daily, benchmark, asof):
    d = daily[daily.date <= asof].set_index('date').sort_index()
    if len(d) < 300 or (asof - d.index[-1]).days > 4:
        return 'DATA_MISSING', ['insufficient_or_stale_price_history'], {}
    # Weekly bars labelled Friday. No current incomplete week, even on Friday.
    # --asof is a completed-session date, never an intraday timestamp.
    w = d.resample('W-FRI').agg({'open': 'first', 'high': 'max', 'low': 'min',
                                  'close': 'last', 'volume': 'sum'}).dropna()
    w = w[w.index <= asof]
    if len(w) < 52 or (asof - w.index[-1]).days > 10:
        return 'DATA_MISSING', ['insufficient_or_stale_completed_weeks'], {}
    c = d.close
    e20 = c.ewm(span=20, adjust=False).mean()
    tr = pd.concat([d.high - d.low, (d.high - c.shift()).abs(),
                    (d.low - c.shift()).abs()], axis=1).max(axis=1)
    atr = rma(tr, 14).iloc[-1]
    e5 = w.close.ewm(span=5, adjust=False).mean()
    e22 = w.close.ewm(span=22, adjust=False).mean()
    wrsi = rsi(w.close)
    trend = w.close.iloc[-1] > e5.iloc[-1] > e22.iloc[-1]
    momentum = 50 <= wrsi.iloc[-1] <= 68 and wrsi.iloc[-1] > wrsi.iloc[-2]
    resistance = d.high.shift(1).rolling(20).max().iloc[-1]
    avg_volume = d.volume.shift(1).rolling(20).mean().iloc[-1]
    if avg_volume <= 0 or atr <= 0:
        return 'DATA_MISSING', ['zero_average_volume_or_atr'], {}
    rv = d.volume.iloc[-1] / avg_volume
    day_range = d.high.iloc[-1] - d.low.iloc[-1]
    close_position = ((c.iloc[-1] - d.low.iloc[-1]) / day_range) if day_range > 0 else 0
    breakout = c.iloc[-1] > resistance
    trigger = breakout and rv >= 1.5 and close_position >= .7
    extended = c.iloc[-1] > e20.iloc[-1] + 2 * atr
    # Ranking bonuses; never represented as success probability.
    score = 0
    macd = w.close.ewm(span=12, adjust=False).mean() - w.close.ewm(span=26, adjust=False).mean()
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    macd_rising = hist.iloc[-1] > hist.iloc[-2]
    weekly_rv = w.volume.iloc[-1] / w.volume.shift(1).rolling(10).mean().iloc[-1]
    band = 4 * c.rolling(20).std(ddof=0) / c.rolling(20).mean()
    # Percentile threshold excludes the day being evaluated.
    squeeze = bool((band.shift(1) <= band.shift(2).rolling(120).quantile(.2)).tail(5).any())
    rel = None
    b = benchmark[benchmark.date <= asof].set_index('date').close
    aligned = pd.concat([c.rename('stock'), b.rename('benchmark')], axis=1, join='inner').dropna()
    if len(aligned) >= 21 and aligned.index[-1] == d.index[-1]:
        rel = float((aligned.stock.iloc[-1] / aligned.stock.iloc[-21] -
                     aligned.benchmark.iloc[-1] / aligned.benchmark.iloc[-21]) * 100)
    score += 10 * int(macd_rising) + 10 * int(weekly_rv >= 1.5) + 10 * int(squeeze)
    score += 10 * int(rel is not None and rel > 0)
    crosses = (e5 > e22) & (e5.shift(1) <= e22.shift(1))
    score += 10 * int(crosses.tail(3).any())
    metrics = dict(price_date=str(d.index[-1].date()), weekly_date=str(w.index[-1].date()),
                   close=float(c.iloc[-1]), weekly_rsi=float(wrsi.iloc[-1]),
                   daily_relative_volume=float(rv), weekly_relative_volume=float(weekly_rv),
                   resistance_20d=float(resistance), atr14=float(atr),
                   ema20=float(e20.iloc[-1]), relative_return_20d_pp=rel,
                   ranking_bonus=score, squeeze=squeeze, macd_histogram_rising=bool(macd_rising))
    if not trend or not momentum:
        return 'WAIT', [k for k, ok in [('weekly_trend', trend), ('weekly_momentum', momentum)] if not ok], metrics
    if extended:
        return 'EXTENDED', ['above_ema20_plus_2atr'], metrics
    if trigger:
        return 'TRIGGERED', [], metrics
    return 'RADAR', [k for k, ok in [('breakout_missing', breakout), ('relative_volume_below_1_5', rv >= 1.5),
                                    ('weak_close_location', close_position >= .7)] if not ok], metrics


def run(args):
    asof = pd.Timestamp(args.asof)
    if asof.tzinfo is not None or asof != asof.normalize() or pd.isna(asof):
        raise ValueError('--asof must be a plain completed-session date')
    universe = pd.read_csv(args.universe, dtype={'symbol': str})
    symbols = universe.symbol.tolist()
    if not symbols or universe.symbol.isna().any() or len(set(symbols)) != len(symbols):
        raise ValueError('universe must contain unique nonempty symbols')
    prices = validate_prices(pd.read_csv(args.prices))
    fin = pd.read_csv(args.fundamentals)
    if 'symbol' not in fin:
        raise ValueError('financial symbol column missing')
    # Multiple snapshots allowed. Never select a publication after asof.
    if 'published_at' in fin:
        fin['_pub'] = pd.to_datetime(fin.published_at, errors='coerce')
    else:
        fin['_pub'] = pd.NaT
    b = prices[prices.symbol == args.benchmark]
    rows = []
    for symbol in symbols:
        candidates = fin[(fin.symbol == symbol) & (fin._pub <= asof)].sort_values('_pub')
        f = candidates.iloc[-1].to_dict() if len(candidates) else None
        fs, reasons, metrics = fundamental(f, asof)
        result = dict(symbol=symbol, status=fs, fundamental_status=fs, reasons=';'.join(reasons), **metrics)
        if fs == 'PASS':
            try:
                ts, tr, tm = technical(prices[prices.symbol == symbol], b, asof)
                result.update(status=ts, reasons=';'.join(tr), **tm)
            except (ValueError, IndexError, KeyError) as exc:
                result.update(status='DATA_MISSING', reasons='technical_error:' + type(exc).__name__)
        rows.append(result)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    report = pd.DataFrame(rows)
    if 'ranking_bonus' in report:
        report = report.sort_values(['status', 'ranking_bonus', 'symbol'], ascending=[True, False, True])
    report.to_csv(out / 'all_results.csv', index=False)
    report[report.status == 'TRIGGERED'].to_csv(out / 'candidates.csv', index=False)
    summary = dict(asof=args.asof, mode='CSV_INPUT_NOT_LIVE', expected=len(symbols), reported=len(rows),
                   status_counts=report.status.value_counts().to_dict(), benchmark_available=not b.empty,
                   note='Ranking bonus is not a probability. No orders/messages sent. No backtest performed.')
    (out / 'status.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# LARUS PEG Momentum', '', 'Tarih: ' + args.asof, '',
             'Her evren hissesi raporlanır. Eksik veri aday sayılmaz. Puan başarı olasılığı değildir.', '',
             '| Durum | Adet |', '|---|---:|']
    lines += [f'| {k} | {v} |' for k, v in summary['status_counts'].items()]
    lines += ['', 'Canlı veri bağlantısı, otomatik Telegram gönderimi ve geçmiş performans testi bu çalıştırmada yoktur.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prices', required=True)
    p.add_argument('--fundamentals', required=True)
    p.add_argument('--universe', required=True)
    p.add_argument('--asof', required=True, help='Completed BIST session date YYYY-MM-DD; run after close')
    p.add_argument('--benchmark', default='XU100')
    p.add_argument('--output', default='peg_momentum_output')
    run(p.parse_args())
