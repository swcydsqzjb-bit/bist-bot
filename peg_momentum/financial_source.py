"""Automatic source acquisition, deliberately separate from verified fundamentals.

The provider exposes cumulative statements and restates some historical columns.
These observations cannot certify TTM EPS growth or recurring earnings on their own.
"""
import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import pandas as pd

PAGE = 'https://www.isyatirim.com.tr/tr-tr/analiz/hisse/Sayfalar/sirket-karti.aspx'
API = 'https://www.isyatirim.com.tr/_layouts/15/IsYatirim.Website/Common/Data.aspx/MaliTablo'
RAW_COLUMNS = ['symbol', 'period', 'item_code', 'description', 'value_try', 'source', 'fetched_at', 'verification']


def get_text(url):
    for attempt in range(2):
        try:
            req = Request(url, headers={'User-Agent': 'LARUS financial research/1.0'})
            with urlopen(req, timeout=25) as r:
                return r.read().decode('utf-8-sig')
        except HTTPError as exc:
            if attempt == 1 or exc.code not in {429, 500, 502, 503, 504}:
                raise
        except (TimeoutError, OSError):
            if attempt == 1:
                raise
        time.sleep(2)


def discover(html, asof):
    soup = BeautifulSoup(html, 'html.parser')
    select = soup.find('select', id='ddlMaliTabloDonem1')
    group = soup.find('select', id='ddlMaliTabloGroup')
    if select is None or group is None:
        raise ValueError('financial_selectors_missing')
    groups = [o.get('value') for o in group.find_all('option') if o.get('value')]
    selected = group.find('option', selected=True)
    if selected is None and len(groups) != 1:
        raise ValueError('ambiguous_financial_group')
    financial_group = selected.get('value') if selected else groups[0]
    periods = set()
    for option in select.find_all('option'):
        value = option.get('value', '')
        if not re.fullmatch(r'20\d{2}/(?:3|6|9|12)', value):
            continue
        year, month = map(int, value.split('/'))
        end = (pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)).date()
        if end <= asof:
            periods.add((year, month))
    periods = sorted(periods, reverse=True)[:4]
    if len(periods) < 4:
        raise ValueError('fewer_than_four_reported_periods')
    return financial_group, periods


def parse_payload(payload, symbol, periods, url, fetched_at):
    if not isinstance(payload, dict) or payload.get('ok') is not True or not isinstance(payload.get('value'), list):
        raise ValueError('financial_response_schema_invalid')
    rows, seen = [], set()
    for item in payload['value']:
        code = item.get('itemCode')
        if not isinstance(code, str) or not code or code in seen:
            raise ValueError('financial_item_code_missing_or_duplicated')
        seen.add(code)
        for n, (year, month) in enumerate(periods, 1):
            raw = item.get(f'value{n}')
            if raw is None or str(raw).strip() in {'', '-'}:
                value = None
            else:
                # API units are TRY, unlike the company's short summary in millions.
                value = float(raw)
                if not math.isfinite(value):
                    raise ValueError('financial_value_nonfinite')
            rows.append(dict(symbol=symbol, period=f'{year}/{month:02d}', item_code=code,
                             description=item.get('itemDescTr', '').strip(), value_try=value,
                             source=url, fetched_at=fetched_at,
                             verification='UNVERIFIED_COMPARISON_BASIS'))
    if not rows:
        raise ValueError('financial_response_empty')
    return rows


def fetch_one(symbol, asof, getter=get_text):
    if not re.fullmatch(r'[A-Z0-9]{3,8}', symbol):
        raise ValueError('invalid_symbol')
    page_url = PAGE + '?' + urlencode({'hisse': symbol})
    fetched = datetime.now(timezone.utc).isoformat()
    try:
        group, periods = discover(getter(page_url), asof)
        params = dict(companyCode=symbol, exchange='TRY', financialGroup=group)
        for n, (year, month) in enumerate(periods, 1):
            params[f'year{n}'], params[f'period{n}'] = str(year), str(month)
        url = API + '?' + urlencode(params)
        rows = parse_payload(json.loads(getter(url)), symbol, periods, url, fetched)
        return rows, dict(symbol=symbol, status='ACQUIRED_UNVERIFIED', period=rows[0]['period'],
                          rows=len(rows), source=page_url, fetched_at=fetched,
                          reason='publication_time_comparison_basis_and_recurring_earnings_unverified')
    except Exception as exc:
        return [], dict(symbol=symbol, status='SOURCE_ERROR', rows=0, source=page_url,
                        fetched_at=fetched, reason=type(exc).__name__)


def acquire(symbols, asof, output, workers=2, fetcher=fetch_one):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    raw, diagnostics = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, (rows, diagnostic) in enumerate(pool.map(lambda s: fetcher(s, asof), symbols), 1):
            raw.extend(rows)
            diagnostics.append(diagnostic)
            if n % 50 == 0:
                print(f'Bilanço kaynak verisi: {n}/{len(symbols)}', flush=True)
    pd.DataFrame(raw, columns=RAW_COLUMNS).to_csv(output/'source_financials.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(output/'financial_source_diagnostics.csv', index=False)
    status = dict(expected=len(symbols), reported=len(diagnostics),
                  acquired=sum(d['status'] == 'ACQUIRED_UNVERIFIED' for d in diagnostics),
                  verified_for_peg=0, requested_asof=asof.isoformat(),
                  point_in_time=False,
                  note='Acquisition is not verification. No TTM/PEG is derived from incompatible cumulative periods.')
    (output/'financial_source_status.json').write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding='utf-8')
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', help='Comma separated explicit SAMPLE; otherwise existing full bot universe')
    parser.add_argument('--output', default='peg_momentum_output')
    parser.add_argument('--workers', type=int, choices=range(1, 5), default=2)
    args = parser.parse_args()
    if args.symbols:
        symbols = sorted(set(s.strip().upper() for s in args.symbols.split(',') if s.strip()))
    else:
        import weekly_scan
        symbols, _ = weekly_scan.universe()
    if not symbols or any(not re.fullmatch(r'[A-Z0-9]{3,8}', s) for s in symbols):
        raise ValueError('invalid_or_empty_symbols')
    print(json.dumps(acquire(symbols, datetime.now(ZoneInfo('Europe/Istanbul')).date(), args.output, args.workers), ensure_ascii=False))
