"""Small KAP cross-check of reported values; never certifies TTM comparability."""
import argparse
import re
from pathlib import Path

from bs4 import BeautifulSoup
import pandas as pd

from .financial_source import get_text

KAP_URLS = {
    'THYAO': 'https://www.kap.org.tr/tr/sirket-finansal-bilgileri/4028e4a140f2ed720140f376bebb01a7',
    'ASELS': 'https://www.kap.org.tr/tr/sirket-finansal-bilgileri/866-aselsan-elektronik-sanayi-ve-ticaret-a-s',
    'BIMAS': 'https://www.kap.org.tr/tr/sirket-finansal-bilgileri/4028e4a140e95be70140ee1b7b030119',
}
ITEMS = {'Hasılat': '3C', 'Esas Faaliyet Kârı (Zararı)': '3DF',
         'Dönem Kârının (Zararının) Dağılımı, Ana Ortaklık Payları': '2OCF'}


def kap_observations(html):
    soup = BeautifulSoup(html, 'html.parser')
    periods, units, observations = [], [], []
    for tr in soup.find_all('tr'):
        cells = [c.get_text(' ', strip=True) for c in tr.find_all(['td','th'], recursive=False)]
        if not cells:
            continue
        possible = [v for v in cells[1:] if re.fullmatch(r'20\d{2}/\d{2}', v)]
        if possible:
            periods, units = possible, []
        elif cells[0] == 'Sunum Para Birimi':
            units = []
            for value in cells[1:]:
                m = re.fullmatch(r'(1|1000|1000000)TL', value.replace(' ', ''))
                if not m:
                    raise ValueError('unsupported_kap_units')
                units.append(int(m.group(1)))
        elif cells[0] in ITEMS:
            if len(cells)-1 != len(periods) or len(units) != len(periods):
                raise ValueError('kap_period_units_alignment_missing')
            for period, unit, text in zip(periods, units, cells[1:]):
                value = float(text.replace('.', '').replace(',', '.')) if text else None
                observations.append(dict(period=period, item_code=ITEMS[cells[0]],
                                         kap_value_try=None if value is None else value*unit,
                                         kap_display_unit=unit))
    if not observations:
        raise ValueError('kap_financial_rows_missing')
    return observations


def compare(raw, observations, symbol, url):
    rows = []
    for item in observations:
        matches = raw[(raw.symbol==symbol) & (raw.period==item['period']) & (raw.item_code==item['item_code'])]
        if matches.empty:
            continue
        if len(matches)!=1:
            raise ValueError('ambiguous_source_observation')
        value = matches.iloc[0].value_try
        official = item['kap_value_try']
        known = pd.notna(value) and official is not None
        tolerance = item['kap_display_unit']
        status = ('MATCH_WITHIN_DISPLAY_PRECISION' if abs(value-official)<=tolerance else 'DIFFERENCE_REQUIRES_BASIS_REVIEW') if known else 'MISSING'
        rows.append(dict(symbol=symbol, period=item['period'], item_code=item['item_code'],
                         provider_value_try=value, **{k:v for k,v in item.items() if k not in {'period','item_code'}},
                         audit=status, kap_source=url,
                         comparison_basis_verified=False))
    return rows


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-csv', required=True)
    p.add_argument('--output', default='peg_momentum_output/kap_sample_audit.csv')
    args=p.parse_args()
    raw=pd.read_csv(args.source_csv, dtype={'period':str})
    rows=[]
    for symbol,url in KAP_URLS.items():
        if symbol in set(raw.symbol):
            rows.extend(compare(raw,kap_observations(get_text(url)),symbol,url))
    if not rows:
        raise ValueError('no_auditable_sample_rows')
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    result=pd.DataFrame(rows)
    result.to_csv(args.output,index=False)
    print(result[['symbol','period','item_code','audit']].to_string(index=False))
