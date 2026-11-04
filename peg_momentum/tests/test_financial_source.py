import json
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from peg_momentum.financial_source import discover, parse_payload, fetch_one, acquire

HTML = '''<select id="ddlMaliTabloGroup"><option value="XI_29">IFRS</option></select>
<select id="ddlMaliTabloDonem1"><option value="2026/9">2026/9</option>
<option value="2026/6">2026/6</option><option value="2026/3">2026/3</option>
<option value="2025/12">2025/12</option><option value="2025/9">2025/9</option></select>'''


class FinancialSourceTests(unittest.TestCase):
    def test_only_available_periods_no_assumed_quarter(self):
        group, periods = discover(HTML, date(2026, 8, 1))
        self.assertEqual(group, 'XI_29')
        self.assertEqual(periods, [(2026,6), (2026,3), (2025,12), (2025,9)])

    def test_missing_selector_and_ambiguous_group_fail(self):
        for html in ['', HTML.replace('<option value="XI_29">IFRS</option>', '<option value="A">A</option><option value="B">B</option>')]:
            with self.assertRaises(ValueError):
                discover(html, date(2026,10,30))

    def test_values_preserve_units_negative_zero_and_null(self):
        p = dict(ok=True, value=[dict(itemCode='3C', itemDescTr='Satış', value1='1000000000', value2='-5', value3='0', value4=None)])
        rows = parse_payload(p, 'TEST', [(2026,6),(2026,3),(2025,12),(2025,9)], 'url', 'now')
        self.assertEqual([r['value_try'] for r in rows], [1e9, -5, 0, None])
        self.assertTrue(all(r['verification']=='UNVERIFIED_COMPARISON_BASIS' for r in rows))

    def test_schema_nonfinite_and_duplicate_fail(self):
        item = dict(itemCode='3C', value1='1')
        for p in [dict(ok=False,value=[]), dict(ok=True,value=[item,item]), dict(ok=True,value=[item | {'value1':'NaN'}])]:
            with self.assertRaises(ValueError):
                parse_payload(p, 'TEST', [(2026,6)], 'url', 'now')

    def test_fetch_does_not_make_verified_fundamentals(self):
        def getter(url):
            return HTML if 'sirket-karti' in url else json.dumps(dict(ok=True,value=[dict(itemCode='3C',value1='1')]))
        rows, diagnostic = fetch_one('TEST', date(2026,10,30), getter)
        self.assertEqual(len(rows), 4)
        self.assertEqual(diagnostic['status'], 'ACQUIRED_UNVERIFIED')
        self.assertNotIn('published_at', diagnostic)

    def test_source_failure_keeps_every_symbol_in_coverage(self):
        def fake(symbol, asof):
            return [], dict(symbol=symbol, status='SOURCE_ERROR',rows=0)
        with tempfile.TemporaryDirectory() as tmp:
            status = acquire(['AAA','BBB'], date(2026,10,30), tmp, fetcher=fake)
            self.assertEqual(status['expected'], status['reported'])
            self.assertEqual(status['verified_for_peg'], 0)
            self.assertEqual(len(pd.read_csv(Path(tmp)/'financial_source_diagnostics.csv')), 2)
            self.assertFalse((Path(tmp)/'fundamentals.csv').exists())


if __name__ == '__main__':
    unittest.main()
