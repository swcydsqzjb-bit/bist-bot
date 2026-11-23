import argparse
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from peg_momentum.engine import fundamental, rsi, run, technical, validate_prices


def good_fin():
    return dict(symbol='TEST', sector_model='nonfinancial', source='SYNTHETIC_TEST',
                comparable=True, recurring_earnings_verified=True,
                comparison_basis='same purchasing power; comparable shares; normalized',
                published_at='2026-10-01', period_end='2026-06-30', valuation_date='2026-10-30',
                pe_ttm=12, eps_ttm=1.3, eps_prior_ttm=1, profit_ttm=130,
                profit_prior_ttm=100, operating_profit_q=125,
                operating_profit_prior_year_q=100, sales_q=1200,
                sales_prior_year_q=1000, operating_cashflow_ttm=120,
                net_debt=100, ebitda_ttm=100)


def prices():
    dates = pd.bdate_range(end='2026-10-30', periods=420)
    t = np.arange(len(dates))
    c = 100 + t * .02 + 2 * np.sin(t / 7)
    return pd.DataFrame(dict(symbol='TEST', date=dates, open=c, high=c+1,
                             low=c-1, close=c, volume=np.full(len(c), 1000.)))


class EngineTests(unittest.TestCase):
    def test_peg_uses_percentage_not_fraction(self):
        status, _, metrics = fundamental(good_fin(), pd.Timestamp('2026-10-30'))
        self.assertEqual(status, 'PASS')
        self.assertAlmostEqual(metrics['peg_historical'], .4)

    def test_missing_negative_and_stale_do_not_pass(self):
        for patch, expected in [({'eps_ttm': np.nan}, 'DATA_MISSING'),
                                ({'eps_prior_ttm': -1}, 'REJECTED'),
                                ({'valuation_date': '2026-10-01'}, 'DATA_MISSING'),
                                ({'comparable': False}, 'DATA_MISSING'),
                                ({'sector_model': 'bank'}, 'SEPARATE_MODEL'),
                                ({'eps_ttm': 3}, 'REVIEW_REQUIRED'),
                                ({'published_at': '2026-11-01'}, 'DATA_MISSING'),
                                ({'operating_cashflow_ttm': 0}, 'REJECTED')]:
            with self.subTest(patch=patch):
                self.assertEqual(fundamental(good_fin() | patch, pd.Timestamp('2026-10-30'))[0], expected)

    def test_flat_and_one_direction_rsi(self):
        self.assertEqual(rsi(pd.Series([1.] * 40)).iloc[-1], 50)
        self.assertEqual(rsi(pd.Series(np.arange(1., 41.))).iloc[-1], 100)
        self.assertEqual(rsi(pd.Series(np.arange(40., 0., -1))).iloc[-1], 0)

    def test_incomplete_week_and_future_prices_excluded(self):
        p = prices()
        asof = pd.Timestamp('2026-10-28')
        before = technical(p, p, asof)
        self.assertEqual(before[2]['weekly_date'], '2026-10-23')
        p.loc[p.date > asof, ['open', 'high', 'low', 'close']] *= 20
        self.assertEqual(before, technical(p, p, asof))

    def test_current_volume_excluded_from_average(self):
        p = prices()
        p.loc[p.index[-1], 'volume'] = 2000
        result = technical(p, p, pd.Timestamp('2026-10-30'))
        self.assertEqual(result[2]['daily_relative_volume'], 2)
        self.assertEqual(result[2]['resistance_20d'], p.high.iloc[-21:-1].max())

    def test_bad_price_duplicate_fails(self):
        p = prices()
        with self.assertRaises(ValueError):
            validate_prices(pd.concat([p, p.tail(1)]))

    def test_trigger_and_extension_classification(self):
        p = prices()
        # Isolate weekly momentum so this test exercises daily trigger boundaries.
        p.loc[p.index[-1], 'volume'] = 2000
        close = p.high.iloc[-21:-1].max() + .1
        p.loc[p.index[-1], ['open', 'close', 'high', 'low']] = [close, close, close+.1, close-.3]
        def controlled_rsi(c):
            return pd.Series(np.linspace(51, 60, len(c)), index=c.index)
        with patch('peg_momentum.engine.rsi', controlled_rsi):
            self.assertEqual(technical(p, p, pd.Timestamp('2026-10-30'))[0], 'TRIGGERED')
            p.loc[p.index[-1], 'volume'] = 1000
            self.assertEqual(technical(p, p, pd.Timestamp('2026-10-30'))[0], 'RADAR')
            p.loc[p.index[-1], ['open', 'close', 'high', 'low']] = [close+30, close+30, close+31, close+29]
            self.assertEqual(technical(p, p, pd.Timestamp('2026-10-30'))[0], 'EXTENDED')

    def test_all_universe_reported_and_future_financial_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pd.DataFrame({'symbol': ['TEST', 'MISSING']}).to_csv(root/'universe.csv', index=False)
            prices().to_csv(root/'prices.csv', index=False)
            pd.DataFrame([good_fin(), good_fin() | {'published_at': '2026-11-01', 'eps_ttm': 99}]).to_csv(root/'fin.csv', index=False)
            run(argparse.Namespace(asof='2026-10-30', universe=root/'universe.csv',
                                   prices=root/'prices.csv', fundamentals=root/'fin.csv',
                                   benchmark='XU100', output=root/'out'))
            r = pd.read_csv(root/'out/all_results.csv').set_index('symbol')
            self.assertEqual(len(r), 2)
            self.assertEqual(r.loc['TEST', 'fundamental_status'], 'PASS')
            self.assertEqual(r.loc['MISSING', 'status'], 'DATA_MISSING')
            self.assertAlmostEqual(r.loc['TEST', 'peg_historical'], .4)
            self.assertEqual(json.loads((root/'out/status.json').read_text())['reported'], 2)


if __name__ == '__main__':
    unittest.main()
