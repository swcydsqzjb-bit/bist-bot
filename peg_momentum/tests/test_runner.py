import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from peg_momentum.run import completed_day, normalize_history, finish_report, PRICE_COLUMNS


class RunnerTests(unittest.TestCase):
    def test_istanbul_cutoff_and_weekend(self):
        tz = ZoneInfo('Europe/Istanbul')
        self.assertEqual(str(completed_day(datetime(2026, 10, 30, 18, tzinfo=tz)).date()), '2026-10-29')
        self.assertEqual(str(completed_day(datetime(2026, 10, 30, 19, tzinfo=tz)).date()), '2026-10-30')
        self.assertEqual(str(completed_day(datetime(2026, 11, 1, 12, tzinfo=tz)).date()), '2026-10-30')

    def test_adjust_all_ohlc_and_remove_future(self):
        h = pd.DataFrame(dict(Open=[100, 101], High=[102, 103], Low=[98, 99],
                              Close=[100, 102], Volume=[200, 200], **{'Adj Close': [50, 51]}),
                         index=pd.date_range('2026-10-29', periods=2, tz='Europe/Istanbul'))
        p = normalize_history(h, 'TEST', pd.Timestamp('2026-10-29'))
        self.assertEqual(len(p), 1)
        self.assertEqual(p.close.iloc[0], 50)
        self.assertEqual(p.high.iloc[0], 51)
        self.assertEqual(p.volume.iloc[0], 200)

    def test_missing_adjustment_rejected(self):
        with self.assertRaises(ValueError):
            normalize_history(pd.DataFrame({'Close': [1]}, index=[pd.Timestamp('2026-10-29')]),
                              'TEST', pd.Timestamp('2026-10-30'))

    def finish(self, status, fundamental_status, technical_status='WAIT'):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            pd.DataFrame([dict(symbol='TEST', status=status, fundamental_status=fundamental_status)]).to_csv(out/'all_results.csv', index=False)
            pd.DataFrame([dict(symbol='TEST', status=status)]).to_csv(out/'candidates.csv', index=False)
            with patch('peg_momentum.run.engine.technical', return_value=(technical_status, [], {})):
                summary = finish_report(out, pd.Timestamp('2026-10-30'), 'TEST', ['TEST'],
                                        pd.DataFrame(columns=PRICE_COLUMNS), [], True)
            candidates = pd.read_csv(out/'candidates.csv')
            self.assertEqual(json.loads((out/'status.json').read_text()), summary)
            return summary, candidates

    def test_completed_zero_candidate_is_not_data_failure(self):
        summary, _ = self.finish('REJECTED', 'REJECTED')
        self.assertEqual(summary['readiness'], 'READY_WITH_REPORTED_COVERAGE')
        self.assertEqual(summary['fundamental_pass'], 0)

    def test_missing_financials_blocks_false_candidates(self):
        summary, candidates = self.finish('TRIGGERED', 'DATA_MISSING')
        self.assertEqual(summary['readiness'], 'INCOMPLETE_DATA')
        self.assertEqual(len(candidates), 0)

    def test_price_failure_blocks_candidates(self):
        summary, candidates = self.finish('TRIGGERED', 'PASS', 'DATA_MISSING')
        self.assertEqual(summary['readiness'], 'INCOMPLETE_DATA')
        self.assertEqual(len(candidates), 0)


if __name__ == '__main__':
    unittest.main()
