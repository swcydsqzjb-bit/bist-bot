import copy
import json
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from peg_momentum.kap_normalize import build,coefficients,share_count,parse_report,TAGS
from peg_momentum.engine import fundamental

# Synthetic note text with numeric coefficients/share counts independently
# checked against the live PDFs. No tests depend on downloading a report.
CURRENT_NOTE='''Dönem Endeks Düzeltme Katsayısı
30 Eylül 2026 137,23 1
31 Aralık 2025 110,39 1,2432
30 Eylül 2025 105,78 1,2973
Adi hisse senetleri (Bin adet) 4.560.000 4.560.000 4.560.000 4.560.000
sulandırılmış hissesi bulunmamaktadır'''
OLD_NOTE='''Adi hisse senetleri (Bin adet) 4.560.000 4.560.000
sulandırılmış hissesi bulunmamaktadır'''


class NormalizeTests(unittest.TestCase):
    def setUp(self):
        self.reports=json.loads((Path(__file__).parent/'fixtures/asels_report_chain.json').read_text())
        for report in self.reports:
            report['published']=datetime.fromisoformat(report['published'])
        self.notes=['30 Eylül 2026 tarihindeki satın alma gücü '+CURRENT_NOTE,
                    '31 Aralık 2025 tarihindeki satın alma gücü '+OLD_NOTE,
                    '30 Eylül 2025 tarihindeki satın alma gücü '+OLD_NOTE]

    def run_build(self,asof='2026-10-29'):
        return build(*self.reports,self.notes,358.5,'2026-10-28',asof)

    def test_real_chain_math_and_quality_gate(self):
        row,audit=self.run_build()
        self.assertAlmostEqual(row['profit_ttm'],48589287534.4,places=1)
        self.assertAlmostEqual(row['profit_prior_ttm'],29389026557.2,places=1)
        self.assertAlmostEqual(audit['peg_historical'],.5149813970951769)
        self.assertEqual(row['published_at'],'2026-10-28')
        status,reasons,metrics=fundamental(row,pd.Timestamp('2026-10-29'))
        self.assertEqual(status,'REVIEW_REQUIRED')
        self.assertIn('recurring_earnings_not_verified',reasons)
        self.assertAlmostEqual(metrics['peg_historical'],audit['peg_historical'])

    def test_after_close_release_not_usable_same_day(self):
        with self.assertRaisesRegex(ValueError,'future_report'):
            self.run_build('2026-10-27')

    def test_restatement_bridge_difference_rejected(self):
        self.reports[2]['statements']['income']['values'][TAGS['profit']][0]*=1.1
        with self.assertRaisesRegex(ValueError,'bridge_mismatch'):
            self.run_build()

    def test_different_cashflow_context_rejected(self):
        self.reports[0]['statements']['cashflow']['contexts'][0][-1]='2026-06-30'
        with self.assertRaisesRegex(ValueError,'cashflow_income'):
            self.run_build()

    def test_single_quarter_not_derived_from_unmatched_ytd(self):
        self.reports[0]['statements']['income']['contexts'][2][0]='2026-04-01'
        with self.assertRaisesRegex(ValueError,'quarterly_contexts'):
            self.run_build()

    def test_changing_weighted_shares_rejected(self):
        self.notes[1]=self.notes[1].replace('4.560.000','2.280.000')
        with self.assertRaisesRegex(ValueError,'share_count_changed'):
            self.run_build()
        with self.assertRaises(ValueError):
            share_count(OLD_NOTE.replace('4.560.000 4.560.000','4.560.000 2.280.000'))

    def test_foreign_functional_currency_is_not_TMS29_group(self):
        with self.assertRaisesRegex(ValueError,'foreign_functional'):
            coefficients('geçerli para birimi ABD Doları '+CURRENT_NOTE,'2026-09-30')

    def test_missing_coefficient_and_wrong_target_rejected(self):
        for note in [CURRENT_NOTE.replace('1,2973',''),CURRENT_NOTE.replace('137,23 1','137,23 2')]:
            with self.assertRaises(ValueError):
                coefficients(note,'2026-09-30')

    def test_unverified_quality_does_not_hide_calculated_metrics(self):
        row,_=self.run_build()
        row['operating_cashflow_ttm']=0
        status,reasons,metrics=fundamental(row,pd.Timestamp('2026-10-29'))
        self.assertEqual(status,'REJECTED')
        self.assertIn('weak_cash_conversion',reasons)
        self.assertIn('peg_historical',metrics)

    def test_missing_publication_time_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'publication_time_missing'):
            parse_report('<html></html>',1)

    def test_wrong_pdf_basis_date_rejected(self):
        self.notes[1]=self.notes[1].replace('31 Aralık 2025','31 Aralık 2024')
        with self.assertRaisesRegex(ValueError,'purchasing_power_date'):
            self.run_build()
