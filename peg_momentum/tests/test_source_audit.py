import unittest
import pandas as pd
from peg_momentum.source_audit import kap_observations,compare


class AuditTests(unittest.TestCase):
    def test_kap_units_and_decimal_comma(self):
        h='<table><tr><th>Report</th><th>2026/06</th></tr><tr><td>Sunum Para Birimi</td><td>1000TL</td></tr><tr><td>Hasılat</td><td>1.234,5</td></tr></table>'
        rows=kap_observations(h)
        self.assertEqual(rows[0]['kap_value_try'],1234500)
        raw=pd.DataFrame([dict(symbol='TEST',period='2026/06',item_code='3C',value_try=1234500)])
        audit=compare(raw,rows,'TEST','url')[0]
        self.assertEqual(audit['audit'],'MATCH_WITHIN_DISPLAY_PRECISION')
        self.assertFalse(audit['comparison_basis_verified'])

    def test_difference_and_missing_are_not_matches(self):
        o=[dict(period='2026/06',item_code='3C',kap_value_try=1000,kap_display_unit=1)]
        for value,expected in [(1200,'DIFFERENCE_REQUIRES_BASIS_REVIEW'),(None,'MISSING')]:
            raw=pd.DataFrame([dict(symbol='TEST',period='2026/06',item_code='3C',value_try=value)])
            self.assertEqual(compare(raw,o,'TEST','url')[0]['audit'],expected)

    def test_unknown_units_rejected(self):
        with self.assertRaises(ValueError):
            kap_observations('<table><tr><td>Sunum Para Birimi</td><td>USD</td></tr></table>')
