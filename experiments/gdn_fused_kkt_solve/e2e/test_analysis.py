"""CPU checks of statistical direction, null handling, and confirmation gates."""
import json
from pathlib import Path
import tempfile
import unittest

from analyze import analyze


class AnalysisTests(unittest.TestCase):
    def report(self, ratio=1, phase='confirmation', count=24):
        blocks=[]
        for i in range(count):
            baseline=100+i
            blocks.append(dict(pair=i, workload=[8192,32], matching_outputs=4,
                arms={'baseline':dict(mean_e2e_ms=baseline,mean_ttft_ms=baseline/2),
                      'fused':dict(mean_e2e_ms=baseline/ratio,mean_ttft_ms=baseline/2/ratio)}))
        return dict(status='complete',phase=phase,inference={},analysis_plan='test',
                    requests_per_arm=4,blocks=blocks)

    def evaluate(self, report):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'results.json'
            path.write_text(json.dumps(report))
            return analyze(path)['workloads'][0]['metrics']['mean_e2e_ms']

    def test_identical_timings_are_not_significant(self):
        result=self.evaluate(self.report())
        self.assertEqual(result['permutation_p_two_sided'],1)
        self.assertEqual(result['speedup_ci95'],[1,1])
        self.assertFalse(result['significant_at_05'])

    def test_speedup_and_slowdown_keep_direction(self):
        for ratio in [2,.5]:
            result=self.evaluate(self.report(ratio))
            self.assertAlmostEqual(result['geometric_mean_speedup'],ratio)
            self.assertLess(result['holm_adjusted_p'],.05)
            for bound in result['speedup_ci95']:
                self.assertAlmostEqual(bound,ratio)

    def test_pilot_is_not_confirmatory(self):
        result=self.evaluate(self.report(2,'pilot',4))
        self.assertFalse(result['confirmatory'])
        self.assertFalse(result['significant_at_05'])

    def test_partial_confirmation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate(self.report(count=23))


if __name__=='__main__':
    unittest.main()
