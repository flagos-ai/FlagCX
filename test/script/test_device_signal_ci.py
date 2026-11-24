"""Regression matrix and failure propagation for Device API signal tests."""
import shlex
import unittest
import test_kunlunxin_ci as fixture


class DeviceSignalCiTest(unittest.TestCase):
    def calls(self, platform, failure=''):
        result, log = fixture.KunlunXinUnitDispatchTest().run_mock_suite(
            'device_api_unified_ir', failure, unified_ir_platform=platform)
        calls = [(label, shlex.split(args))
                 for label, args in (line.split('|', 1) for line in log.splitlines())
                 if label.startswith('signal ownership ')]
        return result, calls

    def test_cuda_covers_context_reuse_and_both_inter_transports(self):
        result, calls = self.calls('cuda')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 12)
        for contexts in range(1, 5):
            selected = [(label, args) for label, args in calls
                        if f'contexts={contexts}' in label]
            self.assertEqual(len(selected), 3)
            for label, args in selected:
                ranks = 2 if ' inter ' in label else 1
                for setting in [f'FLAGCX_TEST_CONTEXTS={contexts}',
                                'FLAGCX_TEST_SIGNAL_REPEATS=100',
                                'FLAGCX_TEST_SIGNAL_SKEW=10000']:
                    self.assertEqual(args.count(setting), ranks)
                self.assertEqual(args[-8:], ['-b', '1K', '-e', '1K', '-f', '2', '-R', '1'])

    def test_klx_retains_intra_only_capability_gate(self):
        result, calls = self.calls('kunlunxin')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(len(calls), 4)
        self.assertTrue(all(' intra ' in label for label, _ in calls))
        self.assertTrue(all('XSHMEM_SYMMETRIC_SIZE' in args for _, args in calls))

    def test_signal_regression_failure_is_final(self):
        failure = 'signal ownership inter contexts=2 net=1'
        result, calls = self.calls('cuda', failure)
        self.assertEqual(result.returncode, 7, result.stdout + result.stderr)
        self.assertEqual(calls[-1][0], failure)
        self.assertFalse(any('inter contexts=3' in label for label, _ in calls))


if __name__ == '__main__':
    unittest.main()
