"""Run the exact OpGuard baseline fixtures in a fresh allocated child process."""

import json
import os
from pathlib import Path
import sys
import unittest


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item.id()


def main():
    contract_path = Path(sys.argv[1])
    out = Path(sys.argv[2])
    out.mkdir(exist_ok=False)
    if sys.platform != 'linux' or not os.environ.get('SLURM_JOB_ID', '').isdecimal():
        raise RuntimeError('Owned Slurm fixture process required')
    contract = json.loads(contract_path.read_text())
    historical = Path(contract['source_path'])
    sys.path.insert(0, str(historical / 'src'))
    os.environ['OPGUARD_TRACE_OUTPUT_BASE'] = str(out / 'official-trace')
    suite = unittest.defaultTestLoader.loadTestsFromName('opguard_baseline.test_allocated')
    ids = list(flatten(suite))
    if sorted(ids) != sorted(contract['fixture_ids']):
        raise RuntimeError('Exact configured-baseline fixture census differs')
    with (out / 'fixtures.log').open('x') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = {'tests_run': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'skipped': len(result.skipped),
               'test_ids': ids, 'successful': result.wasSuccessful()}
    (out / 'fixture-result.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return 0 if result.wasSuccessful() and not result.skipped else 1


if __name__ == '__main__':
    raise SystemExit(main())
