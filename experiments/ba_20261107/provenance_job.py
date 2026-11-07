"""Run the preregistered provenance study only after allocated admission."""

import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import traceback
import unittest

from allocated_common import ROOT, admit, command, retain, sha, verify_file, write


def interrupted(signum, frame):
    raise KeyboardInterrupt('Processing deadline reached; retain partial evidence')


def main():
    checkout = Path(__file__).resolve().parents[2]
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdecimal() or sys.platform != 'linux':
        raise RuntimeError('Real Linux Slurm allocation required')
    out = ROOT / 'artifacts' / job / 'ba-provenance'
    out.mkdir(parents=True, exist_ok=False)
    for signum in (signal.SIGALRM, signal.SIGUSR1, signal.SIGTERM):
        signal.signal(signum, interrupted)
    signal.alarm(3000)
    contract = None
    exit_code = 1
    try:
        path = Path(os.environ['BA_CONTRACT_PATH'])
        contract = admit(checkout, path, out)
        write(out / 'contract.json', contract)
        release_path = Path(os.environ['BA_RELEASE_PATH'])
        write(out / 'execution-release.json', json.loads(release_path.read_text()))
        with tarfile.open(out / 'exact-experiment-source-and-config.tar.gz', 'x:gz') as archive:
            for name in sorted(contract['files']):
                archive.add(checkout / name, arcname=name, recursive=False)
        historical = Path(contract['source_path'])
        original = contract['historical_contract']
        historical_revision = command(out, 'historical-head', ['git', '-C', str(historical), 'rev-parse', 'HEAD']).strip()
        if historical_revision != original['execution_source_revision']:
            raise RuntimeError('Historical source revision differs')
        for name, pin in original['source_files'].items():
            verify_file(historical / name, pin)
        for item in contract['inputs']:
            verify_file(item['path'], item['sha256'])
            if Path(item['path']).stat().st_size != item['bytes']:
                raise RuntimeError('Input byte count differs')
        sys.path.insert(0, str(historical / 'src'))
        import torch
        import numpy
        import transformers

        if torch.__version__ != '2.13.0+cu126' or numpy.__version__ != '2.5.2':
            raise RuntimeError('Pinned torch/numpy runtime differs')
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError('Allocated CUDA device unavailable')
        properties = torch.cuda.get_device_properties(0)
        if 'A100' not in properties.name or properties.major != 8 or properties.minor != 0:
            raise RuntimeError('Registered A100 device required')
        command(out, 'nvidia-smi', ['nvidia-smi'])
        processes = command(out, 'gpu-processes', ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'])
        own_pid = str(os.getpid())
        if any(line.split(',')[0].strip() != own_pid for line in processes.splitlines() if line.strip()):
            raise RuntimeError('Unexpected GPU compute process on exclusive node')
        write(out / 'runtime.json', {'python': sys.version, 'torch': torch.__version__,
              'numpy': numpy.__version__, 'transformers': transformers.__version__,
              'cuda': torch.version.cuda, 'gpu': properties.name,
              'device_memory_bytes': properties.total_memory})
        model_release = out / 'model-work-release.json'
        write(model_release, {'status': 'PASS', 'job_id': job, 'account': 'CCR26063',
              'source_revision': historical_revision, 'contract_sha256': sha(path),
              'scope': 'BA_PROVENANCE_ONLY', 'scientific_acceptance': False})
        os.environ['ACI_MODEL_WORK_RELEASE'] = str(model_release)
        os.environ['ACI_CONTRACT_SHA256'] = sha(path)
        from ac_integrity.research.completion import source_allocation_guard
        source_allocation_guard()
        suite = unittest.defaultTestLoader.loadTestsFromName('provenance.test_provenance_allocated')
        def test_ids(tests):
            for test in tests:
                if isinstance(test, unittest.TestSuite):
                    yield from test_ids(test)
                else:
                    yield test.id()
        actual_ids = list(test_ids(suite))
        if sorted(actual_ids) != sorted(contract['fixture_ids']):
            raise RuntimeError('Exact allocated GPU fixture census differs')
        with (out / 'fixtures.log').open('x') as stream:
            fixture_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        write(out / 'fixture-result.json', {'tests_run': fixture_result.testsRun,
              'failures': len(fixture_result.failures), 'errors': len(fixture_result.errors),
              'skipped': len(fixture_result.skipped), 'test_ids': actual_ids,
              'successful': fixture_result.wasSuccessful()})
        if not fixture_result.wasSuccessful() or fixture_result.skipped:
            raise RuntimeError('Actual allocated CUDA fixtures failed; model study not run')
        from provenance.provenance import run
        results = run(out / 'study', contract)
        write(out / 'results.json', results)
        exit_code = 0 if results.get('status') == 'COMPLETE_PENDING_INDEPENDENT_REVIEW' else 1
    except BaseException as error:
        write(out / 'failure.json', {'type': type(error).__name__, 'message': str(error),
              'traceback': traceback.format_exc(), 'scientific_acceptance': False})
    finally:
        signal.alarm(0)
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(signum, signal.SIG_IGN)
        write(out / 'terminal.json', {'exit_code': exit_code, 'job_id': job,
              'scientific_acceptance': False})
        if contract is not None:
            retain(out, contract['campaign_id'], contract['output_max_bytes'])
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
