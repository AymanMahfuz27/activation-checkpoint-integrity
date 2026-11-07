"""Run the configured official OpGuard study only after allocated admission."""

import json
import os
from pathlib import Path
import signal
import sys
import tarfile
import traceback
import subprocess
import time
import shutil

from allocated_common import ROOT, admit, command, retain, sha, verify_file, write


def interrupted(signum, frame):
    raise KeyboardInterrupt('Processing deadline reached; retain partial evidence')


def bounded_command(out, label, args, *, cwd=None, env=None):
    """Bound one owned subprocess group by this allocation's processing cutoff."""
    remaining = int(float(os.environ['ACI_SCIENCE_DEADLINE_EPOCH']) - time.time())
    if remaining < 30:
        raise TimeoutError('Insufficient processing time before ' + label)
    with (out / (label + '.log')).open('xb') as stream:
        process = subprocess.Popen(args, cwd=cwd, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            status = process.wait(timeout=remaining)
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
    if status:
        raise RuntimeError(label + ' failed with status ' + str(status))
    return (out / (label + '.log')).read_text(errors='replace')


def run_official_study(checkout, out, contract, contract_path):
    """Build and qualify the exact official runtime before four fresh arm processes."""
    source = out / 'official-source'
    bounded_command(out, 'opguard-clone', ['git', 'clone', '--quiet',
                    'https://github.com/OrderLab/OpGuard.git', str(source)])
    bounded_command(out, 'opguard-checkout', ['git', '-C', str(source), 'checkout',
                    '--detach', contract['opguard_revision']])
    revision = bounded_command(out, 'opguard-head', ['git', '-C', str(source),
                               'rev-parse', 'HEAD']).strip()
    if revision != contract['opguard_revision']:
        raise RuntimeError('Official source revision differs')
    for name, pin in contract['official_runtime_files'].items():
        verify_file(source / name, pin)
    # Save official tracked source before build-generated files are introduced.
    with (out / 'official-exact-source.tar').open('xb') as stream:
        subprocess.run(['git', '-C', str(source), 'archive', '--format=tar', revision],
                       stdout=stream, check=True, timeout=30)
    uv = ROOT / 'tools/uv-0.12.3/uv'
    bounded_command(out, 'opguard-native-build', [str(uv), 'pip', 'install',
                    '--python', sys.executable, '--no-build-isolation', '--no-deps', str(source)])
    bounded_command(out, 'installed-packages', [str(uv), 'pip', 'freeze', '--python', sys.executable])
    # Run outside the checkout so local Python files cannot shadow the installed build.
    sandbox = out / 'qualification'
    sandbox.mkdir()
    report_text = bounded_command(out, 'opguard-doctor-json', [sys.executable,
                    '-m', 'opguard.doctor', '--json'], cwd=sandbox)
    doctor = json.loads(report_text)
    if doctor.get('native_ready') is not True:
        raise RuntimeError('Official native extensions not qualified')
    native_files = {}
    binary_out = out / 'installed-native-artifacts'
    binary_out.mkdir()
    prefix = Path(sys.prefix).resolve()
    for label, metadata in doctor['extensions'].items():
        binary = Path(metadata['path']).resolve()
        if not binary.is_relative_to(prefix) or not binary.is_file():
            raise RuntimeError('Native extension is outside the isolated environment')
        destination = binary_out / (label + '-' + binary.name)
        shutil.copyfile(binary, destination)
        if sha(binary) != sha(destination):
            raise RuntimeError('Native binary custody differs')
        native_files[label] = {'installed_path': str(binary), 'sha256': sha(destination),
                               'bytes': destination.stat().st_size}
    write(out / 'native-binary-custody.json', native_files)
    bounded_command(out, 'opguard-doctor-native', [sys.executable,
                    '-m', 'opguard.doctor', '--require-native'], cwd=sandbox)
    smoke_text = bounded_command(out, 'opguard-official-native-smoke', [sys.executable,
                    str(source / 'tests/native_smoke.py')], cwd=sandbox)
    summaries = []
    for line in smoke_text.splitlines():
        try:
            summaries.append(json.loads(line))
        except ValueError:
            pass
    smoke = next((row for row in summaries if isinstance(row, dict) and row.get('verified') is True), None)
    if smoke is None:
        raise RuntimeError('Official native smoke produced no successful verification')
    qualification = {'status': 'PASS', 'opguard_revision': revision,
                     'doctor': doctor, 'native_smoke': smoke,
                     'scientific_acceptance': False}
    write(out / 'native-qualification.json', qualification)
    bounded_command(out, 'baseline-fixtures', [sys.executable,
        str(checkout / 'experiments/ba_20261107/opguard_fixtures.py'),
        str(contract_path), str(out / 'fixtures')], cwd=sandbox)
    arm_paths = {}
    for fault in (False, True):
        for official in (False, True):
            name = ('fault' if fault else 'clean') + ('-official' if official else '-native')
            folder = out / name
            child_env = os.environ.copy()
            child_env.update(OPGUARD_TRACE_OUTPUT_BASE=str(folder / 'official-traces'),
                             OPGUARD_BASELINE_CONTRACT_SHA256=sha(contract_path))
            bounded_command(out, name, [sys.executable,
                str(checkout / 'experiments/ba_20261107/opguard_baseline/run_arm.py'),
                '--contract', str(contract_path), '--out', str(folder),
                '--fault', str(int(fault)), '--official', str(int(official))],
                cwd=sandbox, env=child_env)
            write(folder / 'child-outcome.json', {'returncode': 0, 'normal_exit': True})
            arm_paths[name] = folder
    from opguard_baseline.aggregate import aggregate
    return aggregate(arm_paths, qualification, contract)


def main():
    checkout = Path(__file__).resolve().parents[2]
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdecimal() or sys.platform != 'linux':
        raise RuntimeError('Real Linux Slurm allocation required')
    out = ROOT / 'artifacts' / job / 'ba-opguard'
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
              'scope': 'BA_OPGUARD_ONLY', 'scientific_acceptance': False})
        os.environ['ACI_MODEL_WORK_RELEASE'] = str(model_release)
        os.environ['ACI_CONTRACT_SHA256'] = sha(path)
        from ac_integrity.research.completion import source_allocation_guard
        source_allocation_guard()
        results = run_official_study(checkout, out, contract, path)
        write(out / 'results.json', results)
        exit_code = 0 if results.get('execution_complete') is True else 1
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
