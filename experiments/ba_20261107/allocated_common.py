"""Admission and durable output custody for the November 7 experiments.

Inputs are a committed source checkout and a sealed experiment contract.
Before numerical imports, verify the owned allocation, exact batch and source,
then inspect live storage. Keep the original evidence and every failed attempt.
"""

import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import signal
import subprocess
import time
from datetime import datetime, timezone


ROOT = Path('/scratch/09999/aymanmahfuz/activation-checkpoint-integrity')
WORK = Path('/work/09999/aymanmahfuz/ls6/activation-checkpoint-integrity')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def command(out, label, args):
    result = subprocess.run(args, capture_output=True, text=True, timeout=40)
    (out / (label + '.stdout')).write_text(result.stdout)
    (out / (label + '.stderr')).write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(label + ' failed with status ' + str(result.returncode))
    return result.stdout


def verify_file(path, pin):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or sha(path) != pin:
        raise ValueError('Pinned regular file differs: ' + str(path))


def storage(out, output_bytes=2_000_000_000):
    text = command(out, 'work-quota', ['/usr/bin/lfs', 'quota', '-u', str(os.getuid()), '/work'])
    rows = [line.split() for line in text.splitlines() if line.split() and line.split()[0] == '/work']
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError('One complete native WORK quota row required')
    row = rows[0]
    used, soft, hard = [int(row[i].rstrip('*')) * 1024 for i in (1, 2, 3)]
    limit = min(value for value in (soft, hard) if value > 0)
    file_used, file_soft, file_hard = [int(row[i].rstrip('*')) for i in (5, 6, 7)]
    file_limit = min(value for value in (file_soft, file_hard) if value > 0)
    if used + output_bytes >= .9 * limit or limit - used < 50_000_000_000 + output_bytes:
        raise RuntimeError('WORK headroom or below-90% admission fails')
    if file_limit - file_used < 3000:
        raise RuntimeError('WORK file reserve fails')
    disk = shutil.disk_usage(ROOT)
    if disk.free < 50_000_000_000 + output_bytes:
        raise RuntimeError('Scratch reserve fails')
    write(out / 'storage.json', {'queried_utc': datetime.now(timezone.utc).isoformat(),
          'used_bytes': used, 'limit_bytes': limit, 'file_used': file_used,
          'file_limit': file_limit, 'scratch_free_bytes': disk.free,
          'output_max_bytes': output_bytes, 'raw_quota': text})


def admit(checkout, contract_path, out):
    """Require a real owned job and a source-bound independent precheck."""
    job = os.environ.get('SLURM_JOB_ID', '')
    if not job.isdecimal() or os.getuid() != 893489:
        raise RuntimeError('Owned LS6 allocation required')
    verify_file(contract_path, os.environ['BA_CONTRACT_SHA256'])
    contract = json.loads(Path(contract_path).read_text())
    release_path = Path(os.environ['BA_RELEASE_PATH'])
    verify_file(release_path, os.environ['BA_RELEASE_SHA256'])
    release = json.loads(release_path.read_text())
    if release.get('verdict') != 'PROMOTE' or release.get('contract_sha256') != sha(contract_path):
        raise RuntimeError('Independent exact precheck is absent')
    for name, pin in contract['files'].items():
        member = Path(name)
        if member.is_absolute() or '..' in member.parts:
            raise ValueError('Source member leaves checkout')
        verify_file(checkout / member, pin)
    revision = command(out, 'source-head', ['git', '-C', str(checkout), 'rev-parse', 'HEAD']).strip()
    if revision != release['execution_revision']:
        raise RuntimeError('Executed revision differs from release')
    dirty = command(out, 'source-status', ['git', '-C', str(checkout), 'status', '--porcelain', '--untracked-files=no'])
    if dirty.strip():
        raise RuntimeError('Execution source has tracked modifications')
    descriptor = command(out, 'owned-job', ['scontrol', 'show', 'job', job, '-o'])
    fields = dict(word.split('=', 1) for word in descriptor.split() if '=' in word)
    required = {'JobId': job, 'UserId': 'aymanmahfuz(893489)', 'Account': 'CCR26063',
                'JobState': 'RUNNING', 'JobName': contract['job_name'],
                'Partition': contract['partition'], 'NumNodes': '1', 'NumTasks': '1',
                'CPUs/Task': str(contract['cpus']), 'OverSubscribe': 'NO',
                'Requeue': '0', 'Restarts': '0', 'TimeLimit': contract['time_limit'],
                'Command': str(checkout / contract['batch'])}
    if any(fields.get(key) != value for key, value in required.items()):
        raise RuntimeError('Owned job descriptor differs from contract')
    nodes = command(out, 'owned-nodes', ['scontrol', 'show', 'hostnames', fields['NodeList']]).splitlines()
    if nodes != [socket.gethostname().split('.')[0]]:
        raise RuntimeError('Process is outside its assigned compute node')
    actual = command(out, 'actual-batch', ['scontrol', 'write', 'batch_script', job, '-'])
    if hashlib.sha256(actual.encode()).hexdigest() != contract['files'][contract['batch']]:
        raise RuntimeError('Actual scheduler batch differs from sealed batch')
    remaining = command(out, 'time-left', ['squeue', '-h', '-j', job, '-o', '%L']).strip()
    days, separator, clock_text = remaining.partition('-')
    if not separator:
        clock_text, days = days, '0'
    clock = clock_text.split(':')
    if len(clock) not in (2, 3) or not days.isdecimal() or not all(part.isdecimal() for part in clock):
        raise RuntimeError('Malformed owned allocation time')
    if len(clock) == 2:
        seconds = int(clock[0]) * 60 + int(clock[1])
    else:
        seconds = int(clock[0]) * 3600 + int(clock[1]) * 60 + int(clock[2])
    seconds += int(days) * 86400
    if not 1200 <= seconds <= 3600:
        raise RuntimeError('Insufficient allocated time after environment setup')
    signal.alarm(seconds - 600)
    os.environ['ACI_SCIENCE_DEADLINE_EPOCH'] = str(time.time() + seconds - 600)
    storage(out, contract['output_max_bytes'])
    write(out / 'physical-admission.json', {'job_id': job, 'node': nodes[0],
          'account': fields['Account'], 'revision': revision,
          'contract_sha256': sha(contract_path), 'scientific_acceptance': False})
    return contract


def retain(out, campaign, max_bytes):
    """Copy only this job's bounded outputs; verify both copies without removal."""
    fresh = out / 'retention-admission'
    fresh.mkdir()
    storage(fresh, max_bytes)
    target = WORK / 'completion' / campaign / os.environ['SLURM_JOB_ID']
    target.mkdir(parents=True, exist_ok=False)
    members = [path for path in out.rglob('*') if path.is_file()]
    if len(members) > 3000 or sum(path.stat().st_size for path in members) > max_bytes:
        raise RuntimeError('Output envelope exceeded; Scratch evidence retained')
    files = {}
    for source in members:
        if source.is_symlink():
            raise ValueError('Output symlink is forbidden')
        name = str(source.relative_to(out))
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing)
        pin = sha(source)
        if sha(destination) != pin:
            raise RuntimeError('Durable copy hash differs: ' + name)
        files[name] = {'bytes': source.stat().st_size, 'sha256': pin}
    receipt = {'status': 'VERIFIED_OUTPUT_CUSTODY_PENDING_ACTUAL_REVIEW',
               'source': str(out), 'durable': str(target), 'files': files,
               'scientific_acceptance': False, 'source_removed': False}
    write(out / 'output-custody.json', receipt)
    write(target / 'output-custody.json', receipt)
