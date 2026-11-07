"""Retrieve one terminal job and verify its saved durable-custody manifest.

This is byte verification and metadata retrieval, with no statistical reduction.
An incomplete job or missing durable receipt stops collection without retry.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import tarfile


ROOT = '/scratch/09999/aymanmahfuz/activation-checkpoint-integrity'


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('study', choices=['ba-closeout', 'ba-provenance'])
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    if not args.job.isdecimal():
        raise ValueError('Numeric job ID required')
    args.destination.mkdir(parents=True, exist_ok=False)
    socket = f'/tmp/tacc-control-{os.getuid()}/ctrl-ls6-activation-checkpoint-integrity.sock'
    ssh = ['ssh', '-S', socket, '-o', 'BatchMode=yes', 'aymanmahfuz@ls6.tacc.utexas.edu']
    result = subprocess.run(ssh + ['sacct -X -j ' + args.job + ' -n -P --format=JobIDRaw,Account,JobName,State,ElapsedRaw,NNodes,ExitCode'],
                            capture_output=True, text=True, timeout=40, check=True)
    (args.destination / 'native-accounting.txt').write_text(result.stdout)
    rows = [line.split('|') for line in result.stdout.splitlines() if line.strip()]
    terminal = {'COMPLETED', 'FAILED', 'TIMEOUT', 'CANCELLED', 'OUT_OF_MEMORY', 'NODE_FAIL'}
    if len(rows) != 1 or rows[0][0] != args.job or rows[0][1].upper() != 'CCR26063' or rows[0][3] not in terminal:
        raise RuntimeError('Exact owned job is not terminal; no data copied')
    source = f'{ROOT}/artifacts/{args.job}/{args.study}'
    receipt = subprocess.run(ssh + ['cat ' + shlex.quote(source + '/output-custody.json')],
                             capture_output=True, text=True, timeout=40, check=True)
    custody = json.loads(receipt.stdout)
    (args.destination / 'saved-output-custody.json').write_text(receipt.stdout)
    if custody.get('source') != source or custody.get('source_removed') is not False:
        raise RuntimeError('Unexpected output custody source')
    durable = custody['durable']
    expected_prefix = '/work/09999/aymanmahfuz/ls6/activation-checkpoint-integrity/completion/'
    if not durable.startswith(expected_prefix) or not durable.endswith('/' + args.job):
        raise RuntimeError('Durable custody leaves project namespace')
    actual = subprocess.run(ssh + ['cat ' + shlex.quote(durable + '/output-custody.json')],
                            capture_output=True, text=True, timeout=40, check=True)
    if actual.stdout != receipt.stdout:
        raise RuntimeError('Scratch and durable custody receipts differ')
    archive_path = args.destination / 'job-output.tar.gz'
    with archive_path.open('xb') as outgoing:
        archive = subprocess.run(ssh + ['tar -C ' + shlex.quote(f'{ROOT}/artifacts/{args.job}') + ' -czf - ' + args.study],
                                 stdout=outgoing, stderr=subprocess.PIPE, timeout=240)
    (args.destination / 'archive-stderr.txt').write_bytes(archive.stderr)
    if archive.returncode:
        raise RuntimeError('Archive transfer failed; preserve partial bytes')
    extracted = args.destination / 'extracted'
    extracted.mkdir()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            name = PurePosixPath(member.name)
            if name.is_absolute() or '..' in name.parts or not (member.isfile() or member.isdir()):
                raise RuntimeError('Unsafe archive member')
        archive.extractall(extracted, filter='data')
    folder = extracted / args.study
    for name, pin in custody['files'].items():
        member = PurePosixPath(name)
        if member.is_absolute() or '..' in member.parts:
            raise RuntimeError('Unsafe custody member')
        path = folder / name
        if path.is_symlink() or path.stat().st_size != pin['bytes'] or digest(path) != pin['sha256']:
            raise RuntimeError('Copied output differs: ' + name)
    verified = {'job_id': args.job, 'native_accounting': rows[0],
                'status': 'ALL_RECEIPTED_MEMBERS_BYTE_VERIFIED_LOCALLY',
                'members': len(custody['files']), 'archive_sha256': digest(archive_path),
                'source': source, 'durable': durable, 'scientific_acceptance': False}
    (args.destination / 'retrieval-verification.json').write_text(json.dumps(verified, indent=2) + '\n')
    print(json.dumps(verified))


if __name__ == '__main__':
    main()
