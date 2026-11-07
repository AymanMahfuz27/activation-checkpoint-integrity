"""Submit a sealed experiment once through the already authenticated socket.

The local intent is exclusive and precedes any SSH call. A lost response must
be reconciled from the remote intent and scheduler; this tool never retries.
It performs lightweight source/quota/queue admission only, never computation.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('packet', type=Path)
    parser.add_argument('record', type=Path)
    args = parser.parse_args()
    packet = json.loads(args.packet.read_text())
    args.record.mkdir(parents=True, exist_ok=False)
    (args.record / 'intent.json').write_text(json.dumps({
        'at_utc': datetime.now(timezone.utc).isoformat(), 'packet': packet,
        'packet_sha256': hashlib.sha256(args.packet.read_bytes()).hexdigest(),
        'automatic_retry': False}, indent=2) + '\n')
    remote = r'''
import hashlib,json,os,re,subprocess
from pathlib import Path
p=PACKET
root=Path('/scratch/09999/aymanmahfuz/activation-checkpoint-integrity/slurm/ba-20261107')
root.mkdir(exist_ok=True)
checkout=root/p['checkout_name']
if checkout.exists():
    raise RuntimeError('Exclusive new checkout already exists; reconcile, do not retry')
subprocess.run(['git','clone','--branch','experiment/ba-20261107',
  'https://github.com/AymanMahfuz27/activation-checkpoint-integrity.git',str(checkout)],check=True)
subprocess.run(['git','-C',str(checkout),'pull','--ff-only'],check=True)
subprocess.run(['git','-C',str(checkout),'checkout','--detach',p['revision']],check=True)
if subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()!=p['revision']:
    raise RuntimeError('Revision differs')
for name,pin in p['files'].items():
    f=checkout/name
    if not f.is_file() or f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=pin:
        raise RuntimeError('Sealed control differs: '+name)
status=subprocess.check_output(['/usr/local/etc/taccinfo'],text=True)
if not any('CCR26063' in line and '2027-10-31' in line for line in status.splitlines()):
    raise RuntimeError('ValueGuard membership/expiry not verified')
quota=subprocess.check_output(['/usr/bin/lfs','quota','-u',str(os.getuid()),'/work'],text=True)
rows=[x.split() for x in quota.splitlines() if x.split() and x.split()[0]=='/work']
if len(rows)!=1 or len(rows[0])!=9:
    raise RuntimeError('Quota malformed')
row=rows[0];used=int(row[1].rstrip('*'));limit=min(int(row[i].rstrip('*')) for i in (2,3) if int(row[i].rstrip('*'))>0)
if used*1024+2000000000>=limit*1024*.9 or (limit-used)*1024<52000000000:
    raise RuntimeError('Fresh below90%/52GB quota reserve fails')
queue=subprocess.check_output(['squeue','--me','-h','-o','%i|%a|%j'],text=True)
allowed=p.get('allowed_queue',[])
if any(line.strip().split('|') not in allowed for line in queue.splitlines() if line.strip()):
    raise RuntimeError('Owned queue has unreconciled work; stop before dispatch')
control=root/p['control_name'];control.mkdir(exist_ok=False)
(control/'admission.json').write_text(json.dumps({'taccinfo':status,'quota':quota,'owned_queue':queue,'revision':p['revision']},indent=2)+'\n')
if 'release' in p:
    (control/'execution-release.json').write_text(json.dumps(p['release'],indent=2,sort_keys=True)+'\n')
intent=control/'dispatch-intent.json'
with intent.open('x') as f: json.dump({'packet':p,'automatic_retry':False},f,indent=2)
env=os.environ.copy();env.update(p['environment'])
result=subprocess.run(['sbatch','--parsable',str(checkout/p['batch'])],capture_output=True,text=True,env=env)
receipt={'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'automatic_retry':False}
(control/'response.json').write_text(json.dumps(receipt,indent=2)+'\n')
if result.returncode==0:
    matches=re.findall(r'^(\d+)(?:;[A-Za-z0-9_.-]+)?$',result.stdout,re.MULTILINE)
    if len(matches)!=1: raise RuntimeError('Unparseable sbatch success; reconcile only')
    job=matches[0]
    receipt['job_id']=job
    receipt['immediate_queue']=subprocess.check_output(['squeue','-j',job,'-o','%.18i %.16P %.20a %.28j %.8T %.10M %.10l %.6D %R'],text=True)
    (control/'accepted.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
raise SystemExit(result.returncode)
'''
    remote = remote.replace('PACKET', repr(packet), 1)
    socket = f'/tmp/tacc-control-{os.getuid()}/ctrl-ls6-activation-checkpoint-integrity.sock'
    result = subprocess.run(['ssh', '-S', socket, '-o', 'BatchMode=yes',
            'aymanmahfuz@ls6.tacc.utexas.edu', 'python3', '-'],
            input=remote, capture_output=True, text=True, timeout=150)
    (args.record / 'stdout.txt').write_text(result.stdout)
    (args.record / 'stderr.txt').write_text(result.stderr)
    (args.record / 'returncode.txt').write_text(str(result.returncode) + '\n')
    print(result.stdout)
    print(result.stderr, file=sys.stderr)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
