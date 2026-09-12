"""Lossless, acknowledged transfer from scheduled scratch through a small inbox.

The sender keeps source artifacts until the complete archive is verified on the
laptop. Only temporary transfer chunks are retired after acknowledgement. This
avoids putting a full GPU trace in the UT account's 15 GiB home directory.
"""

import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time

from ac_integrity.state import sha256, write_json


def identical_files(left, right):
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            x, y = a.read(1024 * 1024), b.read(1024 * 1024)
            if x != y:
                return False
            if not x:
                return True


class ChunkSink(io.RawIOBase):
    def __init__(self, inbox, chunk_bytes=512 * 1024**2, timeout=12 * 3600):
        self.inbox = Path(inbox)
        self.inbox.mkdir(parents=True, exist_ok=False)
        self.chunk_bytes, self.timeout = chunk_bytes, timeout
        self.number, self.length, self.total = 0, 0, 0
        self.archive_digest = hashlib.sha256()
        self.digest = hashlib.sha256()
        self.file = (self.inbox / "chunk.partial").open("xb")
        self.records = []

    def writable(self):
        return True

    def write(self, data):
        view = memoryview(data)
        count = len(view)
        while view:
            size = min(len(view), self.chunk_bytes - self.length)
            piece = view[:size]
            self.file.write(piece)
            self.digest.update(piece)
            self.archive_digest.update(piece)
            self.length += size
            self.total += size
            view = view[size:]
            if self.length == self.chunk_bytes:
                self.publish()
        return count

    def wait_for(self, path, expected):
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            if path.exists() and path.read_text().strip() == expected:
                return
            time.sleep(2)
        raise TimeoutError(f"Waiting for verified transfer acknowledgement: {path}")

    def publish(self):
        self.file.flush()
        os.fsync(self.file.fileno())
        self.file.close()
        ready = self.inbox / f"chunk_{self.number:05d}.bin"
        (self.inbox / "chunk.partial").rename(ready)
        record = {"index": self.number, "bytes": self.length, "sha256": self.digest.hexdigest(),
                  "filename": ready.name}
        self.records.append(record)
        write_json(self.inbox / "ready.json", record)
        self.wait_for(self.inbox / f"ack_{self.number:05d}", record["sha256"])
        # The original source remains intact; this is a verified transfer buffer.
        ready.unlink()
        self.number += 1
        self.length = 0
        self.digest = hashlib.sha256()
        self.file = (self.inbox / "chunk.partial").open("xb")

    def finish(self):
        if self.length:
            self.publish()
        self.file.close()
        (self.inbox / "chunk.partial").unlink()
        complete = {"chunks": self.records, "bytes": self.total,
                    "sha256": self.archive_digest.hexdigest()}
        write_json(self.inbox / "complete.json", complete)
        self.wait_for(self.inbox / "verified", complete["sha256"])
        return complete


def send(root, inbox, chunk_bytes=512 * 1024**2):
    root = Path(root).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    manifest = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": sha256(p)} for p in files}
    write_json(root / "archive_manifest.json", manifest)
    files.append(root / "archive_manifest.json")
    sink = ChunkSink(inbox, chunk_bytes=chunk_bytes)
    seen = {}
    with gzip.GzipFile(fileobj=sink, mode="wb", compresslevel=1, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|") as archive:
            for path in files:
                relative = str(path.relative_to(root))
                digest = manifest.get(relative, {}).get("sha256") or sha256(path)
                candidates = seen.setdefault((path.stat().st_size, digest), [])
                previous = next((p for p in candidates if identical_files(p, path)), None)
                info = archive.gettarinfo(str(path), arcname=relative)
                if previous is not None:
                    info.type = tarfile.LNKTYPE
                    info.linkname = str(previous.relative_to(root))
                    info.size = 0
                    archive.addfile(info)
                else:
                    candidates.append(path)
                    with path.open("rb") as source:
                        archive.addfile(info, source)
    return sink.finish()


def receive(host, inbox, output, max_bytes):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inbox = str(Path(inbox))
    if not inbox.startswith("/u/ayman27/activation-checkpoint-integrity/artifacts/"):
        raise ValueError("Inbox must be an experiment transfer directory")
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host]
    def read_json(name):
        process = subprocess.run(ssh + ["cat " + shlex.quote(inbox + "/" + name)],
                                 capture_output=True, text=True, timeout=30)
        return json.loads(process.stdout) if process.returncode == 0 else None
    index, total = 0, 0
    digest = hashlib.sha256()
    deadline = time.monotonic() + 12 * 3600
    with (output / "evidence.tar.gz").open("xb") as target:
        while time.monotonic() < deadline:
            complete = read_json("complete.json")
            if complete and index == len(complete["chunks"]):
                if total != complete["bytes"] or digest.hexdigest() != complete["sha256"]:
                    raise ValueError("Complete archive checksum/length mismatch")
                target.flush()
                os.fsync(target.fileno())
                write_json(output / "verified.json", complete)
                acknowledgement = "printf '%s\\n' " + shlex.quote(complete["sha256"]) + " > " + shlex.quote(inbox + "/verified")
                subprocess.run(ssh + [acknowledgement], check=True, timeout=30)
                return complete
            ready = read_json("ready.json")
            if not ready or ready["index"] != index:
                time.sleep(3)
                continue
            filename = f"chunk_{index:05d}.bin"
            if ready["filename"] != filename:
                raise ValueError("Unexpected transfer chunk path")
            if total + ready["bytes"] > max_bytes or shutil.disk_usage(output).free < ready["bytes"] + 1024**3:
                raise OSError("Local archive budget exceeded; remote source is retained")
            temporary = output / "chunk.download"
            subprocess.run(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                            f"{host}:{inbox}/{filename}", str(temporary)], check=True, timeout=1800)
            if temporary.stat().st_size != ready["bytes"] or sha256(temporary) != ready["sha256"]:
                raise ValueError("Transfer chunk checksum/length mismatch")
            with temporary.open("rb") as part:
                while block := part.read(1024 * 1024):
                    target.write(block)
                    digest.update(block)
            target.flush()
            os.fsync(target.fileno())
            total += ready["bytes"]
            subprocess.run(ssh + ["printf '%s\\n' " + shlex.quote(ready["sha256"]) + " > "
                                  + shlex.quote(inbox + f"/ack_{index:05d}")], check=True, timeout=30)
            temporary.unlink()
            print(json.dumps({"verified_chunk": index, "archive_bytes": total}), flush=True)
            index += 1
    raise TimeoutError("Archive transfer did not complete; source remains on scheduled scratch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["send", "receive"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--inbox", required=True)
    parser.add_argument("--host", default="ayman27@darmok.cs.utexas.edu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-bytes", type=int, default=20_000_000_000)
    args = parser.parse_args()
    result = send(args.root, args.inbox) if args.action == "send" else receive(args.host, args.inbox, args.output, args.max_bytes)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
