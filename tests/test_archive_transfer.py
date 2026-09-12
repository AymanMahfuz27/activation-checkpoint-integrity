"""A complete, chunked round trip must preserve raw files and duplicate entries."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import threading
import time

from ac_integrity.archive_transfer import send


def test_lossless_archive_roundtrip_with_verified_chunk_retirement(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    payload = bytes(range(256)) * 100
    (root / "one.bin").write_bytes(payload)
    (root / "two.bin").write_bytes(payload)
    (root / "other.bin").write_bytes(b"different" * 100)
    inbox = tmp_path / "inbox"
    result = []
    errors = []
    def producer():
        try:
            result.append(send(root, inbox, chunk_bytes=128))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=producer, daemon=True)
    thread.start()
    chunks = []
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        complete = inbox / "complete.json"
        if complete.exists():
            record = json.loads(complete.read_text())
            archive = b"".join(chunks)
            assert hashlib.sha256(archive).hexdigest() == record["sha256"]
            (inbox / "verified").write_text(record["sha256"])
            break
        ready = inbox / "ready.json"
        if ready.exists():
            record = json.loads(ready.read_text())
            if record["index"] == len(chunks):
                chunk = (inbox / record["filename"]).read_bytes()
                assert hashlib.sha256(chunk).hexdigest() == record["sha256"]
                chunks.append(chunk)
                (inbox / f"ack_{record['index']:05d}").write_text(record["sha256"])
        time.sleep(0.02)
    thread.join(timeout=5)
    assert not thread.is_alive() and not errors and result
    assert len(chunks) > 1
    assert not list(inbox.glob("chunk_*.bin"))
    with tarfile.open(fileobj=io.BytesIO(b"".join(chunks)), mode="r:gz") as archive:
        assert archive.extractfile("one.bin").read() == payload
        assert archive.extractfile("two.bin").read() == payload
        assert archive.getmember("two.bin").islnk()
        assert archive.extractfile("other.bin").read() == b"different" * 100
    assert (root / "one.bin").read_bytes() == payload


def test_resume_verifies_prefix_and_rejects_corruption(tmp_path):
    import pytest
    from ac_integrity.archive_transfer import verify_prefix
    archive = tmp_path / "archive"
    archive.write_bytes(b"abcde")
    records = [{"index": i, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
               for i, data in enumerate((b"abc", b"de"))]
    total, digest = verify_prefix(archive, records)
    assert total == 5 and digest.hexdigest() == hashlib.sha256(b"abcde").hexdigest()
    for damaged in (b"abXde", b"abcd", b"abcdef"):
        archive.write_bytes(damaged)
        with pytest.raises(ValueError):
            verify_prefix(archive, records)
